# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Merge multiple Arena teleop HDF5 demonstration files into a single training-ready dataset.

Teleoperators often collect 100+ demonstrations across several sessions rather than in one sitting,
producing one HDF5 per session via ``submodules/IsaacLab/scripts/tools/record_demos.py``.
This script combines those files into a single dataset suitable for downstream consumers
(``submodules/IsaacLab/scripts/tools/replay_demos.py``,
``submodules/IsaacLab/scripts/imitation_learning/isaaclab_mimic/annotate_demos.py``,
``convert_hdf5_to_lerobot.py``).

Compared to upstream ``submodules/IsaacLab/scripts/tools/merge_hdf5_datasets.py``, this script:

- Preserves the root ``format_version`` attribute (otherwise the loader assumes legacy WXYZ
  quaternions and silently corrupts ``root_pose`` data).
- Recomputes ``data.attrs["total"]`` from the per-demo ``num_samples`` sum.
- Validates structural compatibility across inputs (``format_version``, dataset paths, per-key
  shapes/dtypes) and warns on ``env_args`` mismatches.
- Logs an operator-friendly per-file summary and aggregate report.
- Uses a recursive ``h5py.Group.copy`` so new recorder terms added by future Isaac Lab versions
  (new ``obs/*`` keys, new sensor groups, new metadata attrs) round-trip unchanged.
- With ``--success_only``, drops individual ``demo_*`` groups whose ``success`` attribute is
  ``False`` (or missing) while still copying the rest of that same file -- unlike pre-filtering
  the input file list (e.g. via ``filter_successful_demos.py``), this doesn't discard an entire
  file just because one of its demos failed.
- With one or more ``--drop_subtree PATH``, excludes those group paths from both schema
  validation and the merged output -- for recorder terms downstream consumers don't need and
  that may not be uniform across inputs (e.g. per-session teleop-console diagnostics).

The script has zero simulation dependency and only requires ``h5py``.

Example
-------
.. code-block:: bash

    python isaaclab_arena/scripts/imitation_learning/merge_demos.py \\
        -o $DATASET_DIR/combined.hdf5 \\
        $DATASET_DIR/session_a.hdf5 $DATASET_DIR/session_b.hdf5 $DATASET_DIR/session_c.hdf5
"""

from __future__ import annotations

import argparse
import contextlib
import h5py
import json
import os
import sys
from dataclasses import dataclass, field

_TABLE_SEP_CHAR = "-"


@dataclass
class _FileInfo:
    """Summary of one input HDF5 file, populated by :func:`_inspect_file`."""

    path: str
    format_version: int
    env_args: dict
    num_demos: int
    total_steps: int
    size_bytes: int
    schema_fingerprint: dict[str, tuple[tuple[int, ...], str]] = field(default_factory=dict)
    success_count: int = 0
    no_success_attr_count: int = 0
    failed_count: int = 0
    total_steps_successful: int = 0
    """Sum of ``num_samples`` across only the demos with ``success=True`` -- what ``--success_only``
    would keep, used for the dry-run preview."""
    untracked_step_demos: list[str] = field(default_factory=list)


def _format_bytes(num_bytes: float) -> str:
    """Format a byte count as a human-readable string."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PiB"


def _format_int(n: int) -> str:
    """Format an integer with thousands separators."""
    return f"{n:,}"


def _matches_dropped_subtree(path: str, drop_subtrees: list[str]) -> bool:
    """True if ``path`` is at or under one of ``drop_subtrees`` (``/``-separated group paths)."""
    return any(path == subtree or path.startswith(subtree + "/") for subtree in drop_subtrees)


def _build_schema_fingerprint(
    demo_group: h5py.Group, drop_subtrees: list[str] | None = None
) -> dict[str, tuple[tuple[int, ...], str]]:
    """Walk a demo group recursively and produce a ``path -> (shape[1:], dtype_str)`` map.

    The leading time dimension is dropped because episode lengths legitimately vary across demos.
    Everything else (action_dim, obs feature dim, image HxWxC, ...) must match for the merged
    dataset to be usable by downstream consumers. Paths under ``drop_subtrees`` are excluded, since
    ``--drop_subtree`` removes them from the merged output anyway -- their compatibility across
    inputs doesn't matter.
    """
    drop_subtrees = drop_subtrees or []
    fingerprint: dict[str, tuple[tuple[int, ...], str]] = {}

    def _visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset) and not _matches_dropped_subtree(name, drop_subtrees):
            fingerprint[name] = (tuple(obj.shape[1:]), obj.dtype.str)

    demo_group.visititems(_visit)
    return fingerprint


def _sorted_demo_names(data_group: h5py.Group) -> list[str]:
    """Return ``demo_*`` group names sorted by numeric suffix when possible."""
    demo_names = [k for k in data_group.keys() if k.startswith("demo_")]

    def _key(name: str) -> tuple[int, str]:
        suffix = name[len("demo_") :]
        if suffix.isdigit():
            return (0, f"{int(suffix):020d}")
        return (1, name)

    return sorted(demo_names, key=_key)


def _demo_is_successful(demo_attrs) -> bool:
    """True if a demo's ``success`` attribute is set and it has at least one recorded sample.

    A demo can carry success=True with num_samples=0 (an empty/truncated recording mistakenly
    tagged successful, confirmed 2026-08-14 against a real dataset) -- treat that as
    unsuccessful too, since there's no data to merge regardless of the flag.
    """
    if "success" not in demo_attrs or not bool(demo_attrs["success"]):
        return False
    return int(demo_attrs.get("num_samples", 1)) > 0


def _inspect_file(path: str, drop_subtrees: list[str] | None = None) -> _FileInfo:
    """Open an input HDF5 file read-only and build a :class:`_FileInfo` summary."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file does not exist: {path}")
    size_bytes = os.path.getsize(path)

    # Prepend the path to h5py's bare OSError on non-HDF5/truncated/locked inputs.
    try:
        h5_file = h5py.File(path, "r")
    except OSError as e:
        raise ValueError(f"{path}: cannot open as HDF5 ({e})") from e

    with h5_file as f:
        if "data" not in f:
            raise ValueError(f"{path}: missing top-level 'data' group; not a record_demos HDF5 file")

        format_version = int(f.attrs["format_version"]) if "format_version" in f.attrs else 0

        data_group = f["data"]
        env_args: dict = {}
        raw_env_args = data_group.attrs.get("env_args")
        if raw_env_args is not None:
            try:
                env_args = json.loads(raw_env_args)
            except (json.JSONDecodeError, TypeError):
                env_args = {"_raw": str(raw_env_args)}

        demo_names = _sorted_demo_names(data_group)
        num_demos = len(demo_names)
        if num_demos == 0:
            raise ValueError(f"{path}: contains no 'demo_*' groups; nothing to merge")

        # Sample the schema from the first demo only. record_demos.py writes a uniform
        # recorder stack within a single file, so this is the fast path. Intra-file
        # inconsistency (e.g. a file that was manually edited mid-session) is not detected
        # here; cross-file consistency is what _validate_compatibility checks.
        schema_fingerprint = _build_schema_fingerprint(data_group[demo_names[0]], drop_subtrees)

        total_steps = 0
        success_count = 0
        no_success_attr_count = 0
        failed_count = 0
        total_steps_successful = 0
        untracked_step_demos: list[str] = []
        for name in demo_names:
            demo = data_group[name]
            if "num_samples" in demo.attrs:
                demo_steps = int(demo.attrs["num_samples"])
            elif "actions" in demo:
                demo_steps = int(demo["actions"].shape[0])
            else:
                demo_steps = 0
                untracked_step_demos.append(name)
            total_steps += demo_steps

            has_success_attr = "success" in demo.attrs
            # A demo can carry success=True with num_samples=0 (an empty/truncated recording
            # mistakenly tagged successful, confirmed 2026-08-14 against a real dataset) --
            # treat that as failed too, since there's no data to merge regardless of the flag.
            is_success = has_success_attr and bool(demo.attrs["success"]) and demo_steps > 0
            if has_success_attr:
                if is_success:
                    success_count += 1
                else:
                    failed_count += 1
            else:
                no_success_attr_count += 1
            if is_success:
                total_steps_successful += demo_steps

    return _FileInfo(
        path=path,
        format_version=format_version,
        env_args=env_args,
        num_demos=num_demos,
        total_steps=total_steps,
        size_bytes=size_bytes,
        schema_fingerprint=schema_fingerprint,
        success_count=success_count,
        no_success_attr_count=no_success_attr_count,
        failed_count=failed_count,
        total_steps_successful=total_steps_successful,
        untracked_step_demos=untracked_step_demos,
    )


def _diff_schema(
    ref: dict[str, tuple[tuple[int, ...], str]], other: dict[str, tuple[tuple[int, ...], str]]
) -> list[str]:
    """Return human-readable list of differences between two schema fingerprints."""
    out: list[str] = []
    ref_keys = set(ref.keys())
    other_keys = set(other.keys())
    for k in sorted(ref_keys - other_keys):
        out.append(f"only in first: {k} (shape={ref[k][0]}, dtype={ref[k][1]})")
    for k in sorted(other_keys - ref_keys):
        out.append(f"only in other: {k} (shape={other[k][0]}, dtype={other[k][1]})")
    for k in sorted(ref_keys & other_keys):
        if ref[k] != other[k]:
            out.append(
                f"shape/dtype differs at {k}: shape={ref[k][0]} dtype={ref[k][1]}"
                f" vs shape={other[k][0]} dtype={other[k][1]}"
            )
    return out


@dataclass
class _ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    schema_status: str = "OK"
    format_version_status: str = "OK"
    env_args_status: str = "OK"


def _validate_compatibility(infos: list[_FileInfo], success_only: bool = False) -> _ValidationReport:
    """Compare input files for compatibility and produce a :class:`_ValidationReport`."""
    report = _ValidationReport()

    versions = {i.format_version for i in infos}
    if len(versions) > 1:
        report.format_version_status = "MISMATCH"
        report.errors.append(
            f"format_version mismatch across inputs: {sorted(versions)}. Legacy WXYZ files "
            "must be converted first via HDF5DatasetFileHandler.convert_dataset_to_xyzw()."
        )

    if len(infos) > 1:
        ref = infos[0]
        any_schema_diff = False
        for other in infos[1:]:
            diffs = _diff_schema(ref.schema_fingerprint, other.schema_fingerprint)
            if not diffs:
                continue
            any_schema_diff = True
            msg = f"Schema mismatch between {ref.path} and {other.path}:\n  " + "\n  ".join(diffs)
            report.errors.append(msg)
        if any_schema_diff:
            report.schema_status = "MISMATCH"

    env_names = {i.env_args.get("env_name", "") for i in infos}
    sim_args_set = {json.dumps(i.env_args.get("sim_args", {}), sort_keys=True) for i in infos}
    if len(env_names) > 1:
        report.env_args_status = "WARN"
        report.warnings.append(
            f"env_args.env_name differs across inputs: {sorted(env_names)}. "
            "record_demos.py typically writes an empty env_name, so this is often expected."
        )
    if len(sim_args_set) > 1:
        report.env_args_status = "WARN"
        report.warnings.append(
            "env_args.sim_args differ across inputs (e.g. dt, decimation, render_interval, "
            "num_envs). The first file's values will be written to the merged output."
        )

    for i in infos:
        if i.failed_count > 0:
            disposition = "will be dropped (--success_only)" if success_only else "included as-is in the merged file"
            report.warnings.append(
                f"{i.path}: {i.failed_count} demo(s) with success=False -- {disposition}."
            )
        if i.no_success_attr_count > 0:
            disposition = "will be dropped (--success_only)" if success_only else "included as-is, legacy format"
            report.info.append(
                f"{i.path}: {i.no_success_attr_count} demo(s) without @success attribute -- {disposition}."
            )
        if i.untracked_step_demos:
            shown = ", ".join(i.untracked_step_demos[:3])
            suffix = ", ..." if len(i.untracked_step_demos) > 3 else ""
            report.warnings.append(
                f"{i.path}: {len(i.untracked_step_demos)} demo(s) without 'num_samples' attribute"
                f" or 'actions' dataset ({shown}{suffix}); contributed 0 to the reported step total."
            )

    return report


def _print_summary(
    infos: list[_FileInfo],
    output_path: str,
    *,
    output_total_steps: int | None = None,
    output_num_demos: int | None = None,
    output_size_bytes: int | None = None,
    report: _ValidationReport | None = None,
    dry_run: bool = False,
) -> None:
    """Print an operator-friendly per-file table, aggregate row, and validation summary."""
    n = len(infos)
    if n == 0:
        return

    max_name_len = max(len(os.path.basename(i.path)) for i in infos)
    max_name_len = max(max_name_len, len(os.path.basename(output_path)) + len(" (output)"), 25)

    print()
    for idx, i in enumerate(infos, start=1):
        env_name = i.env_args.get("env_name", "")
        print(
            f"[{idx}/{n}] {os.path.basename(i.path):<{max_name_len}}  "
            f"demos={i.num_demos:>4d}  "
            f"steps={_format_int(i.total_steps):>10s}  "
            f"size={_format_bytes(i.size_bytes):>10s}  "
            f'env="{env_name}"  '
            f"v={i.format_version}  "
            f"keys={len(i.schema_fingerprint)}"
        )

    print(_TABLE_SEP_CHAR * (max_name_len + 65))

    total_demos = output_num_demos if output_num_demos is not None else sum(i.num_demos for i in infos)
    total_steps = output_total_steps if output_total_steps is not None else sum(i.total_steps for i in infos)
    if output_size_bytes is not None:
        size_str = _format_bytes(output_size_bytes)
    else:
        size_str = "~" + _format_bytes(sum(i.size_bytes for i in infos))

    label = f"{os.path.basename(output_path)} ({'dry-run' if dry_run else 'output'})"
    print(
        f"        {label:<{max_name_len}}  "
        f"demos={total_demos:>4d}  "
        f"steps={_format_int(total_steps):>10s}  "
        f"size={size_str:>10s}"
    )

    if report is not None:
        print(
            "Validation: "
            f"format_version {report.format_version_status}, "
            f"schema {report.schema_status}, "
            f"env_args {report.env_args_status}"
        )
        for msg in report.info:
            print(f"INFO: {msg}")
        for msg in report.warnings:
            print(f"WARNING: {msg}")
        for msg in report.errors:
            print(f"ERROR: {msg}")

    if total_demos > 0:
        print(f"Demo numbering: demo_0..demo_{total_demos - 1} (input order preserved)")
    print()


def _merge(
    infos: list[_FileInfo],
    output_path: str,
    success_only: bool = False,
    drop_subtrees: list[str] | None = None,
) -> tuple[int, int, int]:
    """Write a merged HDF5 dataset from validated inputs.

    Args:
        infos: Per-input-file summaries from :func:`_inspect_file`.
        output_path: Path to write the merged HDF5 dataset to.
        success_only: Skip demos whose ``success`` attribute is ``False`` or missing.
        drop_subtrees: ``/``-separated group paths (relative to each demo group) to remove from
            every copied demo, e.g. ``["checkpoints", "obs/raw_action"]``.

    Returns:
        A tuple ``(output_size_bytes, total_steps_written, total_demos_written)``.
    """
    drop_subtrees = drop_subtrees or []
    format_version = infos[0].format_version
    merged_env_args = dict(infos[0].env_args)
    for other in infos[1:]:
        for k, v in other.env_args.items():
            if k not in merged_env_args:
                merged_env_args[k] = v

    total_demos_written = 0
    total_steps_written = 0

    with h5py.File(output_path, "w") as out:
        out.attrs["format_version"] = format_version
        data_out = out.create_group("data")
        data_out.attrs["env_args"] = json.dumps(merged_env_args)

        for info in infos:
            with h5py.File(info.path, "r") as src:
                src_data = src["data"]
                for src_demo_name in _sorted_demo_names(src_data):
                    src_demo_attrs = src_data[src_demo_name].attrs
                    if success_only and not _demo_is_successful(src_demo_attrs):
                        continue
                    dst_demo_name = f"demo_{total_demos_written}"
                    try:
                        src.copy(src_data[src_demo_name], data_out, name=dst_demo_name)
                    except (OSError, RuntimeError, ValueError) as e:
                        # h5py error messages from deep recursive copies are notoriously opaque;
                        # surface the offending source file and demo so the operator can
                        # repair / drop the bad input without guessing.
                        raise RuntimeError(
                            f"Failed to copy {info.path}::{src_demo_name} into the merged"
                            f" dataset as {dst_demo_name}: {e}"
                        ) from e
                    dst_demo = data_out[dst_demo_name]
                    for subtree in drop_subtrees:
                        if subtree in dst_demo:
                            del dst_demo[subtree]
                    if "num_samples" in dst_demo.attrs:
                        total_steps_written += int(dst_demo.attrs["num_samples"])
                    elif "actions" in dst_demo:
                        total_steps_written += int(dst_demo["actions"].shape[0])
                    total_demos_written += 1

        data_out.attrs["total"] = total_steps_written

    output_size = os.path.getsize(output_path)
    return output_size, total_steps_written, total_demos_written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple Arena teleop HDF5 demo files into a single training-ready dataset."
            " Validates schema compatibility across inputs, renumbers demos sequentially, and"
            " preserves all recorder data (forward-compatible to new recorder terms)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        type=str,
        help="One or more input HDF5 demo files to merge. Files are merged in the given order.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=True,
        help="Path to write the merged HDF5 dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate inputs and print the merge report without writing the output file.",
    )
    parser.add_argument(
        "--success_only",
        action="store_true",
        help=(
            "Drop individual demo_* groups whose success attribute is False or missing, instead of"
            " copying every demo from every input file. Unlike pre-filtering the input file list,"
            " this keeps the successful demos from a file that also contains failed ones."
        ),
    )
    parser.add_argument(
        "--drop_subtree",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "A '/'-separated group path (relative to each demo group, e.g. 'checkpoints' or"
            " 'obs/raw_action') to exclude from both schema validation and the merged output."
            " Repeatable. Use for recorder terms downstream consumers don't need and that may not"
            " be uniform across inputs."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code (0 success, non-zero on error)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_abs = os.path.abspath(args.output_file)
    tmp_abs = output_abs + ".tmp"
    for inp in args.input_files:
        if not os.path.exists(inp):
            continue
        inp_abs = os.path.abspath(inp)
        if inp_abs == output_abs:
            print(
                f"ERROR: --output_file path equals an input path: {inp}",
                file=sys.stderr,
            )
            return 2
        if inp_abs == tmp_abs:
            print(
                f"ERROR: input path {inp} collides with the internal temp path "
                f"{args.output_file}.tmp. Rename the input or choose a different --output_file.",
                file=sys.stderr,
            )
            return 2
    if not args.dry_run and os.path.exists(args.output_file) and not args.overwrite:
        print(
            f"ERROR: output file {args.output_file} already exists. Pass --overwrite to replace.",
            file=sys.stderr,
        )
        return 2

    infos: list[_FileInfo] = []
    for path in args.input_files:
        try:
            infos.append(_inspect_file(path, drop_subtrees=args.drop_subtree))
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    report = _validate_compatibility(infos, success_only=args.success_only)

    if args.dry_run:
        if args.success_only:
            preview_num_demos = sum(i.success_count for i in infos)
            preview_total_steps = sum(i.total_steps_successful for i in infos)
        else:
            preview_num_demos = None
            preview_total_steps = None
        _print_summary(
            infos,
            args.output_file,
            output_num_demos=preview_num_demos,
            output_total_steps=preview_total_steps,
            report=report,
            dry_run=True,
        )
        return 1 if report.errors else 0

    if report.errors:
        _print_summary(infos, args.output_file, report=report, dry_run=True)
        print(
            "Merge aborted due to validation errors. Re-run with --dry_run to inspect.",
            file=sys.stderr,
        )
        return 1

    output_dir = os.path.dirname(output_abs) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Atomic write: stream into <output>.tmp and rename on success so a mid-merge
    # failure does not leave a partial output or, under --overwrite, destroy the
    # prior file.
    tmp_path = args.output_file + ".tmp"
    success = False
    try:
        output_size, total_steps, total_demos = _merge(
            infos, tmp_path, success_only=args.success_only, drop_subtrees=args.drop_subtree
        )
        os.replace(tmp_path, args.output_file)
        success = True
    except Exception as e:
        print(f"ERROR: Merge failed: {e}", file=sys.stderr)
        return 1
    finally:
        if not success and os.path.exists(tmp_path):
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

    _print_summary(
        infos,
        args.output_file,
        output_total_steps=total_steps,
        output_num_demos=total_demos,
        output_size_bytes=output_size,
        report=report,
        dry_run=False,
    )
    print(f"Merged dataset written to: {args.output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

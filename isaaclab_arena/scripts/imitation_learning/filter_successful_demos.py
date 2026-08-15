# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""List which ``record_demos.py``-style HDF5 files have only successful demos.

Datasets recorded via teleoperation often include aborted or incomplete takes alongside
successful ones, each tagged via the per-demo ``success`` attribute Isaac Lab's
``RecorderManager`` already writes. Feeding a mix of successful and failed demos into
imitation learning is generally undesirable, so this script reports, per input file, how many
of its ``demo_*`` groups are ``success=True`` -- and optionally writes the paths of
fully-successful files to an output file, for use as ``merge_demos.py``'s input file list.

The script has zero simulation dependency and only requires ``h5py``.

Example
-------
.. code-block:: bash

    python isaaclab_arena/scripts/imitation_learning/filter_successful_demos.py \\
        -o $DATASET_DIR/successful_demos.txt \\
        $DATASET_DIR/data/*.hdf5
"""

from __future__ import annotations

import argparse
import h5py
import os
import sys
from dataclasses import dataclass


@dataclass
class _FileSuccessInfo:
    """Per-file demo/success counts, populated by :func:`_inspect_file`."""

    path: str
    num_demos: int = 0
    num_successful: int = 0
    num_failed: int = 0
    num_no_success_attr: int = 0

    @property
    def all_successful(self) -> bool:
        return self.num_demos > 0 and self.num_successful == self.num_demos


def _inspect_file(path: str) -> _FileSuccessInfo:
    """Open an input HDF5 file read-only and count each ``demo_*`` group's ``success`` attribute."""
    try:
        h5_file = h5py.File(path, "r")
    except OSError as e:
        raise ValueError(f"{path}: cannot open as HDF5 ({e})") from e

    info = _FileSuccessInfo(path=path)
    with h5_file as f:
        if "data" not in f:
            raise ValueError(f"{path}: missing top-level 'data' group; not a record_demos HDF5 file")
        for demo_name in f["data"].keys():
            if not demo_name.startswith("demo_"):
                continue
            info.num_demos += 1
            demo = f["data"][demo_name]
            success = demo.attrs.get("success", None)
            # A demo can carry success=True with num_samples=0 (an empty/truncated recording
            # mistakenly tagged successful, confirmed 2026-08-14 against a real dataset) --
            # treat that as failed too, since there's no data to convert regardless of the flag.
            has_samples = int(demo.attrs.get("num_samples", 1)) > 0
            if success is None:
                info.num_no_success_attr += 1
            elif bool(success) and has_samples:
                info.num_successful += 1
            else:
                info.num_failed += 1
    return info


def _print_summary(infos: list[_FileSuccessInfo]) -> None:
    """Print an operator-friendly per-file table and aggregate counts."""
    max_name_len = max((len(os.path.basename(i.path)) for i in infos), default=25)

    print()
    for i in infos:
        flags = []
        if i.num_demos > 1:
            flags.append(f"{i.num_demos} demos")
        if i.num_failed:
            flags.append(f"{i.num_failed} failed")
        if i.num_no_success_attr:
            flags.append(f"{i.num_no_success_attr} no-success-attr")
        status = "OK" if i.all_successful else "SKIP"
        suffix = f"  ({', '.join(flags)})" if flags else ""
        print(f"[{status:<4}] {os.path.basename(i.path):<{max_name_len}}{suffix}")

    total_files = len(infos)
    qualifying = sum(1 for i in infos if i.all_successful)
    total_demos = sum(i.num_demos for i in infos)
    total_successful = sum(i.num_successful for i in infos)
    total_failed = sum(i.num_failed for i in infos)
    captured_successful = sum(i.num_successful for i in infos if i.all_successful)
    print()
    print(
        f"{qualifying}/{total_files} files fully successful "
        f"({total_successful}/{total_demos} demos successful, {total_failed} failed)"
    )
    if captured_successful != total_successful:
        print(
            f"NOTE: whole-file filtering captures {captured_successful}/{total_successful} successful"
            " demos -- the rest sit in otherwise-mixed files (some demos successful, some not) and are"
            " excluded along with their file, since merge_demos.py copies all demo_* groups from a file"
            " it's given, not a chosen subset."
        )
    print()


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code (0 success, non-zero on error)."""
    parser = argparse.ArgumentParser(
        description=(
            "Report which record_demos.py-style HDF5 files have only successful demos, and"
            " optionally write the list of fully-successful file paths for use as merge_demos.py's"
            " input file list."
        )
    )
    parser.add_argument("input_files", nargs="+", type=str, help="HDF5 files to inspect.")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default=None,
        help="If given, write the newline-separated paths of fully-successful files here.",
    )
    args = parser.parse_args(argv)

    infos: list[_FileSuccessInfo] = []
    for path in args.input_files:
        if not os.path.exists(path):
            print(f"ERROR: input file does not exist: {path}", file=sys.stderr)
            return 2
        try:
            infos.append(_inspect_file(path))
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    _print_summary(infos)

    if args.output_file:
        qualifying_paths = [i.path for i in infos if i.all_successful]
        with open(args.output_file, "w") as f:
            f.write("\n".join(qualifying_paths) + "\n")
        print(f"Wrote {len(qualifying_paths)} paths to: {args.output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

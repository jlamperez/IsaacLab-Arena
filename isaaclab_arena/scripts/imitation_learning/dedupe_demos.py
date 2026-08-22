# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Write a deduplicated copy of a ``record_demos.py``-style HDF5 data folder.

Companion to ``find_duplicate_demos.py``: groups demos by an MD5 hash of their ``actions``
dataset (bit-identical content -> same group), keeps exactly one demo per group -- the
first, in (sorted file, sorted demo name) order -- and writes it to a new output directory,
one HDF5 file per source file that still has at least one kept demo. Files whose every demo
was a duplicate of an earlier file's demo are skipped entirely.

Each output file mirrors its source file's schema exactly (``format_version`` root attr,
``env_args`` data-group attr, per-demo groups copied via ``h5py.Group.copy`` -- same
approach ``merge_demos.py`` uses), so every downstream consumer (``filter_successful_demos.py``,
``merge_demos.py``, ``convert_hdf5_to_lerobot.py``) works against the output folder unchanged,
just like the original ``data/`` folder.

Only a demo's own file is dropped from the output when *none* of its demos were kept; a file
with a mix of kept and duplicate demos keeps only the kept ones (the ``data`` group's ``total``
attr is recomputed to match).

The script has zero simulation dependency and only requires ``h5py``/``numpy``.

Example
-------
.. code-block:: bash

    python isaaclab_arena/scripts/imitation_learning/dedupe_demos.py \\
        -o $DATASET_DIR/data_deduplicated \\
        $DATASET_DIR/data/*.hdf5
"""

from __future__ import annotations

import argparse
import h5py
import numpy as np
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class _DemoRef:
    """One ``demo_*`` group's location."""

    file_path: str
    demo_name: str

    def __str__(self) -> str:
        return f"{os.path.basename(self.file_path)}::{self.demo_name}"


def _content_hash(actions: np.ndarray) -> str:
    """MD5 of an ``actions`` array's raw bytes, dtype-normalized so equal values always match."""
    import hashlib

    return hashlib.md5(np.ascontiguousarray(actions, dtype=np.float64).tobytes()).hexdigest()


@dataclass
class _DedupePlan:
    """Which demos to keep per source file, and how many duplicates were dropped."""

    kept_demos_by_file: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    num_groups: int = 0
    num_input_demos: int = 0

    @property
    def num_kept_demos(self) -> int:
        return sum(len(v) for v in self.kept_demos_by_file.values())

    @property
    def num_dropped_duplicates(self) -> int:
        return self.num_input_demos - self.num_kept_demos


def _build_dedupe_plan(paths: list[str]) -> _DedupePlan:
    """Group demos by content hash, keeping the first member of each group (sorted order)."""
    groups: dict[str, list[_DemoRef]] = defaultdict(list)
    for path in sorted(paths):
        with h5py.File(path, "r") as f:
            if "data" not in f:
                raise ValueError(f"{path}: missing top-level 'data' group; not a record_demos HDF5 file")
            for demo_name in sorted(f["data"].keys()):
                if not demo_name.startswith("demo_"):
                    continue
                demo = f["data"][demo_name]
                if not bool(demo.attrs.get("success", False)) or int(demo.attrs.get("num_samples", 0)) <= 0:
                    continue
                h = _content_hash(demo["actions"][:])
                groups[h].append(_DemoRef(file_path=path, demo_name=demo_name))

    plan = _DedupePlan(num_groups=len(groups), num_input_demos=sum(len(g) for g in groups.values()))
    for group in groups.values():
        canonical = group[0]  # first in (sorted file, sorted demo name) order
        plan.kept_demos_by_file[canonical.file_path].append(canonical.demo_name)
    return plan


def _write_deduplicated_file(src_path: str, kept_demo_names: list[str], dst_path: str) -> int:
    """Write one output file containing only ``kept_demo_names`` from ``src_path``.

    Returns:
        Total ``num_samples`` across the kept demos, for the ``data`` group's ``total`` attr.
    """
    total_samples = 0
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as out:
        if "format_version" in src.attrs:
            out.attrs["format_version"] = src.attrs["format_version"]
        src_data = src["data"]
        data_out = out.create_group("data")
        if "env_args" in src_data.attrs:
            data_out.attrs["env_args"] = src_data.attrs["env_args"]
        for demo_name in sorted(kept_demo_names):
            src.copy(src_data[demo_name], data_out, name=demo_name)
            total_samples += int(data_out[demo_name].attrs.get("num_samples", 0))
        data_out.attrs["total"] = total_samples
    return total_samples


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code (0 success, non-zero on error)."""
    parser = argparse.ArgumentParser(
        description=(
            "Write a deduplicated copy of a record_demos.py-style HDF5 data folder: one demo per"
            " unique 'actions' content, dropping byte-identical duplicates."
        )
    )
    parser.add_argument("input_files", nargs="+", type=str, help="HDF5 files to deduplicate.")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write the deduplicated HDF5 files to. Created if it doesn't exist.",
    )
    args = parser.parse_args(argv)

    for path in args.input_files:
        if not os.path.exists(path):
            print(f"ERROR: input file does not exist: {path}", file=sys.stderr)
            return 2

    try:
        plan = _build_dedupe_plan(args.input_files)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    os.makedirs(args.output_dir, exist_ok=True)

    files_written = 0
    for src_path, kept_demo_names in plan.kept_demos_by_file.items():
        dst_path = os.path.join(args.output_dir, os.path.basename(src_path))
        _write_deduplicated_file(src_path, kept_demo_names, dst_path)
        files_written += 1

    print()
    print(f"{plan.num_input_demos} input demos -> {plan.num_groups} unique-content groups")
    print(f"Wrote {files_written} files to {args.output_dir} ({plan.num_kept_demos} demos kept)")
    print(f"Dropped {plan.num_dropped_duplicates} duplicate demos (kept the first occurrence of each)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

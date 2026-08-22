# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Find byte-identical duplicate demos across ``record_demos.py``-style HDF5 files.

Built for the ``LightwheelAI/iros2026-ikea-assembly`` dataset: found 2026-08-16 that 250 of
its 300 "successful" demos are exact duplicates of one another (up to 8 copies of the same
recording under different file timestamps), so only 114 are genuinely distinct. Training on
the raw, undeduplicated set silently overweights whichever recordings happen to be repeated,
without adding any real variety.

Groups demos by an MD5 hash of their ``actions`` dataset (bit-identical content -> same
group). The script has zero simulation dependency and only requires ``h5py``/``numpy``.

Example
-------
.. code-block:: bash

    python isaaclab_arena/scripts/imitation_learning/find_duplicate_demos.py \\
        -o $DATASET_DIR/duplicate_groups.txt \\
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
    """One ``demo_*`` group's location, for display purposes."""

    file_path: str
    demo_name: str

    def __str__(self) -> str:
        return f"{os.path.basename(self.file_path)}::{self.demo_name}"


@dataclass
class _DuplicateReport:
    """Grouping result, populated by :func:`_find_duplicate_groups`."""

    groups_by_hash: dict[str, list[_DemoRef]] = field(default_factory=dict)

    @property
    def duplicate_groups(self) -> list[list[_DemoRef]]:
        """Groups with 2+ members, largest first."""
        groups = [g for g in self.groups_by_hash.values() if len(g) > 1]
        return sorted(groups, key=lambda g: -len(g))

    @property
    def singleton_groups(self) -> list[list[_DemoRef]]:
        """Groups with exactly 1 member (no duplicate anywhere in the input set)."""
        return [g for g in self.groups_by_hash.values() if len(g) == 1]

    @property
    def total_demos(self) -> int:
        return sum(len(g) for g in self.groups_by_hash.values())


def _content_hash(actions: np.ndarray) -> str:
    """MD5 of an ``actions`` array's raw bytes, dtype-normalized so equal values always match."""
    import hashlib

    return hashlib.md5(np.ascontiguousarray(actions, dtype=np.float64).tobytes()).hexdigest()


def _find_duplicate_groups(paths: list[str]) -> _DuplicateReport:
    """Hash each successful demo's ``actions`` array and group identical ones together."""
    groups: dict[str, list[_DemoRef]] = defaultdict(list)
    for path in paths:
        with h5py.File(path, "r") as f:
            if "data" not in f:
                raise ValueError(f"{path}: missing top-level 'data' group; not a record_demos HDF5 file")
            for demo_name in f["data"].keys():
                if not demo_name.startswith("demo_"):
                    continue
                demo = f["data"][demo_name]
                # Same success/num_samples filter as filter_successful_demos.py, so the two
                # scripts agree on what counts as a real demo.
                if not bool(demo.attrs.get("success", False)) or int(demo.attrs.get("num_samples", 0)) <= 0:
                    continue
                h = _content_hash(demo["actions"][:])
                groups[h].append(_DemoRef(file_path=path, demo_name=demo_name))
    return _DuplicateReport(groups_by_hash=dict(groups))


def _print_summary(report: _DuplicateReport) -> None:
    """Print an operator-friendly duplicate-group listing and aggregate counts."""
    dup_groups = report.duplicate_groups
    singletons = report.singleton_groups
    total = report.total_demos
    unique_groups = len(dup_groups) + len(singletons)

    print()
    print(f"{total} demos total, {unique_groups} unique-content groups")
    print(f"  {len(dup_groups)} groups have 2+ identical copies ({sum(len(g) for g in dup_groups)} demos)")
    print(f"  {len(singletons)} groups are singletons (no duplicate anywhere)")
    print()

    if dup_groups:
        print("=== Duplicate groups (2+ identical copies), largest first ===")
        for i, group in enumerate(dup_groups):
            print(f"\nGroup {i + 1} ({len(group)} copies):")
            for ref in group:
                print(f"  {ref}")
    print()


def _write_report_file(report: _DuplicateReport, output_file: str) -> None:
    """Write the full grouping (duplicates and singletons) to a text file for later use."""
    with open(output_file, "w") as f:
        dup_groups = report.duplicate_groups
        singletons = report.singleton_groups
        f.write(
            f"{len(dup_groups)} duplicate groups ({sum(len(g) for g in dup_groups)} demos), "
            f"{len(singletons)} singleton (truly unique) groups\n\n"
        )
        f.write("=== DUPLICATE GROUPS (2+ identical copies), sorted by group size ===\n\n")
        for i, group in enumerate(dup_groups):
            f.write(f"Group {i + 1} ({len(group)} copies):\n")
            for ref in group:
                f.write(f"  {ref}\n")
            f.write("\n")
        f.write("=== SINGLETON GROUPS (truly unique, no duplicate anywhere) ===\n\n")
        for group in singletons:
            f.write(f"  {group[0]}\n")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code (0 success, non-zero on error)."""
    parser = argparse.ArgumentParser(
        description=(
            "Find byte-identical duplicate demos (by their 'actions' dataset content) across"
            " record_demos.py-style HDF5 files, and report them grouped by content."
        )
    )
    parser.add_argument("input_files", nargs="+", type=str, help="HDF5 files to inspect.")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default=None,
        help="If given, write the full duplicate/singleton group listing here.",
    )
    args = parser.parse_args(argv)

    for path in args.input_files:
        if not os.path.exists(path):
            print(f"ERROR: input file does not exist: {path}", file=sys.stderr)
            return 2

    try:
        report = _find_duplicate_groups(args.input_files)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    _print_summary(report)

    if args.output_file:
        _write_report_file(report, args.output_file)
        print(f"Wrote full group listing to: {args.output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

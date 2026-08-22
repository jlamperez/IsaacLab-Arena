# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Copy ``record_demos.py``-style HDF5 files into per-strategy subfolders, for manual review.

Companion to ``classify_hand_strategy.py``: reuses the same first-leg hand-strategy
classification (``right_only`` / ``left_only`` / ``both`` / ``neither`` / ``too_short``, see that
script's module docstring for the exact definitions and the gripper-column convention), but
instead of just printing a report, physically copies each input file into
``<output_dir>/<strategy>/`` -- so a human can browse one strategy at a time in an HDF5 viewer
and hand-pick a clean seed set (e.g. for Mimic data generation) rather than trusting the
classification blindly.

Assumes ~1 demo per input file (true of ``dedupe_demos.py``'s output, which this is meant to run
against). If a file has multiple demos with *different* classifications, the whole file is
copied under its first demo's classification -- a warning is printed so this doesn't happen
silently.

Copies files, never moves or deletes -- the input directory is left untouched.

The script has zero simulation dependency and only requires ``h5py``/``numpy``.

Example
-------
.. code-block:: bash

    python isaaclab_arena/scripts/imitation_learning/split_by_strategy.py \\
        -o $DATASET_DIR/by_strategy \\
        $DATASET_DIR/data_deduplicated/*.hdf5
"""

from __future__ import annotations

import argparse
import h5py
import numpy as np
import os
import shutil
import sys


def _classify_demo(
    actions: np.ndarray, window_start: int, window_end: int
) -> str:
    """Classify a single demo's first-leg strategy. Same logic as classify_hand_strategy.py."""
    if actions.shape[0] < window_end:
        return "too_short"
    window = actions[window_start:window_end]
    left_closed = (window[:, 0] > 0.5).any()
    right_closed = (window[:, 1] > 0.5).any()
    if left_closed and right_closed:
        return "both"
    elif right_closed:
        return "right_only"
    elif left_closed:
        return "left_only"
    return "neither"


def _classify_file(path: str, window_start: int, window_end: int) -> str | None:
    """Return the file's strategy (its first successful demo's classification), or None if it has none."""
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise ValueError(f"{path}: missing top-level 'data' group; not a record_demos HDF5 file")
        demo_names = sorted(k for k in f["data"].keys() if k.startswith("demo_"))
        classifications = []
        for demo_name in demo_names:
            demo = f["data"][demo_name]
            if not bool(demo.attrs.get("success", False)) or int(demo.attrs.get("num_samples", 0)) <= 0:
                continue
            classifications.append(_classify_demo(demo["actions"][:], window_start, window_end))
        if not classifications:
            return None
        if len(set(classifications)) > 1:
            print(
                f"WARNING: {path} has demos with different strategies {classifications} -- "
                f"copying the whole file under '{classifications[0]}' (its first demo's strategy)."
            )
        return classifications[0]


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code (0 success, non-zero on error)."""
    parser = argparse.ArgumentParser(
        description=(
            "Copy record_demos.py-style HDF5 files into per-strategy subfolders "
            "(<output_dir>/right_only/, <output_dir>/both/, ...), for manual review."
        )
    )
    parser.add_argument("input_files", nargs="+", type=str, help="HDF5 files to classify and copy.")
    parser.add_argument("--window-start", type=int, default=250, help="Classification window start (steps).")
    parser.add_argument("--window-end", type=int, default=650, help="Classification window end (steps).")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Directory to create per-strategy subfolders under. Created if it doesn't exist.",
    )
    args = parser.parse_args(argv)

    for path in args.input_files:
        if not os.path.exists(path):
            print(f"ERROR: input file does not exist: {path}", file=sys.stderr)
            return 2

    counts: dict[str, int] = {}
    for path in sorted(args.input_files):
        try:
            strategy = _classify_file(path, args.window_start, args.window_end)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        if strategy is None:
            continue
        dst_dir = os.path.join(args.output_dir, strategy)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(path, dst_dir)
        counts[strategy] = counts.get(strategy, 0) + 1

    print()
    print(f"Copied {sum(counts.values())} files into {args.output_dir}:")
    for strategy, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {strategy}: {count}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Classify which hand-strategy each demo uses for its first pickup-and-place, from gripper actions.

Built for the ``LightwheelAI/iros2026-ikea-assembly`` dataset: found 2026-08-16 that
demonstrations solve the (identical-looking) first leg placement in genuinely different ways --
grasp and insert with the right hand alone, or grasp with the right hand and hand off to the
left before inserting -- with no visual/state signal distinguishing which strategy a given demo
uses. Nothing in the task setup determines this, so a policy trained across all of them has to
guess. This script reports, per demo:

- ``right_only`` / ``left_only`` / ``both`` (handoff): which hand(s) close during a configurable
  early window (default ``[250, 650)`` steps -- the first leg's pickup-and-place). ``left_only``
  should not occur with a wide-enough window; a narrow one can misclassify a handoff as
  ``left_only`` if the right-hand grasp happened before the window starts (confirmed 2026-08-16:
  widening ``--window-start`` from 350 to 250 reclassified all 29 apparent ``left_only`` demos as
  ``both``).
- ``switches_within_episode``: whether the demo's *own full length* (not just the early window)
  has meaningful stretches of both left-hand-only and right-hand-only closure at different
  points -- i.e. whether later legs in the same recording switch which hand does the work,
  independent of the first-leg classification above.

Gripper convention: ``actions[:, 0]`` = left gripper, ``actions[:, 1]`` = right gripper, closed
when > 0.5 (see ``decompose_lightwheel_wbc_action`` in ``convert_hdf5_to_lerobot.py``).

The script has zero simulation dependency and only requires ``h5py``/``numpy``.

Example
-------
.. code-block:: bash

    python isaaclab_arena/scripts/imitation_learning/classify_hand_strategy.py \\
        -o $DATASET_DIR/hand_strategy.txt \\
        $DATASET_DIR/data_deduplicated/*.hdf5
"""

from __future__ import annotations

import argparse
import h5py
import numpy as np
import os
import sys
from dataclasses import dataclass


@dataclass
class _DemoClassification:
    """One demo's hand-strategy classification."""

    file_path: str
    demo_name: str
    num_samples: int
    first_leg_strategy: str  # "right_only" | "left_only" | "both" | "neither" | "too_short"
    switches_within_episode: bool

    def __str__(self) -> str:
        switch_tag = "switches" if self.switches_within_episode else "consistent"
        return f"{os.path.basename(self.file_path)}::{self.demo_name}  {self.first_leg_strategy:10s}  {switch_tag}"


def _classify_demo(
    actions: np.ndarray, window_start: int, window_end: int, switch_frac_threshold: float
) -> tuple[str, bool]:
    """Classify a single demo's ``actions`` array. See module docstring for the definitions."""
    n = actions.shape[0]
    if n < window_end:
        first_leg_strategy = "too_short"
    else:
        window = actions[window_start:window_end]
        left_closed_w = (window[:, 0] > 0.5).any()
        right_closed_w = (window[:, 1] > 0.5).any()
        if left_closed_w and right_closed_w:
            first_leg_strategy = "both"
        elif right_closed_w:
            first_leg_strategy = "right_only"
        elif left_closed_w:
            first_leg_strategy = "left_only"
        else:
            first_leg_strategy = "neither"

    left_closed = actions[:, 0] > 0.5
    right_closed = actions[:, 1] > 0.5
    left_only_frac = np.logical_and(left_closed, ~right_closed).mean()
    right_only_frac = np.logical_and(right_closed, ~left_closed).mean()
    switches_within_episode = bool(left_only_frac > switch_frac_threshold and right_only_frac > switch_frac_threshold)

    return first_leg_strategy, switches_within_episode


def _classify_all(
    paths: list[str], window_start: int, window_end: int, switch_frac_threshold: float
) -> list[_DemoClassification]:
    """Open each input file and classify every successful ``demo_*`` group inside it."""
    results: list[_DemoClassification] = []
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
                actions = demo["actions"][:]
                strategy, switches = _classify_demo(actions, window_start, window_end, switch_frac_threshold)
                results.append(
                    _DemoClassification(
                        file_path=path,
                        demo_name=demo_name,
                        num_samples=actions.shape[0],
                        first_leg_strategy=strategy,
                        switches_within_episode=switches,
                    )
                )
    return results


def _print_summary(results: list[_DemoClassification]) -> None:
    """Print an operator-friendly per-demo listing and aggregate counts."""
    print()
    for r in results:
        print(f"  {r}")

    counts: dict[str, int] = {}
    for r in results:
        counts[r.first_leg_strategy] = counts.get(r.first_leg_strategy, 0) + 1
    num_switching = sum(1 for r in results if r.switches_within_episode)

    print()
    print(f"{len(results)} demos classified")
    for strategy, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  first_leg_strategy={strategy}: {count}")
    print(f"  switches_within_episode: {num_switching} / {len(results)}")
    print()


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code (0 success, non-zero on error)."""
    parser = argparse.ArgumentParser(
        description=(
            "Classify each demo's hand strategy (right-hand-only vs. handoff) for its first"
            " pickup-and-place, from raw gripper actions -- no simulation needed."
        )
    )
    parser.add_argument("input_files", nargs="+", type=str, help="HDF5 files to classify.")
    parser.add_argument(
        "--window-start",
        type=int,
        default=250,
        help="First-leg classification window start (steps). Default 250 -- confirmed wide enough"
        " to avoid misclassifying handoffs as left_only (see module docstring).",
    )
    parser.add_argument("--window-end", type=int, default=650, help="First-leg classification window end (steps).")
    parser.add_argument(
        "--switch-frac-threshold",
        type=float,
        default=0.02,
        help="Minimum fraction of the episode a hand must be closed *alone* (the other hand open)"
        " for that hand to count as having a 'meaningful' single-hand stretch, when deciding"
        " switches_within_episode.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default=None,
        help="If given, write the full per-demo classification listing here.",
    )
    args = parser.parse_args(argv)

    for path in args.input_files:
        if not os.path.exists(path):
            print(f"ERROR: input file does not exist: {path}", file=sys.stderr)
            return 2

    try:
        results = _classify_all(args.input_files, args.window_start, args.window_end, args.switch_frac_threshold)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    _print_summary(results)

    if args.output_file:
        with open(args.output_file, "w") as f:
            for r in results:
                f.write(f"{r}\n")
        print(f"Wrote full classification to: {args.output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

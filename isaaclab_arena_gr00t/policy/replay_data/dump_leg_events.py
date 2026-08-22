# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dump each episode's leg-handling events: who grabs each leg, how it's carried, who inserts it.

Ground-truth successor to ``dump_grasp_events.py``. That script tried to infer everything --
including where one leg's handling ends and the next begins -- from the wrist's own gripper
closure and rotation. It mostly worked, but broke on episodes where a hand releases and
re-grips the same leg mid-fastening (sliding its grip up the shaft to get more twist out of the
wrist): a re-grip looks just like a new pickup in wrist-only signals, so some episodes came out
massively over-segmented (confirmed 2026-08-22 on episode_000003: 10 "legs" detected instead of
4).

The source HDF5 (``iros2026_ikea_assembly_deduplicated.hdf5``, one directory up from the LeRobot
conversion) carries ground truth we don't need to infer: each of the 4 named leg props
(``Leg001_01_Leg001``, ``Leg001_03_Leg001``, ``Leg001_06_Leg001``, ``Leg001_Leg001`` -- confirmed
identical names across all 114 demos) has its own ``states/rigid_object/<name>/root_pose`` track.

A first version tried to bound each leg's handling to a single ``[this leg's first pickup, next
leg's first pickup)`` window. That broke too: a leg can be picked up, set down again mid-table
while a different leg gets handled, and picked back up later (confirmed 2026-08-22 on
episode_000002's 3rd leg: real pickup at frame ~7454, set down and motionless for ~40s at
frames ~9000-11000, picked up again and placed for good by frame ~13000) -- fixed-boundary
windows attribute that resumed handling to whichever leg's window happens to contain it, not the
leg it actually belongs to. Whether the robot carries one leg across to the far side of the table
or two, and where it sets the "spare" one down in between, is itself part of what a demo's
strategy is (this dataset's whole point), so that in-between parking position needs to survive as
real output, not get silently absorbed into whichever leg's window it lands in.

So there's no time-windowing at all now. Instead, for *every* ``action.gripper``-closed interval
of either hand (jitter-tolerant merged, same as ``dump_grasp_events.py``), every leg's
displacement during that interval is checked, and the interval ("touch") is attributed to
whichever leg moved the most -- if that leg moved at least ``--min-transport-displacement``,
otherwise the interval is dropped (this is what a fastening-only closure looks like: the hand
stays shut a long time, but no leg's position changes much). A leg's full handling is then just
the ordered list of touches attributed to it, each reported with its own start/end position --
one touch means picked up once and placed for good; more than one means it was set down somewhere
in between (that in-between end position is where it was "parked") and resumed later.

Each touch is also checked against ``teleop.navigate_command`` for whether the base was moving
during it, and touches are grouped by rough time-overlap into "concurrent episodes" so the report
directly says whether one or two legs were in hand while the base was walking.

Needs both ``h5py`` (ground-truth object states) and ``pandas`` (LeRobot parquet) -- run inside
the Arena dev container (see the ``dev-container`` skill), which has both:

.. code-block:: bash

    docker exec "$ARENA_CONTAINER" su $(id -un) -c \\
        "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \\
        isaaclab_arena_gr00t/policy/replay_data/dump_leg_events.py \\
        --episodes 0 1 2 3 4"
"""

from __future__ import annotations

import argparse
import h5py
import numpy as np
import pandas as pd
import sys
from dataclasses import dataclass
from pathlib import Path
from scipy.spatial.transform import Rotation

_FPS = 50
_GRIPPER_CLOSED_THRESHOLD = 0.5
_WALK_SPEED_THRESHOLD = 0.01  # m/s; below this, navigate_command is treated as noise, not motion
_LEG_OBJECT_NAMES = ("Leg001_01_Leg001", "Leg001_03_Leg001", "Leg001_06_Leg001", "Leg001_Leg001")

_DEFAULT_HDF5 = Path(
    "/datasets/lerobot_cache/LightwheelAI/iros2026-ikea-assembly/merged/iros2026_ikea_assembly_deduplicated.hdf5"
)
_DEFAULT_LEROBOT_DATA_DIR = Path(
    "/datasets/lerobot_cache/LightwheelAI/iros2026-ikea-assembly/merged/iros2026_ikea_assembly_deduplicated/lerobot/data/chunk-000"
)


def _episode_to_demo_name(hdf5_file: h5py.File, episode_index: int) -> str:
    """LeRobot ``episode_index`` -> this HDF5's ``demo_*`` key.

    ``convert_hdf5_to_lerobot.py`` assigns ``episode_index`` by iterating ``hdf5_data.keys()`` in
    whatever order h5py returns them, which is lexicographic (``demo_0, demo_1, demo_10,
    demo_100, ...``), not numeric -- confirmed 2026-08-22 by matching every one of the 114
    episodes' ``meta/episodes.jsonl`` frame count against ``sorted(hdf5 keys)[episode_index]``'s
    ``num_samples`` attr (114/114 matched; a naive ``demo_{episode_index}`` guess does not).
    """
    return sorted(hdf5_file["data"].keys())[episode_index]


def _closed_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """Raw ``[start, end)`` intervals where ``mask`` is True, before any merging/filtering."""
    intervals = []
    start = None
    for i, closed in enumerate(mask):
        if closed and start is None:
            start = i
        elif not closed and start is not None:
            intervals.append((start, i))
            start = None
    if start is not None:
        intervals.append((start, len(mask)))
    return intervals


def _merge_and_filter(intervals: list[tuple[int, int]], merge_gap_frames: int, min_segment_frames: int) -> list[tuple[int, int]]:
    """Bridge intervals separated by a short gap, then drop anything still too short to be real."""
    if not intervals:
        return []
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start - merged[-1][1] < merge_gap_frames:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return [(s, e) for s, e in merged if e - s >= min_segment_frames]


def _walk_distance(nav: np.ndarray, start: int, end: int) -> float:
    """Integrated linear speed (m) from ``teleop.navigate_command``'s [vx, vy] over ``[start, end)``."""
    speed = np.linalg.norm(nav[start:end, :2], axis=1)
    speed = np.where(speed > _WALK_SPEED_THRESHOLD, speed, 0.0)
    return float(speed.sum() / _FPS)


def _world_hand_positions(base_pose: np.ndarray, hand_pos_local: np.ndarray) -> np.ndarray:
    """Compose the base's world pose with the hand's base-relative ``action.eef_pose`` xyz.

    ``action.eef_pose`` is in the (moving) base frame, while leg ``root_pose`` is in world frame
    -- confirmed 2026-08-22 comparing them directly gave nonsense multi-meter "distances" even
    for a hand known to be gripping a leg. ``states/articulation/robot/root_pose`` (7 = xyz +
    xyzw quat, confirmed by getting a ~0.2m hand-to-leg distance -- plausible given ``root_pose``
    is a leg's own origin, not its grip point -- only with xyzw; wxyz gave 0.6-1m nonsense) composes
    them into one frame.
    """
    rotations = Rotation.from_quat(base_pose[:, 3:7])
    return rotations.apply(hand_pos_local) + base_pose[:, :3]


@dataclass
class _Touch:
    """One hand-closed interval attributed to the leg it actually moved."""

    leg: str
    hand: str
    start: int
    end: int
    from_pos: np.ndarray
    to_pos: np.ndarray
    walked_m: float

    def overlaps(self, other: "_Touch") -> bool:
        return self.start < other.end and other.start < self.end


def _attribute_touches(
    hdf5_file: h5py.File,
    demo_name: str,
    hand_closed: dict[str, np.ndarray],
    hand_world_pos: dict[str, np.ndarray],
    nav: np.ndarray,
    merge_gap_frames: int,
    min_segment_frames: int,
    min_transport_displacement: float,
) -> list[_Touch]:
    """Assign each hand-closed interval to whichever leg it's actually holding, if enough to count.

    Two checks, not one:

    1. *Moving enough*: the leg's max distance from its position at ``start``, reached at any
       point during ``[start, end)``, must clear ``min_transport_displacement`` -- this is what a
       fastening-only closure fails (the hand stays shut a long time, but no leg's position
       changes much). Deliberately the *max* excursion, not the net start-to-end distance: a leg
       can be carried out and set back down close to where it started within the same hand-closed
       interval (confirmed 2026-08-22 on episode_000003: net displacement missed a leg entirely
       that visibly traveled most of the way around the table and back within one interval).
    2. *Closest hand*: among legs that pass check 1, the one whose average distance to *this
       specific hand* (in world frame, via ``_world_hand_positions``) is smallest. Needed because
       both hands can be moving a leg at once (confirmed 2026-08-22 on episode_000003: the last
       two legs get carried together, one per hand, around the table) -- "moved the most" alone
       can't tell which hand is holding which leg when both legs are moving throughout the same
       window, but proximity (~0.2-0.3m for the hand actually gripping it, vs. further for the
       other hand's leg) can.
    """
    demo = hdf5_file["data"][demo_name]
    leg_positions = {name: demo["states"]["rigid_object"][name]["root_pose"][:, :3] for name in _LEG_OBJECT_NAMES}

    touches = []
    for hand in ("left", "right"):
        for start, end in _merge_and_filter(_closed_intervals(hand_closed[hand]), merge_gap_frames, min_segment_frames):
            moving_legs = {
                name: pos
                for name, pos in leg_positions.items()
                if np.linalg.norm(pos[start:end] - pos[start], axis=1).max() > min_transport_displacement
            }
            if not moving_legs:
                continue
            mean_hand_dist = {
                name: float(np.linalg.norm(hand_world_pos[hand][start:end] - pos[start:end], axis=1).mean())
                for name, pos in moving_legs.items()
            }
            best_leg = min(mean_hand_dist, key=mean_hand_dist.get)
            touches.append(
                _Touch(
                    leg=best_leg,
                    hand=hand,
                    start=start,
                    end=end,
                    from_pos=leg_positions[best_leg][start],
                    to_pos=leg_positions[best_leg][end - 1],
                    walked_m=_walk_distance(nav, start, end),
                )
            )
    touches.sort(key=lambda t: t.start)
    return touches


def _concurrency_groups(touches: list[_Touch]) -> list[list[_Touch]]:
    """Group touches whose time spans overlap -- i.e. touches happening with both hands at once."""
    groups: list[list[_Touch]] = []
    for touch in touches:
        joined = next((g for g in groups if any(touch.overlaps(t) for t in g)), None)
        if joined is not None:
            joined.append(touch)
        else:
            groups.append([touch])
    return groups


def _describe_episode(
    hdf5_file: h5py.File,
    lerobot_data_dir: Path,
    episode_index: int,
    merge_gap_frames: int,
    min_segment_frames: int,
    min_transport_displacement: float,
) -> str:
    """Human-readable leg-handling summary for one episode."""
    demo_name = _episode_to_demo_name(hdf5_file, episode_index)

    parquet_path = lerobot_data_dir / f"episode_{episode_index:06d}.parquet"
    df = pd.read_parquet(parquet_path, columns=["action.gripper", "action.eef_pose", "teleop.navigate_command"])
    gripper = np.stack(df["action.gripper"].values)  # (n, 2): left, right
    eef = np.stack(df["action.eef_pose"].values)  # (n, 14): left pos+quat(0:7), right pos+quat(7:14)
    nav = np.stack(df["teleop.navigate_command"].values)  # (n, 3): vx, vy, wz
    hand_closed = {"left": gripper[:, 0] > _GRIPPER_CLOSED_THRESHOLD, "right": gripper[:, 1] > _GRIPPER_CLOSED_THRESHOLD}

    base_pose = hdf5_file["data"][demo_name]["states"]["articulation"]["robot"]["root_pose"][:]  # (n, 7): xyz + xyzw quat
    hand_world_pos = {
        "left": _world_hand_positions(base_pose, eef[:, 0:3]),
        "right": _world_hand_positions(base_pose, eef[:, 7:10]),
    }

    touches = _attribute_touches(
        hdf5_file,
        demo_name,
        hand_closed,
        hand_world_pos,
        nav,
        merge_gap_frames,
        min_segment_frames,
        min_transport_displacement,
    )

    lines = [f"episode_{episode_index:06d}  (demo {demo_name}, {int(df.shape[0])} frames)"]
    lines.append("  -- touches, in time order (one line per hand-closed interval that really moved a leg) --")
    for t in touches:
        walk_tag = f"  [walked {t.walked_m:.2f}m]" if t.walked_m > 0.05 else ""
        lines.append(
            f"  {t.start:6d}-{t.end:6d}  {t.hand:5s}  {t.leg:18s}"
            f"  {t.from_pos.round(2).tolist()} -> {t.to_pos.round(2).tolist()}{walk_tag}"
        )

    lines.append("  -- concurrency (touches from both hands overlapping in time) --")
    for group in _concurrency_groups(touches):
        if len(group) == 1:
            continue
        legs_involved = sorted({t.leg for t in group})
        span = f"{min(t.start for t in group)}-{max(t.end for t in group)}"
        lines.append(f"  [{span}]  {len(legs_involved)} leg(s) in hand at once: {', '.join(legs_involved)}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code (0 success, non-zero on error)."""
    parser = argparse.ArgumentParser(
        description="Dump each episode's leg-handling events (hand sequence, walking, place position), "
        "attributing each hand-closed interval to whichever leg it actually displaced."
    )
    parser.add_argument("--episodes", nargs="+", type=int, required=True, help="LeRobot episode indices to inspect.")
    parser.add_argument("--hdf5-file", type=str, default=str(_DEFAULT_HDF5), help="Source HDF5 dataset file.")
    parser.add_argument(
        "--lerobot-data-dir", type=str, default=str(_DEFAULT_LEROBOT_DATA_DIR), help="LeRobot dataset's data/chunk-*/ directory."
    )
    parser.add_argument(
        "--merge-gap-frames",
        type=int,
        default=60,
        help="Bridge closed intervals separated by fewer than this many frames (closure jitter within one grasp).",
    )
    parser.add_argument(
        "--min-segment-frames",
        type=int,
        default=80,
        help="Drop closed intervals shorter than this many frames (jitter, not a real grasp).",
    )
    parser.add_argument(
        "--min-transport-displacement",
        type=float,
        default=0.18,
        help="Minimum net displacement (m) of the best-matching leg during a hand's closed interval for that"
        " interval to count as a real transport (excludes fastening, which barely moves any leg).",
    )
    parser.add_argument(
        "-o", "--output_file", type=str, default=None, help="If given, write the full per-episode summaries here."
    )
    args = parser.parse_args(argv)

    hdf5_path = Path(args.hdf5_file)
    lerobot_data_dir = Path(args.lerobot_data_dir)
    if not hdf5_path.exists():
        print(f"ERROR: HDF5 file does not exist: {hdf5_path}", file=sys.stderr)
        return 2
    if not lerobot_data_dir.exists():
        print(f"ERROR: LeRobot data directory does not exist: {lerobot_data_dir}", file=sys.stderr)
        return 2

    with h5py.File(hdf5_path, "r") as hdf5_file:
        reports = [
            _describe_episode(
                hdf5_file,
                lerobot_data_dir,
                episode_index,
                args.merge_gap_frames,
                args.min_segment_frames,
                args.min_transport_displacement,
            )
            for episode_index in args.episodes
        ]

    print()
    print("\n\n".join(reports))
    print()

    if args.output_file:
        Path(args.output_file).write_text("\n\n".join(reports) + "\n")
        print(f"Wrote full summaries to: {args.output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

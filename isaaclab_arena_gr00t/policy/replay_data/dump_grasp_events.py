# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dump each episode's leg-handling events (which hand(s), whether it walked, where it ends up).

First step towards grouping the ``LightwheelAI/iros2026-ikea-assembly`` demos by which assembly
"recipe" they follow -- which hand grasps each leg, whether it hands off mid-transport, whether
the robot walks to the other side of the table while carrying it, and where it ends up. That
grouping needs each episode reduced to a short label first; this script produces that label's raw
ingredients (a timeline of leg-handling events) so it can be checked against the actual video
before any cross-episode comparison is built on top of it.

Pipeline, per hand:

1. Segment ``action.gripper`` (``[left, right]`` closure, closed when > 0.5) into "closed"
   intervals, jitter-tolerant: a held grip crosses the 0.5 threshold many times a second
   (confirmed 2026-08-22 on episode_000000, where the naive right-hand mask alone produced >100
   segments), so raw crossings within ``--merge-gap-frames`` of each other are bridged, and
   anything shorter than ``--min-segment-frames`` is dropped as jitter, not a real grasp.
2. Split each closed interval into fixed ``--fasten-window-frames`` windows and classify each
   window by its ``action.eef_pose`` quaternion rotation rate: ``fasten`` (screwing the
   already-placed leg in with a twisting motion in ~one spot) spins at ~70-95 deg/s, vs. ~25-42
   deg/s for an actual pickup-to-hole ``transport`` -- confirmed 2026-08-22 against
   episode_000000's video. A single closed interval can be *both* in sequence (grab the leg from
   the other hand, then immediately start screwing without ever releasing -- confirmed on that
   same episode's last leg, whose one continuous right-hand grip starts at ~26 deg/s for 6s then
   jumps to ~85 deg/s), so classification is windowed and adjacent same-class windows are merged
   back into sub-segments, rather than one kind per raw closed interval.
   (Net xyz displacement was tried first for this and isn't reliable: a long fastening motion can
   drift several cm as the leg sinks in, occasionally exceeding a short handoff's displacement.)

``transport`` sub-segments that overlap in time by at least ``--min-handoff-overlap-frames`` are a
handoff (one hand receiving the leg from the other) and get merged into a single leg-handling
event; ``fasten`` sub-segments are reported but excluded from events, since they follow every leg
regardless of strategy and add no discriminative signal. Each event also reports the total
distance walked (integrated ``teleop.navigate_command`` linear speed) during its time span.

Run with the Isaac-GR00T venv (needs pandas):

.. code-block:: bash

    cd Isaac-GR00T && uv run python \\
        ../IsaacLab-Arena/isaaclab_arena_gr00t/policy/replay_data/dump_grasp_events.py \\
        $DATASET_DIR/lerobot/data/chunk-000/episode_00000{0,1,2,3,4}.parquet
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import sys
from dataclasses import dataclass
from pathlib import Path

_FPS = 50
_GRIPPER_CLOSED_THRESHOLD = 0.5
_WALK_SPEED_THRESHOLD = 0.01  # m/s; below this, navigate_command is treated as noise, not motion


@dataclass
class _GripSegment:
    """One hand's continuous, single-``kind`` sub-interval of a "closed" grip."""

    hand: str
    """``"left"`` or ``"right"``."""
    kind: str
    """``"transport"`` or ``"fasten"``, see module docstring."""
    start: int
    end: int
    start_pos: np.ndarray
    """``action.eef_pose`` xyz for ``hand`` at ``start`` (shape ``(3,)``)."""
    end_pos: np.ndarray
    """``action.eef_pose`` xyz for ``hand`` at ``end - 1`` (shape ``(3,)``)."""
    rotation_rate_deg_s: float
    """Average quaternion-change rate over ``[start, end)``, for display/debugging."""

    @property
    def duration_s(self) -> float:
        return (self.end - self.start) / _FPS

    def overlap_frames(self, other: "_GripSegment") -> int:
        return max(0, min(self.end, other.end) - max(self.start, other.start))


def _rotation_rate_deg_s(quat_segment: np.ndarray) -> float:
    """Average degrees/second of quaternion change across consecutive frames in ``quat_segment``."""
    if len(quat_segment) < 2:
        return 0.0
    q = quat_segment / np.linalg.norm(quat_segment, axis=1, keepdims=True)
    dots = np.abs(np.sum(q[:-1] * q[1:], axis=1)).clip(-1, 1)
    total_deg = np.degrees(2 * np.arccos(dots)).sum()
    return float(total_deg / (len(quat_segment) / _FPS))


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


def _merge_and_filter(
    intervals: list[tuple[int, int]], merge_gap_frames: int, min_segment_frames: int
) -> list[tuple[int, int]]:
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


def _split_by_kind(
    start: int, end: int, quat: np.ndarray, window_frames: int, min_fasten_rotation_rate: float
) -> list[tuple[int, int, str]]:
    """Split ``[start, end)`` into ``(sub_start, sub_end, kind)`` runs of one rotation-rate class.

    Windows are fixed-size except the last, which absorbs the remainder so no frames are dropped.
    """
    windows = []
    w0 = start
    while w0 < end:
        w1 = min(w0 + window_frames, end)
        if end - w1 < window_frames:  # fold a too-small remainder into this (last) window
            w1 = end
        rate = _rotation_rate_deg_s(quat[w0:w1])
        kind = "fasten" if rate >= min_fasten_rotation_rate else "transport"
        windows.append((w0, w1, kind))
        w0 = w1

    runs = [windows[0]]
    for w0, w1, kind in windows[1:]:
        if kind == runs[-1][2]:
            runs[-1] = (runs[-1][0], w1, kind)
        else:
            runs.append((w0, w1, kind))
    return runs


def _extract_grip_segments(
    df: pd.DataFrame,
    merge_gap_frames: int,
    min_segment_frames: int,
    fasten_window_frames: int,
    min_fasten_rotation_rate: float,
) -> list[_GripSegment]:
    """Segment both hands' closures, split each by rotation-rate kind, and attach eef xyz/rotation."""
    gripper = np.stack(df["action.gripper"].values)  # (n, 2): left, right
    eef = np.stack(df["action.eef_pose"].values)  # (n, 14): left pos+quat(0:7), right pos+quat(7:14)
    hand_xyz = {"left": eef[:, 0:3], "right": eef[:, 7:10]}
    hand_quat = {"left": eef[:, 3:7], "right": eef[:, 10:14]}
    hand_closed = {"left": gripper[:, 0] > _GRIPPER_CLOSED_THRESHOLD, "right": gripper[:, 1] > _GRIPPER_CLOSED_THRESHOLD}

    segments = []
    for hand in ("left", "right"):
        raw = _closed_intervals(hand_closed[hand])
        for start, end in _merge_and_filter(raw, merge_gap_frames, min_segment_frames):
            for sub_start, sub_end, kind in _split_by_kind(
                start, end, hand_quat[hand], fasten_window_frames, min_fasten_rotation_rate
            ):
                segments.append(
                    _GripSegment(
                        hand=hand,
                        kind=kind,
                        start=sub_start,
                        end=sub_end,
                        start_pos=hand_xyz[hand][sub_start],
                        end_pos=hand_xyz[hand][sub_end - 1],
                        rotation_rate_deg_s=_rotation_rate_deg_s(hand_quat[hand][sub_start:sub_end]),
                    )
                )
    segments.sort(key=lambda s: s.start)
    return segments


def _walk_distance(df: pd.DataFrame, start: int, end: int) -> float:
    """Integrated linear speed (m) from ``teleop.navigate_command``'s [vx, vy] over ``[start, end)``."""
    nav = np.stack(df["teleop.navigate_command"].values[start:end])  # (n, 3): vx, vy, wz
    speed = np.linalg.norm(nav[:, :2], axis=1)
    speed = np.where(speed > _WALK_SPEED_THRESHOLD, speed, 0.0)
    return float(speed.sum() / _FPS)


@dataclass
class _LegEvent:
    """One leg's pickup-to-placement, possibly spanning a handoff between hands."""

    segments: list[_GripSegment]
    """In time order; a single-element list means no handoff."""
    walked_m: float

    @property
    def start(self) -> int:
        return self.segments[0].start

    @property
    def end(self) -> int:
        return max(s.end for s in self.segments)

    @property
    def hand_sequence(self) -> str:
        hands = [self.segments[0].hand]
        for s in self.segments[1:]:
            if s.hand != hands[-1]:
                hands.append(s.hand)
        return "->".join(hands)

    @property
    def pickup_pos(self) -> np.ndarray:
        return self.segments[0].start_pos

    @property
    def place_pos(self) -> np.ndarray:
        return max(self.segments, key=lambda s: s.end).end_pos


def _group_leg_events(
    df: pd.DataFrame, transport_segments: list[_GripSegment], min_handoff_overlap_frames: int
) -> list[_LegEvent]:
    """Chain transport segments connected by a real (handoff) overlap into single leg events."""
    events: list[_LegEvent] = []
    current: list[_GripSegment] = []
    for seg in transport_segments:
        if current and not any(seg.overlap_frames(prev) >= min_handoff_overlap_frames for prev in current):
            events.append(_LegEvent(segments=current, walked_m=_walk_distance(df, current[0].start, max(s.end for s in current))))
            current = []
        current.append(seg)
    if current:
        events.append(_LegEvent(segments=current, walked_m=_walk_distance(df, current[0].start, max(s.end for s in current))))
    return events


def _describe_episode(
    path: Path,
    merge_gap_frames: int,
    min_segment_frames: int,
    fasten_window_frames: int,
    min_fasten_rotation_rate: float,
    min_handoff_overlap_frames: int,
) -> str:
    """Human-readable segment timeline + derived leg-event summary for one episode's parquet file."""
    df = pd.read_parquet(path, columns=["action.gripper", "action.eef_pose", "teleop.navigate_command"])
    segments = _extract_grip_segments(
        df, merge_gap_frames, min_segment_frames, fasten_window_frames, min_fasten_rotation_rate
    )
    transport_segments = [s for s in segments if s.kind == "transport"]
    events = _group_leg_events(df, transport_segments, min_handoff_overlap_frames)

    lines = [f"{path.stem}  ({len(df)} frames, {len(df) / _FPS:.1f}s)", "  -- raw segments --"]
    for seg in segments:
        lines.append(
            f"  {seg.hand:5s} closed  {seg.start:6d}-{seg.end:6d}  ({seg.duration_s:5.1f}s, rot={seg.rotation_rate_deg_s:5.1f}deg/s)"
            f"  [{seg.kind}]"
        )
    lines.append("  -- leg events (transport-only, handoffs merged) --")
    for i, ev in enumerate(events):
        lines.append(
            f"  leg {i + 1}: {ev.hand_sequence:12s}  {ev.start:6d}-{ev.end:6d}  walked={ev.walked_m:5.2f}m"
            f"  pickup {ev.pickup_pos.round(2).tolist()} -> place {ev.place_pos.round(2).tolist()}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code (0 success, non-zero on error)."""
    parser = argparse.ArgumentParser(
        description="Dump each episode's leg-handling events (hand sequence, walking, pickup/place position) from LeRobot parquet action data."
    )
    parser.add_argument("input_files", nargs="+", type=str, help="LeRobot episode parquet files to inspect.")
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
        "--fasten-window-frames",
        type=int,
        default=150,
        help="Window size (frames) for the transport-vs-fasten rotation-rate classification"
        " (default 150 = 3s at 50fps; confirmed clean on episode_000000, no window straddling a"
        " transport/fasten transition).",
    )
    parser.add_argument(
        "--min-fasten-rotation-rate",
        type=float,
        default=55.0,
        help="Rotation rate (deg/s) at or above which a window counts as 'fasten' (screwing in place)"
        " rather than 'transport' -- confirmed gap is ~25-42 deg/s (transport) vs. ~70-95 deg/s (fasten).",
    )
    parser.add_argument(
        "--min-handoff-overlap-frames",
        type=int,
        default=5,
        help="Minimum time-overlap (frames) between two transport segments to count as a real handoff"
        " (a hand-to-hand pass can be as brief as ~15 frames).",
    )
    parser.add_argument(
        "-o", "--output_file", type=str, default=None, help="If given, write the full per-episode timelines here."
    )
    args = parser.parse_args(argv)

    for path in args.input_files:
        if not Path(path).exists():
            print(f"ERROR: input file does not exist: {path}", file=sys.stderr)
            return 2

    reports = [
        _describe_episode(
            Path(path),
            args.merge_gap_frames,
            args.min_segment_frames,
            args.fasten_window_frames,
            args.min_fasten_rotation_rate,
            args.min_handoff_overlap_frames,
        )
        for path in sorted(args.input_files)
    ]

    print()
    print("\n\n".join(reports))
    print()

    if args.output_file:
        Path(args.output_file).write_text("\n\n".join(reports) + "\n")
        print(f"Wrote full timelines to: {args.output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

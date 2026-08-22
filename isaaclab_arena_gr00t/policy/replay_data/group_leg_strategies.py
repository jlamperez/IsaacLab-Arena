# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Group the ``LightwheelAI/iros2026-ikea-assembly`` demos by which assembly "recipe" they use.

Reduces each episode to a short recipe string built from ``dump_leg_events.py``'s per-leg
touches -- for each leg, in the order it was first handled:

- which hand(s) touched it in what order (a handoff shows as e.g. ``right->left``)
- whether the base walked a meaningful distance while handling it
- whether it was set down somewhere short of its final hole and resumed later (``+parked``) --
  needed as its own field, not folded into the hand sequence: two legs can have the identical
  hand-sequence string while differing here (confirmed 2026-08-22: of 18 episodes that all carry
  the last two legs across the table together with an identical ``right``/``right->left``-style
  recipe, 16 park the *left*-hand leg while it waits its turn and 2 -- episodes 19 and 23 -- park
  the *right*-hand one instead; without this field both looked like the same recipe)
- whether it was handled at the same time as a *different* leg (both hands full at once, e.g.
  carrying the last two legs across together -- confirmed 2026-08-22 this is the *majority*
  pattern across the dataset, not a rare one: an earlier version of this attribution under-counted
  it because "which leg moved the most" can't tell which hand is holding which leg when both legs
  are moving at once, see ``dump_leg_events.py``'s ``_attribute_touches`` docstring)
- which of the table's two near/far holes it ends up in (``H1``/``H2``, see ``_hole_labels``) --
  confirmed 2026-08-22 needed as its own field too: two groups of episodes share an identical
  hand-sequence/walked/parked recipe for the last two legs but insert them into the *opposite*
  holes of one another (which hand ends up at which hole isn't determined by anything else this
  recipe already tracks)

Episodes with an identical recipe string get grouped together, the same way
``find_duplicate_demos.py`` groups byte-identical demos -- except the grouping key here is the
strategy, not the raw content.

Like ``dump_leg_events.py``, this needs both ``h5py`` (ground-truth leg object states) and
``pandas`` (LeRobot parquet), so run it inside the Arena dev container (see the ``dev-container``
skill):

.. code-block:: bash

    docker exec "$ARENA_CONTAINER" su $(id -un) -c \\
        "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \\
        isaaclab_arena_gr00t/policy/replay_data/group_leg_strategies.py"
"""

from __future__ import annotations

import argparse
import h5py
import numpy as np
import pandas as pd
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from dump_leg_events import (
    _DEFAULT_HDF5,
    _DEFAULT_LEROBOT_DATA_DIR,
    _GRIPPER_CLOSED_THRESHOLD,
    _attribute_touches,
    _episode_to_demo_name,
    _world_hand_positions,
)

_WALK_DISTANCE_THRESHOLD = 2.0  # m; below this a leg's handling counts as "stays" not "walks" --
# confirmed 2026-08-22 there's a clean gap between same-side repositioning (<=1.8m) and an actual
# walk to the table's far side (>=3.3m) across the episodes checked so far.


@dataclass
class _LegEntry:
    """One leg's recipe entry: its hand sequence, whether it involved walking, and its rank."""

    leg: str
    hand_sequence: str
    walked: bool
    parked: bool
    """True if this leg has more than one touch -- picked up, set down somewhere short of its
    final hole, and resumed later. Two legs can share the exact same ``hand_sequence`` string
    (e.g. both just "right") while differing in this -- one carried straight to its hole in a
    single touch, the other set aside mid-way and finished later -- so it has to be its own field,
    not folded into ``hand_sequence`` (confirmed 2026-08-22: episodes 19 and 23 park the
    *right*-hand leg instead of the *left*-hand leg that the other 16 in their group park,
    despite having an identical hand_sequence-only recipe)."""
    place_pos: np.ndarray
    """xyz at the end of this leg's last touch -- which hole it went in, not just how it got there."""
    first_touch_start: int
    last_touch_end: int

    def overlaps(self, other: "_LegEntry") -> bool:
        return self.first_touch_start < other.last_touch_end and other.first_touch_start < self.last_touch_end


def _leg_entries(touches: list) -> list[_LegEntry]:
    """Collapse touches into one entry per leg (robust to a leg being parked and resumed later)."""
    by_leg: dict[str, list] = defaultdict(list)
    for t in touches:
        by_leg[t.leg].append(t)

    entries = []
    for leg, leg_touches in by_leg.items():
        leg_touches.sort(key=lambda t: t.start)
        hands = [leg_touches[0].hand]
        for t in leg_touches[1:]:
            if t.hand != hands[-1]:
                hands.append(t.hand)
        entries.append(
            _LegEntry(
                leg=leg,
                hand_sequence="->".join(hands),
                walked=sum(t.walked_m for t in leg_touches) >= _WALK_DISTANCE_THRESHOLD,
                parked=len(leg_touches) > 1,
                place_pos=leg_touches[-1].to_pos,
                first_touch_start=leg_touches[0].start,
                last_touch_end=leg_touches[-1].end,
            )
        )
    entries.sort(key=lambda e: e.first_touch_start)
    return entries


def _hole_labels(entries: list[_LegEntry]) -> dict[str, str]:
    """Label each leg's final hole ``H1``/``H2`` by which of the first two legs' holes it's nearer.

    The first two legs handled (chronologically) are always the unpaired, unwalked ones straight
    off the rack -- they define the table's two near-side holes. A later leg's hole isn't fixed to
    "whichever hand carried it": confirmed 2026-08-22 that two groups of episodes share an
    identical hand_sequence/walked/parked recipe for the last two legs but insert them into
    *opposite* holes -- one group's right-hand leg goes in the hole on Leg001_01's side, the
    other's goes in Leg001_Leg001's side. Y (not x) is what separates the two holes on each side
    of the table; x instead separates near-side from far-side (i.e. correlates with the walk).
    """
    reference_y = [entries[0].place_pos[1], entries[1].place_pos[1]]
    labels = {}
    for entry in entries:
        closest = min(range(2), key=lambda i: abs(entry.place_pos[1] - reference_y[i]))
        labels[entry.leg] = f"H{closest + 1}"
    return labels


def _entry_tag(entry: _LegEntry, hole_label: str) -> str:
    walk_tag = "walk" if entry.walked else "stay"
    parked_tag = "+parked" if entry.parked else ""
    return f"{entry.hand_sequence}:{walk_tag}{parked_tag}->{hole_label}"


def _recipe_string(entries: list[_LegEntry]) -> str:
    """One slot per leg, in handling order; two legs sharing overlapping time become one slot."""
    hole_labels = _hole_labels(entries)
    slots = []
    used = [False] * len(entries)
    for i, entry in enumerate(entries):
        if used[i]:
            continue
        paired_with = next(
            (j for j in range(i + 1, len(entries)) if not used[j] and entry.overlaps(entries[j])), None
        )
        if paired_with is None:
            slots.append(_entry_tag(entry, hole_labels[entry.leg]))
        else:
            other = entries[paired_with]
            used[paired_with] = True
            slots.append(
                f"paired({_entry_tag(entry, hole_labels[entry.leg])}+{_entry_tag(other, hole_labels[other.leg])})"
            )
    return " | ".join(slots)


def _episode_recipe(
    hdf5_file: h5py.File,
    lerobot_data_dir: Path,
    episode_index: int,
    merge_gap_frames: int,
    min_segment_frames: int,
    min_transport_displacement: float,
) -> str:
    """This episode's recipe string (see module docstring for the encoding)."""
    demo_name = _episode_to_demo_name(hdf5_file, episode_index)
    parquet_path = lerobot_data_dir / f"episode_{episode_index:06d}.parquet"
    df = pd.read_parquet(parquet_path, columns=["action.gripper", "action.eef_pose", "teleop.navigate_command"])
    gripper = np.stack(df["action.gripper"].values)
    eef = np.stack(df["action.eef_pose"].values)
    nav = np.stack(df["teleop.navigate_command"].values)
    hand_closed = {"left": gripper[:, 0] > _GRIPPER_CLOSED_THRESHOLD, "right": gripper[:, 1] > _GRIPPER_CLOSED_THRESHOLD}

    base_pose = hdf5_file["data"][demo_name]["states"]["articulation"]["robot"]["root_pose"][:]
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
    return _recipe_string(_leg_entries(touches))


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code (0 success, non-zero on error)."""
    parser = argparse.ArgumentParser(
        description="Group episodes of the LightwheelAI/iros2026-ikea-assembly dataset by assembly strategy."
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        default=None,
        help="LeRobot episode indices to inspect. Defaults to every episode in the dataset.",
    )
    parser.add_argument("--hdf5-file", type=str, default=str(_DEFAULT_HDF5), help="Source HDF5 dataset file.")
    parser.add_argument(
        "--lerobot-data-dir", type=str, default=str(_DEFAULT_LEROBOT_DATA_DIR), help="LeRobot dataset's data/chunk-*/ directory."
    )
    parser.add_argument("--merge-gap-frames", type=int, default=60)
    parser.add_argument("--min-segment-frames", type=int, default=80)
    parser.add_argument("--min-transport-displacement", type=float, default=0.18)
    parser.add_argument(
        "-o", "--output_file", type=str, default=None, help="If given, write the full per-episode recipe listing here."
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
        num_episodes = len(hdf5_file["data"].keys())
        episodes = args.episodes if args.episodes is not None else list(range(num_episodes))

        recipes: dict[int, str] = {}
        failures: dict[int, str] = {}
        for episode_index in episodes:
            try:
                recipes[episode_index] = _episode_recipe(
                    hdf5_file,
                    lerobot_data_dir,
                    episode_index,
                    args.merge_gap_frames,
                    args.min_segment_frames,
                    args.min_transport_displacement,
                )
            except Exception as e:  # noqa: BLE001 -- one bad episode shouldn't kill the whole batch
                failures[episode_index] = str(e)

    groups: dict[str, list[int]] = defaultdict(list)
    for episode_index, recipe in recipes.items():
        groups[recipe].append(episode_index)
    ordered_groups = sorted(groups.items(), key=lambda kv: -len(kv[1]))

    lines = [f"{len(recipes)} episodes recipe'd, {len(failures)} failed, {len(ordered_groups)} distinct recipes\n"]
    for recipe, episode_indices in ordered_groups:
        lines.append(f"[{len(episode_indices)}x] {recipe}")
        lines.append(f"      episodes: {episode_indices}")
    if failures:
        lines.append("\nFAILED episodes:")
        for episode_index, error in sorted(failures.items()):
            lines.append(f"  episode_{episode_index:06d}: {error}")

    report = "\n".join(lines)
    print()
    print(report)
    print()

    if args.output_file:
        Path(args.output_file).write_text(report + "\n")
        print(f"Wrote full grouping to: {args.output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

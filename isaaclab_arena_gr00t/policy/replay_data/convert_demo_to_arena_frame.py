# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Remap a ``record_demos.py``-style HDF5's ``initial_state`` and ``actions`` into Arena's frame.

The ``LightwheelAI/iros2026-ikea-assembly`` dataset was recorded in robofinals' own environment
(a full Robocasa kitchen scene, robofinals' own 23-D ``G1-Gripper-Controller-DecoupledWBC``
action), not inside Arena's own ``assemble_table`` scene -- Jorge is porting robofinals' working
setup into Arena, and this dataset predates that port. Getting a recorded demo through Isaac Lab
Mimic's ``annotate_demos.py`` (which calls ``env.reset_to()`` then replays ``actions`` step by
step) needs two independent fixes, found in that order, each confirmed by an actual run against
the ``iros2026_ikea_assembly_mimic_seed13.hdf5`` seed set (2026-08-22):

1. ``env.reset_to()`` raised ``KeyError: 'table001'`` -- the recorded ``initial_state/
   rigid_object`` keys (``Table001_Table001_01``, ``Leg001_01_Leg001``, ...) and poses don't
   match Arena's own scene's asset names/frame. Fixed by ``_convert_initial_state`` below.
2. Once that was fixed, ``env.step()`` raised ``ValueError: Invalid action shape, expected: 25,
   received: 23`` -- the recorded 23-D action doesn't match Arena's default ``g1_wbc_agile_pink_dex1``
   embodiment's 25-D action space at all. Fixed by ``_convert_actions`` below, using the *already
   verified* 23-D -> 27-D mapping from ``isaaclab_arena_gr00t/policy/lightwheel_hdf5_replay_policy.py``
   (confirmed there 2026-08-09/10 against robofinals' own ``decoupled_wbc_action.py`` and
   empirically against this dataset's per-column stats) -- which targets a *different* embodiment,
   ``g1_wbc_agile_pink_dex1_continuous_grip`` (27-D), not the 25-D default. **Anything using this
   converted HDF5 (``annotate_demos.py``, ``generate_dataset.py``, ...) must pass
   ``--embodiment g1_wbc_agile_pink_dex1_continuous_grip``, not the ``assemble_table`` default.**

``initial_state`` rename + position offset (neither is a guess -- both are already in active use
elsewhere in this codebase, for the *same* Scene02.usd source this dataset and Arena's
``assemble_table`` scene both derive from; see ``isaaclab_arena_environments/
assemble_table_environment.py``'s own comments, e.g. "Leg001_01 (2.0258) -> -0.3468"):

.. list-table::
   :header-rows: 1

   * - Dataset ``rigid_object`` name
     - Arena scene asset name
   * - ``Table001_Table001_01``
     - ``table001`` (``fixed_asset``)
   * - ``Table278_Table278``
     - ``table278`` (``background``)
   * - ``Leg001_01_Leg001``
     - ``leg001`` (``held_asset`` -- the only leg ``AssemblyTask`` actually tracks for success)
   * - ``Leg001_Leg001``
     - ``leg001_2`` (cosmetic ``extra_leg_2``)
   * - ``Leg001_03_Leg001``
     - ``leg001_3`` (cosmetic ``extra_leg_3``)
   * - ``Leg001_06_Leg001``
     - ``leg001_4`` (cosmetic ``extra_leg_4``)

Position offset ``(+1.944878, -2.372593)`` (x, y; z unchanged, no rotation) -- confirmed against
the dataset directly this session: ``Table278_Table278``'s recorded pose ``(-1.399, 2.363,
0.3823)`` plus this offset lands on Arena's own hardcoded ``background.set_initial_pose``
position ``(0.5461, -0.0091, 0.3823)`` to within Arena's own rounding, and the same computation
on the robot's own root pose reproduces the 2026-08-15-session-documented start pose
``(-0.4649, 0.0274, 0.78)`` almost exactly. Robot orientation/joint positions/joint velocities are
left untouched -- not world-frame-relative, and this smoke-tested clean (``env.reset_to()``
raised nothing about ``articulation`` once ``rigid_object`` was fixed).

``initial_state`` conversion writes a superset, not the exact registered set: all 6 renamed
rigid-object entries are always written, whether or not each one is actually a registered
``rigid_object`` in Arena's scene -- ``InteractiveScene.reset_to`` only reads the keys it needs,
so extra ones are silently ignored, and guessing the precise registered set wasn't worth the
extra verification round-trip.

``actions`` conversion, exact column mapping (see ``lightwheel_hdf5_replay_policy.py`` for the
original derivation): dataset columns ``[2:23]`` copy straight into Arena ``[2:23]`` (same WBC
field order/convention), except the two quaternion sub-slices (``[5:9]`` left, ``[12:16]`` right)
get reordered wxyz -> xyzw; dataset column 0 (left gripper) duplicates into Arena ``[23:25]``,
column 1 (right gripper) into ``[25:27]``; Arena ``[0:2]`` (hand_state) has no dataset equivalent
and is left zero, unused by the Dex1 embodiment.

Needs ``h5py`` -- run inside the Arena dev container (see the ``dev-container`` skill).

.. code-block:: bash

    docker exec "$ARENA_CONTAINER" su $(id -un) -c \\
        "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \\
        isaaclab_arena_gr00t/policy/replay_data/convert_demo_to_arena_frame.py \\
        --input_file /datasets/.../mimic/iros2026_ikea_assembly_mimic_seed13.hdf5 \\
        --output_file /datasets/.../mimic/iros2026_ikea_assembly_mimic_seed13_arena_frame.hdf5"
"""

from __future__ import annotations

import argparse
import h5py
import numpy as np
import shutil
import sys
from pathlib import Path

_XY_OFFSET = np.array([1.944878, -2.372593])

_RIGID_OBJECT_RENAME = {
    "Table001_Table001_01": "table001",
    "Table278_Table278": "table278",
    "Leg001_01_Leg001": "leg001",
    "Leg001_Leg001": "leg001_2",
    "Leg001_03_Leg001": "leg001_3",
    "Leg001_06_Leg001": "leg001_4",
}

# See module docstring's "actions conversion" section for what each slice means.
_ACTION_DIM = 27
_DATASET_WBC_SLICE = slice(2, 23)
_DATASET_LEFT_GRIPPER_IDX = 0
_DATASET_RIGHT_GRIPPER_IDX = 1
_DATASET_LEFT_QUAT_WXYZ_SLICE = slice(5, 9)
_DATASET_RIGHT_QUAT_WXYZ_SLICE = slice(12, 16)
_ARENA_WBC_SLICE = slice(2, 23)
_ARENA_LEFT_QUAT_XYZW_SLICE = slice(5, 9)
_ARENA_RIGHT_QUAT_XYZW_SLICE = slice(12, 16)
_ARENA_LEFT_GRIPPER_SLICE = slice(23, 25)
_ARENA_RIGHT_GRIPPER_SLICE = slice(25, 27)


def _wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Reorder an ``(n, 4)`` wxyz quaternion array to xyzw."""
    return quat_wxyz[:, [1, 2, 3, 0]]


def _offset_pose_xy(root_pose: np.ndarray) -> np.ndarray:
    """Apply the Scene02.usd -> Arena rigid (x, y) offset to a ``(n, 7)`` xyz+quat pose array."""
    out = root_pose.copy()
    out[:, 0:2] += _XY_OFFSET
    return out


def _convert_initial_state(demo: h5py.Group) -> None:
    """Rewrite one demo's ``initial_state`` in place: rename+offset rigid objects, offset robot."""
    initial_state = demo["initial_state"]

    rigid_object_group = initial_state["rigid_object"]
    for raw_name, arena_name in _RIGID_OBJECT_RENAME.items():
        if raw_name not in rigid_object_group:
            continue
        root_pose = rigid_object_group[raw_name]["root_pose"][:]
        root_velocity = rigid_object_group[raw_name]["root_velocity"][:]
        del rigid_object_group[raw_name]
        new_group = rigid_object_group.create_group(arena_name)
        new_group.create_dataset("root_pose", data=_offset_pose_xy(root_pose))
        new_group.create_dataset("root_velocity", data=root_velocity)

    robot_pose_ds = initial_state["articulation"]["robot"]["root_pose"]
    robot_pose_ds[...] = _offset_pose_xy(robot_pose_ds[:])


def _convert_actions(demo: h5py.Group) -> None:
    """Replace one demo's 23-D ``actions`` with the 27-D ``..._continuous_grip`` layout."""
    raw_actions = demo["actions"][:]
    n_frames = raw_actions.shape[0]

    actions = np.zeros((n_frames, _ACTION_DIM), dtype=np.float32)
    actions[:, _ARENA_WBC_SLICE] = raw_actions[:, _DATASET_WBC_SLICE]
    actions[:, _ARENA_LEFT_QUAT_XYZW_SLICE] = _wxyz_to_xyzw(raw_actions[:, _DATASET_LEFT_QUAT_WXYZ_SLICE])
    actions[:, _ARENA_RIGHT_QUAT_XYZW_SLICE] = _wxyz_to_xyzw(raw_actions[:, _DATASET_RIGHT_QUAT_WXYZ_SLICE])
    actions[:, _ARENA_LEFT_GRIPPER_SLICE] = raw_actions[:, _DATASET_LEFT_GRIPPER_IDX, None]
    actions[:, _ARENA_RIGHT_GRIPPER_SLICE] = raw_actions[:, _DATASET_RIGHT_GRIPPER_IDX, None]

    del demo["actions"]
    demo.create_dataset("actions", data=actions)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code (0 success, non-zero on error)."""
    parser = argparse.ArgumentParser(
        description="Remap a record_demos.py-style HDF5's initial_state and actions from the raw "
        "dataset's frame/embodiment into Arena's assemble_table (continuous-grip) frame/embodiment."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Source HDF5 (e.g. the Mimic seed set).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to write the converted HDF5 to.")
    args = parser.parse_args(argv)

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    if not input_path.exists():
        print(f"ERROR: input file does not exist: {input_path}", file=sys.stderr)
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying {input_path} -> {output_path} ...")
    shutil.copy2(input_path, output_path)

    with h5py.File(output_path, "r+") as f:
        demo_names = sorted(f["data"].keys())
        for demo_name in demo_names:
            demo = f["data"][demo_name]
            _convert_initial_state(demo)
            _convert_actions(demo)

    print()
    print(f"Converted {len(demo_names)} demos' initial_state + actions in {output_path}")
    print("Remember: run downstream tools (annotate_demos.py, generate_dataset.py, ...) with")
    print("  --embodiment g1_wbc_agile_pink_dex1_continuous_grip")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

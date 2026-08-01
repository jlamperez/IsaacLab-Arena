# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Cheap, no-retrain check of the ``NON_EEF`` -> ``EEF`` modality-config fix.

Compares the naive Euler-angle addition currently used to reconstruct an absolute
``ee_action`` from a ``RELATIVE`` prediction (what ``ActionType.NON_EEF`` does today, see
``JointActionChunk.to_absolute_chunking`` in Isaac-GR00T's ``action_chunking.py``) against
proper homogeneous-transform composition (what ``ActionType.EEF`` would do, via
``EndEffectorActionChunk.to_absolute_chunking``), using one real captured chunk
(state + model-predicted relative action) from a closed-loop debug dump.

Uses ``EndEffectorPose``'s direct constructor (``rotation_type="euler"``) rather than
``EndEffectorPose.from_action_format``, since none of ``from_action_format``'s supported
``ActionFormat`` values (``XYZ_ROT6D``, ``XYZ_ROTVEC``, ``DEFAULT``) match our raw
position + Euler-xyz layout -- this script only tests the composition math itself, not
a full pipeline switch.

Run with the Isaac-GR00T venv (needs the ``gr00t`` package):

    cd Isaac-GR00T && uv run python \\
        ../IsaacLab-Arena/isaaclab_arena_gr00t/policy/replay_data/validate_eef_composition.py \\
        --chunk-npz /path/to/gr00t_dex1_chunk_001.npz
"""

from __future__ import annotations

import argparse
import numpy as np

from gr00t.data.state_action.action_chunking import EndEffectorActionChunk
from gr00t.data.state_action.pose import EndEffectorPose

# ee_state / ee_action layout: left arm pos(3)+euler_xyz(3), right arm pos(3)+euler_xyz(3).
_LEFT_POS, _LEFT_ROT = slice(0, 3), slice(3, 6)
_RIGHT_POS, _RIGHT_ROT = slice(6, 9), slice(9, 12)


def _compare_arm(name: str, state_pos, state_euler, delta_pos, delta_euler) -> None:
    naive_pos = state_pos + delta_pos
    naive_euler = state_euler + delta_euler

    ref_pose = EndEffectorPose(
        translation=state_pos, rotation=state_euler, rotation_type="euler", rotation_order="xyz", degrees=False
    )
    rel_pose = EndEffectorPose(
        translation=delta_pos, rotation=delta_euler, rotation_type="euler", rotation_order="xyz", degrees=False
    )
    composed = EndEffectorActionChunk([rel_pose]).to_absolute_chunking(reference_frame=ref_pose)
    abs_pose = composed.poses[0]
    # to_rotation defaults to degrees=True -- our data (and the naive-add comparison) is in radians.
    composed_euler = abs_pose.to_rotation("euler", "xyz", degrees=False)

    print(f"--- {name} ---")
    print(f"  state:            pos={state_pos} euler={state_euler}")
    print(f"  relative (pred):  pos={delta_pos} euler={delta_euler}")
    print(f"  naive add:        pos={naive_pos} euler={naive_euler}")
    print(f"  proper compose:   pos={abs_pose.translation} euler={composed_euler}")
    print(f"  pos diff (naive - compose): {naive_pos - abs_pose.translation}")
    print(f"  euler diff (naive - compose): {naive_euler - composed_euler}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-npz", required=True, help="A gr00t_dex1_chunk_NNN.npz debug dump.")
    args = parser.parse_args()

    data = np.load(args.chunk_npz, allow_pickle=True)
    ee_state = data["ee_state"]
    ee_action = data["ee_action"][0]  # first predicted step of the chunk

    _compare_arm(
        "left arm",
        ee_state[_LEFT_POS], ee_state[_LEFT_ROT],
        ee_action[_LEFT_POS], ee_action[_LEFT_ROT],
    )
    print()
    _compare_arm(
        "right arm",
        ee_state[_RIGHT_POS], ee_state[_RIGHT_ROT],
        ee_action[_RIGHT_POS], ee_action[_RIGHT_ROT],
    )


if __name__ == "__main__":
    main()

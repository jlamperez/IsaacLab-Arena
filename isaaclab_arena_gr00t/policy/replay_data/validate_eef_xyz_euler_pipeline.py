# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end check of the ``NON_EEF`` -> ``EEF`` + ``XYZ_EULER`` modality-config fix.

Unlike ``validate_eef_composition.py`` (which builds ``EndEffectorPose`` objects by hand to
test the composition math in isolation), this script calls
``StateActionProcessor._convert_to_absolute_action`` directly -- the exact function
Isaac-GR00T's inference/training code paths use -- so it also exercises the new
``ActionFormat.XYZ_EULER`` support end-to-end (``EndEffectorPose.from_action_format`` /
``EndEffectorActionChunk.to()``), not just the underlying rotation-composition math.

Run with the Isaac-GR00T venv (needs the ``gr00t`` package):

    cd Isaac-GR00T && uv run python \\
        ../IsaacLab-Arena/isaaclab_arena_gr00t/policy/replay_data/validate_eef_xyz_euler_pipeline.py \\
        --chunk-npz /path/to/gr00t_dex1_chunk_001.npz
"""

from __future__ import annotations

import argparse
import numpy as np

from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.data.types import ActionFormat, ActionType

# ee_state / ee_action layout: left arm pos(3)+euler_xyz(3), right arm pos(3)+euler_xyz(3).
_LEFT, _RIGHT = slice(0, 6), slice(6, 12)


def _compare_arm(name: str, state_arm: np.ndarray, delta_arm: np.ndarray) -> None:
    processor = StateActionProcessor.__new__(StateActionProcessor)  # no normalization needed here

    naive = state_arm + delta_arm

    abs_arm = processor._convert_to_absolute_action(
        action=delta_arm[None, :],
        reference_state=state_arm,
        action_type=ActionType.EEF,
        action_format=ActionFormat.XYZ_EULER,
    )[0]

    print(f"--- {name} ---")
    print(f"  state:          pos={state_arm[:3]} euler={state_arm[3:]}")
    print(f"  relative (pred): pos={delta_arm[:3]} euler={delta_arm[3:]}")
    print(f"  naive add:      pos={naive[:3]} euler={naive[3:]}")
    print(f"  EEF+XYZ_EULER:  pos={abs_arm[:3]} euler={abs_arm[3:]}")
    print(f"  pos diff (naive - fixed): {naive[:3] - abs_arm[:3]}")
    print(f"  euler diff (naive - fixed): {naive[3:] - abs_arm[3:]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-npz", required=True, help="A gr00t_dex1_chunk_NNN.npz debug dump.")
    args = parser.parse_args()

    data = np.load(args.chunk_npz, allow_pickle=True)
    ee_state = data["ee_state"]
    ee_action = data["ee_action"][0]  # first predicted step of the chunk

    _compare_arm("left arm", ee_state[_LEFT], ee_action[_LEFT])
    print()
    _compare_arm("right arm", ee_state[_RIGHT], ee_action[_RIGHT])


if __name__ == "__main__":
    main()

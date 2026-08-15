# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
External data configuration module for the LightwheelAI/iros2026-ikea-assembly dataset
(G1 + Dex1 gripper, robofinals-recorded), for GR00T N1.7.

modality_keys below are the group names from this embodiment's own meta/modality.json (written by
isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py), not the LeRobot parquet column names
directly -- GR00T resolves each key through that file's own start/end/original_key mapping.

All action groups use ActionConfig(rep=ABSOLUTE, type=NON_EEF, format=DEFAULT): the wrist-pose
groups are stored as pos(3)+quat_wxyz(4), not one of GR00T's ActionFormat.XYZ_* rotation formats
(no native quaternion format exists), so ActionType.EEF cannot parse them; ActionType.NON_EEF is a
plain passthrough regardless of width, and is provably safe here since NON_EEF/EEF is only ever
read for ActionRepresentation.RELATIVE (never ABSOLUTE) -- confirmed 2026-08-15 against
gr00t/data/state_action/state_action_processor.py in this GR00T N1.7 clone. This exactly mirrors
../g1/g1_sim_wbc_data_gr00t_n_1_7_config.py's own treatment of its joint-position action groups.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

g1_dex1_ikea_lightwheel_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["first_person", "left_hand", "right_hand"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_leg", "right_leg", "waist", "left_arm", "left_gripper", "right_arm", "right_gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=[
            "left_wrist_pose",
            "right_wrist_pose",
            "gripper",
            "base_height_command",
            "navigate_command",
            "torso_orientation_rpy_command",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(g1_dex1_ikea_lightwheel_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
External data configuration module for the BitRobot G1 Dex1 IKEA assembly dataset
(BitRobot/G1_WBT_Dex1_Building-Children-Table), for GR00T N1.7.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

g1_dex1_ikea_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam_0"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ee_state_left", "ee_state_right", "hand_state"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        # ee_action packs both hands into one 12-D dataset column (pos(3)+euler_xyz(3) per
        # hand). EndEffectorPose models a single end-effector pose, so each hand needs its own
        # modality key -- split via meta/modality.json's ee_action_left (0:6) / ee_action_right
        # (6:12) joint groups, not one combined 12-D EEF key.
        modality_keys=["ee_action_left", "ee_action_right", "hand_cmd"],
        action_configs=[
            # ee_action_{left,right} is a Cartesian end-effector pose (position + xyz-Euler
            # orientation), not independent joint angles -- type=EEF makes relative<->absolute
            # reconstruction use proper rotation-matrix composition (T_ref @ T_relative) instead
            # of naive per-component addition, which silently gives the wrong result whenever the
            # reference orientation is far from identity (e.g. the ~180-deg wrist yaw at reset).
            # format=XYZ_EULER matches the dataset's actual pos(3)+euler_xyz(3) layout (verified
            # against the source repo's own euler-from-quaternion formula, see
            # gr00t_dex1_eef_closedloop_policy.py's module docstring).
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_EULER,
                state_key="ee_state_left",
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_EULER,
                state_key="ee_state_right",
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

register_modality_config(g1_dex1_ikea_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

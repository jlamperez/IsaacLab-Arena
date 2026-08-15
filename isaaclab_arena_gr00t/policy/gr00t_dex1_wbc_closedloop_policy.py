# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T remote closed-loop policy for the G1 + Dex1 gripper embodiment, native WBC action space.

Separate from :mod:`isaaclab_arena_gr00t.policy.gr00t_dex1_eef_closedloop_policy` (the BitRobot
checkpoint's client): that checkpoint's ``ee_action``/``hand_cmd`` protocol encodes rotation as
Euler xyz and predicts *relative* deltas, needing per-step Euler<->quat conversion and delta
reconstruction. The ``LightwheelAI/iros2026-ikea-assembly`` checkpoint's action groups
(``left_wrist_pose``/``right_wrist_pose``/``gripper``/``base_height_command``/``navigate_command``/
``torso_orientation_rpy_command``, see
``isaaclab_arena_gr00t/embodiments/g1_dex1_ikea_lightwheel/g1_dex1_ikea_lightwheel_data_gr00t_n_1_7_config.py``)
are instead an ``ABSOLUTE`` passthrough of robofinals' own ``G1-Gripper-Controller-DecoupledWBC``
command -- pos(3)+quat wxyz(4) per wrist, no Euler encoding, no delta reconstruction -- which is
column-for-column the same layout
:class:`~isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action.G1DecoupledWBCPinkAction`
already expects (see that class's ``process_actions`` docstring), modulo the wxyz->xyzw quaternion
reorder. See ``isaaclab_arena_gr00t/policy/lightwheel_hdf5_replay_policy.py`` for the same mapping
applied to raw dataset replay instead of a live policy.

Targets ``g1_wbc_agile_pink_dex1_continuous_grip`` (27-D action,
:class:`~isaaclab_arena.embodiments.g1.g1.G1WBCAgilePinkDex1ContinuousGripActionCfg`), not the plain
``g1_wbc_agile_pink_dex1``: the dataset's gripper command is a continuous -1=open/+1=close ramp
(robofinals' ``Dex1GripperCfg.process_hand``), which the binary embodiment's
``BinaryJointPositionAction`` would collapse to two extremes.

The dataset was recorded in-sim at 50 Hz (``env_args.sim_args``: ``dt=0.005 x decimation=4``),
matching Arena's own ``assemble_table`` step rate exactly -- unlike the BitRobot EEF client's 30 Hz
dataset, no upsampling is needed here.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import Any

from gr00t.policy.server_client import PolicyClient as Gr00tPolicyClient

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.action_scheduling import ActionChunkScheduler
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

# --------------------------------------------------------------------------------------- #
# g1_wbc_agile_pink_dex1_continuous_grip action tensor layout: G1DecoupledWBCPinkAction's
# 23-D WBC pose command ++ 2x2-wide continuous Dex1 gripper (one scalar per hand, duplicated
# across that hand's 2 finger joints -- see G1WBCAgilePinkDex1ContinuousGripActionCfg).
# --------------------------------------------------------------------------------------- #
_LEFT_HAND_STATE_IDX = 0
_RIGHT_HAND_STATE_IDX = 1
_LEFT_POS_SLICE = slice(2, 5)
_LEFT_QUAT_XYZW_SLICE = slice(5, 9)
_RIGHT_POS_SLICE = slice(9, 12)
_RIGHT_QUAT_XYZW_SLICE = slice(12, 16)
_NAVIGATE_SLICE = slice(16, 19)
_BASE_HEIGHT_IDX = 19
_TORSO_RPY_SLICE = slice(20, 23)
_LEFT_GRIPPER_SLICE = slice(23, 25)
_RIGHT_GRIPPER_SLICE = slice(25, 27)
_ACTION_DIM = 27

# gr00t_policy_joint_space.yaml's per-group joint order (isaaclab_arena_gr00t/embodiments/
# g1_dex1_ikea_lightwheel/) -- the exact 33-wide layout observation.state was trained on.
# Duplicated here rather than parsed from the yaml at runtime, matching
# lightwheel_hdf5_replay_policy.py's own precedent for these joint-order constants.
_STATE_JOINT_GROUPS: dict[str, list[str]] = {
    "left_leg": [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ],
    "right_leg": [
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ],
    "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    "left_arm": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
    "left_gripper": ["left_dex1_finger_joint_1", "left_dex1_finger_joint_2"],
    "right_arm": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    "right_gripper": ["right_dex1_finger_joint_1", "right_dex1_finger_joint_2"],
}

# meta/tasks.jsonl's single fixed instruction for this dataset (see g1_dex1_ikea_lightwheel_config.yaml's
# language_instruction) -- happens to already match assemble_table's own cfg.task_description, so no
# override of set_task_description (unlike the BitRobot EEF client) is needed to reach it in practice;
# kept as an explicit fallback only for a task_description of None.
_DEFAULT_LANGUAGE_INSTRUCTION = "Assemble the table leg into the tabletop"

# Dataset/sim rate both 50 Hz (dt=0.005 x decimation=4) -- see module docstring. Checked once per
# scheduler build rather than assumed, since a silent fps mismatch here was a real, hard-to-diagnose
# bug for the BitRobot client (ran the whole trajectory ~1.67x too fast).
_EXPECTED_SIM_STEP_DT = 0.02

_DEBUG = True
# Directory for per-chunk raw prediction/state dumps while _DEBUG is on -- compared offline
# against the dataset's own recorded action.eef_pose/action.gripper/observation.state to check
# whether the live closed-loop predictions are in the same numeric ballpark as training data,
# same methodology as replay_data/validate_eef_composition.py's own offline checks.
_DEBUG_DUMP_DIR = "/tmp/gr00t_dex1_wbc_debug"


def _wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Reorder a (..., 4) wxyz quaternion array to xyzw."""
    return quat_wxyz[..., [1, 2, 3, 0]]


@dataclass
class Gr00tDex1WBCClosedloopPolicyCfg(PolicyCfg):
    """Configure the GR00T Dex1 native-WBC closed-loop policy."""

    remote_host: str = "localhost"
    """GR00T policy server hostname."""

    remote_port: int = 5555
    """GR00T policy server port."""

    remote_api_token: str | None = None
    """Optional policy-server API token."""

    action_horizon: int = 40
    """Number of actions in the policy's prediction horizon (must match training's action delta_indices length)."""

    action_chunk_length: int = 16
    """Number of actions to execute per inference rollout (<= action_horizon)."""

    head_cam_name_sim: str = "dataset_first_person_cam_rgb"
    """Name of the first-person camera observation in ``obs['camera_obs']``."""

    left_hand_cam_name_sim: str = "left_hand_cam_rgb"
    """Name of the left wrist camera observation in ``obs['camera_obs']``."""

    right_hand_cam_name_sim: str = "right_hand_cam_rgb"
    """Name of the right wrist camera observation in ``obs['camera_obs']``."""

    policy_device: str = "cuda"
    """Device used for Arena-side tensor operations."""


@register_policy
class Gr00tDex1WBCClosedloopPolicy(PolicyBase[Gr00tDex1WBCClosedloopPolicyCfg]):
    """GR00T closed-loop policy for the ``g1_wbc_agile_pink_dex1_continuous_grip`` embodiment.

    Connects to a GR00T policy server (``gr00t/eval/run_gr00t_server.py``) serving the
    ``LightwheelAI/iros2026-ikea-assembly`` checkpoint, and forwards its native WBC pose/gripper
    prediction into Arena's action tensor with only a quaternion reorder -- no Euler conversion or
    relative-delta reconstruction needed (see module docstring).
    """

    name = "gr00t_dex1_wbc_closedloop"

    def __init__(self, config: Gr00tDex1WBCClosedloopPolicyCfg):
        super().__init__(config)
        self.device = config.policy_device
        self.num_envs = 1

        self._chunking_state: ActionChunkScheduler | None = None
        self._joint_ids_per_group: dict[str, list[int]] | None = None

        client = Gr00tPolicyClient(
            host=config.remote_host,
            port=config.remote_port,
            api_token=config.remote_api_token,
            strict=False,
        )
        if not client.ping():
            raise ConnectionError(f"Cannot reach GR00T policy server at {config.remote_host}:{config.remote_port}")
        self._client: Gr00tPolicyClient | None = client

        self.task_description: str | None = None
        self._chunk_count = 0
        if _DEBUG:
            import os

            os.makedirs(_DEBUG_DUMP_DIR, exist_ok=True)

    def _ensure_scheduler(self, env) -> None:
        if self._chunking_state is not None:
            return
        sim_dt = env.unwrapped.step_dt
        assert abs(sim_dt - _EXPECTED_SIM_STEP_DT) < 1e-6, (
            f"Sim step_dt={sim_dt} does not match the checkpoint's recorded 50Hz ({_EXPECTED_SIM_STEP_DT}s) --"
            " this policy assumes a 1:1 dataset-fps to sim-fps mapping, see module docstring."
        )
        self._chunking_state = ActionChunkScheduler(
            num_envs=self.num_envs,
            action_chunk_length=self.config.action_chunk_length,
            action_horizon=self.config.action_horizon,
            action_dim=_ACTION_DIM,
            device=self.device,
            dtype=torch.float,
        )

    def _resolve_joint_ids(self, env) -> None:
        asset = env.unwrapped.scene["robot"]
        self._joint_ids_per_group = {
            group: asset.find_joints(names)[0] for group, names in _STATE_JOINT_GROUPS.items()
        }

    def _build_state(self, env) -> dict[str, np.ndarray]:
        if self._joint_ids_per_group is None:
            self._resolve_joint_ids(env)
        joint_pos = env.unwrapped.scene["robot"].data.joint_pos[0].detach().cpu().numpy()
        return {
            group: joint_pos[ids].astype(np.float32) for group, ids in self._joint_ids_per_group.items()
        }

    def _fetch_action_chunk(self, env, observation: dict[str, Any]) -> torch.Tensor:
        assert self._client is not None, "GR00T Dex1 WBC policy has been closed"
        assert self.task_description is not None, "Task description is not set"

        cam_obs = observation["camera_obs"]
        first_person = cam_obs[self.config.head_cam_name_sim][0].detach().cpu().numpy()
        left_hand = cam_obs[self.config.left_hand_cam_name_sim][0].detach().cpu().numpy()
        right_hand = cam_obs[self.config.right_hand_cam_name_sim][0].detach().cpu().numpy()

        state = self._build_state(env)

        policy_observations = {
            "video": {
                "first_person": first_person[None, None],
                "left_hand": left_hand[None, None],
                "right_hand": right_hand[None, None],
            },
            "state": {group: arr[None, None] for group, arr in state.items()},
            "language": {"annotation.human.task_description": [[self.task_description]]},
        }

        action_dict, _ = self._client.get_action(policy_observations)
        left_wrist_pose = np.asarray(action_dict["left_wrist_pose"])[0]  # (horizon, 7) pos(3)+quat_wxyz(4)
        right_wrist_pose = np.asarray(action_dict["right_wrist_pose"])[0]  # (horizon, 7)
        gripper = np.asarray(action_dict["gripper"])[0]  # (horizon, 2) [left, right], -1=open/+1=close
        base_height_cmd = np.asarray(action_dict["base_height_command"])[0]  # (horizon, 1)
        navigate_cmd = np.asarray(action_dict["navigate_command"])[0]  # (horizon, 3)
        torso_rpy_cmd = np.asarray(action_dict["torso_orientation_rpy_command"])[0]  # (horizon, 3)
        horizon = left_wrist_pose.shape[0]

        if _DEBUG:
            self._chunk_count += 1
            print(f"[gr00t_dex1_wbc_debug] chunk {self._chunk_count}, task_description: {self.task_description!r}")
            print(f"[gr00t_dex1_wbc_debug] left_wrist_pose[0]: {left_wrist_pose[0]}")
            print(f"[gr00t_dex1_wbc_debug] right_wrist_pose[0]: {right_wrist_pose[0]}")
            print(f"[gr00t_dex1_wbc_debug] gripper[0]: {gripper[0]}, navigate_cmd[0]: {navigate_cmd[0]}")
            print(f"[gr00t_dex1_wbc_debug] base_height_cmd[0]: {base_height_cmd[0]}, torso_rpy_cmd[0]: {torso_rpy_cmd[0]}")
            print(
                f"[gr00t_dex1_wbc_debug] left_wrist_pose min/max per-dim: "
                f"{left_wrist_pose.min(axis=0)} / {left_wrist_pose.max(axis=0)}"
            )
            np.savez(
                f"{_DEBUG_DUMP_DIR}/chunk_{self._chunk_count:03d}.npz",
                **{f"state_{group}": arr for group, arr in state.items()},
                left_wrist_pose=left_wrist_pose,
                right_wrist_pose=right_wrist_pose,
                gripper=gripper,
                base_height_cmd=base_height_cmd,
                navigate_cmd=navigate_cmd,
                torso_rpy_cmd=torso_rpy_cmd,
                task_description=self.task_description,
            )

        actions = np.zeros((horizon, _ACTION_DIM), dtype=np.float32)
        actions[:, _LEFT_HAND_STATE_IDX] = 0.0
        actions[:, _RIGHT_HAND_STATE_IDX] = 0.0
        actions[:, _LEFT_POS_SLICE] = left_wrist_pose[:, 0:3]
        actions[:, _LEFT_QUAT_XYZW_SLICE] = _wxyz_to_xyzw(left_wrist_pose[:, 3:7])
        actions[:, _RIGHT_POS_SLICE] = right_wrist_pose[:, 0:3]
        actions[:, _RIGHT_QUAT_XYZW_SLICE] = _wxyz_to_xyzw(right_wrist_pose[:, 3:7])
        actions[:, _NAVIGATE_SLICE] = navigate_cmd
        actions[:, _BASE_HEIGHT_IDX] = base_height_cmd[:, 0]
        actions[:, _TORSO_RPY_SLICE] = torso_rpy_cmd
        # Each hand's single predicted scalar broadcasts across that hand's 2 finger-joint slots,
        # same duplication lightwheel_hdf5_replay_policy.py applies to the raw recorded gripper column.
        actions[:, _LEFT_GRIPPER_SLICE] = gripper[:, 0:1]
        actions[:, _RIGHT_GRIPPER_SLICE] = gripper[:, 1:2]

        return torch.from_numpy(actions).unsqueeze(0).to(self.device)

    def get_action(self, env, observation: dict[str, Any]) -> torch.Tensor:
        self._ensure_scheduler(env)

        def fetch_chunk() -> torch.Tensor:
            return self._fetch_action_chunk(env, observation)

        assert self._chunking_state is not None
        return self._chunking_state.get_action(fetch_chunk, hold_action=None)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        assert self._client is not None, "GR00T Dex1 WBC policy has been closed"
        self._client.reset()
        if self._chunking_state is not None:
            self._chunking_state.reset(env_ids)

    def close(self) -> None:
        client = self._client
        try:
            if client is not None:
                socket = getattr(client, "socket", None)
                context = getattr(client, "context", None)
                try:
                    if socket is not None:
                        socket.close(linger=0)
                finally:
                    if context is not None:
                        context.term()
        finally:
            self._client = None

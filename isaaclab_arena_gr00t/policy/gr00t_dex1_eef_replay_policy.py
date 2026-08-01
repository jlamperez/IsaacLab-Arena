# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Open-loop replay of a recorded BitRobot episode's ``action.ee_action``/``action.hand_cmd``.

Diagnostic-only: bypasses the GR00T model and the remote policy client entirely, to isolate
whether the wrist-orientation instability seen in closed-loop rollouts comes from the model's
predictions or from Arena's own IK/WBC execution of an ``ee_action``-shaped command. If replaying
ground-truth actions from the real dataset also produces the same wrist jump, the bug is in
execution (IK/WBC), not in what the model predicts.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import Any

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena_gr00t.policy.gr00t_dex1_eef_closedloop_policy import (
    _ACTION_DIM,
    _BASE_HEIGHT_IDX,
    _DEFAULT_BASE_HEIGHT,
    _HAND_STATE_SCALE,
    _LEFT_GRIPPER_IDX,
    _LEFT_POS_SLICE,
    _LEFT_QUAT_SLICE,
    _NAVIGATE_SLICE,
    _RIGHT_GRIPPER_IDX,
    _RIGHT_POS_SLICE,
    _RIGHT_QUAT_SLICE,
    _TORSO_RPY_SLICE,
    _euler_xyz_to_quat_xyzw,
    _quat_xyzw_to_euler_xyz,
    _upsample_to_sim_rate,
)

_DATASET_FPS = 30.0


@dataclass
class Gr00tDex1EEFReplayPolicyCfg(PolicyCfg):
    """Configure the ground-truth replay policy."""

    npz_path: str = (
        "/workspaces/isaaclab_arena/isaaclab_arena_gr00t/policy/replay_data/episode_000000_actions.npz"
    )
    """Path to a .npz with ``ee_action`` (N,12), ``hand_cmd`` (N,2), ``navigate_cmd`` (N,3) arrays from one
    recorded episode. Lives under ``policy/replay_data/`` (gitignored, not ``/tmp``) so it survives container
    restarts and doesn't need re-copying in via ``docker cp`` each time."""

    policy_device: str = "cuda"
    """Device used for the action tensor."""


@register_policy
class Gr00tDex1EEFReplayPolicy(PolicyBase[Gr00tDex1EEFReplayPolicyCfg]):
    """Replays a recorded episode's ``ee_action``/``hand_cmd`` open-loop, no model in the loop."""

    name = "gr00t_dex1_eef_replay"

    def __init__(self, config: Gr00tDex1EEFReplayPolicyCfg):
        super().__init__(config)
        self.device = config.policy_device
        self.num_envs = 1
        data = np.load(config.npz_path)
        self._ee_action = data["ee_action"]
        self._hand_cmd = data["hand_cmd"]
        # (N,3) body-frame [lin_vel_x, lin_vel_y, ang_vel_z], derived offline from
        # action.robot_q_desired's root pose -- absent for older cached .npz files.
        self._navigate_cmd = data["navigate_cmd"] if "navigate_cmd" in data else None
        self._sim_actions: torch.Tensor | None = None
        self._step_idx = 0
        self.task_description: str | None = None

    def set_task_description(self, task_description: str | None) -> str:
        self.task_description = task_description or "replay"
        return self.task_description

    def _build_sim_actions(self, env) -> None:
        if self._sim_actions is not None:
            return
        sim_dt = env.unwrapped.step_dt
        fps_ratio = (1.0 / sim_dt) / _DATASET_FPS

        n_frames = self._ee_action.shape[0]
        actions = np.zeros((n_frames, _ACTION_DIM), dtype=np.float32)
        actions[:, _LEFT_POS_SLICE] = self._ee_action[:, 0:3]
        actions[:, _RIGHT_POS_SLICE] = self._ee_action[:, 6:9]
        for t in range(n_frames):
            actions[t, _LEFT_QUAT_SLICE] = _euler_xyz_to_quat_xyzw(self._ee_action[t, 3:6])
            actions[t, _RIGHT_QUAT_SLICE] = _euler_xyz_to_quat_xyzw(self._ee_action[t, 9:12])
        actions[:, _NAVIGATE_SLICE] = 0.0 if self._navigate_cmd is None else self._navigate_cmd
        actions[:, _BASE_HEIGHT_IDX] = _DEFAULT_BASE_HEIGHT
        actions[:, _TORSO_RPY_SLICE] = 0.0
        actions[:, _LEFT_GRIPPER_IDX] = (self._hand_cmd[:, 0] / _HAND_STATE_SCALE) * 2.0 - 1.0
        actions[:, _RIGHT_GRIPPER_IDX] = (self._hand_cmd[:, 1] / _HAND_STATE_SCALE) * 2.0 - 1.0

        actions = _upsample_to_sim_rate(actions, fps_ratio)
        self._sim_actions = torch.from_numpy(actions).to(self.device)
        print(f"[gr00t_dex1_replay] built {actions.shape[0]} sim-rate actions from {n_frames} dataset frames")

    def get_action(self, env, observation: dict[str, Any]) -> torch.Tensor:
        self._build_sim_actions(env)
        assert self._sim_actions is not None
        idx = min(self._step_idx, self._sim_actions.shape[0] - 1)
        if idx < 5:
            try:
                from PIL import Image

                cam = observation["camera_obs"]["robot_head_cam_rgb"][0].detach().cpu().numpy()
                Image.fromarray(cam).save(f"/tmp/replay_camcheck_{idx:02d}.png")
            except Exception as e:  # noqa: BLE001
                print(f"[gr00t_dex1_replay] camcheck save failed: {e}")
        if idx % 5 == 0:
            policy_obs = observation["policy"]
            left_quat = policy_obs["left_eef_quat"][0].detach().cpu().numpy()
            right_quat = policy_obs["right_eef_quat"][0].detach().cpu().numpy()
            left_euler = _quat_xyzw_to_euler_xyz(left_quat)
            right_euler = _quat_xyzw_to_euler_xyz(right_quat)
            print(f"[gr00t_dex1_replay] step {idx}: left_euler={left_euler} right_euler={right_euler}")
        self._step_idx += 1
        return self._sim_actions[idx].unsqueeze(0)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._step_idx = 0

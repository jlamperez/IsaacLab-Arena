# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GR00T remote closed-loop policy for the G1 + Dex1 gripper embodiment.

Unlike :mod:`isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy`, this policy does
not go through Arena's joint-name-keyed ``gr00t_core`` remap machinery. The BitRobot Dex1
IKEA checkpoint (trained via ``action.ee_action`` / ``action.hand_cmd``) predicts Cartesian
end-effector targets, not per-joint positions, matching
:class:`~isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action.G1DecoupledWBCPinkAction`'s
pose-based action tensor directly.

Rotation convention: the dataset's 12-D ``ee_state``/``ee_action`` (per hand: 3-D position +
3-D rotation) uses intrinsic-free ``xyz`` Euler angles (roll, pitch, yaw with pitch bounded to
+/- pi/2 -- verified empirically against the dataset's per-axis stats, and numerically matched
to ``scipy.spatial.transform.Rotation.from_euler("xyz", ...)`` against the quaternion formula in
the sibling ``robofinals`` repo's ``rotation_helper.get_euler_xyz``).
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as SciRotation
from typing import Any

from gr00t.policy.server_client import PolicyClient as Gr00tPolicyClient

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.action_scheduling import ActionChunkScheduler
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena_gr00t.utils.image_conversion import resize_frames_with_padding

# --------------------------------------------------------------------------------------- #
# g1_wbc_agile_pink_dex1 action tensor layout (see G1DecoupledWBCPinkAction.process_actions
# docstring + G1WBCAgilePinkDex1ActionCfg): 23-D WBC pose action ++ 2 Dex1 gripper scalars.
# --------------------------------------------------------------------------------------- #
_LEFT_HAND_STATE_IDX = 0
_RIGHT_HAND_STATE_IDX = 1
_LEFT_POS_SLICE = slice(2, 5)
_LEFT_QUAT_SLICE = slice(5, 9)
_RIGHT_POS_SLICE = slice(9, 12)
_RIGHT_QUAT_SLICE = slice(12, 16)
_NAVIGATE_SLICE = slice(16, 19)
_BASE_HEIGHT_IDX = 19
_TORSO_RPY_SLICE = slice(20, 23)
_LEFT_GRIPPER_IDX = 23
_RIGHT_GRIPPER_IDX = 24
_ACTION_DIM = 25

# Matches G1DecoupledWBCJointAction's own standing-height default -- do not use 0.0 here,
# base_height_command is an absolute target pelvis height, not a delta.
_DEFAULT_BASE_HEIGHT = 0.75

# robofinals' Dex1GripperCfg OPEN_POS/CLOSE_POS (radians), mirrored in g1.py's
# _make_dex1_gripper_variant.
_DEX1_OPEN_POS = 0.0245
_DEX1_CLOSE_POS = -0.02
# BitRobot dataset's hand_state/hand_cmd scale: 0.0 (close) -> 5.5 (open).
_HAND_STATE_SCALE = 5.5

_LEFT_DEX1_JOINTS = ["left_dex1_finger_joint_1", "left_dex1_finger_joint_2"]
_RIGHT_DEX1_JOINTS = ["right_dex1_finger_joint_1", "right_dex1_finger_joint_2"]

# Fallback instruction used only if the runner doesn't supply one via the shared
# --language_instruction CLI flag or the task's own task_description. Must be one of the
# BitRobot dataset's own literal per-frame subtask labels (meta/tasks.jsonl) -- the
# checkpoint was trained on task_index changing frame-by-frame between exactly these 8
# strings ("insert table leg to table base", "move to table", "flip table",
# "rotate leg to tighten", "pick table leg", "rotate table base", "building children table",
# "move table base"), never a fixed whole-assembly instruction. "pick table leg" matches
# the natural starting subtask from assemble_table's reset pose (robot standing, leg placed
# nearby, nothing grasped yet).
_DEFAULT_LANGUAGE_INSTRUCTION = "pick table leg"

# (height, width, channels) the checkpoint was trained on. Not CLI-configurable: the
# dataclass-to-argparse helper doesn't support tuple-typed fields.
_TARGET_IMAGE_SIZE = (480, 640, 3)

# Temporary debug logging while diagnosing closed-loop stability. Flip off once resolved.
_DEBUG = True
# Separate, off-by-default toggle for the per-step camera-frame PNG dump + pose prints in
# get_action() -- that block was for the (now-fixed) camera-tracking investigation and
# writes one file per sim step, so keep it off during normal _DEBUG action-trace runs.
_DEBUG_CAM_PERSTEP = False

# The BitRobot dataset (and this checkpoint's action_horizon) was recorded at 30 Hz
# (confirmed in the dataset's own meta/info.json). Arena's assemble_table env steps at
# 50 Hz (sim dt=0.005 x decimation=4). Without correcting for this, each model-predicted
# "frame" (meant to span 1/30s) gets applied to a single 1/50s sim step, running the
# whole trajectory ~1.67x faster than it was recorded -- a very plausible contributor to
# the WBC balance failures observed in closed-loop rollouts. See
# isaaclab_arena_gr00t/lerobot/config/g1_static_apple_config.yaml for a working reference
# example that has no such mismatch, because it was recorded in-sim at Arena's own 50 Hz.
_DATASET_FPS = 30.0


def _upsample_to_sim_rate(actions: np.ndarray, fps_ratio: float) -> np.ndarray:
    """Stretch a (n_frames, D) array recorded at ``_DATASET_FPS`` to Arena's sim rate.

    Repeats each row a variable number of times (1 or 2 for a ~1.667 ratio) using a
    cumulative-rounding ("Bresenham-style") accumulator, so every ``_DATASET_FPS``-spaced
    frame occupies the correct number of sim steps on average, instead of exactly one.

    Args:
        actions: (n_frames, action_dim) array, one row per dataset-fps frame.
        fps_ratio: sim_fps / _DATASET_FPS (e.g. 50/30 ~= 1.667).

    Returns:
        (n_sim_steps, action_dim) array with rows repeated to match sim_fps timing.
    """
    n_frames = actions.shape[0]
    cumulative_end_step = np.round(np.arange(1, n_frames + 1) * fps_ratio).astype(int)
    cumulative_start_step = np.concatenate([[0], cumulative_end_step[:-1]])
    hold_counts = cumulative_end_step - cumulative_start_step
    return np.repeat(actions, hold_counts, axis=0)


# HISTORY (2026-07-28 to 2026-08-01): the reset-pose wrist read as yaw ~= +/-pi, right at
# the Euler wraparound discontinuity and ~180 deg off from the BitRobot dataset's own
# convention. A _WRIST_FRAME_OFFSET (composing a fixed 180-deg-about-z correction) was added,
# then removed after empirical testing suggested it made replay worse, then this whole
# ~180-deg mismatch was investigated for days as a robot-geometry / training-data-convention
# problem. Root cause, found 2026-08-01: neither -- it was a quaternion component-order bug
# right here. isaaclab_arena.terms.transforms.get_target_link_quaternion_in_target_frame
# (which left_eef_quat/right_eef_quat are built from) returns isaaclab.utils.math.quat_from_matrix's
# output, which its own docstring documents as (x, y, z, w) -- NOT IsaacLab's usual (w, x, y, z).
# This function was treating that already-xyzw quaternion as wxyz and reordering it again,
# which for a near-identity input produces exactly a spurious ~180-deg-about-z reading. No
# frame-offset is needed at all once the ordering itself is correct.
def _quat_xyzw_to_euler_xyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return SciRotation.from_quat(quat_xyzw).as_euler("xyz")


def _euler_xyz_to_quat_xyzw(euler_xyz: np.ndarray) -> np.ndarray:
    return SciRotation.from_euler("xyz", euler_xyz).as_quat()


@dataclass
class Gr00tDex1EEFClosedloopPolicyCfg(PolicyCfg):
    """Configure the GR00T Dex1 end-effector closed-loop policy."""

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

    pov_cam_name_sim: str = "robot_head_cam_rgb"
    """Name of the head camera observation in ``obs['camera_obs']``."""

    policy_device: str = "cuda"
    """Device used for Arena-side tensor operations."""


@register_policy
class Gr00tDex1EEFClosedloopPolicy(PolicyBase[Gr00tDex1EEFClosedloopPolicyCfg]):
    """GR00T closed-loop policy for the ``g1_wbc_agile_pink_dex1`` embodiment.

    Connects to a GR00T policy server (``gr00t/eval/run_gr00t_server.py``) serving an
    ``ee_action``/``hand_cmd`` checkpoint, and translates its Cartesian wrist-pose +
    gripper-command output into Arena's pose-based WBC action tensor each step.
    """

    name = "gr00t_dex1_eef_closedloop"

    def __init__(self, config: Gr00tDex1EEFClosedloopPolicyCfg):
        super().__init__(config)
        self.device = config.policy_device
        self.num_envs = 1

        # Deferred until the first get_action() call: building the scheduler needs
        # env.unwrapped.step_dt to compute the dataset-fps -> sim-fps ratio, and env is
        # not available until then.
        self._chunking_state: ActionChunkScheduler | None = None
        self._fps_ratio: float | None = None

        client = Gr00tPolicyClient(
            host=config.remote_host,
            port=config.remote_port,
            api_token=config.remote_api_token,
            strict=False,
        )
        if not client.ping():
            raise ConnectionError(f"Cannot reach GR00T policy server at {config.remote_host}:{config.remote_port}")
        self._client: Gr00tPolicyClient | None = client

        self._dex1_joint_ids: tuple[list[int], list[int]] | None = None
        self.task_description: str | None = None

    def set_task_description(self, task_description: str | None) -> str:  # noqa: ARG002
        # Ignore whatever the environment passes (assemble_table's own free-text
        # task_description, "Assemble the table leg into the tabletop", is never None, so
        # the old `if task_description is None` fallback never actually triggered -- this
        # checkpoint was fine-tuned on task_index changing between the BitRobot dataset's
        # own literal per-frame subtask strings (see meta/tasks.jsonl), never a fixed
        # whole-assembly instruction, so a free-text description can't match by
        # construction. Always use our own constant instead.
        self.task_description = _DEFAULT_LANGUAGE_INSTRUCTION
        return self.task_description

    def _ensure_scheduler(self, env) -> None:
        """Lazily build the action-chunk scheduler once ``env.unwrapped.step_dt`` is known.

        Scales ``config.action_horizon``/``config.action_chunk_length`` (expressed in
        dataset-fps "frame" units, matching what the model actually predicts) up to the
        sim's real step rate, so the scheduler's buffer length matches the upsampled
        action array built in :meth:`_fetch_action_chunk`.
        """
        if self._chunking_state is not None:
            return
        sim_dt = env.unwrapped.step_dt
        fps_ratio = (1.0 / sim_dt) / _DATASET_FPS
        self._fps_ratio = fps_ratio
        sim_action_horizon = round(self.config.action_horizon * fps_ratio)
        sim_action_chunk_length = round(self.config.action_chunk_length * fps_ratio)
        self._chunking_state = ActionChunkScheduler(
            num_envs=self.num_envs,
            action_chunk_length=sim_action_chunk_length,
            action_horizon=sim_action_horizon,
            action_dim=_ACTION_DIM,
            device=self.device,
            dtype=torch.float,
        )

    def _resolve_dex1_joint_ids(self, env) -> None:
        asset = env.unwrapped.scene["robot"]
        left_ids, _ = asset.find_joints(_LEFT_DEX1_JOINTS)
        right_ids, _ = asset.find_joints(_RIGHT_DEX1_JOINTS)
        self._dex1_joint_ids = (left_ids, right_ids)

    def _build_ee_state_and_hand_state(self, env, observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Build the 12-D ee_state and 2-D hand_state matching the BitRobot dataset's convention."""
        policy_obs = observation["policy"]
        left_pos = policy_obs["left_eef_pos"][0].detach().cpu().numpy()
        left_quat = policy_obs["left_eef_quat"][0].detach().cpu().numpy()
        right_pos = policy_obs["right_eef_pos"][0].detach().cpu().numpy()
        right_quat = policy_obs["right_eef_quat"][0].detach().cpu().numpy()

        left_euler = _quat_xyzw_to_euler_xyz(left_quat)
        right_euler = _quat_xyzw_to_euler_xyz(right_quat)
        ee_state = np.concatenate([left_pos, left_euler, right_pos, right_euler]).astype(np.float32)

        if self._dex1_joint_ids is None:
            self._resolve_dex1_joint_ids(env)
        left_ids, right_ids = self._dex1_joint_ids
        joint_pos = env.unwrapped.scene["robot"].data.joint_pos[0].detach().cpu().numpy()
        left_finger_pos = float(joint_pos[left_ids].mean())
        right_finger_pos = float(joint_pos[right_ids].mean())

        if _DEBUG and not getattr(self, "_printed_arm_joints", False):
            self._printed_arm_joints = True
            names = env.unwrapped.scene["robot"].data.joint_names
            print("[gr00t_dex1_debug] arm/wrist joint angles at reset:")
            for i, n in enumerate(names):
                if any(k in n for k in ("shoulder", "elbow", "wrist", "waist", "torso", "pelvis")):
                    print(f"  {n:35s} = {joint_pos[i]:8.4f}")

            import isaaclab_arena.terms.transforms as transforms_terms

            chain = [
                "torso_link",
                "left_shoulder_pitch_link",
                "left_shoulder_roll_link",
                "left_shoulder_yaw_link",
                "left_elbow_link",
                "left_wrist_roll_link",
                "left_wrist_pitch_link",
                "left_wrist_yaw_link",
            ]
            print("[gr00t_dex1_debug] link orientations relative to pelvis, walking the chain:")
            for link in chain:
                q = transforms_terms.get_target_link_quaternion_in_target_frame(
                    env.unwrapped, target_link_name=link
                )[0].detach().cpu().numpy()
                euler = _quat_xyzw_to_euler_xyz(q)
                print(f"  {link:28s} quat_wxyz={q}  euler_xyz={euler}")

        def _to_hand_state_scale(pos: float) -> float:
            normalized = (pos - _DEX1_CLOSE_POS) / (_DEX1_OPEN_POS - _DEX1_CLOSE_POS)
            return float(np.clip(normalized, 0.0, 1.0)) * _HAND_STATE_SCALE

        hand_state = np.array(
            [_to_hand_state_scale(left_finger_pos), _to_hand_state_scale(right_finger_pos)],
            dtype=np.float32,
        )
        return ee_state, hand_state

    def _fetch_action_chunk(self, env, observation: dict[str, Any]) -> torch.Tensor:
        assert self._client is not None, "GR00T Dex1 EEF policy has been closed"
        assert self.task_description is not None, "Task description is not set"

        cam = observation["camera_obs"][self.config.pov_cam_name_sim][0].detach().cpu().numpy()
        target_h, target_w, _ = _TARGET_IMAGE_SIZE
        if cam.shape[:2] != (target_h, target_w):
            cam = resize_frames_with_padding(
                cam[None], target_image_size=_TARGET_IMAGE_SIZE, bgr_conversion=False, pad_img=True
            )[0]

        ee_state, hand_state = self._build_ee_state_and_hand_state(env, observation)

        if _DEBUG:
            print(f"[gr00t_dex1_debug] task_description (actually sent to GR00T): {self.task_description!r}")
            print(f"[gr00t_dex1_debug] ee_state (current, input to policy): {ee_state}")
            print(f"[gr00t_dex1_debug] hand_state (current, input to policy): {hand_state}")
            try:
                from PIL import Image

                self._debug_frame_count = getattr(self, "_debug_frame_count", 0) + 1
                Image.fromarray(cam).save(f"/tmp/gr00t_dex1_cam0_debug_{self._debug_frame_count:03d}.png")
            except Exception as e:  # noqa: BLE001
                print(f"[gr00t_dex1_debug] failed to save debug camera frame: {e}")

        policy_observations = {
            "video": {"cam_0": cam[None, None]},
            "state": {
                # ee_state is split into per-hand keys (meta/modality.json's ee_state_left
                # (0:6) / ee_state_right (6:12) joint groups) since EndEffectorPose models a
                # single end-effector pose, not two hands packed into one 12-D array.
                "ee_state_left": ee_state[None, None, 0:6],
                "ee_state_right": ee_state[None, None, 6:12],
                "hand_state": hand_state[None, None],
            },
            "language": {"annotation.human.task_description": [[self.task_description]]},
        }

        action_dict, _ = self._client.get_action(policy_observations)
        ee_action_left = np.asarray(action_dict["ee_action_left"])[0]  # (horizon, 6)
        ee_action_right = np.asarray(action_dict["ee_action_right"])[0]  # (horizon, 6)
        ee_action = np.concatenate([ee_action_left, ee_action_right], axis=1)  # (horizon, 12)
        hand_cmd = np.asarray(action_dict["hand_cmd"])[0]  # (horizon, 2)
        horizon = ee_action.shape[0]

        if _DEBUG:
            print(f"[gr00t_dex1_debug] ee_action[0] (predicted, first step of chunk): {ee_action[0]}")
            print(f"[gr00t_dex1_debug] ee_action min/max per-dim: {ee_action.min(axis=0)} / {ee_action.max(axis=0)}")
            print(f"[gr00t_dex1_debug] hand_cmd[0]: {hand_cmd[0]}")
            np.savez(
                f"/tmp/gr00t_dex1_chunk_{self._debug_frame_count:03d}.npz",
                ee_state=ee_state,
                hand_state=hand_state,
                ee_action=ee_action,
                hand_cmd=hand_cmd,
                task_description=self.task_description,
            )

        actions = np.zeros((horizon, _ACTION_DIM), dtype=np.float32)
        actions[:, _LEFT_HAND_STATE_IDX] = 0.0
        actions[:, _RIGHT_HAND_STATE_IDX] = 0.0
        actions[:, _LEFT_POS_SLICE] = ee_action[:, 0:3]
        actions[:, _RIGHT_POS_SLICE] = ee_action[:, 6:9]
        for t in range(horizon):
            actions[t, _LEFT_QUAT_SLICE] = _euler_xyz_to_quat_xyzw(ee_action[t, 3:6])
            actions[t, _RIGHT_QUAT_SLICE] = _euler_xyz_to_quat_xyzw(ee_action[t, 9:12])
        actions[:, _NAVIGATE_SLICE] = 0.0
        actions[:, _BASE_HEIGHT_IDX] = _DEFAULT_BASE_HEIGHT
        actions[:, _TORSO_RPY_SLICE] = 0.0
        # BitRobot hand_cmd: 0 (close) -> 5.5 (open). Arena's BinaryJointPositionActionCfg
        # convention: action > 0 => open, action < 0 => close (see g1.py's
        # G1WBCAgilePinkDex1ActionCfg gripper terms).
        actions[:, _LEFT_GRIPPER_IDX] = (hand_cmd[:, 0] / _HAND_STATE_SCALE) * 2.0 - 1.0
        actions[:, _RIGHT_GRIPPER_IDX] = (hand_cmd[:, 1] / _HAND_STATE_SCALE) * 2.0 - 1.0

        assert self._fps_ratio is not None, "Scheduler must be built (via _ensure_scheduler) before fetching"
        actions = _upsample_to_sim_rate(actions, self._fps_ratio)
        if _DEBUG:
            print(f"[gr00t_dex1_debug] upsampled actions {horizon} frames (30fps) -> {actions.shape[0]} sim steps")

        return torch.from_numpy(actions).unsqueeze(0).to(self.device)

    def get_action(self, env, observation: dict[str, Any]) -> torch.Tensor:
        self._ensure_scheduler(env)

        if _DEBUG_CAM_PERSTEP:
            try:
                from PIL import Image

                cam_step = observation["camera_obs"][self.config.pov_cam_name_sim][0].detach().cpu().numpy()
                self._debug_step_count = getattr(self, "_debug_step_count", 0) + 1
                Image.fromarray(cam_step).save(f"/tmp/gr00t_dex1_cam0_perstep_{self._debug_step_count:04d}.png")
                cam_sensor = env.unwrapped.scene["robot_head_cam"]
                if self._debug_step_count == 1:
                    print(
                        "[gr00t_dex1_debug] cam_sensor.cfg.update_latest_camera_pose="
                        f"{cam_sensor.cfg.update_latest_camera_pose}"
                    )
                cam_pos_w = cam_sensor.data.pos_w[0].detach().cpu().numpy()
                cam_quat_w = cam_sensor.data.quat_w_world[0].detach().cpu().numpy()
                robot_root_pos_w = env.unwrapped.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()

                import warp as wp

                asset = env.unwrapped.scene["robot"]
                if self._debug_step_count == 1:
                    print(f"[gr00t_dex1_debug] body_names: {asset.data.body_names}")
                head_idx = asset.data.body_names.index("torso_link")
                head_state_w = wp.to_torch(asset.data.body_link_state_w)[0, head_idx, :7].detach().cpu().numpy()
                print(
                    f"[gr00t_dex1_debug] step {self._debug_step_count}: "
                    f"cam_pos_w={cam_pos_w} cam_quat_w={cam_quat_w} robot_root_pos_w={robot_root_pos_w} "
                    f"head_link_pos_quat_w={head_state_w}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"[gr00t_dex1_debug] failed to save per-step debug camera frame: {e}")

        def fetch_chunk() -> torch.Tensor:
            return self._fetch_action_chunk(env, observation)

        assert self._chunking_state is not None
        return self._chunking_state.get_action(fetch_chunk, hold_action=None)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        assert self._client is not None, "GR00T Dex1 EEF policy has been closed"
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

# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Open-loop replay of a robofinals/LightwheelAI HDF5 demo's recorded ``actions``.

Diagnostic-only: sanity-checks whether the ``LightwheelAI/iros2026-ikea-assembly`` dataset's
recorded action trajectory (grabbed via robofinals' ``G1-Gripper-Controller-DecoupledWBC``,
23-D per step) drives Arena's own ``g1_wbc_agile_pink_dex1`` embodiment in a physically
plausible way, once remapped into the ``g1_wbc_agile_pink_dex1_continuous_grip`` embodiment's
27-D action layout. Bypasses ``env.reset_to()``
entirely -- the dataset's ``initial_state`` rigid-object keys (e.g. ``Table278_Table278``,
``Leg001_01_Leg001``) and absolute world poses (anchored in a full Robocasa kitchen scene)
don't match Arena's own ``assemble_table`` scene (keys ``table278``/``leg001_2``/..., poses
near the origin). Object placement therefore stays Arena's own tuned layout, not a faithful
reproduction of the recorded episode's exact starting layout -- but the robot's own *start*
pose is reproduced per-demo: since ``assemble_table``'s ``table001`` sits at (almost exactly)
the same pose relative to its own ``leg001`` as the dataset's tabletop does to its own leg
(both ultimately sourced from the same ``Scene02.usd``), the recorded robot-relative-to-table
offset can be re-applied onto Arena's table position directly, no rotation needed. Computed
fresh from whichever ``hdf5_path``/``demo_name`` is configured, so different demos (which
don't all start from the same relative offset) get their own pose.

Only the *start* pose is forced -- tried forcing the robot's full recorded trajectory
(``states/articulation/robot/root_pose``) every step first (2026-08-08), to route around a
real Homie-v2-vs-Agile locomotion mismatch (the dataset was recorded with robofinals' own
G1DecoupledWBCAction, hardcoded to ``wbc_version="homie_v2"`` -- an ONNX checkpoint pair,
``ckpts/nv_wbc_v0904/homie_v2/{stand,walk}.onnx`` -- a different, independently-trained
network from Arena's own ``agile`` lower-body policy; replaying the same ``navigate_cmd``
values through Agile left the robot ~0.53-0.75m from the table throughout, vs. ~0.53m within
4s in the recording. Homie v2 also consumes ``torso_orientation_rpy_cmd`` for balance/reach
lean, which Agile has no input for at all -- silently a no-op). But teleporting the robot's
root every step moves the gripper without moving whatever it's holding (a freely-simulated
rigid body can't teleport along with it), turning every step into a small yank on a grasped
object -- reverted once this was traced as the actual cause of grasped legs slipping out, no
amount of friction/gripper-continuity tuning fixed it because the desync was geometric.
Reset-only sacrifices exactly matching Homie v2's mid-episode walking distance in exchange for
physically consistent contact once Agile takes over for the rest of the episode.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena.utils.pose import Pose

# g1_wbc_agile_pink_dex1_continuous_grip's action[2:23] (arm pos/quat x2, navigate_cmd,
# base_height_cmd, torso_orientation_rpy_cmd) matches robofinals' own
# G1DecoupledWBCAction[0:21] column-for-column -- same fields, same order, same
# pelvis-relative convention (verified against robofinals/core/mdp/actions/
# decoupled_wbc_action.py's process_actions slicing and empirically against this dataset's
# own per-column stats). Arena's [0:2] (hand_state) has no robofinals equivalent and is
# unused for the Dex1 embodiment (only the dexterous-hand WBC controller reads it); the
# dataset's [0:2] (gripper) map to Arena's [23:27] instead -- 2 dims each, not 1, because
# (unlike BinaryJointPositionAction, which accepts one scalar for N joints)
# JointPositionAction's action_dim is one component per joint, so each 2-joint gripper term
# needs its single recorded scalar duplicated across both slots (total action_dim 27, not
# 25 -- confirmed by a real ValueError the first time this ran with a 25-wide tensor).
#
# Quaternion order mismatch: robofinals' own process_actions comments its raw action slices
# as "(quat, wxyz -> converted to scipy xyzw internally)" -- the recorded columns are wxyz.
# Arena's G1DecoupledWBCPinkAction.process_actions docstring instead declares
# "left_arm_quat: dim=4, xyzw quaternion" and feeds it straight to scipy's R.from_quat with
# no reordering. Confirmed empirically too: frame 0's recorded left_arm_quat is [1,0,0,-0] --
# identity as wxyz (a resting arm, physically sensible), but a spurious ~180deg rotation
# about X if misread as xyzw. So the two quat sub-slices need an explicit wxyz->xyzw
# reorder; positions/navigate_cmd/base_height/torso_rpy have no such ambiguity (plain
# floats, not paired into a rotation) and copy straight across.
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
    """Reorder an (N, 4) wxyz quaternion array to xyzw."""
    return quat_wxyz[:, [1, 2, 3, 0]]

# Dataset key for the tabletop rigid object -- fixed by the task definition (same for every
# demo), only its recorded *pose* varies per file/demo.
_DATASET_TABLE_KEY = "Table001_Table001_01"

# assemble_table_environment.py's fixed_asset (table001) pose -- must stay in sync with that
# file's own literal if it's ever re-tuned. No shared constant exists to import (it's a local
# literal inside build()), so this is duplicated deliberately, same as the OPEN_POS/CLOSE_POS
# duplication between robofinals and g1.py.
#
# Z is Arena's literal 0.7994 minus 0.0140: confirmed via Scene02.usd (pxr.UsdGeom.XformCache)
# that /World/Table001 (the parent Xform, z=0.7994 -- what Arena's set_initial_pose positions)
# and /World/Table001/Table001_01 (the child prim, z=0.7854 -- the actual rigid body) differ
# by exactly that much in Z only (x/y agree to sub-mm). The dataset's own
# initial_state/rigid_object root_pose is recorded at the child level, so using the parent's
# Z here would bias every robot/leg offset computed relative to it by 1.4cm.
_ARENA_TABLE_POS = np.array([0.5, 0.0, 0.7994 - 0.0140])

# Column indices, within the 33-wide per-joint layout, of the 14 arm joints -- in
# isaaclab_arena.embodiments.g1.g1.DATASET_ARM_JOINT_NAMES's order. Verified 2026-08-09
# against env_args.action_space_definition's base_action applied_joint_names list (same
# order) and empirically against frame 0's values matching G1_GEARWBC_CFG's own
# init_state.joint_pos. Duplicated here rather than imported -- same reasoning as
# _ARENA_TABLE_POS's own duplication-not-import comment above.
#
# Same 33-wide column layout is shared by two different HDF5 arrays with very different
# meaning: states/articulation/robot/joint_position (realized state) and
# joint_targets/joint_pos_target (PD setpoint, 132-wide = 33 joints x 4 physics substeps
# per decimated step -- confirmed 2026-08-10 by diffing all 4 substep blocks against each
# other, bit-identical at every step, i.e. a ZOH target like Arena's own action-hold).
# Which one to replay as an *action* matters a lot, see the arm_joint_pos_ds load below.
_DATASET_ARM_JOINT_POSITION_COLUMNS = [11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

# waist_yaw_joint, waist_roll_joint, waist_pitch_joint, in the same joint_position layout
# (verified 2026-08-09, same method as the arm columns above).
_DATASET_WAIST_JOINT_POSITION_COLUMNS = [2, 5, 8]


@dataclass
class LightwheelHdf5ReplayPolicyCfg(PolicyCfg):
    """Configure the LightwheelAI HDF5 ground-truth replay policy."""

    hdf5_path: str = "/datasets/lerobot_cache/LightwheelAI/iros2026-ikea-assembly/data/AssembleTableTask_1784627181912351.hdf5"
    """Path to a robofinals-format demo-recording HDF5 file (``data/<demo_name>/actions``)."""

    demo_name: str = "demo_0"
    """Name of the demo group inside ``hdf5_path`` to replay."""

    policy_device: str = "cuda"
    """Device used for the action tensor."""

    replay_recorded_arm_joints: bool = False
    """Append the dataset's recorded arm joint PD targets after the base 27-D action, for
    use with ``g1_wbc_pink_dex1_direct_arm_continuous_grip`` -- see that embodiment's
    action_config docstring for why this bypasses Pink IK's solution multiplicity for the
    arms. Sourced from ``joint_targets/joint_pos_target`` (the setpoint robofinals' own PD
    tracked), not ``states/.../joint_position`` (their realized, already-lagged state) --
    see the ``arm_joint_pos_ds`` load in ``__init__`` for why that distinction matters.
    Leave False for the plain Pink-IK-driven Dex1 embodiments (wrong action_dim
    otherwise)."""


@register_policy
class LightwheelHdf5ReplayPolicy(PolicyBase[LightwheelHdf5ReplayPolicyCfg]):
    """Replays one robofinals HDF5 demo's recorded actions open-loop, no model in the loop."""

    name = "lightwheel_hdf5_replay"

    def __init__(self, config: LightwheelHdf5ReplayPolicyCfg):
        super().__init__(config)
        self.device = config.policy_device
        with h5py.File(config.hdf5_path, "r") as f:
            demo = f["data"][config.demo_name]
            raw_actions = demo["actions"][:]
            robot_pose_traj_ds = demo["states"]["articulation"]["robot"]["root_pose"][:]
            table_pose_ds = demo["initial_state"]["rigid_object"][_DATASET_TABLE_KEY]["root_pose"][0]
            if config.replay_recorded_arm_joints:
                # joint_targets/joint_pos_target, NOT states/.../joint_position: the latter
                # is robofinals' own *realized* joint state, which already carries their own
                # PD tracking lag (confirmed 2026-08-10 by comparing the two arrays during a
                # fast reach -- e.g. right_elbow_joint at frame 203 has state=+0.566 vs.
                # target=+0.306, a 0.26rad/~15deg gap in the recording itself). Replaying
                # the realized state as if it were a fresh target made Arena's own actuator
                # track an already-lagged signal, compounding a second stage of lag on top
                # of the first -- this, not actuator gains, was why arm tracking error grew
                # specifically during fast motion regardless of which actuator model was
                # tried (IdealPD, robofinals' default_implicit, new_implicit). Only the
                # first 33 columns are used -- the 132-wide array's 4 blocks of 33 are the
                # same target held across the decimated step's 4 physics substeps (see
                # _DATASET_ARM_JOINT_POSITION_COLUMNS's comment), so any block gives the
                # same values.
                arm_joint_pos_ds = demo["joint_targets"]["joint_pos_target"][
                    :, _DATASET_ARM_JOINT_POSITION_COLUMNS
                ]
                # DIAGNOSTIC (temporary, 2026-08-10): realized state, kept *separately* from
                # the target used as the actual replayed action above. The tracking-fidelity
                # question ("does Arena's arm end up where robofinals' arm ended up") needs
                # realized-vs-realized -- comparing Arena's live state against robofinals'
                # *target* instead (an earlier version of this diagnostic did exactly that,
                # by reusing arm_joint_pos_ds for both roles) conflates Arena's own PD lag
                # relative to its input with actual replay fidelity, and reads as a big
                # regression that isn't real.
                self._arm_joint_state_ds = demo["states"]["articulation"]["robot"]["joint_position"][
                    :, _DATASET_ARM_JOINT_POSITION_COLUMNS
                ]
            # DIAGNOSTIC (temporary, 2026-08-10): recorded waist trajectory, to check
            # whether Arena's own live waist_yaw/roll/pitch (driven by HOMIE_V2, not
            # replayed directly like the arms) tracks the recording or drifts -- a waist
            # yaw error would rotate the hand cameras sideways relative to the pieces even
            # with perfect arm-joint replay and near-perfect pelvis tracking.
            self._waist_joint_pos_ds = demo["states"]["articulation"]["robot"]["joint_position"][
                :, _DATASET_WAIST_JOINT_POSITION_COLUMNS
            ]

        n_frames = raw_actions.shape[0]
        action_dim = _ACTION_DIM + len(_DATASET_ARM_JOINT_POSITION_COLUMNS) if config.replay_recorded_arm_joints else _ACTION_DIM
        actions = np.zeros((n_frames, action_dim), dtype=np.float32)
        actions[:, _ARENA_WBC_SLICE] = raw_actions[:, _DATASET_WBC_SLICE]
        actions[:, _ARENA_LEFT_QUAT_XYZW_SLICE] = _wxyz_to_xyzw(raw_actions[:, _DATASET_LEFT_QUAT_WXYZ_SLICE])
        actions[:, _ARENA_RIGHT_QUAT_XYZW_SLICE] = _wxyz_to_xyzw(raw_actions[:, _DATASET_RIGHT_QUAT_WXYZ_SLICE])
        # Copied straight across, no sign flip, duplicated into both finger-joint slots
        # (see the _ACTION_DIM comment above for why each gripper is 2-wide here): the
        # embodiment must be "g1_wbc_agile_pink_dex1_continuous_grip"
        # (G1WBCAgilePinkDex1ContinuousGripActionCfg), whose scale/offset reproduce
        # robofinals' Dex1GripperCfg.process_hand formula exactly in the same -1=open/+1=close
        # convention as the recorded columns. The plain "g1_wbc_agile_pink_dex1" embodiment's
        # BinaryJointPositionAction needs a different convention and action_dim -- don't use
        # that embodiment with this policy.
        actions[:, _ARENA_LEFT_GRIPPER_SLICE] = raw_actions[:, _DATASET_LEFT_GRIPPER_IDX, None]
        actions[:, _ARENA_RIGHT_GRIPPER_SLICE] = raw_actions[:, _DATASET_RIGHT_GRIPPER_IDX, None]
        if config.replay_recorded_arm_joints:
            # Appended after the base 27-D action, matching
            # G1WBCPinkDex1DirectArmContinuousGripActionCfg's field order (g1_action, then
            # the two gripper terms, then arm_joint_override_action last).
            actions[:, _ACTION_DIM:] = arm_joint_pos_ds

        self._sim_actions = torch.from_numpy(actions).to(self.device)
        self._step_idx = 0
        self.task_description: str | None = None

        # Robot's recorded *start* pose only, relative to the recorded table, re-applied onto
        # Arena's own table position (see module docstring for why no rotation is needed).
        # Deliberately NOT re-applied every step (tried 2026-08-08, reverted): teleporting the
        # robot's root pose each step moves the gripper without moving whatever it's holding
        # (a free rigid body, integrated by real physics, can't teleport with it) -- every
        # step becomes a small yank on the grasped leg relative to the fingers, which no
        # amount of friction/gripper-continuity tuning can fix, since the desync is
        # geometric, not a contact-parameter problem. Reset-only sacrifices exactly matching
        # Homie v2's mid-episode walking distance (see the Homie-vs-Agile discussion above)
        # in exchange for physically consistent contact once Agile takes over.
        robot_offset_from_table = robot_pose_traj_ds[0, :3] - table_pose_ds[:3]
        target_pos = _ARENA_TABLE_POS + robot_offset_from_table
        self._start_robot_pose = Pose(
            position_xyz=tuple(target_pos.tolist()), rotation_xyzw=tuple(robot_pose_traj_ds[0, 3:7].tolist())
        )
        self._robot_pose_applied = False
        print(f"[lightwheel_hdf5_replay] loaded {n_frames} frames from {config.hdf5_path}:{config.demo_name}")
        print(f"[lightwheel_hdf5_replay] robot start pose: {self._start_robot_pose}")

    def set_task_description(self, task_description: str | None) -> str:
        self.task_description = task_description or "replay"
        return self.task_description

    def get_action(self, env, observation: dict[str, Any]) -> torch.Tensor:
        if not self._robot_pose_applied:
            self._apply_start_robot_pose(env)
            self._robot_pose_applied = True
        idx = min(self._step_idx, self._sim_actions.shape[0] - 1)
        # DIAGNOSTIC (temporary, 2026-08-09): pelvis-to-table xy distance per step, to get a
        # real number for the Agile-vs-Homie walking-offset comparison instead of eyeballing
        # video. Remove once the Homie_v2 Dex1 investigation concludes.
        import warp as wp

        robot_data = env.unwrapped.scene["robot"].data
        pelvis_xy = wp.to_torch(robot_data.root_link_pos_w)[0, :2].cpu().numpy()
        dx, dy = (pelvis_xy - _ARENA_TABLE_POS[:2]).tolist()
        dist = float(np.linalg.norm(pelvis_xy - _ARENA_TABLE_POS[:2]))
        quat_xyzw = wp.to_torch(robot_data.root_link_quat_w)[0].cpu().numpy()
        qx, qy, qz, qw = quat_xyzw
        yaw_deg = float(np.degrees(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))))
        # DIAGNOSTIC (temporary, 2026-08-10): held_asset (leg001) world position, to check
        # whether it's actually being lifted/held vs. staying put on the table.
        leg_pos = wp.to_torch(env.unwrapped.scene["leg001"].data.root_link_pos_w)[0].cpu().numpy()
        print(f"[lightwheel_hdf5_replay] step={self._step_idx} leg001_pos={leg_pos.tolist()}")
        print(
            f"[lightwheel_hdf5_replay] step={self._step_idx} pelvis_xy_dist_to_table={dist:.4f}"
            f" dx={dx:.4f} dy={dy:.4f} yaw_deg={yaw_deg:.2f}"
        )
        # DIAGNOSTIC (temporary, 2026-08-10): live waist_yaw/roll/pitch vs. the recording's
        # own -- these are HOMIE_V2's own output (not part of the direct arm-joint replay),
        # so any drift here would rotate the hand cameras relative to the pieces even with
        # perfect arm-joint replay and near-perfect pelvis tracking.
        if not hasattr(self, "_waist_joint_sim_indices"):
            self._waist_joint_sim_indices = [
                robot_data.joint_names.index(name)
                for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
            ]
        live_waist = wp.to_torch(robot_data.joint_pos)[0, self._waist_joint_sim_indices].cpu().numpy()
        recorded_waist = self._waist_joint_pos_ds[idx]
        print(
            f"[lightwheel_hdf5_replay] step={self._step_idx} waist_yaw/roll/pitch"
            f" live={live_waist.tolist()} recorded={recorded_waist.tolist()}"
        )
        # DIAGNOSTIC (temporary, 2026-08-10): live arm joint positions vs. robofinals'
        # *realized* recorded state (not the target fed in as the action) -- checks replay
        # fidelity to the actual recording, not Arena's own PD tracking of its own input.
        if getattr(self, "_arm_joint_state_ds", None) is not None:
            if not hasattr(self, "_arm_joint_sim_indices"):
                arm_joint_names = [
                    "left_shoulder_pitch_joint",
                    "right_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "right_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "right_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "right_elbow_joint",
                    "left_wrist_roll_joint",
                    "right_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "right_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                    "right_wrist_yaw_joint",
                ]
                self._arm_joint_names_for_diag = arm_joint_names
                self._arm_joint_sim_indices = [robot_data.joint_names.index(name) for name in arm_joint_names]
            live_arm = wp.to_torch(robot_data.joint_pos)[0, self._arm_joint_sim_indices].cpu().numpy()
            recorded_arm = self._arm_joint_state_ds[idx]
            error = live_arm - recorded_arm
            per_joint = ", ".join(
                f"{name}={err:+.4f}" for name, err in zip(self._arm_joint_names_for_diag, error.tolist())
            )
            print(f"[lightwheel_hdf5_replay] step={self._step_idx} arm_tracking_error {per_joint}")
            # DIAGNOSTIC (temporary, 2026-08-10): computed vs. applied (post-clip) effort
            # on the arm joints -- if computed >> applied at the same steps the tracking
            # error above spikes, the PD command is hitting effort_limit and getting
            # clamped (a torque *ceiling* problem, raising stiffness wouldn't fix it),
            # not merely under-stiff gains. Looks each joint up by name across whichever
            # actuator group owns it rather than assuming a single "arms" group, since
            # G1_HOMIE_CFG's "default_implicit" mode splits arm joints across
            # "arms_n5020"/"arms_w4010".
            robot_asset = env.unwrapped.scene["robot"]
            if not hasattr(self, "_arm_joint_actuator_lookup"):
                self._arm_joint_actuator_lookup = {}
                for actuator in robot_asset.actuators.values():
                    for joint_idx, joint_name in enumerate(actuator.joint_names):
                        if joint_name in self._arm_joint_names_for_diag:
                            self._arm_joint_actuator_lookup[joint_name] = (actuator, joint_idx)
            per_joint_effort_parts = []
            for name in self._arm_joint_names_for_diag:
                actuator, joint_idx = self._arm_joint_actuator_lookup[name]
                c = actuator.computed_effort[0, joint_idx].item()
                a = actuator.applied_effort[0, joint_idx].item()
                if abs(c - a) > 0.01:
                    per_joint_effort_parts.append(f"{name}=({c:+.2f}->{a:+.2f})")
            if per_joint_effort_parts:
                joined = ", ".join(per_joint_effort_parts)
                print(f"[lightwheel_hdf5_replay] step={self._step_idx} arm_effort_saturated {joined}")
        # DIAGNOSTIC (temporary, 2026-08-10): right_dex1_finger_joint_1 vs _2 live positions
        # -- Jorge visually spotted the closed gripper looking asymmetric in Arena's replay
        # (one side of the white finger cover exposing more of the black base than in the
        # dataset video). Both fingers get the identical commanded target (verified against
        # robofinals' own Dex1GripperCfg.process_hand, which duplicates one scalar across
        # both joints) and identical actuator gains (single "grippers" IdealPDActuatorCfg
        # group covering all 4 finger joints, isaaclab_arena/embodiments/g1/g1.py) -- so if
        # they read back different positions here, it's a real per-finger divergence (most
        # likely contact-driven: one finger touches the piece and stalls while the other,
        # not yet touching, keeps moving toward the shared target), not a config bug.
        if not hasattr(self, "_right_finger_sim_indices"):
            self._right_finger_sim_indices = [
                robot_data.joint_names.index(name)
                for name in ("right_dex1_finger_joint_1", "right_dex1_finger_joint_2")
            ]
        right_fingers = wp.to_torch(robot_data.joint_pos)[0, self._right_finger_sim_indices].cpu().numpy()
        print(
            f"[lightwheel_hdf5_replay] step={self._step_idx} right_finger_joint_1={right_fingers[0]:+.5f}"
            f" right_finger_joint_2={right_fingers[1]:+.5f} delta={right_fingers[0] - right_fingers[1]:+.5f}"
        )
        self._step_idx += 1
        return self._sim_actions[idx].unsqueeze(0)

    def _apply_start_robot_pose(self, env) -> None:
        """Overwrite the robot's just-reset pose with the recorded demo's own start pose.

        Runs once per episode, before the first ``env.step()`` -- see the module docstring
        for why this isn't repeated every step.
        """
        from isaaclab.managers import SceneEntityCfg

        from isaaclab_arena.terms.events import set_object_pose

        unwrapped = env.unwrapped
        env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device)
        set_object_pose(unwrapped, env_ids, SceneEntityCfg("robot"), self._start_robot_pose)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._step_idx = 0
        self._robot_pose_applied = False

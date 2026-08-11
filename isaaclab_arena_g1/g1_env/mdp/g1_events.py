# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_decoupled_wbc_joint_policy(env: ManagerBasedEnv, env_ids: torch.Tensor):
    # Reset lower body RL-based policy
    policy = env.action_manager.get_term("g1_action").get_wbc_policy
    policy.lower_body_policy.reset(env_ids)


def reset_decoupled_wbc_pink_policy(env: ManagerBasedEnv, env_ids: torch.Tensor):
    # Reset upper body IK solver
    env.action_manager.get_term("g1_action").upperbody_controller.body_ik_solver.initialize()
    env.action_manager.get_term("g1_action").upperbody_controller.in_warmup = True

    # Reset lower body RL-based policy
    policy = env.action_manager.get_term("g1_action").get_wbc_policy
    policy.lower_body_policy.reset(env_ids)

    # Reset P-controller
    env.action_manager.get_term("g1_action")._is_navigating = False
    env.action_manager.get_term("g1_action")._navigation_goal_reached = False
    env.action_manager.get_term("g1_action")._num_navigation_subgoals_reached = -1


def fix_pelvis_contour_link_mass(env: ManagerBasedEnv, env_ids: torch.Tensor, mass: float = 10.0) -> None:
    """Patch ``pelvis_contour_link``'s mass on the USD stage before PhysX parses it.

    Diagnosed 2026-08-09 while investigating why neither AGILE nor HOMIE_V2 could
    reproduce robofinals' recorded walking distance during ``iros2026-ikea-assembly``
    HDF5 replay: Arena's Nucleus-hosted ``g1_29dof_with_hand_rev_1_0.usd`` (the dexterous-hand
    asset used by the non-Dex1 G1 embodiments) ships ``pelvis_contour_link`` at 0.001kg (a
    near-massless placeholder), while robofinals' independently-authored
    ``g1_29dof_with_hand_rev_1_0_wbc.usd`` gives it 10.0kg (almost certainly the G1's
    pelvis-mounted battery pack; verified by directly comparing ``UsdPhysics.MassAPI`` on
    both USDs, every other rigid body's mass/inertia matched within noise).
    ``pelvis_contour_link`` is welded to ``pelvis`` via a ``PhysicsFixedJoint``, so this
    mass adds straight to the robot's effective base inertia.

    Deliberately does **not** touch ``pelvis`` directly when ``pelvis_contour_link`` isn't a
    mass-bearing rigid body -- the case for the Dex1 gripper's spawn asset
    (``G1_GRIPPER.usd``, see ``_make_dex1_gripper_variant``), whose ``pelvis_contour_link``
    is a plain visual Xform with no physics APIs at all. Confirmed (2026-08-09, byte-for-byte
    md5 match) that Arena's ``G1_GRIPPER.usd`` *is* the exact file robofinals' own
    ``G1DecoupledWBCAction``-based ``iros2026-ikea-assembly`` recording used -- so the Dex1
    embodiments are already mass-matched to the actual recording (~34kg, no separate
    contour body on either side); adding the ballast there would move *away* from parity,
    not toward it. This fix is therefore only relevant to G1's dexterous-hand embodiments.
    """
    del env_ids
    from pxr import UsdPhysics

    stage = env.sim.stage
    patched_count = 0
    for env_prim_path in env.scene.env_prim_paths:
        robot_prim_path = f"{env_prim_path}/Robot"
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if not prim_path.startswith(robot_prim_path):
                continue
            if prim.GetName() != "pelvis_contour_link":
                continue
            if not prim.HasAPI(UsdPhysics.MassAPI):
                continue
            UsdPhysics.MassAPI(prim).GetMassAttr().Set(mass)
            patched_count += 1


def apply_high_friction_to_g1_fingers(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    material_path: str,
    static_friction: float,
    dynamic_friction: float,
    prim_name_markers: Sequence[str],
) -> None:
    """Bind a high-friction contact material to G1 hand/finger collision prims."""
    del env_ids
    from isaaclab.sim import bind_physics_material
    from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
    from pxr import UsdPhysics, UsdShade

    stage = env.sim.stage
    material_cfg = RigidBodyMaterialCfg(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        friction_combine_mode="max",
    )
    material_cfg.func(material_path, material_cfg)

    bound_count = 0
    for env_prim_path in env.scene.env_prim_paths:
        robot_prim_path = f"{env_prim_path}/Robot"
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if not prim_path.startswith(robot_prim_path):
                continue
            prim_path_lower = prim_path.lower()
            if not any(marker in prim_path_lower for marker in prim_name_markers):
                continue
            applied_schemas = set(prim.GetAppliedSchemas())
            has_collision_api = prim.HasAPI(UsdPhysics.CollisionAPI) or "PhysicsCollisionAPI" in applied_schemas
            if not has_collision_api:
                continue

            bind_physics_material(prim_path, material_path, stage=stage)
            # bind_physics_material is decorated with apply_nested and returns None, so
            # verify the resulting direct physics material binding explicitly.
            material_binding_api = UsdShade.MaterialBindingAPI(prim)
            direct_binding = material_binding_api.GetDirectBinding("physics")
            if str(direct_binding.GetMaterialPath()) == material_path:
                bound_count += 1

    if bound_count == 0:
        warnings.warn(
            "apply_high_friction_to_g1_fingers: no G1 hand/finger collision prims were found; "
            "G1 hand friction may be unchanged.",
            stacklevel=1,
        )

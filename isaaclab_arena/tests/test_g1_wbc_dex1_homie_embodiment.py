# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import tqdm
import traceback

import pytest
import warp as wp

from isaaclab_arena.tests.test_g1_wbc_embodiment import STANDING_POSITION_XY_EPS, WBC_PINK_IDLE_ACTION
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

NUM_STEPS = 10
HEADLESS = True
ENABLE_CAMERAS = True


def get_dex1_homie_test_environment(embodiment_name: str):
    """Returns a scene with the given HOMIE_V2 Dex1 embodiment, mirroring test_g1_wbc_embodiment's setup."""

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    background = asset_registry.get_asset_by_name("kitchen")()
    embodiment = asset_registry.get_asset_by_name(embodiment_name)(enable_cameras=ENABLE_CAMERAS)

    scene = Scene(assets=[background])
    # NOTE: matches test_g1_wbc_embodiment's initial pose -- keeps the robot from dropping to
    # the ground on reset, which would make the WBC unstable.
    robot_init_base_pose = np.array([0, 0, 0])
    embodiment.set_initial_pose(Pose(position_xyz=tuple(robot_init_base_pose), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    isaaclab_arena_environment = IsaacLabArenaEnvironment(
        name="g1_dex1_homie_standing_test",
        embodiment=embodiment,
        scene=scene,
    )

    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    env_builder = ArenaEnvBuilder(isaaclab_arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    env = env_builder.make_registered()
    env.reset()

    return env, robot_init_base_pose


def _test_dex1_homie_standing_idle_actions(simulation_app, embodiment_name: str) -> bool:

    from isaaclab.envs.manager_based_env import ManagerBasedEnv

    env, robot_init_base_pose = get_dex1_homie_test_environment(embodiment_name)

    def assert_standing_idle(env: ManagerBasedEnv, robot_init_base_pose: np.ndarray):
        robot_base_pose = wp.to_torch(env.unwrapped.scene["robot"].data.root_link_pose_w)[0, :3].cpu().numpy()
        robot_xy_error = np.linalg.norm(robot_base_pose[:2] - robot_init_base_pose[:2])
        assert robot_xy_error < STANDING_POSITION_XY_EPS, "Robot moved away from initial position."

    try:
        # WBC_PINK_IDLE_ACTION covers g1_action's 23 dims (base_height_cmd already set to
        # 0.75m within it); pad with zeros for the Dex1 gripper term(s) appended after --
        # this is a basic wiring smoke test, not a gripper-behavior test, so their exact
        # value doesn't matter.
        action_dim = env.action_space.shape[-1]
        idle_action = WBC_PINK_IDLE_ACTION + [0.0] * (action_dim - len(WBC_PINK_IDLE_ACTION))
        for _ in tqdm.tqdm(range(NUM_STEPS)):
            with torch.inference_mode():
                actions = torch.tensor(idle_action, device=env.unwrapped.device).unsqueeze(0)
                _, _, _, _, _ = env.step(actions)
                assert_standing_idle(env, robot_init_base_pose)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

    finally:
        env.close()

    return True


def _test_g1_wbc_pink_dex1_standing_idle_actions(simulation_app) -> bool:
    return _test_dex1_homie_standing_idle_actions(simulation_app, "g1_wbc_pink_dex1")


@pytest.mark.with_cameras
def test_g1_wbc_pink_dex1_standing_idle_actions_single_env():
    result = run_function_with_persistent_simulation_app(
        _test_g1_wbc_pink_dex1_standing_idle_actions,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )
    assert result, f"Test {_test_g1_wbc_pink_dex1_standing_idle_actions.__name__} failed"


def _test_g1_wbc_pink_dex1_continuous_grip_standing_idle_actions(simulation_app) -> bool:
    return _test_dex1_homie_standing_idle_actions(simulation_app, "g1_wbc_pink_dex1_continuous_grip")


@pytest.mark.with_cameras
def test_g1_wbc_pink_dex1_continuous_grip_standing_idle_actions_single_env():
    result = run_function_with_persistent_simulation_app(
        _test_g1_wbc_pink_dex1_continuous_grip_standing_idle_actions,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
    )
    assert result, f"Test {_test_g1_wbc_pink_dex1_continuous_grip_standing_idle_actions.__name__} failed"


if __name__ == "__main__":
    test_g1_wbc_pink_dex1_standing_idle_actions_single_env()
    test_g1_wbc_pink_dex1_continuous_grip_standing_idle_actions_single_env()

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from isaaclab_arena_gr00t.lerobot.config.dataset_config import Gr00tDatasetConfig
from isaaclab_arena_gr00t.lerobot.convert_hdf5_to_lerobot import (
    convert_hdf5_to_lerobot,
    decompose_lightwheel_wbc_action,
)
from isaaclab_arena_gr00t.tests.utils.constants import TestConstants
from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml

pytestmark = pytest.mark.gr00t_policy


def test_g1_convert_hdf5_to_lerobot():
    # Load expected data for comparison
    expected_g1_parquet = pd.read_parquet(
        TestConstants.test_data_dir + "/test_g1_locomanip_lerobot/data/chunk-000/episode_000000.parquet"
    )
    g1_ds_config = create_config_from_yaml(
        TestConstants.test_data_dir + "/test_g1_locomanip_lerobot/test_g1_locomanip_config.yaml", Gr00tDatasetConfig
    )

    # Clean up any existing output directory
    if g1_ds_config.lerobot_data_dir.exists():

        shutil.rmtree(g1_ds_config.lerobot_data_dir)

    # Run conversion
    convert_hdf5_to_lerobot(g1_ds_config)

    # assert it has episodes.jsonl file
    assert (g1_ds_config.lerobot_data_dir / "meta" / "episodes.jsonl").exists()

    # assert it has tasks.jsonl file
    assert (g1_ds_config.lerobot_data_dir / "meta" / "tasks.jsonl").exists()

    # assert it has info.json file
    assert (g1_ds_config.lerobot_data_dir / "meta" / "info.json").exists()

    # assert it has modality.json file
    assert (g1_ds_config.lerobot_data_dir / "meta" / "modality.json").exists()

    # assert it has data/ folder has parquet files
    parquet_files = list((g1_ds_config.lerobot_data_dir / "data").glob("**/*.parquet"))
    assert len(parquet_files) == 1

    # assert it has videos/ folder has mp4 files
    mp4_files = list((g1_ds_config.lerobot_data_dir / "videos").glob("**/*.mp4"))
    assert len(mp4_files) == 1
    # check parquet file contains expected columns
    actual_df = pd.read_parquet(parquet_files[0])
    expected_columns = set(expected_g1_parquet.columns)
    actual_columns = set(actual_df.columns)
    assert expected_columns.issubset(actual_columns), f"Missing columns: {expected_columns - actual_columns}"
    # check parquet file data is the same as expected
    assert actual_df.equals(expected_g1_parquet)

    # remove lerobot_data_dir
    shutil.rmtree(g1_ds_config.lerobot_data_dir.parent)


def test_decompose_lightwheel_wbc_action():
    # Distinct value per column so every slice boundary is independently checkable.
    raw_actions = np.tile(np.arange(23, dtype=np.float32), (4, 1))
    parts = decompose_lightwheel_wbc_action(raw_actions)

    np.testing.assert_array_equal(parts["left_gripper"], raw_actions[:, 0:1])
    np.testing.assert_array_equal(parts["right_gripper"], raw_actions[:, 1:2])
    np.testing.assert_array_equal(parts["left_eef_pos"], raw_actions[:, 2:5])
    np.testing.assert_array_equal(parts["left_eef_quat"], raw_actions[:, 5:9])
    np.testing.assert_array_equal(parts["right_eef_pos"], raw_actions[:, 9:12])
    np.testing.assert_array_equal(parts["right_eef_quat"], raw_actions[:, 12:16])
    np.testing.assert_array_equal(parts["navigate_cmd"], raw_actions[:, 16:19])
    np.testing.assert_array_equal(parts["base_height_cmd"], raw_actions[:, 19:20])
    np.testing.assert_array_equal(parts["torso_rpy_cmd"], raw_actions[:, 20:23])


def _write_tiny_lightwheel_hdf5(path: Path, num_frames: int, image_size: tuple[int, int, int]) -> None:
    """Write a tiny HDF5 matching LightwheelAI/iros2026-ikea-assembly's schema (not Mimic-generated):
    state under states/articulation/robot/, raw 23-D actions, 3 cameras under obs/.
    """
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        demo = f.create_group("data/demo_0")
        # state[t, i] == i for every frame t, so a remapped policy-space column's value reveals
        # exactly which raw sim column it was read from (see the remap assertions below).
        state = np.tile(np.arange(33, dtype=np.float32), (num_frames, 1))
        demo.create_dataset("states/articulation/robot/joint_position", data=state)
        actions = np.tile(np.arange(23, dtype=np.float32), (num_frames, 1))
        demo.create_dataset("actions", data=actions)
        for cam_name in ("first_person_camera_rgb", "left_hand_camera_rgb", "right_hand_camera_rgb"):
            frames = rng.integers(0, 255, size=(num_frames, *image_size), dtype=np.uint8)
            demo.create_dataset(f"obs/{cam_name}", data=frames)
        demo.attrs["num_samples"] = num_frames
        demo.attrs["success"] = True


def test_lightwheel_wbc_convert_hdf5_to_lerobot(tmp_path):
    num_frames = 8
    image_size = (64, 64, 3)
    hdf5_name = "tiny_lightwheel.hdf5"
    _write_tiny_lightwheel_hdf5(tmp_path / hdf5_name, num_frames, image_size)

    embodiment_dir = Path(TestConstants.repo_root) / "isaaclab_arena_gr00t" / "embodiments" / "g1_dex1_ikea_lightwheel"
    config = Gr00tDatasetConfig(
        data_root=tmp_path,
        hdf5_name=hdf5_name,
        language_instruction="test instruction",
        task_index=0,
        state_name_sim="joint_position",
        state_group_path="states/articulation/robot",
        action_name_sim="actions",
        action_source="lightwheel_wbc_command",
        trim_last_state_frame=False,
        pov_cam_names_sim=["first_person_camera_rgb", "left_hand_camera_rgb", "right_hand_camera_rgb"],
        video_names_lerobot=[
            "observation.images.ego_view",
            "observation.images.left_hand",
            "observation.images.right_hand",
        ],
        fps=50,
        modality_template_path=embodiment_dir / "modality.json",
        info_template_path=embodiment_dir / "info.json",
        policy_joints_config_path=embodiment_dir / "gr00t_policy_joint_space.yaml",
        action_joints_config_path=embodiment_dir / "33dof_joint_space.yaml",
        state_joints_config_path=embodiment_dir / "33dof_joint_space.yaml",
        robot_type="G1-Gripper-Controller-DecoupledWBC",
        original_image_size=image_size,
        target_image_size=image_size,
    )

    convert_hdf5_to_lerobot(config)

    lerobot_dir = config.lerobot_data_dir
    assert (lerobot_dir / "meta" / "episodes.jsonl").exists()
    assert (lerobot_dir / "meta" / "tasks.jsonl").exists()
    assert (lerobot_dir / "meta" / "info.json").exists()
    assert (lerobot_dir / "meta" / "modality.json").exists()

    parquet_files = list((lerobot_dir / "data").glob("**/*.parquet"))
    assert len(parquet_files) == 1
    mp4_files = list((lerobot_dir / "videos").glob("**/*.mp4"))
    assert len(mp4_files) == 3  # one per camera

    df = pd.read_parquet(parquet_files[0])
    assert len(df) == num_frames  # trim_last_state_frame=False -- no frame should be dropped

    # "action" is the raw 23-D array, unmodified.
    action = np.stack(df["action"].values)
    np.testing.assert_array_equal(action, np.tile(np.arange(23, dtype=np.float32), (num_frames, 1)))

    # action.gripper is decomposed from the same raw array's first 2 columns.
    gripper = np.stack(df["action.gripper"].values)
    np.testing.assert_array_equal(gripper, action[:, 0:2])

    # State remap: gr00t_policy_joint_space.yaml puts left_gripper at policy columns 22-23 and
    # right_gripper at 31-32, sourced from 33dof_joint_space.yaml's raw sim columns 29-30/31-32
    # (left/right_dex1_finger_joint_1/2). Since this test's synthetic state has state[t, i] == i,
    # the remapped value at each policy position directly names its source sim column.
    state = np.stack(df["observation.state"].values)
    assert (state[:, 22] == 29).all()
    assert (state[:, 23] == 30).all()
    assert (state[:, 31] == 31).all()
    assert (state[:, 32] == 32).all()


if __name__ == "__main__":
    test_g1_convert_hdf5_to_lerobot()
    test_decompose_lightwheel_wbc_action()
    test_lightwheel_wbc_convert_hdf5_to_lerobot(Path("/tmp/test_lightwheel_wbc_convert"))

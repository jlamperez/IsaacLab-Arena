# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import h5py
import json
import multiprocessing as mp
import numpy as np
import shutil
import subprocess
import time
import torchvision
import traceback
from dataclasses import fields
from pathlib import Path
from tqdm import tqdm
from typing import Any

import pandas as pd

from isaaclab_arena_gr00t.lerobot.config.dataset_config import Gr00tDatasetConfig
from isaaclab_arena_gr00t.utils.image_conversion import resize_frames_with_padding
from isaaclab_arena_gr00t.utils.io_utils import (
    create_config_from_yaml,
    dump_json,
    dump_jsonl,
    load_json,
    load_robot_joints_config_from_yaml,
)
from isaaclab_arena_gr00t.utils.joints_conversion import remap_sim_joints_to_policy_joints
from isaaclab_arena_gr00t.utils.robot_eef_pose import EefPose
from isaaclab_arena_gr00t.utils.robot_joints import JointsAbsPosition

# robofinals' G1-Gripper-Controller-DecoupledWBC packed 23-D action column layout, ported from
# isaaclab_arena_gr00t/policy/lightwheel_hdf5_replay_policy.py's _DATASET_* slice constants (see
# that file's module docstring for the full derivation/verification notes). Unlike that file's own
# use (which feeds Arena's xyzw-expecting action term), the quaternions here are left in their
# recorded wxyz order -- EefPose.from_array expects wxyz natively, so no reorder is needed here.
_LIGHTWHEEL_ACTION_DIM = 23
_LIGHTWHEEL_LEFT_GRIPPER_SLICE = slice(0, 1)
_LIGHTWHEEL_RIGHT_GRIPPER_SLICE = slice(1, 2)
_LIGHTWHEEL_LEFT_EEF_POS_SLICE = slice(2, 5)
_LIGHTWHEEL_LEFT_EEF_QUAT_WXYZ_SLICE = slice(5, 9)
_LIGHTWHEEL_RIGHT_EEF_POS_SLICE = slice(9, 12)
_LIGHTWHEEL_RIGHT_EEF_QUAT_WXYZ_SLICE = slice(12, 16)
_LIGHTWHEEL_NAVIGATE_CMD_SLICE = slice(16, 19)
_LIGHTWHEEL_BASE_HEIGHT_CMD_SLICE = slice(19, 20)
_LIGHTWHEEL_TORSO_RPY_CMD_SLICE = slice(20, 23)
# Column labels for the raw 23-D action, in order -- used to name the "action" feature in
# get_feature_info since it isn't a named-joint array like the joint_array action_source.
_LIGHTWHEEL_ACTION_COLUMN_NAMES = [
    "left_gripper",
    "right_gripper",
    "left_eef_pos_x",
    "left_eef_pos_y",
    "left_eef_pos_z",
    "left_eef_quat_w",
    "left_eef_quat_x",
    "left_eef_quat_y",
    "left_eef_quat_z",
    "right_eef_pos_x",
    "right_eef_pos_y",
    "right_eef_pos_z",
    "right_eef_quat_w",
    "right_eef_quat_x",
    "right_eef_quat_y",
    "right_eef_quat_z",
    "navigate_cmd_x",
    "navigate_cmd_y",
    "navigate_cmd_yaw",
    "base_height_cmd",
    "torso_rpy_cmd_roll",
    "torso_rpy_cmd_pitch",
    "torso_rpy_cmd_yaw",
]


def decompose_lightwheel_wbc_action(raw_actions: np.ndarray) -> dict[str, np.ndarray]:
    """Split robofinals' packed 23-D G1-Gripper-Controller-DecoupledWBC action into named parts.

    Args:
        raw_actions: The dataset's raw action array, shape ``(T, 23)``.

    Returns:
        A dict with keys ``left_gripper``/``right_gripper`` (``(T, 1)``), ``left_eef_pos``/
        ``right_eef_pos`` (``(T, 3)``), ``left_eef_quat``/``right_eef_quat`` (``(T, 4)``, wxyz),
        ``navigate_cmd`` (``(T, 3)``), ``base_height_cmd`` (``(T, 1)``), ``torso_rpy_cmd``
        (``(T, 3)``).
    """
    assert raw_actions.shape[1] == _LIGHTWHEEL_ACTION_DIM, (
        f"expected a {_LIGHTWHEEL_ACTION_DIM}-D lightwheel WBC action, got shape {raw_actions.shape}"
    )
    return {
        "left_gripper": raw_actions[:, _LIGHTWHEEL_LEFT_GRIPPER_SLICE],
        "right_gripper": raw_actions[:, _LIGHTWHEEL_RIGHT_GRIPPER_SLICE],
        "left_eef_pos": raw_actions[:, _LIGHTWHEEL_LEFT_EEF_POS_SLICE],
        "left_eef_quat": raw_actions[:, _LIGHTWHEEL_LEFT_EEF_QUAT_WXYZ_SLICE],
        "right_eef_pos": raw_actions[:, _LIGHTWHEEL_RIGHT_EEF_POS_SLICE],
        "right_eef_quat": raw_actions[:, _LIGHTWHEEL_RIGHT_EEF_QUAT_WXYZ_SLICE],
        "navigate_cmd": raw_actions[:, _LIGHTWHEEL_NAVIGATE_CMD_SLICE],
        "base_height_cmd": raw_actions[:, _LIGHTWHEEL_BASE_HEIGHT_CMD_SLICE],
        "torso_rpy_cmd": raw_actions[:, _LIGHTWHEEL_TORSO_RPY_CMD_SLICE],
    }


def wait_for_video_completion(video_path: str, max_wait_time: int = 60, check_interval: float = 0.5) -> bool:
    """
    Wait for a video file to be completely written and accessible.

    Args:
        video_path: Path to the video file.
        max_wait_time: Maximum time to wait in seconds.
        check_interval: How often to check the file status in seconds.

    Returns:
        True if video is complete and accessible, False if timeout.
    """
    video_path = Path(video_path)
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        # Check if file exists
        if not video_path.exists():
            time.sleep(check_interval)
            continue

        # Check if file is still being written by checking if size is stable
        try:
            size1 = video_path.stat().st_size
            time.sleep(check_interval)
            size2 = video_path.stat().st_size

            # If size is stable and > 0, file is likely complete
            if size1 == size2 and size1 > 0:
                # Try to open file to ensure it's not locked
                try:
                    with open(video_path, "rb") as f:
                        # Try to read first few bytes to ensure file is accessible
                        f.read(1024)
                    return True
                except OSError:
                    # File still locked, continue waiting
                    pass
        except OSError:
            # File might be in process of being created
            pass

        time.sleep(check_interval)

    return False


def get_video_metadata(video_path: str) -> dict[str, Any] | None:
    """
    Get video metadata in the specified format with robust file completion checking.

    Args:
        video_path: Path to the video file.

    Returns:
        Video metadata including shape, names, and video_info, or None if an error occurs.
    """
    # Wait for video file to be completely written
    if not wait_for_video_completion(video_path, max_wait_time=60):
        print(f"Timeout waiting for video completion: {video_path}")
        return None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=height,width,codec_name,pix_fmt,r_frame_rate",
        "-of",
        "json",
        video_path,
    ]

    try:
        output = subprocess.check_output(cmd).decode("utf-8")
        probe_data = json.loads(output)
        stream = probe_data["streams"][0]

        # Parse frame rate (comes as fraction like "15/1")
        num, den = map(int, stream["r_frame_rate"].split("/"))
        fps = num / den

        # Check for audio streams
        audio_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            video_path,
        ]
        audio_output = subprocess.check_output(audio_cmd).decode("utf-8")
        audio_data = json.loads(audio_output)
        has_audio = len(audio_data.get("streams", [])) > 0

        metadata = {
            "dtype": "video",
            "shape": [stream["height"], stream["width"], 3],  # Assuming 3 channels
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.width": stream["width"],
                "video.height": stream["height"],
                "video.fps": fps,
                "video.codec": stream["codec_name"],
                "video.pix_fmt": stream["pix_fmt"],
                "video.channels": 3,
                "video.is_depth_map": False,
                "has_audio": has_audio,
            },
        }

        print(f"Successfully extracted metadata for {video_path}")
        return metadata

    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing ffprobe output: {e}")
        return None


def get_feature_info(
    step_data: pd.DataFrame, video_paths: dict[str, str], config: Gr00tDatasetConfig
) -> dict[str, Any]:
    """
    Get feature info from each frame of the video.

    Args:
        step_data: DataFrame containing data of an episode.
        video_paths: Dictionary mapping video keys to their file paths.
        config: Configuration object containing dataset and path information.
    Returns:
        Dictionary containing feature information for each column and video.
    """
    policy_joints_config = load_robot_joints_config_from_yaml(config.policy_joints_config_path)
    # flatten dict of dict into a single dict, perseving the order of the keys
    policy_joints_names = []
    for joint_group in policy_joints_config.keys():
        for joint_name in policy_joints_config[joint_group]:
            policy_joints_names.append(joint_name)
    features = {}
    for video_key, video_path in video_paths.items():
        video_metadata = get_video_metadata(video_path)
        features[video_key] = video_metadata
    assert isinstance(step_data, pd.DataFrame)
    for column in step_data.columns:
        column_data = np.stack(step_data[column], axis=0)
        shape = column_data.shape
        if len(shape) == 1:
            shape = (1,)
        else:
            shape = shape[1:]
        features[column] = {
            "dtype": column_data.dtype.name,
            "shape": shape,
        }
        # State & action
        if column == config.lerobot_keys["state"] or (
            column == config.lerobot_keys["action"] and config.action_source == "joint_array"
        ):
            dof = column_data.shape[1]
            assert dof == len(policy_joints_names)
            features[column]["names"] = [f"{policy_joints_names[i]}" for i in range(dof)]
        elif column == config.lerobot_keys["action"] and config.action_source == "lightwheel_wbc_command":
            dof = column_data.shape[1]
            assert dof == len(_LIGHTWHEEL_ACTION_COLUMN_NAMES)
            features[column]["names"] = list(_LIGHTWHEEL_ACTION_COLUMN_NAMES)

    return features


def extract_teleop_command(trajectory: h5py.Dataset, teleop_key: str, config: Gr00tDatasetConfig) -> dict[str, Any]:
    """
    Extract the teleop command from the trajectory.
    """
    assert "action" in trajectory.keys()
    assert teleop_key in config.hdf5_keys
    trim = slice(None, -1) if config.trim_last_state_frame else slice(None)
    teleop_command = trajectory["action"][config.hdf5_keys[teleop_key]][trim]
    return [row for row in teleop_command]


def generate_info(
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_videos: int,
    total_chunks: int,
    config: Gr00tDatasetConfig,
    step_data: pd.DataFrame,
    video_paths: dict[str, str],
) -> dict[str, Any]:
    """
    Generate the info.json data field.

    Args:
        total_episodes: Total number of episodes in the dataset.
        total_frames: Total number of frames across all episodes.
        total_tasks: Total number of tasks in the dataset.
        total_videos: Total number of videos in the dataset.
        total_chunks: Total number of data chunks.
        config: Configuration object containing dataset and path information.
        step_data: DataFrame containing step data for an example episode.
        video_paths: Dictionary mapping video keys to their file paths.

    Returns:
        Dictionary containing the info.json data field.
    """

    info_template = load_json(config.info_template_path)

    info_template["robot_type"] = config.robot_type
    info_template["total_episodes"] = total_episodes
    info_template["total_frames"] = total_frames
    info_template["total_tasks"] = total_tasks
    info_template["total_videos"] = total_videos
    info_template["total_chunks"] = total_chunks
    info_template["chunks_size"] = config.chunks_size
    info_template["fps"] = config.fps

    info_template["data_path"] = config.data_path
    info_template["video_path"] = config.video_path

    features = get_feature_info(step_data, video_paths, config)

    info_template["features"] = features
    return info_template


def write_video_job(queue: mp.Queue, error_queue: mp.Queue, config: Gr00tDatasetConfig) -> None:
    """
    Write frames to videos in mp4 format.

    Args:
        queue: Multiprocessing queue containing video frame data to be written.
        error_queue: Multiprocessing queue for reporting errors from worker processes.
        config: Configuration object containing dataset and path information.

    Returns:
        None
    """
    while True:
        job = queue.get()
        if job is None:
            break
        try:
            video_path, frames, fps, video_type = job
            if video_type == "image":
                # Create parent directory if it doesn't exist
                video_path.parent.mkdir(parents=True, exist_ok=True)
                assert (
                    frames.shape[1:] == config.original_image_size
                ), f"frames.shape[1:] {frames.shape[1:]} != config.original_image_size {config.original_image_size}"
                if config.target_image_size != config.original_image_size:
                    frames = resize_frames_with_padding(
                        frames, target_image_size=config.target_image_size, bgr_conversion=False, pad_img=True
                    )
                # h264 codec encoding
                torchvision.io.write_video(video_path, frames, fps, video_codec="h264")

        except Exception as e:
            # Get the traceback and put in error queue
            error_msg = f"Error creating video {video_path}: {e}\n{traceback.format_exc()}"
            print(error_msg)
            error_queue.put(error_msg)


def convert_trajectory_to_df(
    trajectory: h5py.Dataset,
    episode_index: int,
    index_start: int,
    config: Gr00tDatasetConfig,
) -> dict[str, Any]:
    """
    Convert a single trajectory from HDF5 to a pandas DataFrame.

    Args:
        trajectory: HDF5 dataset containing trajectory data.
        episode_index: Index of the current episode.
        index_start: Starting index for the episode.
        config: Configuration object containing dataset and path information.

    Returns:
        Dictionary containing the DataFrame, annotation set, and episode length.
    """

    return_dict = {}
    data = {}

    policy_modality_config = load_json(config.modality_template_path)

    policy_joints_config = load_robot_joints_config_from_yaml(config.policy_joints_config_path)
    action_joints_config = load_robot_joints_config_from_yaml(config.action_joints_config_path)
    state_joints_config = load_robot_joints_config_from_yaml(config.state_joints_config_path)

    """Get joints state/action/timestamp from HDF5 file"""
    length = None
    assert "obs" in trajectory.keys()
    trim = slice(None, -1) if config.trim_last_state_frame else slice(None)
    if config.state_group_path is not None:
        state_group = trajectory
        for part in config.state_group_path.split("/"):
            state_group = state_group[part]
    else:
        state_group = trajectory["obs"]

    for key, hdf5_key_name in config.hdf5_keys.items():
        if key not in ["state", "action"]:
            continue
        if key == "action" and config.action_source == "lightwheel_wbc_command":
            # Handled separately below -- not a named-joint array, doesn't go through the
            # policy_joints_config remap this loop applies to state (and joint_array actions).
            continue
        lerobot_key_name = config.lerobot_keys[key]
        # state
        if key == "state":
            assert hdf5_key_name in state_group.keys()
            joints = state_group[hdf5_key_name]
        # action target
        else:
            assert hdf5_key_name in trajectory.keys()
            joints = trajectory[hdf5_key_name]
        # state
        if key == "state":
            # NOTE(xinjieyao, 2025-09-25): remove the last obs due to Lab reports observations
            joints = joints[trim]
            input_joints_config = state_joints_config
        # action target
        elif key == "action":
            # NOTE(xinjieyao, 2025-09-25): remove the last idle action due to Lab reports actions
            joints = joints[trim]
            input_joints_config = action_joints_config
        else:
            raise ValueError(f"Unknown key: {key}")
        assert joints.ndim == 2
        assert joints.shape[1] == len(input_joints_config)

        # 1.1. Remap the joints from Lab order to the LeRobot-GR00T order
        joints = JointsAbsPosition.from_array(joints, input_joints_config, device="cpu")
        remapped_joints = remap_sim_joints_to_policy_joints(joints, policy_joints_config)

        # 1.2. Fill in the missing joints with zeros
        ordered_joints = []
        for joint_group in policy_modality_config[key].keys():
            # NOTE(xinjieyao, 2025-09-25): Those are not joint position commands, which do not need remapping orders
            if (
                joint_group == "left_wrist_pose"
                or joint_group == "right_wrist_pose"
                or joint_group == "base_height_command"
                or joint_group == "navigate_command"
                or joint_group == "torso_orientation_rpy_command"
            ):
                continue
            num_joints = (
                policy_modality_config[key][joint_group]["end"] - policy_modality_config[key][joint_group]["start"]
            )

            if joint_group not in remapped_joints.keys():
                remapped_joints[joint_group] = np.zeros(
                    (joints.get_joints_pos().shape[0], num_joints), dtype=np.float64
                )
            else:
                assert remapped_joints[joint_group].shape[1] == num_joints
            ordered_joints.append(remapped_joints[joint_group])

        # 1.3. Concatenate the arrays for parquets
        concatenated = np.concatenate(ordered_joints, axis=1)
        data[lerobot_key_name] = [row for row in concatenated]

    if config.action_source == "lightwheel_wbc_command":
        raw_action_hdf5_key = config.hdf5_keys["action"]
        assert raw_action_hdf5_key in trajectory.keys()
        raw_actions = np.asarray(trajectory[raw_action_hdf5_key])[trim]
        data[config.lerobot_keys["action"]] = [row for row in raw_actions]

        wbc_parts = decompose_lightwheel_wbc_action(raw_actions)
        left_eef_pose = EefPose.from_array(wbc_parts["left_eef_pos"], wbc_parts["left_eef_quat"], device="cpu")
        right_eef_pose = EefPose.from_array(wbc_parts["right_eef_pos"], wbc_parts["right_eef_quat"], device="cpu")
        eef_pose = np.concatenate(
            [left_eef_pose.get_eef_pose().numpy(), right_eef_pose.get_eef_pose().numpy()], axis=1
        ).astype(np.float64)
        data[config.lerobot_keys["action_eef_pose"]] = [row for row in eef_pose]

        gripper = np.concatenate([wbc_parts["left_gripper"], wbc_parts["right_gripper"]], axis=1)
        data[config.lerobot_keys["action_gripper"]] = [row for row in gripper]

        data[config.lerobot_keys["teleop_navigate_command"]] = [row for row in wbc_parts["navigate_cmd"]]
        data[config.lerobot_keys["teleop_base_height_command"]] = [row for row in wbc_parts["base_height_cmd"]]
        data[config.lerobot_keys["teleop_torso_orientation_rpy_command"]] = [
            row for row in wbc_parts["torso_rpy_cmd"]
        ]

    assert len(data[config.lerobot_keys["action"]]) == len(data[config.lerobot_keys["state"]])
    length = len(data[config.lerobot_keys["action"]])
    data["timestamp"] = np.arange(length).astype(np.float64) * (1.0 / config.fps)

    """Get eef pose for state/action from HDF5 file(optional)"""

    for key in ["obs", "action"]:
        # break for loop if key not in trajectory.keys()
        if key not in trajectory.keys():
            continue
        eef_pose = {}
        for side in ["left", "right"]:
            if f"{side}_eef_pos" in config.hdf5_keys and f"{side}_eef_quat" in config.hdf5_keys:
                side_eef_pos = trajectory[key][config.hdf5_keys[f"{side}_eef_pos"]][trim]
                side_eef_quat = trajectory[key][config.hdf5_keys[f"{side}_eef_quat"]][trim]
                side_eef_pose = EefPose.from_array(side_eef_pos, side_eef_quat, device="cpu")
                eef_pose[side] = side_eef_pose.get_eef_pose()
        if "left" in eef_pose and "right" in eef_pose:
            eef_pose = np.concatenate([eef_pose["left"].numpy(), eef_pose["right"].numpy()], axis=1).astype(np.float64)

            assert eef_pose.shape == (length, 14), f"{eef_pose.shape} != ({length}, 14)"
            assert f"{key}_eef_pose" in config.lerobot_keys, f"{key}_eef_pose not in config.lerobot_keys"
            lerobot_key_name = config.lerobot_keys[f"{key}_eef_pose"]
            data[lerobot_key_name] = [row for row in eef_pose]

    """Get teleop command for action from HDF5 file(optional)"""

    teleop_command_keys = [
        "teleop_base_height_command",
        "teleop_navigate_command",
        "teleop_torso_orientation_rpy_command",
    ]
    for teleop_key in teleop_command_keys:
        if teleop_key in config.hdf5_keys:
            assert teleop_key in config.lerobot_keys
            data[config.lerobot_keys[teleop_key]] = extract_teleop_command(trajectory, teleop_key, config)

    # 2. Get the annotation
    assert "annotation" in config.lerobot_keys
    annotation_keys = config.lerobot_keys["annotation"]
    # task selection
    data[annotation_keys[0]] = np.ones(length, dtype=int) * config.task_index

    # 3. Other data
    data["episode_index"] = np.ones(length, dtype=int) * episode_index
    data["task_index"] = np.ones(length, dtype=int) * config.task_index
    data["index"] = np.arange(length, dtype=int) + index_start
    data["frame_index"] = np.arange(length, dtype=int)
    # last frame is successful
    reward = np.zeros(length, dtype=np.float64)
    reward[-1] = 1
    done = np.zeros(length, dtype=bool)
    done[-1] = True
    data["next.reward"] = reward
    data["next.done"] = done
    data["observation.img_state_delta"] = np.zeros(length, dtype=np.float64)

    dataframe = pd.DataFrame(data)

    return_dict["data"] = dataframe
    return_dict["length"] = length
    return_dict["annotation"] = set(data[annotation_keys[0]])
    return return_dict


def convert_hdf5_to_lerobot(config: Gr00tDatasetConfig):
    """
    Convert the MimcGen HDF5 dataset to Gr00t-LeRobot format.

    Args:
        config: Configuration object containing dataset and path information.

    Returns:
        None
    """
    # Create a queue to communicate with the worker processes
    max_queue_size = 10
    num_workers = 4
    queue = mp.Queue(maxsize=max_queue_size)
    error_queue = mp.Queue()  # for error handling
    # Start the worker processes
    workers = []
    for _ in range(num_workers):
        worker = mp.Process(target=write_video_job, args=(queue, error_queue, config))
        worker.start()
        workers.append(worker)

    assert Path(config.hdf5_file_path).exists()
    hdf5_handler = h5py.File(config.hdf5_file_path, "r")
    hdf5_data = hdf5_handler["data"]

    # 1. Generate meta/ folder
    config.lerobot_data_dir.mkdir(parents=True, exist_ok=True)
    lerobot_meta_dir = config.lerobot_data_dir / "meta"
    lerobot_meta_dir.mkdir(parents=True, exist_ok=True)

    tasks = {}
    tasks.update({config.task_index: f"{config.language_instruction}"})

    # 2. Generate data/
    total_length = 0
    example_data = None
    video_paths = {}

    trajectory_ids = list(hdf5_data.keys())

    episodes_info = []
    # A trajectory that fails to convert (e.g. a truncated recording) is skipped, not written --
    # episode_index must count only what's actually written, or downstream consumers expecting
    # contiguous episode_0..total_episodes-1 numbering hit a silent gap at the failed index.
    episode_index = 0
    for trajectory_id in tqdm(trajectory_ids):

        try:
            trajectory = hdf5_data[trajectory_id]

            df_ret_dict = convert_trajectory_to_df(
                trajectory=trajectory, episode_index=episode_index, index_start=total_length, config=config
            )
        except Exception as e:
            print(f"Error loading trajectory {trajectory_id}: {e}")
            continue

        # 2.1. Save the episode data
        dataframe = df_ret_dict["data"]
        episode_chunk = episode_index // config.chunks_size
        save_relpath = config.data_path.format(episode_chunk=episode_chunk, episode_index=episode_index)
        save_path = config.lerobot_data_dir / save_relpath
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(save_path)

        # 2.2. Update total length, episodes_info
        length = df_ret_dict["length"]
        total_length += length
        episodes_info.append({
            "episode_index": episode_index,
            "tasks": [tasks[task_index] for task_index in df_ret_dict["annotation"]],
            "length": length,
        })
        # 2.3. Generate videos/
        if config.pov_cam_names_sim is not None:
            camera_pairs = list(zip(config.pov_cam_names_sim, config.video_names_lerobot))
            camera_group = trajectory["obs"]
        else:
            camera_pairs = [(config.pov_cam_name_sim, config.video_name_lerobot)]
            camera_group = trajectory["camera_obs"]

        for cam_key, video_key in camera_pairs:
            assert cam_key in camera_group, f"{cam_key} not found in {camera_group}"

            new_video_relpath = config.video_path.format(
                episode_chunk=episode_chunk, video_key=video_key, episode_index=episode_index
            )
            new_video_path = config.lerobot_data_dir / new_video_relpath
            if video_key not in video_paths:
                video_paths[video_key] = new_video_path

            frames = np.array(camera_group[cam_key])
            if config.trim_last_state_frame:
                # remove last frame due to how Lab reports observations
                frames = frames[:-1]
            assert len(frames) == length
            queue.put((new_video_path, frames, config.fps, "image"))

        if example_data is None:
            example_data = df_ret_dict

        episode_index += 1

    # 3. Generate the rest of meta/
    # 3.1. Generate tasks.json
    tasks_path = lerobot_meta_dir / config.tasks_fname
    task_jsonlines = [{"task_index": task_index, "task": task} for task_index, task in tasks.items()]
    dump_jsonl(task_jsonlines, tasks_path)

    # 3.2. Generate episodes.jsonl
    episodes_path = lerobot_meta_dir / config.episodes_fname
    dump_jsonl(episodes_info, episodes_path)

    # 3.3. Generate modality.json
    modality_path = lerobot_meta_dir / config.modality_fname
    shutil.copy(config.modality_template_path, modality_path)

    try:
        # Check for errors in the error queue
        while not error_queue.empty():
            error_message = error_queue.get()
            print(f"Error in worker process for video creation: {error_message}")

        # Stop the worker processes and wait for all videos to complete
        for _ in range(num_workers):
            queue.put(None)
        for worker in workers:
            worker.join()

        # 3.4. Generate info.json (AFTER all videos are created). Uses episode_index (the count
        # of trajectories actually written), not len(trajectory_ids) (the raw source count) --
        # they differ whenever one or more trajectories failed to convert and were skipped.
        info_json = generate_info(
            total_episodes=episode_index,
            total_frames=total_length,
            total_tasks=len(tasks),
            total_videos=episode_index,
            total_chunks=-(-episode_index // config.chunks_size),  # ceiling division
            step_data=example_data["data"],
            video_paths=video_paths,
            config=config,
        )
        dump_json(info_json, lerobot_meta_dir / "info.json", indent=4)
        print("Successfully generated info.json with video metadata!")

        # Close the HDF5 file handler
        hdf5_handler.close()

    except Exception as e:
        print(f"Error in main process: {e}")
        # Make sure to clean up even if there's an error
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        if not hdf5_handler.closed:
            hdf5_handler.close()
        raise  # Re-raise the exception after cleanup


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Dataset from HDF5 to GR00T LeRobot Format")
    parser.add_argument("--yaml_file", help="Path to YAML configuration file", required=True)
    args = parser.parse_args()

    config = create_config_from_yaml(args.yaml_file, Gr00tDatasetConfig)
    # Print the config
    print("\n" + "=" * 50)
    print("GR00T LEROBOT DATASET CONFIGURATION:")
    print("=" * 50)
    for field in fields(Gr00tDatasetConfig):
        if field.init:  # Only show init fields
            value = getattr(config, field.name)
            print(f"  {field.name}: {value}")
    print("=" * 50 + "\n")
    convert_hdf5_to_lerobot(config)

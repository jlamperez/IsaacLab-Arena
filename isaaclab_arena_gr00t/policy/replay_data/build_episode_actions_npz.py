# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build the ``.npz`` that ``gr00t_dex1_eef_replay_policy.py`` replays open-loop.

Extracts one recorded episode's ``action.ee_action``, ``action.hand_cmd``, ``task_index``
from the BitRobot LeRobot dataset, plus a body-frame ``navigate_cmd`` (linear x/y velocity,
yaw rate) derived from ``action.robot_q_desired``'s root pose via finite differences.

Needs pandas/pyarrow/scipy, which the Isaac Sim container's python does not have -- run this
with the Isaac-GR00T submodule's own uv-managed venv instead, e.g. from the host:

    cd Isaac-GR00T && uv run python \\
        ../IsaacLab-Arena/isaaclab_arena_gr00t/policy/replay_data/build_episode_actions_npz.py

The output lands next to this script (gitignored, see ``.gitignore``'s
``/isaaclab_arena_gr00t/policy/replay_data/`` entry) and is visible inside the Arena
container automatically, since the repo root is mounted at ``/workspaces/isaaclab_arena`` --
no ``docker cp`` needed.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R

_DATASET_ROOT = Path(
    "/mnt/ata-Samsung_SSD_870_QVO_8TB_S5SSNF0W200198W/lerobot_cache/BitRobot/G1_WBT_Dex1_Building-Children-Table"
)
_DATASET_FPS = 30.0
# Root-pose finite differences use a wider-than-1-frame span rather than a naive 1-frame
# difference. Originally justified as working around action.robot_q_desired holding stale
# values for ~3 frames between updates -- but validate_navigate_cmd.py's dataset-wide run
# (see its report) measured a mean stale-run length of ~1.0003 frames, i.e. the root pose
# changes almost every single frame; that "~3 frames" was a one-episode observation, not a
# real recording artifact. Kept at 5 anyway since it's a cheap, harmless smoothing window (long
# genuine stationary holds during manipulation -- up to 99 frames, seen in 9/533 episodes --
# still correctly average to ~0 velocity within it) and re-validating a narrower span isn't
# worth the churn.
_VELOCITY_SPAN_FRAMES = 5


def _build_navigate_cmd(q_desired: np.ndarray) -> np.ndarray:
    """Derive a (N,3) body-frame [lin_vel_x, lin_vel_y, ang_vel_z] from robot_q_desired's root pose."""
    root_pos = q_desired[:, 0:3].copy()
    root_quat_wxyz = q_desired[:, 3:7].copy()

    # A handful of frames carry an all-zero placeholder quaternion; fall back to the nearest
    # valid frame (by absolute distance, searched among all valid frames -- not just the
    # immediate neighbor) so scipy doesn't choke on a zero-norm quat. A neighbor-only fallback
    # breaks on consecutive bad runs: fixing bad_idx from bad_idx+1 before bad_idx+1 has itself
    # been fixed just copies another zero (seen dataset-wide, e.g. episode_000023 has frames
    # 0 and 1 both zero).
    norms = np.linalg.norm(root_quat_wxyz, axis=1)
    valid_mask = norms >= 1e-6
    if not valid_mask.all():
        valid_idx = np.where(valid_mask)[0]
        for bad_idx in np.where(~valid_mask)[0]:
            nearest = valid_idx[np.argmin(np.abs(valid_idx - bad_idx))]
            root_quat_wxyz[bad_idx] = root_quat_wxyz[nearest]
            root_pos[bad_idx] = root_pos[nearest]

    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    rotations = R.from_quat(root_quat_xyzw)

    n_frames = root_pos.shape[0]
    dt = 1.0 / _DATASET_FPS
    half_span = _VELOCITY_SPAN_FRAMES // 2
    lin_vel_body = np.zeros((n_frames, 2), dtype=np.float32)
    ang_vel_z = np.zeros((n_frames,), dtype=np.float32)
    for t in range(n_frames):
        t0 = max(0, t - half_span)
        t1 = min(n_frames - 1, t + half_span)
        span_dt = (t1 - t0) * dt
        if span_dt < 1e-6:
            continue
        dpos_body = rotations[t0].inv().apply(root_pos[t1] - root_pos[t0])
        lin_vel_body[t, 0] = dpos_body[0] / span_dt
        lin_vel_body[t, 1] = dpos_body[1] / span_dt
        rel_rot = rotations[t0].inv() * rotations[t1]
        ang_vel_z[t] = rel_rot.as_rotvec()[2] / span_dt

    return np.concatenate([lin_vel_body, ang_vel_z[:, None]], axis=1).astype(np.float32)


def build(episode_index: int, dataset_root: Path, output_path: Path) -> None:
    episode_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    df = pd.read_parquet(episode_path)

    ee_action = np.stack(df["action.ee_action"].values).astype(np.float32)
    hand_cmd = np.stack(df["action.hand_cmd"].values).astype(np.float32)
    task_index = df["task_index"].values.astype(np.int64)
    q_desired = np.stack(df["action.robot_q_desired"].values).astype(np.float64)
    navigate_cmd = _build_navigate_cmd(q_desired)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        ee_action=ee_action,
        hand_cmd=hand_cmd,
        task_index=task_index,
        navigate_cmd=navigate_cmd,
    )
    print(f"Wrote {output_path} ({ee_action.shape[0]} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--dataset-root", type=Path, default=_DATASET_ROOT)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    output_path = args.output_path or (
        Path(__file__).parent / f"episode_{args.episode_index:06d}_actions.npz"
    )
    build(args.episode_index, args.dataset_root, output_path)

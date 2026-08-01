# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Fast, vectorized replacement for ``gr00t/data/stats.py``'s ``generate_rel_stats`` for the
``ee_action_left`` / ``ee_action_right`` keys added when splitting ``ee_action`` (see
``g1_dex1_ikea_data_gr00t_n_1_7_config.py``).

Isaac-GR00T's own ``RelativeActionLoader.load_relative_actions`` constructs one
``EndEffectorPose`` (a scipy ``Rotation`` object) per single frame per single delta-index,
one at a time in a Python loop -- for a 40-step action horizon over ~533 episodes this measured
at ~80-120s/episode (extrapolates to >24h total). This script computes the exact same
``T_ref^-1 @ T_action`` relative-pose math (matching ``EndEffectorPose._compute_relative`` /
``relative_chunking`` exactly) but batched across an entire episode's frames per delta-index
using scipy's vectorized ``Rotation`` operations, cutting the same computation to seconds.

Run with the Isaac-GR00T venv (needs pandas/scipy/tqdm):

    cd Isaac-GR00T && uv run python \\
        ../IsaacLab-Arena/isaaclab_arena_gr00t/policy/replay_data/generate_relative_stats_fast.py
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation
from tqdm import tqdm

_DATASET_ROOT = Path(
    "/mnt/ata-Samsung_SSD_870_QVO_8TB_S5SSNF0W200198W/lerobot_cache/BitRobot/G1_WBT_Dex1_Building-Children-Table"
)
_HORIZON = 40  # matches g1_dex1_ikea_config's action ModalityConfig(delta_indices=list(range(40)))

# (state_column_slice, action_column_slice) -- both index into the same 12-D
# observation.state.ee_state / action.ee_action columns.
_KEYS = {
    "ee_action_left": slice(0, 6),
    "ee_action_right": slice(6, 12),
}


def _relative_trajectory_for_episode(state6: np.ndarray, action6: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of RelativeActionLoader.load_relative_actions for one 6-D arm.

    Args:
        state6: (N, 6) pos(3)+euler_xyz(3) array (observation.state.ee_state's arm slice).
        action6: (N, 6) pos(3)+euler_xyz(3) array (action.ee_action's arm slice), same frame
            indexing as state6.

    Returns:
        (usable_length * _HORIZON, 6) array of relative pos(3)+euler_xyz(3) samples, in the
        same order RelativeActionLoader.load_relative_actions would produce (all deltas for
        frame 0, then all deltas for frame 1, ...).
    """
    n_frames = state6.shape[0]
    usable_length = n_frames - (_HORIZON - 1)
    if usable_length <= 0:
        return np.empty((0, 6), dtype=np.float32)

    t_state = state6[:, 0:3]
    r_state_all = Rotation.from_euler("xyz", state6[:, 3:6], degrees=False)
    t_action = action6[:, 0:3]
    r_action_all = Rotation.from_euler("xyz", action6[:, 3:6], degrees=False)

    samples = np.empty((usable_length, _HORIZON, 6), dtype=np.float32)
    ref_idx = np.arange(usable_length)
    r_ref_inv = r_state_all[ref_idx].inv()
    t_ref = t_state[ref_idx]

    for d in range(_HORIZON):
        action_idx = ref_idx + d
        r_rel = r_ref_inv * r_action_all[action_idx]
        t_rel = r_ref_inv.apply(t_action[action_idx] - t_ref)
        euler_rel = r_rel.as_euler("xyz", degrees=False)
        samples[:, d, 0:3] = t_rel
        samples[:, d, 3:6] = euler_rel

    return samples.reshape(-1, 6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=_DATASET_ROOT)
    args = parser.parse_args()

    dataset_path: Path = args.dataset_path
    parquet_files = sorted(dataset_path.glob("data/*/*.parquet"))
    assert parquet_files, f"No parquet files found under {dataset_path}/data"

    pooled = {key: [] for key in _KEYS}

    for parquet_path in tqdm(parquet_files, desc="Episodes"):
        df = pd.read_parquet(parquet_path, columns=["observation.state.ee_state", "action.ee_action"])
        state12 = np.stack(df["observation.state.ee_state"].to_numpy()).astype(np.float64)
        action12 = np.stack(df["action.ee_action"].to_numpy()).astype(np.float64)

        for key, arm_slice in _KEYS.items():
            traj = _relative_trajectory_for_episode(state12[:, arm_slice], action12[:, arm_slice])
            if traj.shape[0] > 0:
                pooled[key].append(traj)

    stats_path = dataset_path / "meta" / "relative_stats.json"
    existing = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    for key, chunks in pooled.items():
        all_samples = np.concatenate(chunks, axis=0)
        existing[key] = {
            "max": np.max(all_samples, axis=0).tolist(),
            "min": np.min(all_samples, axis=0).tolist(),
            "q01": np.quantile(all_samples, 0.01, axis=0).tolist(),
            "q99": np.quantile(all_samples, 0.99, axis=0).tolist(),
            "mean": np.mean(all_samples, axis=0).tolist(),
            "std": np.std(all_samples, axis=0).tolist(),
        }
        print(f"{key}: {all_samples.shape[0]} samples pooled from {len(chunks)} episodes")

    stats_path.write_text(json.dumps(existing, indent=4))
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()

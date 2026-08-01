# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check that generate_relative_stats_fast.py's vectorized math matches Isaac-GR00T's own
(slow, per-frame) ``RelativeActionLoader.load_relative_actions`` exactly, on one real episode,
before trusting it to (re)generate ``meta/relative_stats.json`` for the whole dataset.

Run with the Isaac-GR00T venv:

    cd Isaac-GR00T && uv run python \\
        ../IsaacLab-Arena/isaaclab_arena_gr00t/policy/replay_data/test_relative_stats_fast.py
"""

from __future__ import annotations

import importlib.util
import numpy as np
import pandas as pd
import time
from pathlib import Path

from gr00t.data.stats import RelativeActionLoader
from gr00t.data.types import EmbodimentTag

_DATASET_ROOT = Path(
    "/mnt/ata-Samsung_SSD_870_QVO_8TB_S5SSNF0W200198W/lerobot_cache/BitRobot/G1_WBT_Dex1_Building-Children-Table"
)
_CONFIG_PATH = Path(
    "/mnt/ata-Samsung_SSD_870_QVO_8TB_S5SSNF0W200198W/IsaacLabArena/IsaacLab-Arena/isaaclab_arena_gr00t"
    "/embodiments/g1/g1_dex1_ikea_data_gr00t_n_1_7_config.py"
)
_FAST_SCRIPT_PATH = Path(__file__).parent / "generate_relative_stats_fast.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    # Registers g1_dex1_ikea_config under EmbodimentTag.NEW_EMBODIMENT as a side effect.
    _load_module(_CONFIG_PATH, "g1_dex1_ikea_data_gr00t_n_1_7_config")
    fastgen = _load_module(_FAST_SCRIPT_PATH, "generate_relative_stats_fast")

    t0 = time.time()
    official_loader = RelativeActionLoader(_DATASET_ROOT, EmbodimentTag.NEW_EMBODIMENT, "ee_action_left")
    official = np.stack(official_loader.load_relative_actions(0))
    print(f"official: shape={official.shape} time={time.time() - t0:.2f}s")

    df = pd.read_parquet(
        _DATASET_ROOT / "data" / "chunk-000" / "episode_000000.parquet",
        columns=["observation.state.ee_state", "action.ee_action"],
    )
    state12 = np.stack(df["observation.state.ee_state"].to_numpy()).astype(np.float64)
    action12 = np.stack(df["action.ee_action"].to_numpy()).astype(np.float64)

    t0 = time.time()
    mine = fastgen._relative_trajectory_for_episode(state12[:, 0:6], action12[:, 0:6])
    mine = mine.reshape(official.shape)
    print(f"fast:     shape={mine.shape} time={time.time() - t0:.4f}s")

    max_abs_diff = np.max(np.abs(official - mine))
    print(f"max abs diff: {max_abs_diff}")
    print(f"official[0][0]:  {official[0][0]}")
    print(f"mine[0][0]:      {mine[0][0]}")
    print(f"official[10][5]: {official[10][5]}")
    print(f"mine[10][5]:     {mine[10][5]}")
    assert max_abs_diff < 1e-5, "Vectorized implementation diverges from the official one!"
    print("MATCH: vectorized implementation is numerically equivalent.")


if __name__ == "__main__":
    main()

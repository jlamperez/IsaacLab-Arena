# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sanity-check ``_build_navigate_cmd`` against all 533 episodes before writing anything.

``build_episode_actions_npz.py``'s ``_build_navigate_cmd`` was only exercised against one
episode (the one used for open-loop replay validation). Two of its assumptions were empirical
observations on that single episode, not guarantees:

  - the all-zero placeholder quaternion only ever appears at frame 0
  - ``action.robot_q_desired``'s root pose holds a stale value for ~3 frames between updates
    (which is why ``_VELOCITY_SPAN_FRAMES = 5`` was chosen)

This script re-derives ``navigate_cmd`` for every episode via the same (unmodified) function,
and reports whether those assumptions hold dataset-wide, plus pooled ranges and any NaN/Inf,
so a broken assumption surfaces now instead of after a full retrain. Read-only: does not touch
the dataset, only writes a summary JSON next to this script.

Run with the Isaac-GR00T venv (needs pandas/scipy/tqdm):

    cd Isaac-GR00T && uv run python \\
        ../IsaacLab-Arena/isaaclab_arena_gr00t/policy/replay_data/validate_navigate_cmd.py
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from build_episode_actions_npz import _build_navigate_cmd, _VELOCITY_SPAN_FRAMES

_DATASET_ROOT = Path(
    "/mnt/ata-Samsung_SSD_870_QVO_8TB_S5SSNF0W200198W/lerobot_cache/BitRobot/G1_WBT_Dex1_Building-Children-Table"
)
# Flag frames whose derived command exceeds these as likely derivation artifacts rather than
# real commanded motion (generous thresholds -- a G1 walking/turning at speed shouldn't get
# near these).
_OUTLIER_LIN_VEL = 3.0  # m/s
_OUTLIER_ANG_VEL = 6.0  # rad/s


def _stale_hold_run_lengths(root_pose7: np.ndarray) -> np.ndarray:
    """Run lengths of consecutive frames with an identical (stale) root pose."""
    same_as_prev = np.all(np.isclose(root_pose7[1:], root_pose7[:-1]), axis=1)
    run_lengths = []
    run = 1
    for same in same_as_prev:
        run = run + 1 if same else run
        if not same:
            run_lengths.append(run - 1 if run > 1 else 1)
            run = 1
    run_lengths.append(run)
    return np.array(run_lengths, dtype=np.int64)


def validate_episode(parquet_path: Path) -> tuple[dict, np.ndarray]:
    df = pd.read_parquet(parquet_path, columns=["action.robot_q_desired"])
    q_desired = np.stack(df["action.robot_q_desired"].values).astype(np.float64)
    assert q_desired.shape[1] == 36, f"{parquet_path}: expected 36-D robot_q_desired, got {q_desired.shape[1]}"

    root_pose7 = q_desired[:, 0:7]
    quat_norms = np.linalg.norm(root_pose7[:, 3:7], axis=1)
    zero_quat_idx = np.where(quat_norms < 1e-6)[0].tolist()

    run_lengths = _stale_hold_run_lengths(root_pose7)

    navigate_cmd = _build_navigate_cmd(q_desired)
    outlier_mask = (np.abs(navigate_cmd[:, 0]) > _OUTLIER_LIN_VEL) | (
        np.abs(navigate_cmd[:, 1]) > _OUTLIER_LIN_VEL
    ) | (np.abs(navigate_cmd[:, 2]) > _OUTLIER_ANG_VEL)

    summary = {
        "episode": parquet_path.stem,
        "n_frames": int(q_desired.shape[0]),
        "zero_quat_idx": zero_quat_idx,
        "zero_quat_only_at_frame0": zero_quat_idx in ([], [0]),
        "max_stale_run_length": int(run_lengths.max()),
        "mean_stale_run_length": float(run_lengths.mean()),
        "span_frames": _VELOCITY_SPAN_FRAMES,
        "has_nan": bool(np.isnan(navigate_cmd).any()),
        "has_inf": bool(np.isinf(navigate_cmd).any()),
        "n_outlier_frames": int(outlier_mask.sum()),
        "navigate_cmd_min": navigate_cmd.min(axis=0).tolist(),
        "navigate_cmd_max": navigate_cmd.max(axis=0).tolist(),
    }
    return summary, navigate_cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=_DATASET_ROOT)
    parser.add_argument(
        "--output-path", type=Path, default=Path(__file__).parent / "navigate_cmd_validation.json"
    )
    args = parser.parse_args()

    parquet_files = sorted((args.dataset_path / "data").glob("*/*.parquet"))
    assert parquet_files, f"No parquet files found under {args.dataset_path}/data"

    episode_summaries = []
    pooled = []
    for path in tqdm(parquet_files, desc="Episodes"):
        summary, navigate_cmd = validate_episode(path)
        episode_summaries.append(summary)
        pooled.append(navigate_cmd)

    pooled_arr = np.concatenate(pooled, axis=0)
    zero_quat_beyond_frame0 = [s["episode"] for s in episode_summaries if not s["zero_quat_only_at_frame0"]]
    nan_episodes = [s["episode"] for s in episode_summaries if s["has_nan"] or s["has_inf"]]
    at_risk_span = [
        s["episode"] for s in episode_summaries if s["max_stale_run_length"] > _VELOCITY_SPAN_FRAMES
    ]
    outlier_episodes = sorted(
        (s for s in episode_summaries if s["n_outlier_frames"] > 0),
        key=lambda s: -s["n_outlier_frames"],
    )

    report = {
        "n_episodes": len(episode_summaries),
        "n_frames_total": int(pooled_arr.shape[0]),
        "pooled_min": pooled_arr.min(axis=0).tolist(),
        "pooled_max": pooled_arr.max(axis=0).tolist(),
        "pooled_mean": pooled_arr.mean(axis=0).tolist(),
        "pooled_std": pooled_arr.std(axis=0).tolist(),
        "episodes_with_zero_quat_beyond_frame0": zero_quat_beyond_frame0,
        "episodes_with_nan_or_inf": nan_episodes,
        "episodes_with_stale_run_exceeding_span": at_risk_span,
        "n_outlier_episodes": len(outlier_episodes),
        "top_outlier_episodes": [
            {"episode": s["episode"], "n_outlier_frames": s["n_outlier_frames"]} for s in outlier_episodes[:20]
        ],
        "episodes": episode_summaries,
    }

    args.output_path.write_text(json.dumps(report, indent=2))

    print(f"\nChecked {report['n_episodes']} episodes, {report['n_frames_total']} frames total.")
    print(f"navigate_cmd pooled range: min={report['pooled_min']} max={report['pooled_max']}")
    print(f"navigate_cmd pooled mean={report['pooled_mean']} std={report['pooled_std']}")
    print(f"Episodes with zero-quat frames beyond frame 0: {len(zero_quat_beyond_frame0)}")
    if zero_quat_beyond_frame0:
        print(f"  {zero_quat_beyond_frame0[:10]}{'...' if len(zero_quat_beyond_frame0) > 10 else ''}")
    print(f"Episodes with NaN/Inf in navigate_cmd: {len(nan_episodes)}")
    if nan_episodes:
        print(f"  {nan_episodes[:10]}{'...' if len(nan_episodes) > 10 else ''}")
    print(f"Episodes with a stale-hold run longer than the {_VELOCITY_SPAN_FRAMES}-frame span: {len(at_risk_span)}")
    if at_risk_span:
        print(f"  {at_risk_span[:10]}{'...' if len(at_risk_span) > 10 else ''}")
    print(f"Episodes with >={_OUTLIER_LIN_VEL} m/s or >={_OUTLIER_ANG_VEL} rad/s outlier frames: {len(outlier_episodes)}")
    print(f"Wrote full report to {args.output_path}")


if __name__ == "__main__":
    main()

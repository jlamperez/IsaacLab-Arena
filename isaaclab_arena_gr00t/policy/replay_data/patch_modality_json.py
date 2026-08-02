# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Add per-hand joint groups to the BitRobot dataset's own ``meta/modality.json``.

``ee_state``/``ee_action`` are each 12-D dual-arm columns (left hand pos+euler, right hand
pos+euler concatenated), but ``EndEffectorPose`` models a single end-effector -- see
``g1_dex1_ikea_data_gr00t_n_1_7_config.py``'s per-arm ``ActionConfig``s, which reference the
``ee_state_left``/``ee_state_right``/``ee_action_left``/``ee_action_right`` groups this script
adds. Isaac-GR00T's own joint-group splitting mechanism
(``lerobot_episode_loader.py:_extract_joint_groups``) slices an existing raw column
(``original_key``) by ``start``/``end`` -- no new data is written, this only adds named views
into the existing ``observation.state.ee_state`` / ``action.ee_action`` columns.

This file lives in Arena (not the dataset cache) so the patch is reproducible even if the
dataset gets re-downloaded fresh -- meta/modality.json itself is not tracked in any repo.

Run with plain Python (only needs the stdlib):

    python isaaclab_arena_gr00t/policy/replay_data/patch_modality_json.py \\
        --dataset-path /path/to/BitRobot/G1_WBT_Dex1_Building-Children-Table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_DATASET_ROOT = Path(
    "/mnt/ata-Samsung_SSD_870_QVO_8TB_S5SSNF0W200198W/lerobot_cache/BitRobot/G1_WBT_Dex1_Building-Children-Table"
)

# (modality_type, new_group_name) -> (start, end, original_key)
_NEW_GROUPS = {
    ("state", "ee_state_left"): (0, 6, "observation.state.ee_state"),
    ("state", "ee_state_right"): (6, 12, "observation.state.ee_state"),
    ("action", "ee_action_left"): (0, 6, "action.ee_action"),
    ("action", "ee_action_right"): (6, 12, "action.ee_action"),
}


def patch(dataset_path: Path) -> None:
    modality_path = dataset_path / "meta" / "modality.json"
    modality = json.loads(modality_path.read_text())

    added = []
    for (modality_type, group_name), (start, end, original_key) in _NEW_GROUPS.items():
        if group_name in modality[modality_type]:
            continue
        modality[modality_type][group_name] = {"start": start, "end": end, "original_key": original_key}
        added.append(f"{modality_type}.{group_name}")

    if not added:
        print(f"{modality_path} already has all per-hand groups, nothing to do.")
        return

    modality_path.write_text(json.dumps(modality, indent=4) + "\n")
    print(f"Added {added} to {modality_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=_DATASET_ROOT)
    args = parser.parse_args()
    patch(args.dataset_path)

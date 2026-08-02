# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Add per-hand joint groups and ``navigate_cmd`` to the BitRobot dataset's own meta files.

``ee_state``/``ee_action`` are each 12-D dual-arm columns (left hand pos+euler, right hand
pos+euler concatenated), but ``EndEffectorPose`` models a single end-effector -- see
``g1_dex1_ikea_data_gr00t_n_1_7_config.py``'s per-arm ``ActionConfig``s, which reference the
``ee_state_left``/``ee_state_right``/``ee_action_left``/``ee_action_right`` groups this script
adds. Isaac-GR00T's own joint-group splitting mechanism
(``lerobot_episode_loader.py:_extract_joint_groups``) slices an existing raw column
(``original_key``) by ``start``/``end`` -- no new data is written, this only adds named views
into the existing ``observation.state.ee_state`` / ``action.ee_action`` columns.

``navigate_cmd`` is different: it's a genuinely new raw column (written by
``write_navigate_cmd_column.py``, run that first), not a view into an existing one, so besides
the ``meta/modality.json`` group it also needs a ``meta/info.json`` ``features`` entry (that's
where the column's ``dtype``/``shape`` live -- see e.g. the existing ``action.hand_cmd`` entry).

This file lives in Arena (not the dataset cache) so the patch is reproducible even if the
dataset gets re-downloaded fresh -- meta/modality.json and meta/info.json's ``features`` aren't
tracked in any repo themselves.

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

# navigate_cmd is a top-level group over its own raw column (like hand_cmd), not a split view --
# no need for left/right-style entries.
_NAVIGATE_CMD_GROUP = ("action", "navigate_cmd", 0, 3, "action.navigate_cmd")
_NAVIGATE_CMD_FEATURE = {"dtype": "float32", "shape": [3]}


def patch(dataset_path: Path) -> None:
    modality_path = dataset_path / "meta" / "modality.json"
    info_path = dataset_path / "meta" / "info.json"
    modality = json.loads(modality_path.read_text())
    info = json.loads(info_path.read_text())

    added = []
    for (modality_type, group_name), (start, end, original_key) in _NEW_GROUPS.items():
        if group_name in modality[modality_type]:
            continue
        modality[modality_type][group_name] = {"start": start, "end": end, "original_key": original_key}
        added.append(f"modality.{modality_type}.{group_name}")

    modality_type, group_name, start, end, original_key = _NAVIGATE_CMD_GROUP
    if group_name not in modality[modality_type]:
        modality[modality_type][group_name] = {"start": start, "end": end, "original_key": original_key}
        added.append(f"modality.{modality_type}.{group_name}")
    if original_key not in info["features"]:
        info["features"][original_key] = dict(_NAVIGATE_CMD_FEATURE)
        added.append(f"info.features.{original_key}")

    if not added:
        print(f"{modality_path} / {info_path} already patched, nothing to do.")
        return

    modality_path.write_text(json.dumps(modality, indent=4) + "\n")
    info_path.write_text(json.dumps(info, indent=4) + "\n")
    print(f"Added {added} to {modality_path} / {info_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=_DATASET_ROOT)
    args = parser.parse_args()
    patch(args.dataset_path)

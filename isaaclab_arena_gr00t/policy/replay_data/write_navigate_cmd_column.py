# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Write ``action.navigate_cmd`` as a real column into the BitRobot IKEA dataset's parquet files.

Unlike ``patch_modality_json.py``'s ``ee_state_left``/``ee_action_left`` groups (named views into
an *existing* raw column, no new data written), ``navigate_cmd`` doesn't exist in the raw
recording at all -- it's derived from ``action.robot_q_desired`` via
``build_episode_actions_npz.py``'s ``_build_navigate_cmd`` (validated dataset-wide, see
``validate_navigate_cmd.py`` and its report). This script actually rewrites each episode's
parquet file to add that derived column, then run ``patch_modality_json.py`` to register it in
``meta/info.json``/``meta/modality.json``.

Appends the column via a raw ``pyarrow.Table`` (not a pandas round-trip), so every other column
keeps its exact original arrow type (``fixed_size_list<float>[N]``, not the plain ``list<float>``
pandas would infer) and the parquet file's embedded schema metadata is untouched -- confirmed
this doesn't matter to GR00T's own loader (``lerobot_episode_loader.py``'s ``_load_parquet_data``
just does ``pd.read_parquet``, ignores that metadata), but there's no reason to drop it either.

This mutates the dataset in place. Run ``backup_before_navigate_cmd_write.sh`` first, and
``--dry-run`` on one episode before the full 533 -- this dataset was itself produced by a slow
lerobot v3->v2 conversion, not something to casually redo.

Idempotent: episodes that already have ``action.navigate_cmd`` are skipped.

Run with the Isaac-GR00T venv (needs pandas/pyarrow/scipy):

    cd Isaac-GR00T && uv run python \\
        ../IsaacLab-Arena/isaaclab_arena_gr00t/policy/replay_data/write_navigate_cmd_column.py \\
        [--dry-run] [--episode-index N]
"""

from __future__ import annotations

import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

from build_episode_actions_npz import _build_navigate_cmd

_DATASET_ROOT = Path(
    "/mnt/ata-Samsung_SSD_870_QVO_8TB_S5SSNF0W200198W/lerobot_cache/BitRobot/G1_WBT_Dex1_Building-Children-Table"
)
_COLUMN_NAME = "action.navigate_cmd"


def process_episode(parquet_path: Path, dry_run: bool) -> str:
    table = pq.read_table(parquet_path)
    if _COLUMN_NAME in table.column_names:
        return "skipped (already has column)"

    q_desired = np.array(table.column("action.robot_q_desired").to_pylist(), dtype=np.float64)
    navigate_cmd = _build_navigate_cmd(q_desired)  # (N, 3) float32

    if dry_run:
        return (
            f"dry-run ok: {navigate_cmd.shape[0]} frames, "
            f"min={navigate_cmd.min(axis=0)}, max={navigate_cmd.max(axis=0)}"
        )

    flat = pa.array(navigate_cmd.reshape(-1), type=pa.float32())
    new_col = pa.FixedSizeListArray.from_arrays(flat, 3)
    new_table = table.append_column(_COLUMN_NAME, new_col)
    new_table = new_table.replace_schema_metadata(table.schema.metadata)
    pq.write_table(new_table, parquet_path, compression="snappy")
    return f"wrote {navigate_cmd.shape[0]} frames"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=_DATASET_ROOT)
    parser.add_argument("--episode-index", type=int, default=None, help="Only process one episode.")
    parser.add_argument("--dry-run", action="store_true", help="Compute and report, write nothing.")
    args = parser.parse_args()

    if args.episode_index is not None:
        parquet_files = [args.dataset_path / "data" / "chunk-000" / f"episode_{args.episode_index:06d}.parquet"]
    else:
        parquet_files = sorted((args.dataset_path / "data").glob("*/*.parquet"))
    assert parquet_files, f"No parquet files found under {args.dataset_path}/data"

    for path in tqdm(parquet_files, desc="Episodes"):
        result = process_episode(path, args.dry_run)
        tqdm.write(f"{path.stem}: {result}")

    if args.dry_run:
        print("\nDry run only -- nothing written. Rerun without --dry-run to write for real.")
    else:
        print(f"\nDone. Now run patch_modality_json.py to register {_COLUMN_NAME} in meta/info.json + meta/modality.json.")


if __name__ == "__main__":
    main()

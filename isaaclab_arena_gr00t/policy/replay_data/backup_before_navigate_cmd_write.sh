#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Back up the two parts of the BitRobot IKEA dataset that write_navigate_cmd_column.py and
# patch_modality_json.py actually mutate (data/ -- the parquet files -- and meta/), before
# running either for real. Skips videos/ (~195G) on purpose: neither script touches video data,
# only the parquet + the small meta/*.json files (~3.1G total), so backing up videos/ too would
# just waste time and disk for no safety benefit.
#
# This dataset was itself produced by a slow lerobot v3->v2 conversion from a _v3.0 sibling
# directory -- not something to casually redo, hence backing up before an in-place parquet
# rewrite rather than trusting a re-download/re-convert as the fallback.
#
# Usage:
#   ./backup_before_navigate_cmd_write.sh [dataset_path] [backup_root]
#
# Restore:
#   rm -rf <dataset_path>/data <dataset_path>/meta
#   cp -a <backup_root>/data <backup_root>/meta <dataset_path>/

set -euo pipefail

DATASET_PATH="${1:-/mnt/ata-Samsung_SSD_870_QVO_8TB_S5SSNF0W200198W/lerobot_cache/BitRobot/G1_WBT_Dex1_Building-Children-Table}"
BACKUP_ROOT="${2:-${DATASET_PATH}_backup_pre_navigate_cmd_$(date +%Y%m%d_%H%M%S)}"

if [ -e "${BACKUP_ROOT}" ]; then
    echo "Refusing to overwrite existing backup at ${BACKUP_ROOT}" >&2
    exit 1
fi

mkdir -p "${BACKUP_ROOT}"
echo "Backing up ${DATASET_PATH}/data -> ${BACKUP_ROOT}/data"
cp -a "${DATASET_PATH}/data" "${BACKUP_ROOT}/data"
echo "Backing up ${DATASET_PATH}/meta -> ${BACKUP_ROOT}/meta"
cp -a "${DATASET_PATH}/meta" "${BACKUP_ROOT}/meta"

echo "Done. Backup at ${BACKUP_ROOT}"
du -sh "${BACKUP_ROOT}"

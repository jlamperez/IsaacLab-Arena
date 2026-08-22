# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Write a new HDF5 containing only the chosen LeRobot episode indices, e.g. as an Isaac Lab
Mimic seed set.

Companion to ``group_leg_strategies.py``: once that script's grouping has been checked against
real video and a specific recipe group picked (see its module docstring), this writes just those
episodes out to their own ``record_demos.py``-style HDF5 -- same schema as the source
(``format_version`` root attr, ``env_args``/``total`` on the ``data`` group, per-demo groups
copied via ``h5py.Group.copy``, same approach ``dedupe_demos.py`` and ``merge_demos.py`` use), so
it's a drop-in input anywhere the full deduplicated dataset is (Mimic's ``annotate_demos.py``,
``convert_hdf5_to_lerobot.py``, etc.).

Uses ``dump_leg_events.py``'s ``_episode_to_demo_name`` to resolve LeRobot episode indices to
this HDF5's ``demo_*`` keys -- episode_index and demo name are *not* the same ordering (see that
function's docstring).

Run with the Arena dev container (see the ``dev-container`` skill), needs only ``h5py``:

.. code-block:: bash

    docker exec "$ARENA_CONTAINER" su $(id -un) -c \\
        "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \\
        isaaclab_arena_gr00t/policy/replay_data/extract_episode_subset.py \\
        --episodes 7 15 31 32 36 55 60 62 65 72 84 99 112 \\
        -o /datasets/lerobot_cache/LightwheelAI/iros2026-ikea-assembly/merged/iros2026_ikea_assembly_mimic_seed13.hdf5
"""

from __future__ import annotations

import argparse
import h5py
import sys
from pathlib import Path

from dump_leg_events import _DEFAULT_HDF5, _episode_to_demo_name


def _write_subset(src: h5py.File, demo_names: list[str], dst_path: Path) -> int:
    """Copy just ``demo_names`` from ``src`` into a new HDF5 at ``dst_path``.

    Returns:
        Total ``num_samples`` across the copied demos, for the ``data`` group's ``total`` attr.
    """
    total_samples = 0
    with h5py.File(dst_path, "w") as out:
        if "format_version" in src.attrs:
            out.attrs["format_version"] = src.attrs["format_version"]
        src_data = src["data"]
        data_out = out.create_group("data")
        if "env_args" in src_data.attrs:
            data_out.attrs["env_args"] = src_data.attrs["env_args"]
        for demo_name in demo_names:
            src.copy(src_data[demo_name], data_out, name=demo_name)
            total_samples += int(data_out[demo_name].attrs.get("num_samples", 0))
        data_out.attrs["total"] = total_samples
    return total_samples


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code (0 success, non-zero on error)."""
    parser = argparse.ArgumentParser(
        description="Write a new HDF5 containing only the given LeRobot episode indices from the source dataset."
    )
    parser.add_argument("--episodes", nargs="+", type=int, required=True, help="LeRobot episode indices to keep.")
    parser.add_argument("--hdf5-file", type=str, default=str(_DEFAULT_HDF5), help="Source HDF5 dataset file.")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="Path to write the subset HDF5 to.")
    args = parser.parse_args(argv)

    hdf5_path = Path(args.hdf5_file)
    if not hdf5_path.exists():
        print(f"ERROR: input file does not exist: {hdf5_path}", file=sys.stderr)
        return 2

    dst_path = Path(args.output_file)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as src:
        demo_names = [_episode_to_demo_name(src, episode_index) for episode_index in args.episodes]
        total_samples = _write_subset(src, demo_names, dst_path)

    print()
    print(f"Wrote {len(demo_names)} episodes ({total_samples} total samples) to {dst_path}")
    print(f"  episode indices: {args.episodes}")
    print(f"  demo names:      {demo_names}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

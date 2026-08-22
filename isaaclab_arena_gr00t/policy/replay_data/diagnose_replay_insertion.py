# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic-only: replay one converted demo's full action sequence headlessly and report
where ``leg001`` ends up relative to ``table001``, and where the leg was when the right gripper
started closing.

Built to debug 2026-08-22's "grasps the leg but doesn't get it in the hole" finding on
``iros2026_ikea_assembly_mimic_seed13_arena_frame.hdf5`` (converted by
``convert_demo_to_arena_frame.py``) replayed through the ``g1_wbc_pink_dex1_continuous_grip``
embodiment. Unlike ``annotate_demos.py``'s manual mode, this steps through every recorded action
with no keyboard pause, so it runs to completion unattended and prints numbers instead of
requiring a human to watch the Kit viewport. Env construction mirrors ``annotate_demos.py``'s own
(``environment_registration_callback`` -> ``parse_env_cfg`` -> ``gym.make(..., cfg=env_cfg)``),
since that's the one already confirmed to work this session.

Run with the Arena dev container (see the ``dev-container`` skill), headless is fine here since
nothing needs to be watched -- only h5py/torch/gymnasium, all already in the container.

.. code-block:: bash

    docker exec "$ARENA_CONTAINER" su $(id -un) -c \\
        "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \\
        isaaclab_arena_gr00t/policy/replay_data/diagnose_replay_insertion.py \\
        --demo-index 0 \\
        --embodiment g1_wbc_pink_dex1_continuous_grip \\
        --input_file /datasets/.../mimic/iros2026_ikea_assembly_mimic_seed13_arena_frame.hdf5"
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input_file", type=str, required=True, help="Converted (Arena-frame) seed HDF5.")
parser.add_argument("--demo-index", type=int, default=0, help="Index into sorted(f['data'].keys()) to replay.")
parser.add_argument("--task", type=str, default="assemble_table")
parser.add_argument("--embodiment", type=str, default="g1_wbc_pink_dex1_continuous_grip")
parser.add_argument("--print-every", type=int, default=200, help="Print a progress line every N steps.")
parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many steps (default: full episode).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import numpy as np
import sys
import torch

import isaaclab_arena_environments  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main() -> int:
    sys.argv = ["prog", "--task", args_cli.task, "--embodiment", args_cli.embodiment]
    from isaaclab_arena.environments.isaaclab_interop import environment_registration_callback

    environment_registration_callback()

    with h5py.File(args_cli.input_file, "r") as f:
        demo_name = sorted(f["data"].keys())[args_cli.demo_index]
    print(f"Replaying {demo_name} ({args_cli.input_file})")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)

    env_cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    episode = dataset_file_handler.load_episode(demo_name, env.device)
    initial_state = episode.data["initial_state"]
    actions = episode.data["actions"]

    env.sim.reset()
    env.reset_to(initial_state, None, is_relative=True)

    leg_asset = env.scene["leg001"]
    table_asset = env.scene["table001"]
    right_gripper_prev_closed = False

    for step, action in enumerate(actions):
        if args_cli.max_steps is not None and step >= args_cli.max_steps:
            print(f"Stopping early at step={step} (--max-steps={args_cli.max_steps})")
            break
        action_tensor = torch.as_tensor(action, dtype=torch.float32).reshape(1, -1)
        env.step(action_tensor)

        # Arena action[25:27] (right gripper, duplicated) uses the dataset's own -1=open/+1=close
        # convention as the *input* to the joint's scale/offset formula -- see module docstring.
        right_gripper_closed = bool(action[25] > 0.5) if action.shape[0] > 26 else False
        if right_gripper_closed and not right_gripper_prev_closed:
            leg_pos = leg_asset.data.root_link_pos_w[0].cpu().numpy()
            table_pos = table_asset.data.root_link_pos_w[0].cpu().numpy()
            print(
                f"step={step:6d}  RIGHT GRIPPER STARTS CLOSING  leg001_pos={leg_pos.round(3).tolist()}"
                f"  table001_pos={table_pos.round(3).tolist()}"
            )
        right_gripper_prev_closed = right_gripper_closed

        if step % args_cli.print_every == 0:
            leg_pos = leg_asset.data.root_link_pos_w[0].cpu().numpy()
            table_pos = table_asset.data.root_link_pos_w[0].cpu().numpy()
            dist = float(np.linalg.norm(leg_pos - table_pos))
            print(f"step={step:6d}  leg001_pos={leg_pos.round(3).tolist()}  dist_to_table={dist:.3f}")

    final_leg_pos = leg_asset.data.root_link_pos_w[0].cpu().numpy()
    final_table_pos = table_asset.data.root_link_pos_w[0].cpu().numpy()
    print()
    print(f"FINAL leg001_pos={final_leg_pos.round(4).tolist()}")
    print(f"FINAL table001_pos={final_table_pos.round(4).tolist()}")
    print(f"FINAL xy_separation={np.linalg.norm(final_leg_pos[:2] - final_table_pos[:2]):.4f} m")
    print(f"FINAL z_separation={abs(final_leg_pos[2] - final_table_pos[2]):.4f} m")

    env.close()
    return 0


if __name__ == "__main__":
    exit_code = main()
    simulation_app.close()
    sys.exit(exit_code)

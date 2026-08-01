# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_asset, register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


# ---------------------------------------------------------------------------
# Custom assets for this task -- not part of the shared public asset library,
# so they're registered locally here instead of in isaaclab_arena/assets/object_library.py.
#
# Registration happens lazily in _register_local_assets(), not via module-level
# @register_asset classes: isaaclab_arena_environments/__init__.py eagerly imports every
# environment module (to fire @register_environment below) before SimulationApp exists,
# and importing LibraryObject at module scope would drag isaaclab.sim/pxr in at that point
# too -- see isaaclab_arena/assets/registries.py's ensure_assets_registered() docstring.
# ---------------------------------------------------------------------------

_IKEA_ASSETS_DIR = Path(__file__).resolve().parents[1] / "submodules" / "IROS_IKEA_V13_20260702" / "Assets"


def _register_local_assets(asset_registry: AssetRegistry) -> None:
    """Register this environment's local IKEA assets on first use."""
    if asset_registry.is_registered("leg001", ensure_loaded=False):
        return

    from isaaclab_arena.assets.background_library import LibraryBackground
    from isaaclab_arena.assets.object_library import LibraryObject

    @register_asset
    class Leg001(LibraryObject):
        """A single IKEA table leg (the piece that gets picked up and inserted)."""

        name = "leg001"
        tags = ["object", "ikea"]
        usd_path = str(_IKEA_ASSETS_DIR / "Leg001" / "Leg001.usd")

    @register_asset
    class Table001(LibraryObject):
        """The IKEA tabletop that the leg gets assembled into."""

        name = "table001"
        tags = ["object", "ikea"]
        usd_path = str(_IKEA_ASSETS_DIR / "Table001" / "Table001.usd")

    @register_asset
    class Table278(LibraryBackground):
        """The IKEA kit's own workbench -- what Table001/Leg001 rest on in the reference scene."""

        name = "table278"
        tags = ["background", "ikea"]
        usd_path = str(_IKEA_ASSETS_DIR / "Table278" / "Table278.usd")
        # Workbench top sits at world z~0.76 in its Scene02.usd placement (preserved
        # in build()'s set_initial_pose); ~0.15 m of clearance below that before an
        # object is considered dropped.
        object_min_z = 0.6


@dataclass
class AssembleTableEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the IKEA table-leg assembly environment."""

    enable_cameras: bool = False
    fixed_object: str = "table001"  # tabletop -- stays put, the leg is inserted into it
    held_object: str = "leg001"  # table leg -- picked up and inserted
    background: str = "table278"  # the kit's own workbench -- Table001/Leg001 rest on this in Scene02.usd
    embodiment: str = "g1_wbc_agile_pink_dex1"


@register_environment
class AssembleTableEnvironment(ArenaEnvironmentFactory[AssembleTableEnvironmentCfg]):
    """Registered provider for the IKEA table-leg assembly environment."""

    name: str = "assemble_table"
    _legacy_argparse_cfg_type = AssembleTableEnvironmentCfg

    def build(self, cfg: AssembleTableEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        _register_local_assets(self.asset_registry)

        import isaaclab.sim as sim_utils

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.assembly_task import AssemblyTask
        from isaaclab_arena.utils.pose import Pose

        # Step 1: Retrieve assets from the registry
        background = self.asset_registry.get_asset_by_name(cfg.background)()
        fixed_asset = self.asset_registry.get_asset_by_name(cfg.fixed_object)()  # tabletop
        held_asset = self.asset_registry.get_asset_by_name(cfg.held_object)()  # leg
        light = self.asset_registry.get_asset_by_name("light")()
        # Reference scene also has a distant/sun light alongside its dome-ish
        # fill light; DomeLight alone left harder shadows than Scene_enabled.usd.
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()
        # GroundPlaneCfg tints its grid material black by default (color=(0,0,0));
        # override so it isn't rendered pitch black.
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")(
            spawner_cfg=sim_utils.GroundPlaneCfg(color=(0.05, 0.25, 0.5))
        )

        # Step 2: Select the embodiment
        # g1.py's own default init_state (pos=(0.8, -1.38, 0.78), rot=(0,0,1,0)) puts
        # the robot far off in y from table001/leg001 (y=0) and facing away from this
        # layout. Identity rotation (opposite of the default's 180 deg about z) faces
        # the table correctly.
        #
        # Position tuned 2026-07-30 after measuring the robot genuinely could not reach
        # leg001 from the original (-0.35, 0.0) spot: pelvis-to-leg distance was ~0.95 m,
        # over 2x the arm's max straight-line reach (shoulder-to-fingertip ~0.42 m,
        # measured from G1_GRIPPER.usd's link offsets). (0.05, -0.3) keeps ~0.12 m
        # clearance from Table278's near edge (the workbench the robot must not stand
        # inside of, world x >= 0.171 -- see Table278.usd's rotated bounding box) while
        # shifting toward leg001's own y so the right hand (already the hand that grips
        # the leg, see gr00t_dex1_eef_closedloop_policy.py's verified left/right mapping)
        # only needs ~0.54 m of reach -- still beyond pure straight-arm length, but
        # closeable with the WBC's torso lean/crouch, unlike the original ~0.95 m gap.
        #
        # Nudged again 2026-07-30 per Jorge's visual read of a closed-loop rollout at
        # (0.05, -0.3): the right hand does reach leg001, but the left hand can't keep up
        # once the right is committed to the grasp. Backed off slightly (x: 0.05 -> -0.05)
        # and added a 15 deg yaw toward leg001's side, matching the more turned-in stance
        # visible in the BitRobot reference footage's first frame.
        #
        # Nudged again 2026-07-30 per Jorge: shift a bit further along the robot's OWN
        # left side (its local +y axis, rotated by the 15 deg yaw above -> world
        # (+0.026, +0.097) per 0.1 m of shift) to give the left hand more room, since it
        # couldn't keep up once the right hand committed to the grasp.
        #
        # Backed off again 2026-07-30 per Jorge, comparing our head-cam frame against the
        # BitRobot reference footage's first frame side by side: ours has the hands right
        # up against the leg/frame filling the shot, the real one shows noticeably more
        # standoff distance and table context around the hands. x: -0.024 -> -0.15, then
        # eased back in slightly to -0.08 (per Jorge, -0.15 overshot -- too far now).
        #
        # Re-tuned 2026-07-31 per Jorge, now reasoning from measured geometry instead of
        # visual read alone: computed the right shoulder's world position (pelvis + its
        # local (0,-0.1,~0.29) offset, rotated by yaw) and its distance to leg001, at
        # several candidate x's, trading off against clearance from Table278's near edge
        # (world x >= 0.171, see the reach-fix comment above). Also reduced the yaw from
        # 15 to 10 deg per Jorge (wanted some turn-in, just not as much). x=0.10 (~7cm
        # table clearance) + 10 deg yaw gave a shoulder-to-leg distance of ~0.52m, down
        # from ~0.69m at the previous (-0.08, 15 deg) pose -- but visually too close to
        # the table once actually placed, per Jorge. Backed off to x=0.02 (~0.59m to the
        # leg, ~15cm table clearance) -- still too close per Jorge. Backed off again to
        # x=-0.05 (~0.65m to the leg, ~22cm table clearance), keeping the 10 deg yaw.
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
            initial_pose=Pose(position_xyz=(-0.05, -0.203, 0.78), rotation_xyzw=(0.0, 0.0, -0.08715574274765817, 0.9961946980917455)),
        )
        # Step 3: Place the support surface and the two assembly parts.
        # These poses are read directly out of the kit's own reference scene
        # (submodules/IROS_IKEA_V13_20260702/Scene02.usd), which already has
        # Table278/Table001/Leg001 in a valid, non-overlapping arrangement --
        # extracted with pxr.UsdGeom.XformCache and rigidly translated (no
        # re-rotation) so table001 lands at the (0.5, 0.0) spot fixed_asset
        # used to occupy. Still a first pass: verify robot reach in the viewer.
        background.set_initial_pose(
            Pose(position_xyz=(0.5461, -0.0091, 0.3823), rotation_xyzw=(0.0, 0.0, 0.7071, 0.7071))
        )

        fixed_asset.set_initial_pose(
            Pose(position_xyz=(0.5, 0.0, 0.7994), rotation_xyzw=(0.7071, 0.7071, 0.0, 0.0)),
        )
        held_asset.set_initial_pose(
            Pose(position_xyz=(0.5111, -0.4067, 0.7905), rotation_xyzw=(-0.5, -0.5, 0.5, 0.5)),
        )
        # Room floor -- Table278's placement above preserves its own floor
        # contact from Scene02.usd (its base sits at world z~0), so the floor
        # itself belongs at z=0.
        ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))

        # Step 4: Compose the scene
        scene = Scene(assets=[background, fixed_asset, held_asset, ground_plane, light, directional_light])

        # Step 5: Define the task
        task = AssemblyTask(
            task_description="Assemble the table leg into the tabletop",
            fixed_asset=fixed_asset,
            held_asset=held_asset,
            auxiliary_asset_list=[],
            background_scene=background,
            # NOTE: AssemblyTask's defaults (2 cm tolerance) were tuned for
            # mm-scale factory pegs (see Peg/Hole in object_library.py).
            # Table001/Leg001 are furniture-scale -- measure the real
            # leg-socket geometry and tighten/loosen these before relying on
            # the success signal.
            max_x_separation=0.02,
            max_y_separation=0.02,
            max_z_separation=0.02,
            # randomize_poses_and_align_auxiliary_assets applies this range as an
            # absolute position (env-origin-relative, not relative to fixed_asset) to
            # BOTH fixed_asset and held_asset on every reset -- z must match table278's
            # actual resting height (~0.7994), not the old table-at-z=0 convention.
            # Box widened for furniture scale: Table001 alone is 0.42x0.58m, bigger
            # than the old 0.3x0.4m box, leaving no room for leg001 to land clear of it.
            pose_range={
                "x": (0.3, 0.9),
                "y": (-0.4, 0.4),
                "z": (0.79, 0.80),
                "yaw": (-3.14, 3.14),
            },
            # table001's largest half-extent is ~0.29 m -- min_separation must clear
            # that (plus leg001's own size) or the sampler can still land the leg on
            # top of the tabletop.
            min_separation=0.4,
        )

        # AssemblyTask always randomizes fixed_asset/held_asset independently within
        # pose_range on every reset (see events.py's randomize_poses_and_align_auxiliary_assets).
        # That's a poor fit here: table001 is much bigger than the sampling box, so the two
        # can land far apart, out of the (held_asset-following) camera's view. Until this
        # environment needs real randomization, pin both to the Scene02.usd-derived poses
        # above instead, using the same set_object_pose event Arena already uses elsewhere.
        from isaaclab.managers import EventTermCfg, SceneEntityCfg

        from isaaclab_arena.terms.events import set_object_pose

        task.events_cfg.randomize_asset_positions = EventTermCfg(
            func=set_object_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(fixed_asset.name),
                "pose": Pose(position_xyz=(0.5, 0.0, 0.7994), rotation_xyzw=(0.7071, 0.7071, 0.0, 0.0)),
            },
        )
        task.events_cfg.reset_held_asset_pose = EventTermCfg(
            func=set_object_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(held_asset.name),
                "pose": Pose(position_xyz=(0.5111, -0.4067, 0.7905), rotation_xyzw=(-0.5, -0.5, 0.5, 0.5)),
            },
        )

        # AssemblyTask.get_viewer_cfg() hardcodes an offset tuned for the peg_insert
        # layout; with the robot now in front of the table instead of off to the
        # side, override on the instance to frame this scene instead. First pass --
        # verify the angle/side in the viewer and adjust.
        import numpy as np

        from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

        task.get_viewer_cfg = lambda: get_viewer_cfg_look_at_object(
            lookat_object=held_asset, offset=np.array([-3.0, 2.5, 3.0])
        )

        # Step 6: Assemble the environment
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
        )
        return isaaclab_arena_environment

    # TODO(cvolk, 2026-07-03): [typed-config-migration] Delete this CLI-only option when teleoperation runners
    # receive typed configuration instead of the environment subparser namespace.
    @staticmethod
    def _add_legacy_cli_only_args(parser: argparse.ArgumentParser) -> None:
        # Consumed directly by teleop.py and record_demos.py, not by build(cfg).
        parser.add_argument("--teleop_device", type=str, default=None)

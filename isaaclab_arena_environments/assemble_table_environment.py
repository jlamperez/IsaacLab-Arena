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

    import isaaclab.sim as sim_utils
    from isaaclab.utils import configclass

    from isaaclab_arena.assets.background_library import LibraryBackground
    from isaaclab_arena.assets.object_library import LibraryObject, LightBase

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

    @configclass
    class _RectLightCfg(sim_utils.LightCfg):
        """A rectangular area light -- Isaac Lab's light spawners don't include one.

        ``spawn_light`` (the generic light-spawner function every ``LightCfg`` subclass
        uses via its inherited ``func`` default) creates a prim of ``prim_type`` and then
        sets every other dataclass field as a matching ``inputs:<field>`` USD attribute --
        so a plain ``LightCfg`` subclass with ``prim_type="RectLight"`` plus ``width``/
        ``height`` (mirroring how ``DiskLightCfg`` adds ``radius``) works with no custom
        spawn function needed.
        """

        prim_type = "RectLight"
        width: float = 1.0
        height: float = 1.0

    @register_asset
    class RobofinalsRectLight(LightBase):
        """Large overhead area light matching robofinals/LightwheelAI's Scene02.usd rig.

        Scene02.usd lights this same workbench with a single ``RectLight`` (100x100m,
        ~14m up) instead of Arena's usual DomeLight+DistantLight combo -- no dome light
        at all, which is also why the sky renders black with no separate
        ``visible_in_primary_ray`` trick needed. See this asset's call site in build()
        for how the exact position/orientation/intensity were extracted via pxr from
        Scene02.usd's ``/World/RectLight_01`` and re-anchored onto Arena's own coordinate
        frame (the same rigid translation Table278/Table001/Leg001's poses already use).
        """

        name = "robofinals_rect_light"
        tags = ["light", "ikea"]
        default_prim_path = "/World/RobofinalsRectLight"
        default_intensity = 800.0
        default_spawner_cfg = _RectLightCfg(intensity=default_intensity, width=100.0, height=100.0)

        spawner_cfg: _RectLightCfg

        def __init__(
            self,
            instance_name: str | None = None,
            prim_path: str | None = default_prim_path,
            initial_pose=None,
            spawner_cfg: _RectLightCfg = default_spawner_cfg,
        ):
            super().__init__(
                instance_name=instance_name,
                prim_path=prim_path,
                initial_pose=initial_pose,
                spawner_cfg=spawner_cfg,
            )

    @register_asset
    class RobofinalsSphereLight(LightBase):
        """Small, very bright point light matching Scene02.usd's ground-plane SphereLight.

        Isaac Sim's stock grid ground-plane asset (``default_environment.usd``, what
        ``ground_plane``/``GroundPlaneCfg`` spawns) comes with its own small SphereLight
        baked in at ``{prim_path}/SphereLight`` -- Isaac Lab's own ``spawn_ground_plane``
        deliberately hides it right after spawning (its comment: "isn't bright enough and
        messes up with the user's lighting settings"), which is exactly why our own
        ground plane never showed it. Scene02.usd is a static export of that same stock
        asset, though, made before/without ever going through ``spawn_ground_plane`` --
        so its copy (``/World/FlatGrid/SphereLight``) was never hidden and stayed lit,
        and it's what reads as the sharp point-light highlight on Table278's glossy top
        in robofinals' renders (per Jorge, it looked like an actual light, not just a
        specular reflection off the RectLight above -- confirmed by finding this second,
        independent light prim directly in the USD). See this asset's call site in
        build() for the source values (extracted via pxr).
        """

        name = "robofinals_sphere_light"
        tags = ["light", "ikea"]
        default_prim_path = "/World/RobofinalsSphereLight"
        default_intensity = 100000.0
        default_spawner_cfg = sim_utils.SphereLightCfg(intensity=default_intensity, radius=0.25)

        spawner_cfg: sim_utils.SphereLightCfg

        def __init__(
            self,
            instance_name: str | None = None,
            prim_path: str | None = default_prim_path,
            initial_pose=None,
            spawner_cfg: sim_utils.SphereLightCfg = default_spawner_cfg,
        ):
            super().__init__(
                instance_name=instance_name,
                prim_path=prim_path,
                initial_pose=initial_pose,
                spawner_cfg=spawner_cfg,
            )


def _spawn_dark_ground_plane(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the stock grid ground plane, then dim it to match robofinals' floor.

    GroundPlaneCfg only exposes the grid material's ``diffuse_tint`` (via its ``color``
    field); robofinals' own reference scene leaves that at its default (1,1,1) and instead
    overrides ``inputs:albedo_brightness`` (0.19 vs. the stock ~1.0) on the same OmniPBR
    grid material -- see the comment above this function's call site in build() for how
    that was confirmed. Patched here via USD directly since the dataclass has no field
    for it.
    """
    from pxr import Sdf  # noqa: PLC0415

    import isaaclab.sim as sim_utils  # noqa: PLC0415

    prim = sim_utils.spawn_ground_plane(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    sim_utils.change_prim_property(
        prop_path=f"{prim_path}/Looks/theGrid/Shader.inputs:albedo_brightness",
        value=0.19,
        type_to_create_if_not_exist=Sdf.ValueTypeNames.Float,
    )
    return prim


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
        # Tried a DomeLight (uniform gray-white, visible_in_primary_ray=False to hide its
        # color from camera views) + DistantLight combo first (2026-08-08), to get a black
        # sky without going pitch-black. But that read as a workaround, not the real
        # answer, and left a hard directional shadow under fixed_asset that robofinals'
        # own renders don't show. Inspected Scene02.usd directly via pxr instead: it has
        # no dome light or sun at all -- lighting comes from a single huge RectLight
        # (/World/RectLight_01, 100x100m, ~14m up, intensity=800), which is why the sky
        # is black with nothing to suppress and shadows are soft (huge area light, not a
        # point/directional source). RobofinalsRectLight (registered above) reproduces
        # that light exactly: its world position/orientation were read from
        # /World/RectLight_01's xformOps and re-anchored with the same rigid (x, y)
        # offset (+1.944878, -2.372593) that Table278/Table001/Leg001's own poses already
        # apply to go from Scene02.usd's raw coordinates to Arena's -- confirmed by
        # computing that offset independently from both Table278's and Table001's raw vs.
        # Arena poses and getting the same value (to ~1e-4, Arena's own rounding) each
        # time. RectLight_01's rotation is a 90 deg spin about its own -Z (already
        # facing down in this Z-up scene) so no re-derivation was needed there.
        rect_light = self.asset_registry.get_asset_by_name("robofinals_rect_light")()
        rect_light.set_initial_pose(
            Pose(
                position_xyz=(-2.114823, -2.372593, 13.926534),
                rotation_xyzw=(0.0, 0.0, 0.7071067811865475, 0.7071067811865476),
            )
        )
        # Per Jorge, robofinals' render also shows a sharp point-light highlight on
        # Table278's top that the RectLight alone doesn't explain -- traced to a second,
        # independent light: /World/FlatGrid/SphereLight in Scene02.usd. It's the small
        # SphereLight the *stock* grid ground-plane asset (default_environment.usd, what
        # our own "ground_plane" spawns) comes with baked in -- IsaacLab's own
        # spawn_ground_plane() always hides it right after spawning (see its "isn't
        # bright enough and messes up with the user's lighting settings" comment), which
        # is exactly why our ground plane never showed one. Scene02.usd is a static
        # export made without going through that hiding step, so its copy stayed lit.
        # First tried its raw local position (0, 0, 2.5) unmodified (2026-08-08), reasoning
        # that /World/FlatGrid sits at Scene02.usd's own world origin same as our
        # ground_plane -- too close to the table per Jorge's visual read (the highlight
        # landed well short of where robofinals shows it). That reasoning didn't actually
        # hold: FlatGrid's own origin in Scene02.usd is just wherever that asset happened
        # to be dropped, not something meaningfully tied to Table278's position, so there
        # was no reason to treat it as a landmark shared with Arena's independently-chosen
        # ground_plane origin. Applying the *same* rigid (x, y) offset used for every other
        # Scene02.usd-derived pose in this file (Table278/Table001/Leg001/RectLight_01)
        # instead -- consistent with all of them, and rigid translation preserves the
        # ~2.75m horizontal separation from Table278 that the raw scene actually has.
        sphere_light = self.asset_registry.get_asset_by_name("robofinals_sphere_light")()
        sphere_light.set_initial_pose(
            Pose(position_xyz=(1.944878, -2.372593, 2.5), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )
        # GroundPlaneCfg tints its grid material black by default (color=(0,0,0));
        # override so it isn't rendered pitch black.
        #
        # Tried matching robofinals/LightwheelAI's dark-floor look purely via
        # GroundPlaneCfg.color (2026-08-08) -- that field only sets the grid material's
        # diffuse_tint, and several tint values were tried (down to (0.008, 0.02, 0.04))
        # without landing on the right look, per Jorge's visual read each time.
        # Inspected robofinals' own reference scene (submodules/IROS_IKEA_V13_20260702/
        # Scene02.usd, prim /World/FlatGrid/Looks/theGrid/Shader) directly via pxr to stop
        # guessing: it uses the *exact same* OmniPBR grid material as Isaac Sim's stock
        # default_environment.usd (same Wireframe_blue.png diffuse/emissive textures,
        # same emissive_intensity=1000 self-lit blue grid lines -- confirmed identical
        # inputs:* values on both, diffuse_tint included). The only value the reference
        # scene actually overrides is inputs:albedo_brightness=0.19 (vs. the ~1.0 default),
        # with diffuse_tint left untouched at (1,1,1) -- so diffuse_tint was never the
        # right knob to turn. GroundPlaneCfg has no albedo_brightness field, so
        # _spawn_dark_ground_plane below spawns the stock plane via
        # sim_utils.spawn_ground_plane and then patches that one extra shader input via
        # USD directly, matching robofinals' value exactly instead of eyeballing a tint.
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")(
            spawner_cfg=sim_utils.GroundPlaneCfg(color=(1.0, 1.0, 1.0), func=_spawn_dark_ground_plane)
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
        # Back to the tuned reach pose (see the long history above) for grasp testing with
        # all 4 legs now present -- the wider x=0.0/-0.2 positions were temporary, used only
        # for the navigate_cmd/"move to table" experiments on 2026-08-02/06.
        #
        # Nudged closer again, 2026-08-06: the 4-leg rollout got a genuine reach toward the
        # nearest leg (see gr00t_dex1_eef_closedloop_policy.py's session notes) before losing
        # balance/camera framing around t=6-7s -- shortening the remaining reach distance may
        # reduce how far the arm/torso has to commit. Per Jorge, pushed past the -0.02 first
        # try, back to x=0.02 -- the same value rejected as "too close" on 2026-07-31, but
        # that was before the quaternion-convention fix, the EEF-composition fix, and the
        # 4-leg scene composition above, so worth re-testing now rather than assuming it
        # still holds.
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
            initial_pose=Pose(position_xyz=(0.02, -0.203, 0.78), rotation_xyzw=(0.0, 0.0, -0.08715574274765817, 0.9961946980917455)),
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

        # Extra legs (cosmetic only -- NOT tracked by AssemblyTask's fixed/held-asset
        # success check, see the multi-socket TODO on that task above): the BitRobot
        # dataset always shows all 4 legs laid out next to the tabletop, never just 1,
        # so a policy fine-tuned on that footage is seeing an out-of-distribution scene
        # here otherwise. Laid out in a row alongside held_asset (same x/z/rotation).
        # y-spacing corrected 2026-08-08 from an initial 0.12 m guess to the kit's own
        # real spacing (~0.06 m), read directly from Scene02.usd's Leg001/Leg001_01/
        # Leg001_03/Leg001_06 world transforms via pxr.UsdGeom.XformCache (matches the
        # LightwheelAI/iros2026-ikea-assembly HDF5 dataset's initial_state exactly, since
        # that dataset was recorded in this same scene) -- the 0.12 m guess left the extra
        # legs standing in empty space relative to where a replayed grasp trajectory
        # actually reaches. held_asset (leg001, y=-0.4067) is untouched -- it already
        # matched Leg001's own real position; only the 3 cosmetic extras move.
        extra_leg_2 = self.asset_registry.get_asset_by_name(cfg.held_object)(instance_name="leg001_2")
        extra_leg_3 = self.asset_registry.get_asset_by_name(cfg.held_object)(instance_name="leg001_3")
        extra_leg_4 = self.asset_registry.get_asset_by_name(cfg.held_object)(instance_name="leg001_4")
        extra_leg_2.set_initial_pose(Pose(position_xyz=(0.5111, -0.3468, 0.7905), rotation_xyzw=(-0.5, -0.5, 0.5, 0.5)))
        extra_leg_3.set_initial_pose(Pose(position_xyz=(0.5111, -0.4666, 0.7905), rotation_xyzw=(-0.5, -0.5, 0.5, 0.5)))
        extra_leg_4.set_initial_pose(Pose(position_xyz=(0.5111, -0.5296, 0.7905), rotation_xyzw=(-0.5, -0.5, 0.5, 0.5)))

        # Step 4: Compose the scene
        scene = Scene(
            assets=[
                background,
                fixed_asset,
                held_asset,
                extra_leg_2,
                extra_leg_3,
                extra_leg_4,
                ground_plane,
                rect_light,
                sphere_light,
            ]
        )

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

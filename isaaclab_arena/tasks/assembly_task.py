# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

import isaaclab_arena_environments.mdp as mdp
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.common.mimic_default_params import MIMIC_DATAGEN_CONFIG_DEFAULTS
from isaaclab_arena.tasks.events import randomize_poses_and_align_auxiliary_assets
from isaaclab_arena.tasks.predicates.spatial import objects_in_proximity
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


@register_task
class AssemblyTask(TaskBase):
    """
    Assembly task where an object needs to be assembled with a base object, like peg insert, gear mesh, etc.

    The default Mimic cfg is the generic ``FactoryAssemblyMimicEnvCfg`` (no subtasks -- concrete
    tasks must subclass it, see that class's docstring). To wire up a specific subtask sequence,
    pass ``mimic_env_cfg_factory`` -- same pattern as ``PickAndPlaceTask``, except this one
    receives ``arm_mode`` (what ``ArenaEnvBuilder.compose_manager_cfg`` actually calls
    ``get_mimic_env_cfg`` with), not an embodiment name string::

        def _factory(arm_mode):
            return MyCustomMimicEnvCfg(embodiment_name=..., ...)

        AssemblyTask(..., mimic_env_cfg_factory=_factory)
    """

    def __init__(
        self,
        fixed_asset: Asset,
        held_asset: Asset,
        auxiliary_asset_list: list[Asset],
        background_scene: Asset,
        episode_length_s: float | None = None,
        max_x_separation: float = 0.020,
        max_y_separation: float = 0.020,
        max_z_separation: float = 0.020,
        task_description: str | None = None,
        pose_range: dict[str, tuple[float, float]] | None = None,
        min_separation: float = 0.10,
        randomization_mode: Literal["held_and_fixed_only", "held_fixed_and_auxiliary"] = "held_and_fixed_only",
        mimic_env_cfg_factory: Callable[[ArmMode], MimicEnvCfg] | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.fixed_asset = fixed_asset
        self.held_asset = held_asset
        self.auxiliary_asset_list = auxiliary_asset_list
        self.background_scene = background_scene
        self.mimic_env_cfg_factory = mimic_env_cfg_factory
        self.scene_config = None
        # We use specialize randomization at reset for this task. So disable default pose resets.
        self.disable_default_pose_resets()
        self.events_cfg = EventsCfg(
            pose_range=pose_range if pose_range is not None else {},
            min_separation=min_separation,
            asset_cfgs=[SceneEntityCfg(asset.name) for asset in [self.fixed_asset, self.held_asset]],
            fixed_asset_cfg=SceneEntityCfg(self.fixed_asset.name),
            auxiliary_asset_cfgs=[SceneEntityCfg(asset.name) for asset in self.auxiliary_asset_list],
            randomization_mode=randomization_mode,
        )
        self.termination_cfg = self._make_termination_cfg(
            max_x_separation=max_x_separation,
            max_y_separation=max_y_separation,
            max_z_separation=max_z_separation,
        )
        self.task_description = (
            f"Assemble the {self.held_asset.name} with the {self.fixed_asset.name}"
            if task_description is None
            else task_description
        )

    def disable_default_pose_resets(self):
        for asset in [self.fixed_asset, self.held_asset, *self.auxiliary_asset_list]:
            asset.disable_reset_pose()

    def get_scene_cfg(self):
        """Get scene configuration."""
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def _make_termination_cfg(
        self,
        max_x_separation: float,
        max_y_separation: float,
        max_z_separation: float,
    ):
        """
        Create termination configuration for the assembly task.

        Args:
            max_x_separation: Maximum allowed separation in x-axis for success.
            max_y_separation: Maximum allowed separation in y-axis for success.
            max_z_separation: Maximum allowed separation in z-axis for success.

        Returns:
            TerminationsCfg: The termination configuration.
        """
        success = TerminationTermCfg(
            func=objects_in_proximity,
            params={
                "object_cfg": SceneEntityCfg(self.held_asset.name),
                "target_object_cfg": SceneEntityCfg(self.fixed_asset.name),
                "max_x_separation": max_x_separation,  # Tolerance for assembly alignment
                "max_y_separation": max_y_separation,
                "max_z_separation": max_z_separation,
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.held_asset.name),
            },
        )
        return TerminationsCfg(
            success=success,
            object_dropped=object_dropped,
        )

    def get_events_cfg(self):
        """Get events configuration for assembly task."""
        return self.events_cfg

    def get_prompt(self):
        raise NotImplementedError("Function not implemented yet.")

    def get_mimic_env_cfg(self, arm_mode):
        """Build the Mimic env cfg for this task.

        ``arm_mode`` (an ``ArmMode``) is what ``ArenaEnvBuilder.compose_manager_cfg`` actually
        passes here (confirmed 2026-08-22 -- this method's signature previously said
        ``embodiment_name: str``, which doesn't match that real call site at all; nothing had
        ever exercised this path before, since ``FactoryAssemblyMimicEnvCfg`` was an unused
        stub). If ``mimic_env_cfg_factory`` was passed at construction, invoke it with
        ``arm_mode`` and return its result. Otherwise build the generic (subtask-less)
        ``FactoryAssemblyMimicEnvCfg``.
        """
        if self.mimic_env_cfg_factory is not None:
            return self.mimic_env_cfg_factory(arm_mode)
        return FactoryAssemblyMimicEnvCfg()

    def get_metrics(self) -> list[MetricBase]:
        return [
            SuccessRateMetric(),
            ObjectMovedRateMetric(self.held_asset),
        ]

    def get_viewer_cfg(self) -> ViewerCfg:
        """Get viewer configuration to look at the held asset.

        Camera is positioned at right-back-top of the object for better view of assembly operations.
        """
        return get_viewer_cfg_look_at_object(
            lookat_object=self.held_asset,
            offset=np.array([1.5, -0.5, 1.0]),  # Rotated 180° around z-axis from original view
        )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)

    success: TerminationTermCfg = MISSING

    object_dropped: TerminationTermCfg = MISSING


@configclass
class EventsCfg:
    """
    Configuration for assembly task events.
    """

    reset_all: EventTermCfg = MISSING
    randomize_asset_positions: EventTermCfg = MISSING

    def __init__(
        self,
        pose_range: dict[str, tuple[float, float]],
        min_separation: float,
        asset_cfgs: list[SceneEntityCfg],
        fixed_asset_cfg: SceneEntityCfg,
        auxiliary_asset_cfgs: list[SceneEntityCfg],
        randomization_mode: Literal["held_and_fixed_only", "held_fixed_and_auxiliary"] = "held_and_fixed_only",
    ):
        self.reset_all = EventTermCfg(
            func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True}
        )

        self.randomize_asset_positions = EventTermCfg(
            func=randomize_poses_and_align_auxiliary_assets,
            mode="reset",
            params={
                "pose_range": pose_range,
                "min_separation": min_separation,
                "asset_cfgs": asset_cfgs,
                "fixed_asset_cfg": fixed_asset_cfg,
                "auxiliary_asset_cfgs": auxiliary_asset_cfgs,
                "randomization_mode": randomization_mode,
            },
        )


@configclass
class FactoryAssemblyMimicEnvCfg(MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for assembly task.

    Note:
        This is a base configuration class. Specific assembly tasks
        (e.g., PegInsert, GearMesh) should create their own subclasses with
        appropriate asset names.
    """

    embodiment_name: str = MISSING
    fixed_asset_name: str = MISSING
    held_asset_name: str = MISSING
    assist_asset_list_names: list[str] = MISSING


@configclass
class G1AssemblySingleLegMimicEnvCfg(FactoryAssemblyMimicEnvCfg):
    """Mimic env cfg for a G1 dual-arm+locomotion assembly task, right-hand-only, no handoff.

    Built for ``assemble_table`` (see ``isaaclab_arena_environments/assemble_table_environment.py``,
    used via ``AssemblyTask(..., mimic_env_cfg_factory=...)``), but nothing here is
    assemble_table-specific -- it only reaches for ``self.held_asset_name``/``self.fixed_asset_name``
    (from the base class), not any hardcoded object name. ``assemble_table``'s own
    ``AssemblyTask`` only tracks *one* held/fixed asset pair for success (``held_asset`` is
    pinned to Leg001_01's slot; the other 3 legs are cosmetic only, "NOT tracked by AssemblyTask's
    fixed/held-asset success check" per that environment's own scene-building comment) -- so this
    matches, with subtasks for one grasp+insert, not a multi-object sequence. Extending to more
    objects needs ``AssemblyTask`` itself to support multiple held/fixed asset pairs first (not
    done yet).

    Matches the recipe every one of the 13 hand-picked ``iros2026_ikea_assembly_mimic_seed13.hdf5``
    seed episodes (``isaaclab_arena_gr00t/policy/replay_data/group_leg_strategies.py``'s output)
    uses for their first leg: right hand only, no handoff. 3 subtask groups, mirroring
    ``G1PickAndPlaceMimicEnvCfg`` (``isaaclab_arena/tasks/pick_and_place_task.py`` -- the only
    other G1 locomotion+manipulation Mimic cfg in this repo) rather than the simpler single-arm
    ``PickPlaceMimicEnvCfg``, because a walk-to-the-object phase is needed even for this
    single-object scope (confirmed 2026-08-22 against the source HDF5: every recorded demo starts
    with the robot ~1m from the table, not already in reach):

    - ``right``: ``idle_right`` (walking, arm not yet reaching) -> ``grasp_object`` (picks up
      ``held_asset_name``) -> final (inserts into ``fixed_asset_name``).
    - ``left``: a single idle placeholder (this hand never touches anything) -- deliberately
      *not* mirroring ``G1PickAndPlaceMimicEnvCfg``'s 3-phase idle-arm pattern, since a
      truly-idle eef with an empty subtask signal list is skipped entirely during manual
      annotation (see ``annotate_demos.py``'s ``annotate_episode_in_manual_mode``), instead of
      making the annotator mark 2 meaningless points.
    - ``body``: ``navigate_to_object`` (the initial walk) -> final (no more walking needed once
      the object is in reach).

    ``mimic_recorder_config`` is left at the base-class default (unset), NOT
    ``G1LocomanipRecorderManagerCfg`` (what the proven galileo_g1_locomanip_pick_and_place
    example uses) -- that recorder's ``navigate_cmd`` patch looks up a specific
    action-manager-term attribute that was only verified against ``g1_wbc_pink``, not
    ``assemble_table``'s embodiment (``g1_wbc_agile_pink_dex1``). Flagged as an open question,
    not yet checked: if ``navigate_to_object`` annotation/generation doesn't produce sane
    navigate_cmd values, look at ``G1LocomanipRecorderManagerCfg``
    (isaaclab_arena_g1/g1_env/mdp/recorders/g1_locomanip_recorder_cfg.py) next.
    """

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = f"{self.held_asset_name}_into_{self.fixed_asset_name}_D0"
        for key, value in MIMIC_DATAGEN_CONFIG_DEFAULTS.items():
            setattr(self.datagen_config, key, value)

        self.subtask_configs["right"] = [
            SubTaskConfig(
                object_ref=self.held_asset_name,
                subtask_term_signal="idle_right",
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=0,
            ),
            SubTaskConfig(
                object_ref=self.held_asset_name,
                subtask_term_signal="grasp_object",
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
            ),
            SubTaskConfig(
                object_ref=self.fixed_asset_name,
                subtask_term_signal=None,
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
            ),
        ]
        self.subtask_configs["left"] = [
            SubTaskConfig(
                object_ref=self.held_asset_name,
                subtask_term_signal=None,
                action_noise=0.0,
                num_interpolation_steps=0,
            ),
        ]
        self.subtask_configs["body"] = [
            SubTaskConfig(
                object_ref=self.held_asset_name,
                subtask_term_signal="navigate_to_object",
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=0,
            ),
            SubTaskConfig(
                object_ref=self.held_asset_name,
                subtask_term_signal=None,
                action_noise=0.0,
                num_interpolation_steps=0,
            ),
        ]

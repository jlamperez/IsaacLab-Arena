# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Wireframe overlay of relation-solver / placement-validator geometry in Kit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.relations.collision_mode import CollisionMode, object_uses_mesh_collision
from isaaclab_arena.relations.placement_events import get_placement_pool
from isaaclab_arena.utils.isaac_sim_debug_draw import IsaacSimDebugDraw
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    import gymnasium as gym

    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer

IDENTITY_XYZW = (0.0, 0.0, 0.0, 1.0)

# One RGBA color per scene-entity kind.
KIND_COLORS: dict[str, tuple[float, float, float, float]] = {
    "background": (1.0, 0.55, 0.0, 1.0),  # orange
    "embodiment": (0.2, 0.6, 1.0, 1.0),  # blue
    "object": (0.2, 0.9, 0.3, 1.0),  # green
    "object_reference": (0.95, 0.35, 0.9, 1.0),  # magenta
    "other": (0.85, 0.85, 0.2, 1.0),  # yellow
}

MAX_MESH_EDGES = 8000


@dataclass(frozen=True)
class _OverlayEntry:
    """One entity scheduled for debug drawing."""

    name: str
    kind: str
    color: tuple[float, float, float, float]
    draw_aabb: bool
    draw_mesh: bool
    asset: CollisionObject


def classify_entity_kind(asset: CollisionObject) -> str:
    """Return the legend kind for a collision / placement asset."""
    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.relations.background_collision_object import FixedCollisionObject

    if isinstance(asset, (FixedCollisionObject, Background)):
        return "background"
    if isinstance(asset, EmbodimentBase):
        return "embodiment"
    if isinstance(asset, ObjectReference):
        return "object_reference"
    if isinstance(asset, Object):
        return "object"
    return "other"


def _quat_wxyz_to_xyzw(quat_wxyz) -> tuple[float, float, float, float]:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return (x, y, z, w)


def _live_pose_for_asset(env, asset: CollisionObject) -> Pose | None:
    """Return the world pose used to place solver geometry for ``asset``."""
    from isaaclab_arena.relations.background_collision_object import FixedCollisionObject
    from isaaclab_arena.relations.placement_asset import PlaceableAsset

    if isinstance(asset, FixedCollisionObject):
        # Mesh vertices are already baked in world coordinates.
        return Pose.identity()

    if not isinstance(asset, PlaceableAsset):
        return None

    scene = env.unwrapped.scene
    scene_key = asset.get_scene_key()
    try:
        scene_asset = scene[scene_key]
    except KeyError:
        scene_asset = None
    if scene_asset is not None and hasattr(scene_asset, "data") and hasattr(scene_asset.data, "root_pos_w"):
        pos = scene_asset.data.root_pos_w[0].detach().cpu().tolist()
        quat_wxyz = scene_asset.data.root_quat_w[0].detach().cpu().tolist()
        return Pose(position_xyz=tuple(float(v) for v in pos), rotation_xyzw=_quat_wxyz_to_xyzw(quat_wxyz))

    initial = asset.get_initial_pose()
    if isinstance(initial, Pose):
        return initial
    return Pose.identity()


class PlacementDebugOverlay:
    """Draw placement AABBs and MESH-mode collision meshes as Kit wireframe overlays."""

    def __init__(self, pool: PooledObjectPlacer, max_mesh_edges: int = MAX_MESH_EDGES):
        self._pool = pool
        self._max_mesh_edges = max_mesh_edges
        self._draw = IsaacSimDebugDraw()
        self._default_mode: CollisionMode = pool.default_collision_mode
        self._entries = self._build_entries(pool)
        self._legend_printed = False

    @classmethod
    def from_env(cls, env: gym.Env) -> PlacementDebugOverlay | None:
        """Build an overlay from the env placement pool, or ``None`` when absent."""
        pool = get_placement_pool(env)
        if pool is None:
            return None
        # Unwrap handle if present (branches that store PlacementPoolHandle).
        resolve = getattr(pool, "pool", None)
        if resolve is not None and not hasattr(pool, "objects"):
            pool = resolve
        if not hasattr(pool, "objects"):
            return None
        return cls(pool)

    def _build_entries(self, pool: PooledObjectPlacer) -> list[_OverlayEntry]:
        assets: list[CollisionObject] = list(pool.objects)
        assets.extend(pool.collision_objects)
        entries: list[_OverlayEntry] = []
        seen: set[int] = set()
        for asset in assets:
            asset_id = id(asset)
            if asset_id in seen:
                continue
            seen.add(asset_id)
            kind = classify_entity_kind(asset)
            uses_mesh = object_uses_mesh_collision(asset, self._default_mode)
            mesh = asset.get_collision_mesh() if uses_mesh else None
            entries.append(
                _OverlayEntry(
                    name=getattr(asset, "name", kind),
                    kind=kind,
                    color=KIND_COLORS.get(kind, KIND_COLORS["other"]),
                    draw_aabb=True,
                    draw_mesh=uses_mesh and mesh is not None,
                    asset=asset,
                )
            )
        return entries

    def print_legend(self) -> None:
        """Print color and geometry kind for each overlay entity once."""
        if self._legend_printed:
            return
        self._legend_printed = True
        print(
            f"[placement_debug] default collision_mode={self._default_mode.value}; "
            f"drawing {len(self._entries)} entit(y/ies)",
            flush=True,
        )
        for entry in self._entries:
            geoms = []
            if entry.draw_aabb:
                geoms.append("AABB")
            if entry.draw_mesh:
                geoms.append("mesh")
            geom_label = "+".join(geoms) if geoms else "none"
            print(
                f"[placement_debug] {entry.kind} '{entry.name}' {geom_label} color={entry.color}",
                flush=True,
            )

    def redraw(self, env: gym.Env) -> None:
        """Clear and redraw all scheduled overlays from live (or fixed) poses."""
        self._draw.clear()
        for entry in self._entries:
            pose = _live_pose_for_asset(env, entry.asset)
            if pose is None:
                continue
            if entry.draw_aabb:
                try:
                    bbox = entry.asset.get_bounding_box()
                except Exception as exc:  # noqa: BLE001 — debug overlay must not crash the runner
                    print(f"[placement_debug] skip AABB for '{entry.name}': {exc}", flush=True)
                else:
                    min_pt = tuple(float(v) for v in bbox.min_point[0].tolist())
                    max_pt = tuple(float(v) for v in bbox.max_point[0].tolist())
                    self._draw.draw_oriented_bbox(
                        min_pt,
                        max_pt,
                        pose.position_xyz,
                        pose.rotation_xyzw,
                        color=entry.color,
                        thickness=3.0,
                    )
            if entry.draw_mesh:
                mesh = entry.asset.get_collision_mesh()
                if mesh is None:
                    continue
                self._draw.draw_trimesh_wireframe(
                    mesh,
                    color=entry.color,
                    thickness=1.5,
                    max_edges=self._max_mesh_edges,
                    position_xyz=pose.position_xyz,
                    rotation_xyzw=pose.rotation_xyzw,
                )

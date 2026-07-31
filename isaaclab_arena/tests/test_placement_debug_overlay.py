# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for placement debug overlay geometry helpers (no Kit required)."""

from __future__ import annotations

import numpy as np

import pytest

from isaaclab_arena.utils.isaac_sim_debug_draw import (
    oriented_bbox_corners,
    oriented_bbox_edge_segments,
    rotate_points_xyzw,
    transform_trimesh_vertices,
    trimesh_edge_segments,
)
from isaaclab_arena.utils.placement_debug_overlay import KIND_COLORS, classify_entity_kind


def test_oriented_bbox_corners_identity_pose():
    corners = oriented_bbox_corners(
        min_point=(-1.0, -2.0, -3.0),
        max_point=(1.0, 2.0, 3.0),
        position_xyz=(0.0, 0.0, 0.0),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    assert corners.shape == (8, 3)
    assert np.allclose(corners.min(axis=0), [-1.0, -2.0, -3.0])
    assert np.allclose(corners.max(axis=0), [1.0, 2.0, 3.0])


def test_oriented_bbox_corners_translation():
    corners = oriented_bbox_corners(
        min_point=(0.0, 0.0, 0.0),
        max_point=(1.0, 1.0, 1.0),
        position_xyz=(10.0, 20.0, 30.0),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    assert np.allclose(corners.min(axis=0), [10.0, 20.0, 30.0])
    assert np.allclose(corners.max(axis=0), [11.0, 21.0, 31.0])


def test_oriented_bbox_edge_segments_count():
    starts, ends = oriented_bbox_edge_segments(
        min_point=(0.0, 0.0, 0.0),
        max_point=(1.0, 1.0, 1.0),
        position_xyz=(0.0, 0.0, 0.0),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    assert len(starts) == 12
    assert len(ends) == 12


def test_rotate_points_xyzw_180_about_z():
    # 180 deg about Z: (x,y,z) -> (-x,-y,z); quat xyzw = (0,0,1,0)
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
    out = rotate_points_xyzw(pts, (0.0, 0.0, 1.0, 0.0))
    assert np.allclose(out, [[-1.0, 0.0, 0.0], [0.0, -2.0, 0.0]], atol=1e-6)


def test_trimesh_edge_segments_unique_and_decimated():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    starts, ends = trimesh_edge_segments(vertices, faces, max_edges=100)
    assert len(starts) == len(ends)
    # Two triangles sharing an edge => 5 unique edges.
    assert len(starts) == 5

    starts_cap, ends_cap = trimesh_edge_segments(vertices, faces, max_edges=2)
    assert len(starts_cap) == 2
    assert len(ends_cap) == 2


def test_transform_trimesh_vertices_identity():
    verts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    out = transform_trimesh_vertices(verts, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    assert np.allclose(out, verts)


def test_kind_colors_cover_expected_kinds():
    for kind in ("background", "embodiment", "object", "object_reference", "other"):
        assert kind in KIND_COLORS
        assert len(KIND_COLORS[kind]) == 4


def test_classify_entity_kind_fixed_collision_object():
    import trimesh

    from isaaclab_arena.relations.background_collision_object import FixedCollisionObject

    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    fixed = FixedCollisionObject(mesh, name="fixed_collision_mesh")
    assert classify_entity_kind(fixed) == "background"


def test_solver_world_aabb_fixed_collision_object_is_world_baked():
    import trimesh

    from isaaclab_arena.relations.background_collision_object import FixedCollisionObject
    from isaaclab_arena.utils.placement_debug_overlay import solver_world_aabb
    from isaaclab_arena.utils.pose import Pose

    mesh = trimesh.creation.box(extents=(0.2, 0.4, 0.6))
    mesh.apply_translation([1.0, 2.0, 3.0])
    fixed = FixedCollisionObject(mesh, name="fixed")
    # Pose must be ignored: mesh / AABB are already in world frame.
    pose = Pose(position_xyz=(9.0, 9.0, 9.0), rotation_xyzw=(0.0, 0.0, 0.70710678, 0.70710678))
    world = solver_world_aabb(fixed, pose)
    expected = fixed.get_world_bounding_box()
    assert np.allclose(world.min_point.numpy(), expected.min_point.numpy())
    assert np.allclose(world.max_point.numpy(), expected.max_point.numpy())


def test_solver_world_aabb_live_placeable_matches_rotated_then_translated():
    """Non-anchor placeables: solver refits AABB under quat, then translates (not OBB draw)."""
    import math

    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.placement_debug_overlay import solver_world_aabb
    from isaaclab_arena.utils.pose import Pose

    class _FakePlaceable:
        name = "fake"

        def get_bounding_box(self):
            return AxisAlignedBoundingBox(min_point=(-2.0, -0.5, 0.0), max_point=(2.0, 0.5, 1.0))

        def get_world_bounding_box(self):
            raise AssertionError("live non-anchors must not use get_world_bounding_box in overlay")

        def get_relations(self):
            return []

    yaw = math.pi / 2
    # xyzw for +90 deg about Z
    quat = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    pose = Pose(position_xyz=(10.0, 20.0, 30.0), rotation_xyzw=quat)
    asset = _FakePlaceable()
    world = solver_world_aabb(asset, pose)
    expected = asset.get_bounding_box().rotated_by_quat(quat).translated(pose.position_xyz)
    assert np.allclose(world.min_point.numpy(), expected.min_point.numpy())
    assert np.allclose(world.max_point.numpy(), expected.max_point.numpy())
    # Must remain axis-aligned in world (solver style), not an OBB at the live pose.
    # After 90° Z, the long axis (was X) becomes Y: extents ~1 on X, ~4 on Y.
    size = (world.max_point - world.min_point)[0].tolist()
    assert size[0] == pytest.approx(1.0, abs=1e-5)
    assert size[1] == pytest.approx(4.0, abs=1e-5)

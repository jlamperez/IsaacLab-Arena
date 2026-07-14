# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


def test_discover_replicator_kitchen_usda_paths_finds_flat_usda_files():
    from isaaclab_arena.assets.background_library import _discover_replicator_kitchen_usda_paths

    paths = _discover_replicator_kitchen_usda_paths()
    assert paths["replicator_kitchen_l_shape"].name == "kitchen_l_shape.usda"
    assert all(path.parent.name == "Replicator" for path in paths.values())
    replicator_dir = paths["replicator_kitchen_l_shape"].parent
    assert len(paths) == len(list(replicator_dir.glob("*.usda")))


def test_replicator_kitchen_backgrounds_registered():
    from isaaclab_arena.assets.registries import AssetRegistry, ensure_assets_registered

    ensure_assets_registered()
    registry = AssetRegistry()
    assert registry.is_registered("replicator_kitchen_l_shape")
    cls = registry.get_asset_by_name("replicator_kitchen_l_shape")
    assert "replicator" in cls.tags
    assert cls.usd_path.endswith("kitchen_l_shape.usda")

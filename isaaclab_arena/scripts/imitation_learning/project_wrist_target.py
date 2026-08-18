# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Draw a GR00T WBC policy's predicted wrist target and current hand position on their own camera frame.

Built while chasing an intermittent ``g1_dex1_ikea_lightwheel`` rollout failure (2026-08-18): the
policy sometimes never finds the table leg and stalls near the right edge of the tabletop. Reading
``left_wrist_pose``'s raw (x, y, z) from the ``_DEBUG`` dumps in
``isaaclab_arena_gr00t/policy/gr00t_dex1_wbc_closedloop_policy.py`` doesn't say anything by eye --
three numbers in an unknown frame. This script instead draws where the model *thinks* it's
reaching, as a dot on the actual camera frame from that same chunk, so a failure is visible instead
of inferred.

All the frame math (camera pose reconstruction, world<->local composition, RDF conversion) happens
inside that policy's ``_DEBUG`` block, using IsaacLab's own tested utilities
(``combine_frame_transforms``/``subtract_frame_transforms``/
``convert_camera_frame_orientation_convention``) -- not reimplemented here. Two earlier hand-rolled
numpy versions of this script each visibly misplaced points (2026-08-18), which is what motivated
moving the geometry into the simulation, where the real library code and full torch/IsaacLab
context are available. This script only does the final, trivial step: a pinhole projection of
already-camera-RDF-frame (x=right, y=down, z=forward) coordinates.

Two things had to be worked out before the numbers in the dump could be trusted, both confirmed by
direct evidence rather than assumption:

- ``dataset_first_person_cam``'s own ``pos_w``/``quat_w_*`` are **stale** -- ``cam_pos_w`` was
  bit-identical across 4 consecutive chunks spanning real robot movement (``CameraCfg
  .update_latest_camera_pose`` defaults to ``False`` and isn't overridden for this camera). The
  policy reconstructs the camera pose fresh every chunk from ``torso_link``'s live pose instead.
- ``left_wrist_pose``/``right_wrist_pose`` (the policy's predicted target) are **pelvis-relative,
  not world frame** -- confirmed numerically: adding ``root_pos_w`` (pelvis) alone to the raw
  target landed within a few cm of the actual current hand position on a sanity-check chunk. The
  policy composes it through ``root_pos_w``/``root_quat_w`` before projecting.

Camera intrinsics (``FX``/``FY``) come from ``dataset_first_person_cam``'s ``PinholeCameraCfg`` in
``isaaclab_arena/embodiments/g1/g1.py`` (focal_length=19.3mm, horizontal/vertical aperture
48.53mm/35.37mm, 224x224) -- only valid for frames actually rendered from that camera.

The script has zero simulation dependency and only requires ``numpy``/``Pillow``.

Example
-------
.. code-block:: bash

    python isaaclab_arena/scripts/imitation_learning/project_wrist_target.py \\
        /tmp/gr00t_dex1_wbc_debug_host/chunk_030.npz \\
        /tmp/gr00t_dex1_wbc_debug_host/chunk_030_first_person.png \\
        -o /tmp/chunk_030_projected.png
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image, ImageDraw

# dataset_first_person_cam intrinsics (isaaclab_arena/embodiments/g1/g1.py:1050-1062):
# focal_length=19.3mm, horizontal_aperture=48.53mm, vertical_aperture=35.37mm, 224x224.
_FX = 19.3 * 224 / 48.53
_FY = 19.3 * 224 / 35.37


def project_rdf_to_pixel(point_rdf: np.ndarray, width: int, height: int) -> tuple[float, float] | None:
    """Pinhole-project an RDF-frame (x=right, y=down, z=forward) point to pixel coordinates.

    Returns ``None`` if the point is behind the camera (``z <= 0``).
    """
    x, y, z = point_rdf
    if z <= 0:
        return None
    return _FX * x / z + width / 2, _FY * y / z + height / 2


def wrist_target_pixels(d, width: int, height: int) -> dict[str, tuple[float, float] | None]:
    """Project one chunk dump's wrist/target RDF points to pixels: ``{side}_wrist_px``/``{side}_target_px``.

    ``d`` is a loaded ``chunk_NNN.npz`` (or any mapping with the same ``rdf_*``/``target_rdf_*``
    keys). wrist_yaw_link, not the finger midpoint: confirmed 2026-08-18 in
    ``g1_supplemental_info.py:264-266`` that this is the exact frame name the Pink IK task
    (``hand_frame_names``) targets, so it's the correct apples-to-apples comparison against the
    predicted target -- the finger midpoint used earlier added its own extra offset from that
    real target frame, which only muddied the comparison.
    """
    pixels: dict[str, tuple[float, float] | None] = {}
    for side in ("left", "right"):
        wrist_key = f"rdf_{side}_wrist_yaw_link"
        target_key = f"target_rdf_{side}"
        pixels[f"{side}_wrist_px"] = project_rdf_to_pixel(d[wrist_key], width, height) if wrist_key in d else None
        pixels[f"{side}_target_px"] = project_rdf_to_pixel(d[target_key], width, height) if target_key in d else None
    return pixels


def annotate_frame(image: Image.Image, pixels: dict[str, tuple[float, float] | None]) -> Image.Image:
    """Draw wrist (dot) / predicted-target (cross) markers from :func:`wrist_target_pixels`.

    Returns a new RGB image; ``image`` is not modified.
    """
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for side, color in [("left", "red"), ("right", "cyan")]:
        wrist_px = pixels.get(f"{side}_wrist_px")
        target_px = pixels.get(f"{side}_target_px")
        if wrist_px is not None:
            u, v = wrist_px
            draw.ellipse([u - 5, v - 5, u + 5, v + 5], outline=color, fill=color)
            draw.text((u + 7, v - 6), f"{side} wrist", fill=color)
        if target_px is not None:
            u, v = target_px
            draw.line([u - 6, v, u + 6, v], fill=color, width=2)
            draw.line([u, v - 6, u, v + 6], fill=color, width=2)
            draw.text((u + 7, v - 6), f"{side} wrist target", fill=color)
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("chunk_npz", help="Path to a chunk_NNN.npz debug dump.")
    parser.add_argument("frame_png", help="Path to that same chunk's chunk_NNN_first_person.png.")
    parser.add_argument("-o", "--output", required=True, help="Where to write the annotated image.")
    args = parser.parse_args()

    d = np.load(args.chunk_npz)
    image = Image.open(args.frame_png)
    pixels = wrist_target_pixels(d, *image.size)

    for side in ("left", "right"):
        wrist_key, target_key = f"rdf_{side}_wrist_yaw_link", f"target_rdf_{side}"
        if wrist_key in d:
            print(f"{side} current hand (wrist_yaw_link, the actual IK target frame): "
                  f"rdf={d[wrist_key]} pixel={pixels[f'{side}_wrist_px']}")
        if target_key in d:
            print(f"{side} predicted wrist target: rdf={d[target_key]} pixel={pixels[f'{side}_target_px']}")

    annotate_frame(image, pixels).save(args.output)
    print(f"Wrote annotated image to {args.output}")


if __name__ == "__main__":
    main()

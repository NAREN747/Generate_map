#!/usr/bin/env python3
"""
quality_metrics.py
-------------------
Objective, deterministic quality checks that complement the AI critic's
subjective scoring. These don't need an API call, a GPU, or a model's
opinion — they're measured directly from geometry and pixels, so they're
fast, free, reproducible, and catch a different class of problem than
"does this look good": specifically structural defects (UV distortion,
low texture resolution) that a vision model might not reliably flag but
that show up as visible artifacts in-engine.

Three metrics:
  - UV stretch: how distorted the box-projection UV mapping is per
    triangle. 1.0 = no distortion, higher = visible texture stretching.
  - Texel density: texture pixels per meter of real-world surface —
    too low and textures look blurry up close; too high wastes memory.
  - Texture sharpness: Laplacian-variance-style edge-response measure.
    Low values mean a blurry/flat texture; useful for catching a failed
    or degenerate AI generation pass before it ships.
"""

import numpy as np
from PIL import Image, ImageFilter


def triangle_uv_stretch(tri_3d, tri_uv):
    """
    Distortion for one triangle: compare the ratio of two edge lengths in
    3D space to the ratio of the same two edges in UV space. If the UV
    mapping is a uniform (non-distorting) scaling of the 3D surface, both
    ratios match and stretch == 1.0. The further from 1.0, the more the
    texture will visibly stretch/skew on that triangle.

    tri_3d: 3 x (x,y,z) world-space vertex positions
    tri_uv: 3 x (u,v) texture-space vertex positions
    """
    tri_3d = np.array(tri_3d, dtype=float)
    tri_uv = np.array(tri_uv, dtype=float)

    e3d_01 = np.linalg.norm(tri_3d[1] - tri_3d[0])
    e3d_02 = np.linalg.norm(tri_3d[2] - tri_3d[0])
    euv_01 = np.linalg.norm(tri_uv[1] - tri_uv[0])
    euv_02 = np.linalg.norm(tri_uv[2] - tri_uv[0])

    eps = 1e-9
    if euv_01 < eps or euv_02 < eps or e3d_01 < eps or e3d_02 < eps:
        return None  # degenerate triangle in one of the spaces, skip it

    ratio_01 = e3d_01 / euv_01
    ratio_02 = e3d_02 / euv_02

    hi, lo = max(ratio_01, ratio_02), min(ratio_01, ratio_02)
    return hi / lo if lo > eps else None


def mesh_uv_stretch_report(vertices, faces, uvs, stretch_threshold=2.0):
    """
    Run triangle_uv_stretch across every face of a mesh. Returns summary
    stats plus the fraction of faces exceeding stretch_threshold — that
    fraction is the actionable number: a handful of stretched triangles
    on a hidden face is fine, a third of the mesh distorted is not.
    """
    stretches = []
    for face in faces:
        tri_3d = [vertices[i] for i in face]
        tri_uv = [uvs[i] for i in face]
        s = triangle_uv_stretch(tri_3d, tri_uv)
        if s is not None:
            stretches.append(s)

    if not stretches:
        return {"mean_stretch": None, "max_stretch": None, "pct_over_threshold": None}

    stretches = np.array(stretches)
    return {
        "mean_stretch": float(stretches.mean()),
        "max_stretch": float(stretches.max()),
        "pct_over_threshold": float((stretches > stretch_threshold).mean() * 100),
    }


def texel_density(surface_area_m2, texture_resolution_px):
    """
    Texture pixels per square meter of real surface, then converted to a
    more intuitive px-per-linear-meter figure (sqrt of the areal density).
    Games generally target somewhere in the 128-1024 px/m range depending
    on how close the camera gets; well outside that range usually means
    either wasted texture memory (too high) or visible blur up close
    (too low).
    """
    if surface_area_m2 <= 0:
        return None
    px_total = texture_resolution_px[0] * texture_resolution_px[1]
    areal_density = px_total / surface_area_m2
    return float(np.sqrt(areal_density))  # px per linear meter, roughly


def texture_sharpness(image: Image.Image, border_crop=2):
    """
    Laplacian-variance-style sharpness measure: convolve with an edge
    kernel, take the variance of the response. Sharp/detailed textures
    have high-variance edge response; a blank, flat, or heavily blurred
    texture (e.g. a failed generation that came back nearly solid color)
    has low variance — this is what catches that failure mode
    automatically instead of a human eyeballing every texture.

    border_crop excludes the outer ring of pixels before measuring:
    PIL's edge filter always reads the image boundary as an edge
    regardless of actual content (it treats out-of-bounds as black
    rather than extending), which would otherwise put a constant floor
    under this metric and make it less discriminating.
    """
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(edges, dtype=float)

    if border_crop > 0 and arr.shape[0] > 2 * border_crop and arr.shape[1] > 2 * border_crop:
        arr = arr[border_crop:-border_crop, border_crop:-border_crop]

    return float(arr.var())


def evaluate_building(vertices, faces, uvs, texture_image, mesh_surface_area_m2):
    """Bundle all three objective metrics for one building into a single report."""
    uv_report = mesh_uv_stretch_report(vertices, faces, uvs)
    density = texel_density(mesh_surface_area_m2, texture_image.size)
    sharpness = texture_sharpness(texture_image)

    return {
        "uv_stretch": uv_report,
        "texel_density_px_per_m": density,
        "texture_sharpness": sharpness,
    }

#!/usr/bin/env python3
"""
osm_to_gltf.py
--------------
v0 pipeline: pull building footprints + roads from OpenStreetMap for a
bounding box, extrude buildings into simple 3D blockouts, lay down roads
as flat ribbons, and export the whole scene as a single .glb file that
can be dropped straight into Unity / Unreal / Godot.

This is intentionally the "procedural skeleton" stage of the pipeline we
discussed — no AI texture/detail pass yet. Get this working first, then
layer generative texturing on top of the exported meshes.

Install deps (in a venv on Arch — see notes below):
    pip install requests shapely trimesh pyproj numpy

Usage:
    python osm_to_gltf.py --bbox 12.9716,77.5946,12.9816,77.6046 --out city.glb
    python osm_to_gltf.py --bbox 12.9716,77.5946,12.9816,77.6046 --out city.glb --terrain

    bbox format: min_lat,min_lon,max_lat,max_lon
    (the example above is a ~1km box in Bengaluru)

    --terrain enables real elevation: samples a grid of heights from the
    free Open-Elevation API, displaces the ground mesh to match, and lifts
    each building/road to sit on the terrain at its location instead of
    floating on a flat plane. Adds extra API calls, so it's off by default.
"""

import argparse
import sys
import json
import time

import requests
import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, LineString
import trimesh

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
ELEVATION_BATCH_SIZE = 100   # Open-Elevation's public instance is happier with small batches
MAX_RETRIES = 3
RETRY_BACKOFF_S = 5   # multiplied by attempt number

# overpass-api.de (and public APIs generally) increasingly reject requests
# that "look programmatic" — missing/default User-Agent and Accept headers
# are a common trigger for 406 Not Acceptable, independent of your query
# being valid or your usage being reasonable. Sending real headers avoids
# that entirely; a custom User-Agent also identifies your tool politely,
# which OSM's usage guidelines ask for.
REQUEST_HEADERS = {
    "User-Agent": "citygen/1.0 (https://github.com/yourusername/citygen)",
    "Accept": "application/json",
}

DEFAULT_BUILDING_HEIGHT_M = 9.0   # ~3 storeys, used when OSM has no height tag
LEVEL_HEIGHT_M = 3.0              # assumed height per building:levels
ROAD_WIDTH_M = {
    "motorway": 12.0, "trunk": 10.0, "primary": 9.0, "secondary": 7.0,
    "tertiary": 6.0, "residential": 5.0, "service": 3.5, "footway": 1.5,
}
DEFAULT_ROAD_WIDTH_M = 5.0
ROAD_Z_OFFSET = 0.02  # lift roads slightly above ground plane to avoid z-fighting


def _request_with_retries(method, url, description, **kwargs):
    """
    Shared retry/backoff wrapper for the two public APIs this tool depends
    on (Overpass, Open-Elevation). Both are free community instances and
    do rate-limit or time out under load — this is what makes the tool
    usable for someone hitting it fresh rather than failing on the first
    hiccup.
    """
    kwargs.setdefault("headers", {})
    kwargs["headers"] = {**REQUEST_HEADERS, **kwargs["headers"]}

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = method(url, timeout=kwargs.pop("timeout", 90), **kwargs)
            if resp.status_code == 429:
                wait = RETRY_BACKOFF_S * attempt
                print(f"  {description}: rate limited (429), waiting {wait}s before retry "
                      f"{attempt}/{MAX_RETRIES}...")
                time.sleep(wait)
                continue
            if resp.status_code == 406:
                # Not a transient issue — retrying the same URL won't help.
                # Bail out immediately so fetch_osm_data can try the next mirror.
                last_error = "406 Not Acceptable (server is blocking this request/client)"
                break
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            last_error = "timed out"
        except requests.exceptions.ConnectionError:
            last_error = "connection failed"
        except requests.exceptions.HTTPError as e:
            last_error = str(e)

        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF_S * attempt
            print(f"  {description}: {last_error}, retrying in {wait}s "
                  f"({attempt}/{MAX_RETRIES})...")
            time.sleep(wait)

    return None  # caller decides whether to try another mirror or give up


def fetch_osm_data(min_lat, min_lon, max_lat, max_lon):
    """
    Query Overpass for buildings and roads (ways) inside the bbox. Tries
    each known mirror in turn — overpass-api.de in particular has become
    prone to blocking programmatic-looking requests (406 errors) under
    load, independent of query correctness, so a single point of failure
    here would make the tool unreliable for no fault of the user's.
    """
    query = f"""
    [out:json][timeout:60];
    (
      way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["highway"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out body;
    >;
    out skel qt;
    """

    for i, mirror_url in enumerate(OVERPASS_MIRRORS):
        label = f"Overpass API query ({mirror_url.split('/')[2]})"
        resp = _request_with_retries(requests.post, mirror_url, label,
                                      data={"data": query}, timeout=90)
        if resp is not None:
            return resp.json()
        if i < len(OVERPASS_MIRRORS) - 1:
            print(f"  Trying next mirror...")

    sys.exit(
        f"\nAll Overpass mirrors failed for this bbox.\n"
        f"Try:\n"
        f"  - a smaller bbox (a few city blocks instead of a full district)\n"
        f"  - waiting a few minutes and re-running — the public instances get "
        f"overloaded and rate-limit/block traffic in bursts\n"
        f"  - checking https://overpass-api.de/api/status for current server load\n"
    )


def parse_osm(osm_json):
    """Split raw Overpass JSON into node lookup, building ways, road ways."""
    nodes = {}
    buildings = []
    roads = []

    for el in osm_json["elements"]:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lat"], el["lon"])

    for el in osm_json["elements"]:
        if el["type"] != "way":
            continue
        tags = el.get("tags", {})
        node_ids = el.get("nodes", [])
        if "building" in tags:
            buildings.append({"nodes": node_ids, "tags": tags})
        elif "highway" in tags:
            roads.append({"nodes": node_ids, "tags": tags})

    return nodes, buildings, roads


def latlon_to_local_xy(lat, lon, transformer, origin_xy):
    """Project lat/lon to a local metric plane centered on the bbox."""
    x, y = transformer.transform(lon, lat)
    return x - origin_xy[0], y - origin_xy[1]


def building_height_m(tags):
    if "height" in tags:
        try:
            return float(str(tags["height"]).split()[0])
        except ValueError:
            pass
    if "building:levels" in tags:
        try:
            return float(tags["building:levels"]) * LEVEL_HEIGHT_M
        except ValueError:
            pass
    return DEFAULT_BUILDING_HEIGHT_M


def fetch_elevation_grid(min_lat, min_lon, max_lat, max_lon, resolution=20):
    """
    Sample a resolution x resolution grid of elevations across the bbox
    using the free Open-Elevation API. Returns:
        heights   -- (resolution, resolution) numpy array of meters
        grid_lats -- 1D array of the sample latitudes (south to north)
        grid_lons -- 1D array of the sample longitudes (west to east)
    """
    grid_lats = np.linspace(min_lat, max_lat, resolution)
    grid_lons = np.linspace(min_lon, max_lon, resolution)

    locations = [{"latitude": lat, "longitude": lon}
                 for lat in grid_lats for lon in grid_lons]

    heights_flat = []
    n_batches = (len(locations) + ELEVATION_BATCH_SIZE - 1) // ELEVATION_BATCH_SIZE
    for i in range(0, len(locations), ELEVATION_BATCH_SIZE):
        batch = locations[i:i + ELEVATION_BATCH_SIZE]
        batch_num = i // ELEVATION_BATCH_SIZE + 1
        resp = _request_with_retries(
            requests.post, ELEVATION_URL, f"Elevation batch {batch_num}/{n_batches}",
            json={"locations": batch}, timeout=60
        )
        if resp is None:
            sys.exit(
                f"\nElevation batch {batch_num}/{n_batches} failed after {MAX_RETRIES} attempts.\n"
                f"Try:\n"
                f"  - a smaller bbox or lower --terrain-resolution\n"
                f"  - waiting a few minutes and re-running (the free instance can be overloaded)\n"
                f"  - running without --terrain for now\n"
            )
        results = resp.json()["results"]
        heights_flat.extend(r["elevation"] for r in results)

    heights = np.array(heights_flat).reshape(resolution, resolution)
    return heights, grid_lats, grid_lons


def sample_height(heights, grid_lats, grid_lons, lat, lon):
    """Bilinear-interpolate terrain height at an arbitrary lat/lon."""
    lat = np.clip(lat, grid_lats[0], grid_lats[-1])
    lon = np.clip(lon, grid_lons[0], grid_lons[-1])

    i = np.searchsorted(grid_lats, lat) - 1
    j = np.searchsorted(grid_lons, lon) - 1
    i = np.clip(i, 0, len(grid_lats) - 2)
    j = np.clip(j, 0, len(grid_lons) - 2)

    lat0, lat1 = grid_lats[i], grid_lats[i + 1]
    lon0, lon1 = grid_lons[j], grid_lons[j + 1]
    tlat = (lat - lat0) / (lat1 - lat0) if lat1 != lat0 else 0.0
    tlon = (lon - lon0) / (lon1 - lon0) if lon1 != lon0 else 0.0

    h00, h01 = heights[i, j], heights[i, j + 1]
    h10, h11 = heights[i + 1, j], heights[i + 1, j + 1]
    h0 = h00 * (1 - tlon) + h01 * tlon
    h1 = h10 * (1 - tlon) + h11 * tlon
    return h0 * (1 - tlat) + h1 * tlat


def build_terrain_mesh(heights, grid_lats, grid_lons, transformer, origin_xy):
    """Build a displaced ground mesh from the elevation grid, in local meters."""
    res_lat, res_lon = heights.shape
    base_elevation = heights.min()

    verts = np.zeros((res_lat * res_lon, 3))
    idx = 0
    for i, lat in enumerate(grid_lats):
        for j, lon in enumerate(grid_lons):
            x, y = latlon_to_local_xy(lat, lon, transformer, origin_xy)
            z = heights[i, j] - base_elevation  # relative height, terrain min = 0
            verts[idx] = [x, y, z]
            idx += 1

    faces = []
    for i in range(res_lat - 1):
        for j in range(res_lon - 1):
            a = i * res_lon + j
            b = i * res_lon + (j + 1)
            c = (i + 1) * res_lon + j
            d = (i + 1) * res_lon + (j + 1)
            faces.append([a, b, d])
            faces.append([a, d, c])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
    return mesh, base_elevation


def build_scene(nodes, buildings, roads, min_lat, min_lon, max_lat, max_lon,
                 use_terrain=False, terrain_resolution=20):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    origin_xy = transformer.transform(center_lon, center_lat)

    scene = trimesh.Scene()

    # --- Terrain: fetch + build displaced ground mesh, or fall back to flat ---
    heights = grid_lats = grid_lons = None
    base_elevation = 0.0
    if use_terrain:
        print(f"Sampling {terrain_resolution}x{terrain_resolution} elevation grid...")
        heights, grid_lats, grid_lons = fetch_elevation_grid(
            min_lat, min_lon, max_lat, max_lon, resolution=terrain_resolution)
        terrain_mesh, base_elevation = build_terrain_mesh(
            heights, grid_lats, grid_lons, transformer, origin_xy)
        scene.add_geometry(terrain_mesh, node_name="terrain")
    else:
        span = 1200  # meters, generous flat pad under the bbox content
        ground = trimesh.creation.box(extents=[span, span, 0.1])
        ground.apply_translation([0, 0, -0.05])
        scene.add_geometry(ground, node_name="ground_plane")

    def terrain_z_at(lat, lon):
        if not use_terrain:
            return 0.0
        return sample_height(heights, grid_lats, grid_lons, lat, lon) - base_elevation

    # --- Buildings: extrude each footprint polygon up to its height ---
    n_ok, n_skipped = 0, 0
    for b in buildings:
        latlons, coords = [], []
        for nid in b["nodes"]:
            if nid not in nodes:
                continue
            lat, lon = nodes[nid]
            latlons.append((lat, lon))
            coords.append(latlon_to_local_xy(lat, lon, transformer, origin_xy))

        if len(coords) < 3:
            n_skipped += 1
            continue

        try:
            poly = Polygon(coords)
            if not poly.is_valid or poly.area < 1.0:
                n_skipped += 1
                continue
        except Exception:
            n_skipped += 1
            continue

        height = building_height_m(b["tags"])
        mesh = trimesh.creation.extrude_polygon(poly, height=height)

        centroid_lat = sum(l for l, _ in latlons) / len(latlons)
        centroid_lon = sum(o for _, o in latlons) / len(latlons)
        mesh.apply_translation([0, 0, terrain_z_at(centroid_lat, centroid_lon)])

        scene.add_geometry(mesh, node_name=f"building_{n_ok}")
        n_ok += 1

    # --- Roads: turn each way into a flat ribbon mesh of the right width ---
    r_ok = 0
    for r in roads:
        latlons, coords = [], []
        for nid in r["nodes"]:
            if nid not in nodes:
                continue
            lat, lon = nodes[nid]
            latlons.append((lat, lon))
            coords.append(latlon_to_local_xy(lat, lon, transformer, origin_xy))

        if len(coords) < 2:
            continue

        width = ROAD_WIDTH_M.get(r["tags"].get("highway"), DEFAULT_ROAD_WIDTH_M)
        line = LineString(coords)
        ribbon = line.buffer(width / 2.0, cap_style=2)
        if ribbon.is_empty or ribbon.area < 0.5:
            continue

        try:
            mesh = trimesh.creation.extrude_polygon(ribbon, height=0.15)
            centroid_lat = sum(l for l, _ in latlons) / len(latlons)
            centroid_lon = sum(o for _, o in latlons) / len(latlons)
            mesh.apply_translation([0, 0, terrain_z_at(centroid_lat, centroid_lon) + ROAD_Z_OFFSET])
            scene.add_geometry(mesh, node_name=f"road_{r_ok}")
            r_ok += 1
        except Exception:
            continue

    print(f"Buildings extruded: {n_ok} (skipped: {n_skipped})")
    print(f"Roads extruded: {r_ok}")
    return scene


def main():
    parser = argparse.ArgumentParser(description="OSM bbox -> extruded city -> glTF")
    parser.add_argument("--bbox", required=True,
                         help="min_lat,min_lon,max_lat,max_lon")
    parser.add_argument("--out", default="city.glb", help="output .glb path")
    parser.add_argument("--terrain", action="store_true",
                         help="fetch real elevation and displace the ground mesh")
    parser.add_argument("--terrain-resolution", type=int, default=20,
                         help="elevation grid size (NxN samples), default 20")
    args = parser.parse_args()

    try:
        min_lat, min_lon, max_lat, max_lon = map(float, args.bbox.split(","))
    except ValueError:
        sys.exit("bbox must be 4 comma-separated numbers: min_lat,min_lon,max_lat,max_lon")

    print("Querying Overpass API...")
    osm_json = fetch_osm_data(min_lat, min_lon, max_lat, max_lon)

    print("Parsing OSM elements...")
    nodes, buildings, roads = parse_osm(osm_json)
    print(f"Found {len(buildings)} buildings, {len(roads)} road segments, {len(nodes)} nodes")

    print("Building 3D scene...")
    scene = build_scene(nodes, buildings, roads, min_lat, min_lon, max_lat, max_lon,
                         use_terrain=args.terrain,
                         terrain_resolution=args.terrain_resolution)

    print(f"Exporting to {args.out} ...")
    scene.export(args.out)
    print("Done.")


if __name__ == "__main__":
    main()

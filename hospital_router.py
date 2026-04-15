"""Core routing and reporting functions for the hospital routing project."""

import os
import sys
import time
import argparse
import math
import heapq
import json
import re
import warnings
from pathlib import Path
from typing import Optional
from urllib import error as urlerror
from urllib import request as urlrequest

import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import HeatMap
from shapely.geometry import Point, MultiPolygon

warnings.filterwarnings("ignore")
ox.settings.use_cache = True

# ──────────────────────────────────────────────
# CONFIGURATION  (edit here — no hardcoded paths)
# ──────────────────────────────────────────────
DEFAULT_CONFIG = {
    "road_geojson":     r"ROAD 1\ROAD.geojson",  # path to your road GeoJSON
    "hospital_geojson": "hospital_directory.csv", # path to your hospital data file
    "output_map":       "hospital_route.html",
    "output_report":    "routing_report.txt",
    "routing_provider": "mappls",            # "mappls" (primary) or "local"
    "mappls_api_key":   "oqsngxuojndarlqtunbkiinxyjhbstcklbwj",
    "mappls_profile":   "driving",
    "mappls_route_resource": "route_eta",
    "mappls_timeout_s": 20,
    "mappls_fallback_to_local": True,
    "density_col":      "density",            # column name in road GeoJSON
    "search_radius_m":  5000,                 # OSM graph radius in metres
    "top_k_hospitals":  3,                    # how many hospitals to rank
    "isochrone_minutes": [5, 10, 15],         # reachability rings
    "blocked_road_ids":  [],                  # list of (u,v) edges to simulate block
}

# Road type → speed limit (km/h)  — used for REAL travel time
ROAD_SPEEDS = {
    "motorway":    110,
    "trunk":        90,
    "primary":      60,
    "secondary":    50,
    "tertiary":     40,
    "residential":  30,
    "unclassified": 25,
}

# Road type → base cost factor (lower = preferred)
ROAD_BASE_FACTOR = {
    "motorway":    1.0,
    "trunk":       1.0,
    "primary":     1.2,
    "secondary":   1.4,
    "tertiary":    1.6,
    "residential": 3.5,
}

# Time-of-day peak multipliers  { hour_range: multiplier }
PEAK_HOURS = {
    (7, 10):  1.8,   # morning rush
    (17, 20): 1.6,   # evening rush
}

# Hospital specialty scoring bonus (lower = better)
SPECIALTY_BONUS = {
    "trauma":    0.85,
    "general":   1.00,
    "children":  1.10,
    "eye":       1.20,
}


# ──────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────

def get_peak_multiplier(hour: int) -> float:
    """Return traffic peak multiplier for given hour (0-23)."""
    for (start, end), mult in PEAK_HOURS.items():
        if start <= hour < end:
            return mult
    return 1.0


def haversine_heuristic(G: nx.MultiDiGraph, target_node: int):
    """
    Returns an A* heuristic function.
    Uses haversine distance (great-circle) as lower-bound estimate.
    Time complexity: O(1) per call.
    """
    tx = G.nodes[target_node]["x"]
    ty = G.nodes[target_node]["y"]

    def heuristic(u, v):
        ux = G.nodes[u]["x"]
        uy = G.nodes[u]["y"]
        # Haversine formula
        R = 6371000  # Earth radius in metres
        phi1, phi2 = math.radians(uy), math.radians(ty)
        dphi = math.radians(ty - uy)
        dlam = math.radians(tx - ux)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
        dist_m = 2 * R * math.asin(math.sqrt(a))
        # Convert to travel-time minutes assuming max road speed (optimistic).
        # Edge weights are stored in minutes, so the heuristic must match.
        return (dist_m / (110 * 1000 / 3600)) / 60.0
    return heuristic


def compute_travel_time_minutes(length_m: float, highway: str,
                                 density_norm: float, hour: int = 8) -> float:
    """
    REAL travel time in minutes.
    Formula: time = distance / speed, penalised by density and peak hour.

    Args:
        length_m:     edge length in metres
        highway:      OSM road type string
        density_norm: normalised traffic density [0.0 – 1.0]
        hour:         hour of day (0-23) for peak multiplier

    Returns:
        travel time in minutes (float)
    """
    speed_kmh = ROAD_SPEEDS.get(highway, 25)
    speed_ms  = speed_kmh * 1000 / 3600          # m/s

    # Density slows speed: at density=1.0 speed drops to 30% of free-flow
    density_slowdown = 1.0 + 2.0 * density_norm  # range [1.0, 3.0]
    effective_speed  = speed_ms / density_slowdown

    peak_mult = get_peak_multiplier(hour)
    effective_speed = effective_speed / peak_mult  # rush hour slows further

    travel_sec = length_m / max(effective_speed, 0.5)   # floor: never divide by 0
    return travel_sec / 60.0


def normalise_density(roads_gdf: gpd.GeoDataFrame, col: str) -> gpd.GeoDataFrame:
    """
    Normalise density values to [0, 1] using 95th percentile cap.
    This prevents outlier roads from dominating the cost function.
    """
    valid = roads_gdf[col].dropna()
    p95   = valid.quantile(0.95) if len(valid) > 0 else 1.0
    if p95 == 0:
        p95 = 1.0
    roads_gdf = roads_gdf.copy()
    roads_gdf[col + "_norm"] = (roads_gdf[col].fillna(0) / p95).clip(0, 1)
    return roads_gdf


# ──────────────────────────────────────────────
# GRAPH LOADING
# ──────────────────────────────────────────────

def load_road_graph(lat: float, lon: float, radius_m: int) -> nx.MultiDiGraph:
    """
    Download OSM road network as a weighted directed multigraph.
    Filters to main road types only.
    """
    cache_dir = Path("cache") / "graphs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"lat_{lat:.5f}_lon_{lon:.5f}_r_{radius_m}.graphml".replace("-", "m")
    cache_path = cache_dir / cache_key

    if cache_path.exists():
        print("Loading road network from local cache...")
        G = ox.load_graphml(cache_path)
        print(f"   Cached graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    cf = '["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]'
    print("Downloading road network from OpenStreetMap...")
    G = ox.graph_from_point((lat, lon), dist=radius_m, custom_filter=cf)
    ox.save_graphml(G, cache_path)
    print(f"   Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def load_traffic_data(road_geojson: str, lat: float, lon: float,
                      buffer: float = 0.05) -> gpd.GeoDataFrame:
    """
    Load road GeoJSON, spatially filtered to bounding box.
    Returns empty GDF with fallback column if file not found.
    """
    path = Path(road_geojson)
    if not path.exists():
        print(f"   Road GeoJSON not found at '{road_geojson}'. Using neutral traffic weights.")
        return gpd.GeoDataFrame(columns=["geometry", "density", "density_norm"],
                                 geometry="geometry", crs="EPSG:4326")
    roads = gpd.read_file(
        road_geojson,
        bbox=(lon - buffer, lat - buffer, lon + buffer, lat + buffer)
    ).to_crs(epsg=4326)
    print(f"   Loaded {len(roads)} road segments from traffic GeoJSON")
    return roads


def load_hospital_points(hospital_geojson: str) -> gpd.GeoDataFrame:
    """
    Load hospital points without graph snapping.
    Supports GeoJSON and CSV sources.
    """
    path = Path(hospital_geojson)
    if not path.exists():
        raise FileNotFoundError(
            f"Hospital GeoJSON not found: '{hospital_geojson}'\n"
            "Please set the correct path in config['hospital_geojson']."
        )

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
        if "Location_Coordinates" not in df.columns or "Hospital_Name" not in df.columns:
            raise ValueError(
                "CSV hospital file must contain 'Hospital_Name' and 'Location_Coordinates' columns."
            )

        coords = df["Location_Coordinates"].astype(str).str.split(",", n=1, expand=True)
        df = df.assign(
            latitude=pd.to_numeric(coords[0], errors="coerce"),
            longitude=pd.to_numeric(coords[1], errors="coerce") if coords.shape[1] > 1 else pd.NA,
        )
        df = df.dropna(subset=["latitude", "longitude"]).copy()
        df = df[
            df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)
        ].copy()

        hospitals = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326",
        )
        hospitals["name"] = hospitals["Hospital_Name"].astype(str)
        hospitals["amenity"] = hospitals["Hospital_Category"].astype(str).replace({"nan": "general"})
    else:
        hospitals = gpd.read_file(hospital_geojson).to_crs(epsg=4326)
        if "name" not in hospitals.columns and "Hospital_Name" in hospitals.columns:
            hospitals["name"] = hospitals["Hospital_Name"].astype(str)
        if "amenity" not in hospitals.columns:
            hospitals["amenity"] = "general"

    print(f"   Loaded {len(hospitals)} hospitals")
    return hospitals


def _normalise_hospital_name(name: str) -> str:
    """Create a stable comparable key for hospital names."""
    text = re.sub(r"[^a-z0-9\s]", " ", str(name).lower())
    return re.sub(r"\s+", " ", text).strip()


def deduplicate_hospitals(hospitals: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Collapse spelling variants and exact coordinate duplicates from directory-style datasets.
    """
    if hospitals.empty:
        return hospitals.copy()

    deduped = hospitals.copy()
    deduped["_name_key"] = deduped["name"].map(_normalise_hospital_name)
    deduped["_lat_round"] = deduped.geometry.y.round(4)
    deduped["_lon_round"] = deduped.geometry.x.round(4)
    deduped["_dup_key"] = (
        deduped["_name_key"] + "|"
        + deduped["_lat_round"].astype(str) + "|"
        + deduped["_lon_round"].astype(str)
    )
    deduped = deduped.drop_duplicates(subset="_dup_key", keep="first").copy()
    return deduped.drop(
        columns=["_name_key", "_lat_round", "_lon_round", "_dup_key"],
        errors="ignore",
    )


def load_hospitals(hospital_geojson: str, G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """
    Load hospital GeoJSON and snap each hospital to nearest graph node.
    """
    hospitals = load_hospital_points(hospital_geojson)
    hospitals["node"] = hospitals.apply(
        lambda r: ox.nearest_nodes(G, r.geometry.x, r.geometry.y), axis=1
    )
    return hospitals


def filter_nearby_hospitals(hospitals: gpd.GeoDataFrame, lat: float, lon: float,
                            radius_m: int, max_candidates: int = 75) -> gpd.GeoDataFrame:
    """
    Reduce the hospital search space before graph snapping/routing.
    This is critical for large CSV datasets.
    """
    if hospitals.empty:
        return hospitals

    buffer_deg = max(radius_m / 111320.0, 0.01)
    lon_scale = max(math.cos(math.radians(lat)), 0.2)
    lon_buffer = buffer_deg / lon_scale

    nearby = hospitals[
        hospitals.geometry.y.between(lat - buffer_deg, lat + buffer_deg) &
        hospitals.geometry.x.between(lon - lon_buffer, lon + lon_buffer)
    ].copy()
    if nearby.empty:
        nearby = hospitals.copy()

    nearby["_crow_km"] = nearby.geometry.apply(
        lambda geom: ox.distance.great_circle(lat, lon, geom.y, geom.x) / 1000.0
    )
    nearby = nearby.sort_values("_crow_km").copy()
    nearby = deduplicate_hospitals(nearby)
    nearby = nearby.head(max_candidates).copy()
    return nearby.drop(columns=["_crow_km"], errors="ignore")


def snap_hospitals_to_graph(hospitals: gpd.GeoDataFrame, G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """
    Snap an already-filtered hospital set to the nearest graph node.
    """
    if hospitals.empty:
        return hospitals.copy()
    hospitals = hospitals.copy()
    hospitals["node"] = hospitals.apply(
        lambda r: ox.nearest_nodes(G, r.geometry.x, r.geometry.y), axis=1
    )
    hospitals = hospitals.drop_duplicates(subset=["node"], keep="first").copy()
    return hospitals


# ──────────────────────────────────────────────
# MAPPLS ROUTING
# ──────────────────────────────────────────────

def mappls_directions_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    api_key: str,
    profile: str = "driving",
    route_resource: str = "route_eta",
    timeout_s: int = 20,
) -> dict:
    """
    Mappls-based routing:
      - real route geometry
      - real route duration
      - real route distance
    """
    if not api_key.strip():
        raise ValueError("Mappls API key is missing.")

    url = (
        f"https://route.mappls.com/route/direction/{route_resource}/{profile}/"
        f"{start_lon},{start_lat};{end_lon},{end_lat}"
        f"?steps=false&geometries=geojson&rtype=0&region=ind&access_token={api_key}"
    )

    try:
        with urlrequest.urlopen(url, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Mappls directions HTTP {exc.code}: {detail}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"Mappls directions request failed: {exc.reason}") from exc

    routes = data.get("routes") or []
    if not routes:
        raise RuntimeError("Mappls directions response did not include any routes.")

    route = routes[0]
    geometry = (route.get("geometry") or {}).get("coordinates") or []
    if not geometry:
        raise RuntimeError("Mappls directions response did not include route geometry.")

    route_coords = [(coord[1], coord[0]) for coord in geometry]
    distance_km = round(float(route.get("distance", 0.0)) / 1000.0, 3)
    travel_min = round(float(route.get("duration", 0.0)) / 60.0, 2)

    return {
        "travel_min": travel_min,
        "dist_km": distance_km,
        "route_geometry": route_coords,
        "mappls_raw": data,
    }


# ──────────────────────────────────────────────
# EDGE WEIGHT ASSIGNMENT
# ──────────────────────────────────────────────

def assign_edge_weights(G: nx.MultiDiGraph, roads: gpd.GeoDataFrame,
                         density_col: str, hour: int = 8) -> nx.MultiDiGraph:
    """
    Assign travel_time (minutes) to every edge using:
      - Road length
      - Road type speed limit
      - Normalised traffic density
      - Peak-hour multiplier

    Also stores raw_cost for Dijkstra vs A* comparison.
    """
    print("Building traffic-aware edge weights...")

    # Spatial join: match each OSM edge to nearest road in GeoJSON
    density_lookup = {}
    if len(roads) > 0 and density_col + "_norm" in roads.columns:
        edges_gdf = ox.graph_to_gdfs(G, nodes=False).reset_index()
        edges_gdf = edges_gdf.to_crs(epsg=4326)
        joined = gpd.sjoin_nearest(
            edges_gdf[["u", "v", "key", "geometry"]],
            roads[["geometry", density_col + "_norm"]],
            how="left",
            max_distance=0.001
        )
        density_lookup = {
            (row["u"], row["v"], row["key"]): row[density_col + "_norm"]
            for _, row in joined.iterrows()
        }

    for u, v, k, data in G.edges(keys=True, data=True):
        length  = data.get("length", 10)
        highway = data.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0]

        density_norm = density_lookup.get((u, v, k), 0.0)
        if density_norm != density_norm:  # NaN check
            density_norm = 0.0

        tt = compute_travel_time_minutes(length, highway, density_norm, hour)
        data["travel_time"]  = tt          # minutes — real value
        data["raw_cost"]     = tt          # alias for comparison

    print("   Edge weights assigned (travel_time in minutes)")
    return G


def simulate_road_block(G: nx.MultiDiGraph,
                         blocked: list[tuple]) -> nx.MultiDiGraph:
    """
    Simulate blocked roads by setting their travel_time to infinity.
    This forces Dijkstra to route around them.
    """
    G2 = G.copy()
    for u, v in blocked:
        if G2.has_edge(u, v):
            for k in G2[u][v]:
                G2[u][v][k]["travel_time"] = float("inf")
            print(f"   Road blocked: {u} -> {v}")
    return G2


# ──────────────────────────────────────────────
# ROUTING ALGORITHMS
# ──────────────────────────────────────────────

def dijkstra_route(G: nx.MultiDiGraph, source: int, target: int,
                   weight: str = "travel_time") -> tuple[float, list, float]:
    """
    Dijkstra's shortest path.
    Time complexity: O((V + E) log V) with binary heap.

    Returns:
        (cost, path_node_list, runtime_seconds)
    """
    t0 = time.perf_counter()
    try:
        cost  = nx.shortest_path_length(G, source, target, weight=weight)
        path  = nx.shortest_path(G, source, target, weight=weight)
    except nx.NetworkXNoPath:
        return float("inf"), [], time.perf_counter() - t0
    return cost, path, time.perf_counter() - t0


def astar_route(G: nx.MultiDiGraph, source: int, target: int,
                weight: str = "travel_time") -> tuple[float, list, float]:
    """
    A* shortest path using haversine heuristic.
    Time complexity: O((V + E) log V) — faster than Dijkstra in practice
    because the heuristic prunes the search space.

    Returns:
        (cost, path_node_list, runtime_seconds)
    """
    t0 = time.perf_counter()
    heuristic = haversine_heuristic(G, target)
    try:
        path = nx.astar_path(G, source, target, heuristic=heuristic, weight=weight)
        cost = sum(
            min(G[u][v][k].get(weight, 0) for k in G[u][v])
            for u, v in zip(path[:-1], path[1:])
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf"), [], time.perf_counter() - t0
    return cost, path, time.perf_counter() - t0


# ──────────────────────────────────────────────
# HOSPITAL SCORING & RANKING
# ──────────────────────────────────────────────

def score_hospital(travel_min: float, dist_km: float,
                    row: gpd.GeoSeries) -> float:
    """
    Multi-criteria score (lower = better).

    Criteria:
      - 60% weight on travel time
      - 30% weight on physical distance
      - 10% weight on specialty bonus

    This models real-world decisions better than pure distance.
    """
    specialty = str(row.get("amenity", "general")).lower()
    bonus = SPECIALTY_BONUS.get(specialty, 1.0)

    score = (0.60 * travel_min +
             0.30 * dist_km * 10 +   # normalise km to minutes scale
             0.10 * bonus * travel_min)
    return round(score, 4)


def find_top_k_hospitals(G: nx.MultiDiGraph, orig_node: int,
                          hospitals: gpd.GeoDataFrame, k: int = 3,
                          algorithm: str = "dijkstra") -> list[dict]:
    """
    Find top-K hospitals ranked by multi-criteria score.
    Returns a list of result dicts sorted best-first.

    Supports algorithm = 'dijkstra' | 'astar' | 'both'
    """
    results = []

    for idx, row in hospitals.iterrows():
        target = row["node"]
        chosen_algo = algorithm
        astar_cost = astar_path = astar_rt = None

        if algorithm == "both":
            d_cost, d_path, d_rt = dijkstra_route(G, orig_node, target)
            a_cost, a_path, a_rt = astar_route(G, orig_node, target)
            if (d_cost == float("inf") or not d_path) and (a_cost == float("inf") or not a_path):
                continue

            pick_astar = False
            if d_cost == float("inf") or not d_path:
                pick_astar = True
            elif a_cost != float("inf") and a_path:
                if a_cost < d_cost:
                    pick_astar = True
                elif math.isclose(a_cost, d_cost, rel_tol=0.0, abs_tol=1e-9) and a_rt < d_rt:
                    pick_astar = True

            if pick_astar:
                cost, path, rt = a_cost, a_path, a_rt
                chosen_algo = "astar"
            else:
                cost, path, rt = d_cost, d_path, d_rt
                chosen_algo = "dijkstra"

            astar_cost, astar_path, astar_rt = a_cost, a_path, a_rt
            dijkstra_cost, dijkstra_rt = d_cost, d_rt
        else:
            route_fn = dijkstra_route if algorithm == "dijkstra" else astar_route
            cost, path, rt = route_fn(G, orig_node, target)
            if cost == float("inf") or not path:
                continue

        dist_km = nx.shortest_path_length(
            G, orig_node, target, weight="length"
        ) / 1000

        score = score_hospital(cost, dist_km, row)

        result = {
            "name":        row.get("name", f"Hospital_{idx}"),
            "travel_min":  round(cost, 2),
            "dist_km":     round(dist_km, 3),
            "score":       score,
            "path":        path,
            "runtime_s":   round(rt, 5),
            "algorithm":   chosen_algo,
            "route_source": "local",
            "specialty":   row.get("amenity", "general"),
            "lat":         row.geometry.y,
            "lon":         row.geometry.x,
        }

        # If 'both' algorithms requested, also run A* for comparison
        if algorithm == "both":
            result["selected_algorithm"] = "both"
            result["dijkstra_min"] = round(dijkstra_cost, 2)
            result["dijkstra_runtime_s"] = round(dijkstra_rt, 5)
            result["astar_min"] = round(astar_cost, 2)
            result["astar_runtime_s"] = round(astar_rt, 5)
            result["astar_path"] = astar_path

        results.append(result)

    results.sort(key=lambda x: x["score"])
    return results[:k]


def mappls_distance_matrix(
    lat: float,
    lon: float,
    hospitals: gpd.GeoDataFrame,
    api_key: str,
    profile: str = "driving",
    timeout_s: int = 20,
) -> dict:
    """
    Mappls matrix API:
      - one request for many hospital distances/durations
      - durations are returned in seconds
      - distances are returned in meters
    """
    if not api_key.strip():
        raise ValueError("Mappls API key is missing.")
    if hospitals.empty:
        return {}

    coord_parts = [f"{lon},{lat}"] + [
        f"{row.geometry.x},{row.geometry.y}" for _, row in hospitals.iterrows()
    ]
    url = (
        f"https://route.mappls.com/route/dm/distance_matrix/{profile}/"
        f"{';'.join(coord_parts)}?rtype=0&region=ind&access_token={api_key}"
    )
    try:
        with urlrequest.urlopen(url, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Mappls matrix HTTP {exc.code}: {detail}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"Mappls matrix request failed: {exc.reason}") from exc

    results = data.get("results") or {}
    distances = results.get("distances") or []
    durations = results.get("durations") or []
    if not distances or not durations:
        raise RuntimeError("Mappls matrix response did not include distances/durations.")
    return {
        "distances": distances[0],
        "durations": durations[0],
        "raw": data,
    }


def find_top_k_hospitals_mappls(
    lat: float,
    lon: float,
    hospitals: gpd.GeoDataFrame,
    api_key: str,
    k: int = 3,
    profile: str = "driving",
    route_resource: str = "route_eta",
    timeout_s: int = 20,
    selected_algorithm: str = "dijkstra",
) -> list[dict]:
    """
    Mappls-based routing:
      - ranks hospitals by Mappls matrix travel time
      - fetches real route geometry only for the top-ranked hospitals

    Local NetworkX/OSMnx routing remains available elsewhere.
    """
    if hospitals.empty:
        return []

    results = []
    hospitals = hospitals.copy()
    # Fast ranking path: shortlist by crow-flight distance first, then use one matrix call.
    hospitals["_crow_km"] = hospitals.geometry.apply(
        lambda geom: ox.distance.great_circle(lat, lon, geom.y, geom.x) / 1000.0
    )
    candidate_limit = min(len(hospitals), max(k * 4, 10))
    candidate_hospitals = hospitals.nsmallest(candidate_limit, "_crow_km")
    matrix = mappls_distance_matrix(
        lat, lon, candidate_hospitals, api_key=api_key, profile=profile, timeout_s=timeout_s
    )
    durations = matrix["durations"]
    distances = matrix["distances"]

    ranked_candidates = []
    for offset, (idx, row) in enumerate(candidate_hospitals.iterrows(), start=1):
        duration_s = durations[offset] if offset < len(durations) else None
        distance_m = distances[offset] if offset < len(distances) else None
        if duration_s is None or distance_m is None:
            continue
        if duration_s == 0 and distance_m == 0:
            continue
        travel_min = round(float(duration_s) / 60.0, 2)
        dist_km = round(float(distance_m) / 1000.0, 3)
        ranked_candidates.append((idx, row, travel_min, dist_km))

    ranked_candidates.sort(key=lambda item: item[2])
    top_candidates = ranked_candidates[:k]

    for idx, row, travel_min, dist_km in top_candidates:
        try:
            route_result = mappls_directions_route(
                start_lat=lat,
                start_lon=lon,
                end_lat=row.geometry.y,
                end_lon=row.geometry.x,
                api_key=api_key,
                profile=profile,
                route_resource=route_resource,
                timeout_s=timeout_s,
            )
            route_geometry = route_result["route_geometry"]
        except Exception as exc:
            print(f"   Mappls geometry failed for hospital '{row.get('name', idx)}': {exc}")
            route_geometry = []

        score = score_hospital(travel_min, dist_km, row)
        results.append({
            "name":        row.get("name", f"Hospital_{idx}"),
            "travel_min":  travel_min,
            "dist_km":     dist_km,
            "score":       score,
            "path":        [],
            "route_geometry": route_geometry,
            "runtime_s":   None,
            "algorithm":   selected_algorithm,
            "route_source": "mappls",
            "specialty":   row.get("amenity", "general"),
            "lat":         row.geometry.y,
            "lon":         row.geometry.x,
        })

    return results


# ──────────────────────────────────────────────
# ISOCHRONE (REACHABILITY MAP)
# ──────────────────────────────────────────────

def compute_isochrone(G: nx.MultiDiGraph, orig_node: int,
                       max_minutes: float) -> list[tuple[float, float]]:
    """
    Return list of (lat, lon) for all nodes reachable within max_minutes.
    Uses Dijkstra's ego_subgraph from the origin node.

    Time complexity: O((V + E) log V) — same as Dijkstra.
    """
    lengths = nx.single_source_dijkstra_path_length(
        G, orig_node, cutoff=max_minutes, weight="travel_time"
    )
    coords = []
    for node_id in lengths:
        nd = G.nodes[node_id]
        coords.append((nd["y"], nd["x"]))
    return coords


# ──────────────────────────────────────────────
# GRAPH ANALYTICS
# ──────────────────────────────────────────────

def compute_bottleneck_roads(G: nx.MultiDiGraph,
                              top_n: int = 10) -> list[tuple]:
    """
    Compute edge betweenness centrality to identify bottleneck roads.
    A high-centrality edge = many shortest paths pass through it.
    Blocking such a road would cause the largest re-routing impact.

    Time complexity: O(V * E) — expensive on large graphs, use on subgraph.
    """
    print("Computing edge betweenness centrality...")
    # Use a 1000-node sample for speed on large graphs
    nodes_sample = list(G.nodes)[:min(500, G.number_of_nodes())]
    subG = G.subgraph(nodes_sample)

    centrality = nx.edge_betweenness_centrality(subG, weight="travel_time",
                                                 normalized=True)
    top_edges = sorted(centrality.items(), key=lambda x: -x[1])[:top_n]
    print(f"   Top {top_n} bottleneck edges identified")
    return top_edges


# ──────────────────────────────────────────────
# MAP BUILDING
# ──────────────────────────────────────────────

def build_map(G: nx.MultiDiGraph, lat: float, lon: float,
              top_results: list[dict],
              isochrone_data: dict,
              blocked_edges: list[tuple] = []) -> folium.Map:
    """
    Build a multi-layer Folium map showing:
      - User location
      - Best hospital route (green)
      - 2nd and 3rd best hospitals
      - Isochrone rings (5/10/15 min reachability)
      - Blocked roads (red dashes)

    Route lines come from:
      - Mappls route_geometry when Mappls routing is used
      - local graph node paths when local routing is used
    """
    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="CartoDB positron")

    def marker_location(base_lat: float, base_lon: float, rank_idx: int) -> list[float]:
        # Slightly separate overlapping hospital markers so stacked duplicates stay visible.
        offset = 0.00018 * rank_idx
        return [base_lat + offset, base_lon + offset]

    # ── Isochrone rings ──
    iso_colors = ["#00cc6633", "#ff990033", "#cc000033"]
    iso_labels = ["15 min", "10 min", "5 min"]
    for i, (minutes, coords) in enumerate(sorted(
            isochrone_data.items(), reverse=True)):
        if coords:
            fg = folium.FeatureGroup(name=f"{minutes} min reachability")
            HeatMap(
                coords, min_opacity=0.15, max_opacity=0.35,
                radius=18, blur=25
            ).add_to(fg)
            fg.add_to(m)

    # ── Hospital routes ──
    route_colors = ["#00cc66", "#ff9900", "#cc0055", "#3366ff", "#663399"]
    for i, res in enumerate(top_results):
        if res.get("route_geometry"):
            route_coords = res["route_geometry"]
        else:
            route_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in res["path"]]
        label = f"#{i+1} {res['name']} | {res['travel_min']} min | {res['dist_km']} km"
        route_color = route_colors[i % len(route_colors)]

        if route_coords:
            folium.PolyLine(
                route_coords,
                color=route_color,
                weight=6 - i,
                opacity=0.9 - i * 0.15,
                tooltip=label,
                dash_array=None if i == 0 else "8 4"
            ).add_to(m)

        icon_color = "#e53935" if i == 0 else "#fb8c00" if i == 1 else "#8e24aa"
        badge = str(i + 1)
        marker_lat, marker_lon = marker_location(res["lat"], res["lon"], i)
        folium.Marker(
            [marker_lat, marker_lon],
            popup=folium.Popup(
                f"<b>{res['name']}</b><br>"
                f"Travel: {res['travel_min']} min<br>"
                f"Distance: {res['dist_km']} km<br>"
                f"Score: {res['score']}<br>"
                f"Algorithm: {res['algorithm'].title()}<br>"
                f"Specialty: {res['specialty']}",
                max_width=220
            ),
            tooltip=res["name"],
            icon=folium.DivIcon(
                html=(
                    f"<div style='position:relative;width:34px;height:46px;'>"
                    f"<div style='position:absolute;top:0;left:0;width:34px;height:34px;"
                    f"border-radius:50%;background:{icon_color};border:3px solid white;"
                    f"box-shadow:0 2px 8px rgba(0,0,0,0.35);color:white;font-weight:700;"
                    f"font-size:16px;line-height:28px;text-align:center;'>{badge}</div>"
                    f"<div style='position:absolute;left:12px;top:29px;width:0;height:0;"
                    f"border-left:5px solid transparent;border-right:5px solid transparent;"
                    f"border-top:13px solid {icon_color};'></div>"
                    f"</div>"
                )
            )
        ).add_to(m)

    # ── Blocked roads ──
    for u, v in blocked_edges:
        if G.has_edge(u, v):
            uy, ux = G.nodes[u]["y"], G.nodes[u]["x"]
            vy, vx = G.nodes[v]["y"], G.nodes[v]["x"]
            folium.PolyLine(
                [[uy, ux], [vy, vx]],
                color="red", weight=5, dash_array="6 3",
                tooltip=f"Blocked: {u}->{v}"
            ).add_to(m)

    # ── User location ──
    folium.Marker(
        [lat, lon],
        popup="User location",
        tooltip="You are here",
        icon=folium.Icon(color="blue", icon="user", prefix="fa")
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


# ──────────────────────────────────────────────
# REPORT GENERATION
# ──────────────────────────────────────────────

def generate_report(G: nx.MultiDiGraph, lat: float, lon: float,
                     top_results: list[dict],
                     bottleneck_edges: list[tuple],
                     hour: int, config: dict) -> str:
    """
    Generate a plain-text routing report covering:
      - System parameters
      - Top-K hospital rankings
      - Dijkstra vs A* comparison (if run with algorithm='both')
      - Big-O complexity analysis
      - Graph statistics
      - Bottleneck road list
    """
    sep  = "═" * 52
    sep2 = "─" * 52
    route_source = (
        str(top_results[0].get("route_source", "local")).upper()
        if top_results else "LOCAL"
    )
    routing_provider = "LOCAL" if route_source == "LOCAL" else config.get("routing_provider", "local").upper()
    selected_algorithm = config.get("selected_algorithm", "dijkstra").upper()
    using_local_graph = G.number_of_nodes() > 0 and G.number_of_edges() > 0

    lines = [
        sep,
        "  HOSPITAL ROUTE REPORT",
        sep,
        f"  User location  : ({lat}, {lon})",
        f"  Routing engine : {routing_provider}",
        f"  Algorithm mode : {selected_algorithm}",
        f"  Route source   : {route_source}",
        f"  Time of day    : {hour}:00  (peak multiplier: {get_peak_multiplier(hour):.1f}×)",
        f"  Search radius  : {config['search_radius_m']} m",
    ]

    if using_local_graph:
        lines.append(f"  Graph size     : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        lines.append("  Graph size     : Local graph not built for this run")

    lines += [
        "  TOP HOSPITALS",
        sep2,
        f"  {'Rank':<5} {'Name':<25} {'Time(min)':<11} {'Dist(km)':<10} {'Score':<8} {'Algo':<9} {'Source':<8} {'Runtime(s)'}",
        sep2,
    ]

    for i, r in enumerate(top_results, 1):
        runtime_value = r["runtime_s"] if r.get("runtime_s") is not None else "-"
        lines.append(
            f"  #{i:<4} {r['name']:<25} {r['travel_min']:<11.2f} "
            f"{r['dist_km']:<10.3f} {r['score']:<8.4f} {r['algorithm']:<9} {str(r.get('route_source', '-')):<8} {runtime_value}"
        )

    if top_results and "astar_min" in top_results[0]:
        lines += [
            "",
            "  DIJKSTRA AND A* COMPARISON",
            sep2,
            f"  Displayed route algorithm: {top_results[0]['algorithm'].upper()}",
            f"  Dijkstra  — time: {top_results[0]['dijkstra_min']} min | "
            f"runtime: {top_results[0]['dijkstra_runtime_s']}s",
            f"  A*        — time: {top_results[0]['astar_min']} min | "
            f"runtime: {top_results[0]['astar_runtime_s']}s",
            "  The displayed route is chosen by lower travel cost, then lower runtime.",

        ]

    lines += [
        "  SCORE",
        sep2,
        "  Lower score is better.",
        "  Score = 0.60 × travel time + 0.30 × distance factor + 0.10 × specialty factor",
        "  ",
    ]

    if using_local_graph:
        lines += [
            "",
            "  COMPLEXITY",
            sep2,
            "  Algorithm       | Time Complexity   | Space  | Notes",
            sep2,
            "  Dijkstra        | O((V+E) log V)    | O(V)   | Binary heap; exact optimal",
            "  A*              | O((V+E) log V)    | O(V)   | Faster in practice; needs heuristic",
            "  BFS             | O(V+E)            | O(V)   | Unweighted only; not used here",
            "  Brute-force     | O(V! × E)         | O(V)   | Infeasible for any real graph",
            "  Betweenness     | O(V × E)          | O(V+E) | Used for bottleneck analysis only",
            "",
            f"  Graph V={G.number_of_nodes()}, E={G.number_of_edges()}",
            f"  Dijkstra worst-case ops ≈ {int((G.number_of_nodes() + G.number_of_edges()) * math.log2(max(G.number_of_nodes(),1))):,}",
            "",
            "  DATA STRUCTURES",
            sep2,
            "  MultiDiGraph    - road network",
            "  Min-heap        - priority queue used by shortest-path search",
            "  dict            - O(1) density lookup by (u,v,key)",
            "  GeoDataFrame    - spatial road and hospital data",
            "  list            - ordered route node sequence",
        ]
    else:
        lines += [
            "",
            "  DATA STRUCTURES",
            sep2,
            "  GeoDataFrame    - hospital and traffic data",
            "  dict            - parsed API responses and config lookups",
            "  list            - ranked hospital results and route coordinates",
            "",
            "  NOTE",
            sep2,
            "  Local graph analysis was skipped for this run.",

        ]

    if bottleneck_edges:
        lines += [
            "",
            "  BOTTLENECK ROADS",
            sep2,
            "  Centrality   Edge (u → v)",
            sep2,
        ]
        for edge_ref, c in bottleneck_edges[:5]:
            if len(edge_ref) == 3:
                u, v, _ = edge_ref
            elif len(edge_ref) == 2:
                u, v = edge_ref
            else:
                lines.append(f"  {c:.4f}       {edge_ref}")
                continue
            lines.append(f"  {c:.4f}       {u} → {v}")

    lines += ["", sep, "  END OF REPORT", sep]
    return "\n".join(lines)


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def run(lat: float, lon: float, hour: int = 8,
        algorithm: str = "both",
        blocked_roads: list[tuple] = [],
        config: dict = DEFAULT_CONFIG) -> None:
    """
    Full pipeline:
      1. Load road graph from OSM
      2. Load + normalise traffic data
      3. Assign real travel-time edge weights
      4. Optionally simulate road blocks
      5. Load hospitals + snap to graph
      6. Find top-K hospitals (Dijkstra + A*)
      7. Compute isochrone reachability rings
      8. Compute bottleneck roads
      9. Build Folium map
      10. Generate + save report
    """
    print("\n" + "═"*52)
    print("  HOSPITAL ROUTE FINDER — STARTING")
    print("═"*52 + "\n")

    # 1. Road network
    G = load_road_graph(lat, lon, config["search_radius_m"])

    # 2. Traffic data
    roads = load_traffic_data(config["road_geojson"], lat, lon)
    if len(roads) > 0 and config["density_col"] in roads.columns:
        roads = normalise_density(roads, config["density_col"])
    elif len(roads) > 0:
        print(f"   Density column '{config['density_col']}' not found - "
              f"available: {roads.columns.tolist()}\n"
              "       Using neutral traffic weights.")
        roads[config["density_col"] + "_norm"] = 0.0

    # 3. Edge weights
    G = assign_edge_weights(G, roads, config["density_col"], hour)

    # 4. Road block simulation
    if blocked_roads:
        print(f"\nSimulating {len(blocked_roads)} blocked road(s)...")
        G = simulate_road_block(G, blocked_roads)

    # 5. Hospitals
    print("\nLoading nearby hospital data...")
    hospitals = load_hospital_points(config["hospital_geojson"])
    hospitals = filter_nearby_hospitals(
        hospitals, lat, lon, radius_m=config["search_radius_m"], max_candidates=75
    )
    hospitals = snap_hospitals_to_graph(hospitals, G)

    # 6. Routing
    orig_node = ox.nearest_nodes(G, lon, lat)
    provider = config.get("routing_provider", "local").lower()
    if provider == "mappls":
        print(f"\nRunning Mappls routing to all hospitals with profile '{config["mappls_profile"]}'...")
        top_results = find_top_k_hospitals_mappls(
            lat, lon, hospitals,
            api_key=config.get("mappls_api_key", ""),
            k=config["top_k_hospitals"],
            profile=config.get("mappls_profile", "driving"),
            route_resource=config.get("mappls_route_resource", "route_eta"),
            timeout_s=config.get("mappls_timeout_s", 20),
        )
        if not top_results and config.get("mappls_fallback_to_local", True):
            print("   Mappls returned no ranked hospitals. Falling back to local graph routing.")
            top_results = find_top_k_hospitals(
                G, orig_node, hospitals,
                k=config["top_k_hospitals"],
                algorithm=algorithm
            )
    else:
        print(f"\nRunning {algorithm.upper()} routing to all hospitals...")
        top_results = find_top_k_hospitals(
            G, orig_node, hospitals,
            k=config["top_k_hospitals"],
            algorithm=algorithm
        )

    if not top_results:
        print("\nNo reachable hospitals found. Try increasing search_radius_m in config.")
        return

    # 7. Isochrones
    print("\nComputing reachability isochrones...")
    isochrone_data = {}
    for mins in config["isochrone_minutes"]:
        isochrone_data[mins] = compute_isochrone(G, orig_node, mins)
        print(f"   {mins}-minute ring: {len(isochrone_data[mins])} reachable nodes")

    # 8. Bottleneck analysis
    bottleneck_edges = compute_bottleneck_roads(G)

    # 9. Map
    print("\nBuilding interactive map...")
    m = build_map(G, lat, lon, top_results, isochrone_data, blocked_roads)
    m.save(config["output_map"])
    print(f"   Map saved: {config["output_map"]}")

    # 10. Report
    report = generate_report(G, lat, lon, top_results,
                              bottleneck_edges, hour, config)
    print("\n" + report)
    with open(config["output_report"], "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n   Report saved: {config["output_report"]}")
    print("\n" + "═"*52 + "\n")


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Hospital Route Finder"
    )
    p.add_argument("--lat",       type=float, required=True,  help="Latitude")
    p.add_argument("--lon",       type=float, required=True,  help="Longitude")
    p.add_argument("--hour",      type=int,   default=8,      help="Hour of day 0-23 (default 8)")
    p.add_argument("--algo",      type=str,   default="both",
                   choices=["dijkstra", "astar", "both"],     help="Routing algorithm")
    p.add_argument("--road",      type=str,   default=DEFAULT_CONFIG["road_geojson"],
                   help="Path to road GeoJSON")
    p.add_argument("--hospitals", type=str,   default=DEFAULT_CONFIG["hospital_geojson"],
                   help="Path to hospital GeoJSON")
    p.add_argument("--output",    type=str,   default=DEFAULT_CONFIG["output_map"],
                   help="Output HTML map path")
    p.add_argument("--topk",      type=int,   default=3,      help="Number of hospitals to rank")
    p.add_argument("--provider",  type=str,   default=DEFAULT_CONFIG["routing_provider"],
                   choices=["mappls", "local"], help="Primary routing engine")
    p.add_argument("--mappls-key", type=str,  default=DEFAULT_CONFIG["mappls_api_key"],
                   help="Mappls API key")
    p.add_argument("--mappls-profile", type=str, default=DEFAULT_CONFIG["mappls_profile"],
                   help="Mappls routing profile (e.g. driving)")
    p.add_argument("--block",     type=str,   default="",
                   help="Comma-separated blocked edges e.g. '123456,789012'")
    p.add_argument("--radius",    type=int,   default=5000,   help="Search radius in metres")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg["road_geojson"]     = args.road
    cfg["hospital_geojson"] = args.hospitals
    cfg["output_map"]       = args.output
    cfg["top_k_hospitals"]  = args.topk
    cfg["routing_provider"] = args.provider
    cfg["mappls_api_key"]   = args.mappls_key
    cfg["mappls_profile"]   = args.mappls_profile
    cfg["search_radius_m"]  = args.radius

    # Parse blocked road node pairs (optional)
    blocked = []
    if args.block:
        ids = [int(x) for x in args.block.split(",")]
        blocked = list(zip(ids[0::2], ids[1::2]))

    run(
        lat=args.lat,
        lon=args.lon,
        hour=args.hour,
        algorithm=args.algo,
        blocked_roads=blocked,
        config=cfg
    )

"""Streamlit interface for the hospital routing project."""

import os
import sys

import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

sys.path.insert(0, os.path.dirname(__file__))

from hospital_router import (
    DEFAULT_CONFIG,
    assign_edge_weights,
    build_map,
    compute_bottleneck_roads,
    compute_isochrone,
    filter_nearby_hospitals,
    generate_report,
    get_peak_multiplier,
    load_hospital_points,
    load_road_graph,
    load_traffic_data,
    normalise_density,
    simulate_road_block,
    snap_hospitals_to_graph,
    find_top_k_hospitals,
)


st.set_page_config(page_title="Hospital Route Finder", layout="wide")

st.title("Hospital Route Finder")
st.caption("Data Structures and Algorithms project")


@st.cache_resource(show_spinner=False)
def get_cached_weighted_graph(
    lat: float, lon: float, radius: int, road_path: str, density_col: str, hour: int
):
    """Load and weight the local road graph."""
    graph = load_road_graph(lat, lon, radius, road_path)
    roads = load_traffic_data(road_path, lat, lon)
    if len(roads) > 0 and density_col in roads.columns:
        roads = normalise_density(roads, density_col)
    graph = assign_edge_weights(graph, roads, density_col, hour)
    return graph


@st.cache_data(show_spinner=False)
def get_cached_nearby_hospital_points(
    hosp_path: str, lat: float, lon: float, radius: int, max_candidates: int = 75
) -> gpd.GeoDataFrame:
    """Load and filter nearby hospitals from the dataset."""
    hospitals = load_hospital_points(hosp_path)
    return filter_nearby_hospitals(
        hospitals, lat, lon, radius_m=radius, max_candidates=max_candidates
    )


def format_travel_time_label(minutes: float) -> str:
    total_seconds = int(round(float(minutes) * 60))
    if total_seconds < 60:
        return f"{total_seconds} sec"
    if total_seconds < 3600:
        mins, secs = divmod(total_seconds, 60)
        return f"{mins} min" if secs == 0 else f"{mins} min {secs} sec"
    hours, rem = divmod(total_seconds, 3600)
    mins = rem // 60
    return f"{hours} hr {mins} min"


def format_distance_label(km: float) -> str:
    km = float(km)
    if km < 1:
        return f"{int(round(km * 1000))} m"
    return f"{km:.2f} km"


with st.sidebar:
    st.header("Parameters")

    lat = st.number_input("Latitude", value=18.5204, format="%.6f")
    lon = st.number_input("Longitude", value=73.8567, format="%.6f")

    hour = st.slider("Time of day (hour)", min_value=0, max_value=23, value=8)
    peak_mult = get_peak_multiplier(hour)
    st.info(f"Peak multiplier at {hour}:00: {peak_mult:.1f}x")

    algorithm = st.selectbox("Routing algorithm", ["both", "dijkstra", "astar"])

    routing_provider = st.selectbox(
        "Map tiles / visualization provider",
        ["mappls", "local"],
        index=0 if DEFAULT_CONFIG["routing_provider"] == "mappls" else 1,
        help="This affects map display only. Routing uses the local graph.",
    )
    mappls_api_key = st.text_input(
        "Mappls API key",
        value=DEFAULT_CONFIG["mappls_api_key"],
        type="password",
    )
    mappls_profile = st.text_input("Mappls profile", value=DEFAULT_CONFIG["mappls_profile"])
    mappls_route_resource = st.selectbox(
        "Mappls route mode",
        ["route_eta", "route_adv"],
        index=0 if DEFAULT_CONFIG["mappls_route_resource"] == "route_eta" else 1,
    )
    use_local_fallback = st.checkbox(
        "Use local routing backend",
        value=DEFAULT_CONFIG["mappls_fallback_to_local"],
    )
    show_isochrones = st.checkbox("Show isochrone layers", value=False)
    include_bottlenecks = st.checkbox("Enable bottleneck analysis", value=False)

    top_k = st.slider("Top hospitals", 1, 5, 3)
    radius = st.slider("Search radius (metres)", 1000, 10000, 5000, step=500)

    st.subheader("Data Files")
    road_path = st.text_input("Road GeoJSON path", DEFAULT_CONFIG["road_geojson"])
    hosp_path = st.text_input("Hospital data path", DEFAULT_CONFIG["hospital_geojson"])

    st.subheader("Road Block Simulation")
    enable_block = st.checkbox("Enable road block")
    block_input = st.text_input(
        "Blocked road node IDs (u1,v1,u2,v2,...)",
        disabled=not enable_block,
        placeholder="e.g. 123456,789012",
    )

    run_btn = st.button("Find Best Hospital", type="primary", use_container_width=True)


if run_btn:
    blocked = []
    if enable_block and block_input:
        try:
            ids = [int(x.strip()) for x in block_input.split(",")]
            blocked = list(zip(ids[0::2], ids[1::2]))
        except ValueError:
            st.error("Invalid node IDs. Use comma-separated integers.")
            st.stop()

    cfg = DEFAULT_CONFIG.copy()
    cfg.update(
        {
            "road_geojson": road_path,
            "hospital_geojson": hosp_path,
            "routing_provider": routing_provider,
            "mappls_api_key": mappls_api_key,
            "mappls_profile": mappls_profile,
            "mappls_route_resource": mappls_route_resource,
            "mappls_fallback_to_local": use_local_fallback,
            "selected_algorithm": algorithm,
            "top_k_hospitals": top_k,
            "search_radius_m": radius,
        }
    )

    progress = st.progress(0, text="Preparing data")

    try:
        density_col = DEFAULT_CONFIG["density_col"]
        graph = get_cached_weighted_graph(lat, lon, radius, road_path, density_col, hour).copy()
        progress.progress(45, "Road graph ready")

        if blocked:
            graph = simulate_road_block(graph, blocked)

        import osmnx as ox

        orig_node = ox.nearest_nodes(graph, lon, lat)

        progress.progress(60, "Loading hospitals")
        hospitals = get_cached_nearby_hospital_points(hosp_path, lat, lon, radius)

        progress.progress(70, "Computing routes")
        hospitals_local = snap_hospitals_to_graph(hospitals, graph)
        top_results = find_top_k_hospitals(graph, orig_node, hospitals_local, top_k, algorithm)

        if not top_results:
            st.error("No reachable hospitals found. Increase the search radius or check the dataset.")
            st.stop()

        if show_isochrones and orig_node is not None:
            progress.progress(82, "Computing isochrones")
            isochrone_data = {
                minutes: compute_isochrone(graph, orig_node, minutes)
                for minutes in cfg["isochrone_minutes"]
            }
        else:
            isochrone_data = {minutes: [] for minutes in cfg["isochrone_minutes"]}

        progress.progress(92, "Building map")
        route_map = build_map(graph, lat, lon, top_results, isochrone_data, blocked)

        progress.progress(100, "Done")
        progress.empty()

    except FileNotFoundError as exc:
        progress.empty()
        st.error(str(exc))
        st.stop()
    except ValueError as exc:
        progress.empty()
        st.error(str(exc))
        st.info("If you are testing Andhra Pradesh locations, make sure the road file also contains Andhra Pradesh roads.")
        st.stop()
    except Exception as exc:
        progress.empty()
        st.error(f"Unexpected error: {exc}")
        raise

    best = top_results[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Hospital", best["name"])
    c2.metric("Travel Time", format_travel_time_label(best["travel_min"]))
    c3.metric("Distance", format_distance_label(best["dist_km"]))
    c4.metric("Score", best["score"])
    st.caption(
        f"Selected algorithm: {algorithm.title()} | "
        f"Displayed route source: {str(best.get('route_source', routing_provider)).title()} | "
        f"Displayed route algorithm: {str(best.get('algorithm', algorithm)).title()}"
    )

    st.subheader("Ranked Hospitals")
    rows = []
    for i, result in enumerate(top_results, 1):
        row = {
            "Rank": f"#{i}",
            "Hospital": result["name"],
            "Travel": format_travel_time_label(result["travel_min"]),
            "Distance": format_distance_label(result["dist_km"]),
            "Score": result["score"],
            "Algorithm": result["algorithm"].title(),
            "Route Source": str(result.get("route_source", routing_provider)).title(),
            "Runtime (s)": result["runtime_s"] if result.get("runtime_s") is not None else "-",
        }
        if "astar_min" in result:
            row["Dijkstra Time"] = format_travel_time_label(result["dijkstra_min"])
            row["Dijkstra Runtime (s)"] = result["dijkstra_runtime_s"]
            row["A* Time"] = format_travel_time_label(result["astar_min"])
            row["A* Runtime (s)"] = result["astar_runtime_s"]
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if "astar_min" in top_results[0]:
        best = top_results[0]
        st.subheader("Algorithm Comparison")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.write("Dijkstra")
            st.write(f"Time: {best['dijkstra_min']} min")
            st.write(f"Runtime: {best['dijkstra_runtime_s']} s")
        with cc2:
            st.write("A*")
            st.write(f"Time: {best.get('astar_min', '-')} min")
            st.write(f"Runtime: {best.get('astar_runtime_s', '-')} s")

    st.subheader("Route Map")
    st.caption("Green: best route | Orange: second | Red: third")
    st_html(route_map._repr_html_(), height=520, scrolling=False)

    with st.expander("Analysis Report"):
        if include_bottlenecks and graph.number_of_nodes() > 0:
            if st.button("Generate full report with bottleneck analysis", use_container_width=True):
                try:
                    bottleneck_edges = compute_bottleneck_roads(graph)
                except Exception:
                    bottleneck_edges = []
                report = generate_report(graph, lat, lon, top_results, bottleneck_edges, hour, cfg)
                st.code(report, language="text")
            else:
                st.info("Generate the full report only when needed.")
        else:
            report = generate_report(graph, lat, lon, top_results, [], hour, cfg)
            st.code(report, language="text")
else:
    st.info("Enter the location values in the sidebar and click Find Best Hospital.")

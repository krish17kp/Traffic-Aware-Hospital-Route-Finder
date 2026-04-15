# Traffic-Aware Hospital Route Finder

This project finds suitable hospital routes from a user location using a road network graph, travel-time based edge weights, and shortest path algorithms. It includes a Streamlit interface, hospital ranking, map output, and a text report for analysis.

## Project Structure

`app.py` - Streamlit interface  
`hospital_router.py` - routing, ranking, map generation, and report generation  
`hospital_directory.csv` - hospital dataset  
`ROAD 1/ROAD.geojson` - road dataset  
`cache/` - cached graph files

## Installation

```bash
pip install osmnx networkx geopandas folium shapely streamlit pandas
```

## How to Run

```bash
streamlit run app.py
```

## Required Files

- `hospital_directory.csv`
- `ROAD 1/ROAD.geojson`

The hospital file should contain `Hospital_Name` and `Location_Coordinates`.

## Algorithms and Data Structures

- `Dijkstra`: shortest path on the weighted road graph
- `A*`: shortest path using a haversine-based heuristic
- `Edge betweenness centrality`: optional bottleneck analysis

Data structures used:

- `NetworkX MultiDiGraph` for the road network
- `GeoDataFrame` for hospital and road data
- `dict` for edge density and configuration data
- `list` for ranked results and route paths

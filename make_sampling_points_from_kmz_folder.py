#!/usr/bin/env python3
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import glob
import os

# -----------------------------
# PARAMETERS
# -----------------------------
KMZ_FOLDER = "regions_kmz/"      # folder containing *.kmz files
STEP = 0.125                      # ERA5 grid spacing in degrees is 0.25
OUT_CSV = "sampling_points_all_regions_8thdeg.csv"

# -----------------------------
# FUNCTION TO PROCESS ONE REGION
# -----------------------------
def points_for_region(kmz_path, step=STEP):
    """Generate 0.25° sampling points inside a KMZ polygon."""
    
    region_name = os.path.splitext(os.path.basename(kmz_path))[0]
    print(f"\nProcessing region: {region_name}")

    # Load KMZ → GeoDataFrame
    poly = gpd.read_file(kmz_path)
    poly = poly.to_crs("EPSG:4326")  # ensure lat/lon
    region = poly.unary_union        # dissolve into single polygon

    # Get bounding box
    min_lon, min_lat, max_lon, max_lat = region.bounds

    # Build ERA5-style aligned lat/lon sequences
    lons = np.arange(
        np.floor(min_lon * 4) / 4,
        np.ceil(max_lon * 4) / 4 + 1e-12,
        step
    )
    lats = np.arange(
        np.floor(min_lat * 4) / 4,
        np.ceil(max_lat * 4) / 4 + 1e-12,
        step
    )

    # Test each grid point
    rows = []
    for lat in lats:
        for lon in lons:
            pt = Point(lon, lat)
            if region.contains(pt):
                rows.append({
                    "region": region_name,
                    "lat": lat,
                    "lon": lon
                })

    print(f"  → {len(rows)} points inside polygon.")
    return rows


# -----------------------------
# MAIN SCRIPT: PROCESS ALL KMZ FILES
# -----------------------------
def main():
    kmz_files = sorted(glob.glob(os.path.join(KMZ_FOLDER, "*.kmz")))
    if not kmz_files:
        print("No KMZ files found.")
        return

    all_rows = []

    for kmz in kmz_files:
        rows = points_for_region(kmz)
        all_rows.extend(rows)

    # Save combined CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote combined grid to {OUT_CSV}")
    print(f"Total points across all regions: {len(df)}")


if __name__ == "__main__":
    main()


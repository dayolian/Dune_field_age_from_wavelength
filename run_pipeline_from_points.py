#!/usr/bin/env python3
"""
run_pipeline_from_points.py

Pipeline:

1. Read a CSV of points with columns: region, lat, lon.
2. For each point:
   - Build a small lat/lon box around it.
   - Call download_s2_box_SZA.py to download a single-band Sentinel-2 GeoTIFF.
   - Reject tiles with >50% nodata.
   - Run analyze_geotiff_multimode(...) on the TIF.
3. Write out:
   - A CSV of all modes (good_points).
   - A CSV of failed points with reasons (failed_points).

No ZIP handling: the downloader writes TIFs directly.
"""

import os
import sys
import csv
import subprocess
import argparse

import pandas as pd
import rasterio

from fft_full_geotiff_multimode import analyze_geotiff_multimode


# --------------------------
# Configuration / constants
# --------------------------

DOWNLOAD_SCRIPT = "download_s2_box_SZA_luminance.py"
EE_PROJECT = "dunefieldage"

SCALE      = 20.0          # m/pixel

# Defaults for Sentinel-2 filters; can override via CLI
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE   = "2022-12-31"
DEFAULT_MAX_CLOUD  = 30.0
DEFAULT_MIN_SZA    = 30.0

# Default half box size around each point
DEFAULT_BOX_HALF_DEG_LAT = 0.125
DEFAULT_BOX_HALF_DEG_LON = 0.125

# Nodate threshold: if < this fraction of pixels is valid, we skip
VALID_FRACTION_MIN = 0.75


# --------------------------
# Utility functions
# --------------------------

def make_tile_basename(region: str, lat: float, lon: float) -> str:
    """Build a readable tile basename from region + lat/lon."""
    r = region.replace(" ", "_")
    return f"{r}_lat{lat:+05.2f}_lon{lon:+06.2f}".replace(".", "p").replace("+", "")


def compute_valid_fraction(tif_path: str) -> float:
    """
    Open a TIF and estimate the fraction of valid pixels using rasterio's
    masked read (nodata -> masked).
    """
    with rasterio.open(tif_path) as src:
        band = src.read(1, masked=True)  # MaskedArray: mask==True where nodata
    mask = getattr(band, "mask", None)
    if mask is None:
        # No mask information; assume fully valid
        return 1.0
    total = mask.size
    if total == 0:
        return 0.0
    invalid = mask.sum()  # True -> 1
    valid = total - invalid
    return float(valid) / float(total)


def run_download(min_lat, max_lat, min_lon, max_lon, out_tif_path,
                 start_date, end_date, max_cloud, min_sza):
    """
    Call download_s2_box_SZA.py via subprocess to download a single-band TIF.
    """
    cmd = [
        sys.executable, DOWNLOAD_SCRIPT,
        "--min_lat", str(min_lat),
        "--max_lat", str(max_lat),
        "--min_lon", str(min_lon),
        "--max_lon", str(max_lon),
        "--start_date", start_date,
        "--end_date", end_date,
        "--out_path", out_tif_path,
        "--scale", str(SCALE),
        "--max_cloud", str(max_cloud),
        "--min_sza", str(min_sza),
        "--ee_project", EE_PROJECT,
    ]

    print("  Running download command:")
    print("   ", " ".join(cmd))

    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Download script failed with return code {res.returncode}")


def run_fft_and_collect_modes(tif_path, region, lat, lon, diag_dir):
    """
    Run the multimode FFT on tif_path and return a list of result rows (dicts).
    No internal try/except: errors are handled in main().
    """
    base = make_tile_basename(region, lat, lon)
    diag_prefix = os.path.join(diag_dir, base)

    print(f"  Running FFT multimode on {tif_path}...")
    modes = analyze_geotiff_multimode(tif_path, out_prefix=diag_prefix)

    rows = []
    if not modes:
        print("    No significant modes found.")
        return rows

    for i, m in enumerate(modes, start=1):
        row = {
            "region": region,
            "lat": lat,
            "lon": lon,
            "tile_basename": base,
            "mode_index": i,
            "lambda_m": m.get("lambda_m", None),
            "crest_az_deg": m.get("crest_az", None),
            "rel_power": m.get("rel_power", None),
            "spectral_intensity": m.get("spectral_intensity", None),
            "prom_frac": m.get("prom_frac", None),
            "theta_image_deg": m.get("theta_image_deg", None),
        }
        rows.append(row)

    print(f"    Found {len(rows)} modes.")
    return rows


# --------------------------
# CLI
# --------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Run S2 download + FFT multimode pipeline from a CSV of points."
    )
    ap.add_argument(
        "--points_csv", required=True,
        help="Input CSV with at least columns: region, lat, lon"
    )
    ap.add_argument(
        "--out_csv", required=True,
        help="Output CSV for all modes"
    )
    ap.add_argument(
        "--tiles_dir", default="tiles",
        help="Directory to store downloaded GeoTIFF tiles (default: tiles)"
    )
    ap.add_argument(
        "--diag_dir", default="outputs",
        help="Directory to store diagnostic FFT figures (default: outputs)"
    )
    ap.add_argument(
        "--fail_csv", default="failed_points.csv",
        help="CSV to log points that failed download or FFT (default: failed_points.csv)"
    )

    # Sentinel-2 filters
    ap.add_argument("--start_date", default=DEFAULT_START_DATE,
                    help="Start date for S2 search (YYYY-MM-DD)")
    ap.add_argument("--end_date", default=DEFAULT_END_DATE,
                    help="End date for S2 search (YYYY-MM-DD)")
    ap.add_argument("--max_cloud", type=float, default=DEFAULT_MAX_CLOUD,
                    help="Max CLOUDY_PIXEL_PERCENTAGE (default 30)")
    ap.add_argument("--min_sza", type=float, default=DEFAULT_MIN_SZA,
                    help="Min solar zenith angle in degrees (0 = no SZA filter)")

    # spatial box size
    ap.add_argument("--box_half_deg_lat", type=float, default=DEFAULT_BOX_HALF_DEG_LAT,
                    help="Half height of lat box around each point (deg)")
    ap.add_argument("--box_half_deg_lon", type=float, default=DEFAULT_BOX_HALF_DEG_LON,
                    help="Half width of lon box around each point (deg)")

    return ap.parse_args()


# --------------------------
# Main pipeline
# --------------------------

def main():
    args = parse_args()

    os.makedirs(args.tiles_dir, exist_ok=True)
    os.makedirs(args.diag_dir, exist_ok=True)

    df = pd.read_csv(args.points_csv)
    required_cols = {"region", "lat", "lon"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    all_rows = []
    failures = []

    total_points = len(df)
    for idx, row in df.iterrows():
        region = str(row["region"])
        lat = float(row["lat"])
        lon = float(row["lon"])

        print(f"\n=== Processing point {idx+1}/{total_points}: "
              f"{region}, lat={lat:.3f}, lon={lon:.3f} ===")

        # Build small box around the point
        min_lat = lat - args.box_half_deg_lat
        max_lat = lat + args.box_half_deg_lat
        min_lon = lon - args.box_half_deg_lon
        max_lon = lon + args.box_half_deg_lon

        tile_base = make_tile_basename(region, lat, lon)
        tif_path = os.path.join(args.tiles_dir, f"{tile_base}.tif")

        # ---- Download step ----
        if not os.path.exists(tif_path):
            try:
                run_download(
                    min_lat, max_lat, min_lon, max_lon, tif_path,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    max_cloud=args.max_cloud,
                    min_sza=args.min_sza,
                )
            except Exception as e:
                msg = str(e)
                print(f"  Download failed for this point: {msg}")
                failures.append({
                    "region": region,
                    "lat": lat,
                    "lon": lon,
                    "tile_basename": tile_base,
                    "stage": "download",
                    "reason": msg,
                    "valid_fraction": None,
                })
                continue
        else:
            print(f"  TIF already exists, skipping download: {tif_path}")

        if not os.path.exists(tif_path):
            msg = "TIF not found after download attempt"
            print(f"  {msg}: {tif_path}")
            failures.append({
                "region": region,
                "lat": lat,
                "lon": lon,
                "tile_basename": tile_base,
                "stage": "download",
                "reason": msg,
                "valid_fraction": None,
            })
            continue

        # ---- Nodate / valid-fraction check ----
        try:
            valid_fraction = compute_valid_fraction(tif_path)
        except Exception as e:
            msg = f"failed to compute valid_fraction: {e}"
            print("  " + msg)
            failures.append({
                "region": region,
                "lat": lat,
                "lon": lon,
                "tile_basename": tile_base,
                "stage": "nodata_check",
                "reason": msg,
                "valid_fraction": None,
            })
            continue

        print(f"  valid_fraction = {valid_fraction:.3f}")
        if valid_fraction < VALID_FRACTION_MIN:
            msg = (f"valid_fraction={valid_fraction:.3f} < "
                   f"{VALID_FRACTION_MIN}, skipping FFT.")
            print("  " + msg)
            failures.append({
                "region": region,
                "lat": lat,
                "lon": lon,
                "tile_basename": tile_base,
                "stage": "nodata_check",
                "reason": msg,
                "valid_fraction": valid_fraction,
            })
            continue

        # ---- FFT step ----
        try:
            rows = run_fft_and_collect_modes(
                tif_path, region, lat, lon, args.diag_dir
            )
        except Exception as e:
            msg = str(e)
            print(f"  FFT failed for this tile: {msg}")
            failures.append({
                "region": region,
                "lat": lat,
                "lon": lon,
                "tile_basename": tile_base,
                "stage": "fft",
                "reason": msg,
                "valid_fraction": valid_fraction,
            })
            continue

        all_rows.extend(rows)

    # ---- Write main output CSV ----
    if all_rows:
        out_dir = os.path.dirname(args.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        print(f"\nWriting {len(all_rows)} mode records to {args.out_csv}")
        fieldnames = [
            "region", "lat", "lon", "tile_basename", "mode_index",
            "lambda_m", "crest_az_deg", "rel_power",
            "spectral_intensity", "prom_frac", "theta_image_deg",
        ]
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)
    else:
        print("\nNo modes found for any point; nothing written to out_csv.")

    # ---- Write failures CSV ----
    if failures:
        fail_dir = os.path.dirname(args.fail_csv)
        if fail_dir:
            os.makedirs(fail_dir, exist_ok=True)
        print(f"Writing {len(failures)} failed points to {args.fail_csv}")
        fail_fields = [
            "region", "lat", "lon", "tile_basename",
            "stage", "reason", "valid_fraction",
        ]
        with open(args.fail_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fail_fields)
            writer.writeheader()
            for r in failures:
                writer.writerow(r)
    else:
        print("No failures to log; no fail_csv written.")


if __name__ == "__main__":
    main()


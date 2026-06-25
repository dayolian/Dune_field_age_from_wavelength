#!/usr/bin/env python3
"""
download_s2_box_SZA.py

Download a RGB Sentinel-2 image for a given lat/lon box and
date range, with optional cloud and solar zenith angle (SZA) filtering.

Output is a ZIP file from Earth Engine's getDownloadURL, which contains a GeoTIFF.

Basic usage: 
python download_s2_box_SZA.py \
  --min_lat 26.0 --max_lat 28.0 \
  --min_lon 39.0 --max_lon 41.0 \
  --start_date 2019-01-01 \
  --end_date   2019-12-31 \
  --out_path   data/nafud_tile_2019.tif

"""

import argparse
import os
import requests
import ee


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--min_lat", type=float, required=True)
    ap.add_argument("--max_lat", type=float, required=True)
    ap.add_argument("--min_lon", type=float, required=True)
    ap.add_argument("--max_lon", type=float, required=True)

    ap.add_argument("--start_date", type=str, required=True,
                    help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end_date", type=str, required=True,
                    help="End date (YYYY-MM-DD)")

    ap.add_argument("--out_path", type=str, required=True,
                    help="Path to save the downloaded ZIP (e.g. data/tile.zip)")

    ap.add_argument("--scale", type=float, default=30.0,
                    help="Output pixel size in meters (default 30)")

    ap.add_argument("--max_cloud", type=float, default=30.0,
                    help="Max CLOUDY_PIXEL_PERCENTAGE (default 30)")

    ap.add_argument("--min_sza", type=float, default=0.0,
                    help="Min MEAN_SOLAR_ZENITH_ANGLE in degrees "
                         "(default 0 = no SZA filter)")

    ap.add_argument("--ee_project", type=str, default="dunefieldage",
                    help="Earth Engine project ID for ee.Initialize(project=...). "
                         "Default: dunefieldage")

    return ap.parse_args()


def init_ee(project_id: str):
    """
    Initialize Earth Engine with an explicit project. This avoids the
    'no project found' EEException you were seeing.
    """
    try:
        ee.Initialize(project=project_id)
        print(f"Initialized Earth Engine with project='{project_id}'.")
    except Exception:
        print("EE initialization failed; attempting authentication...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print(f"Authenticated and initialized Earth Engine with project='{project_id}'.")


def build_s2_image(roi, start_date, end_date, max_cloud, min_sza):
    """
    Build a single Sentinel-2 image over the ROI, preferring:
      - low cloud
      - optionally high solar zenith angle (min_sza)

    If no image passes the SZA filter, we fall back to the least-cloudy
    image from the cloud-filtered collection.
    """

    collection_base = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(roi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
    )

    base_size = collection_base.size().getInfo()
    if base_size == 0:
        raise RuntimeError(
            "No Sentinel-2 images found for this ROI/date/cloud filter."
        )

    if min_sza > 0.0:
        col_sza = collection_base.filter(
            ee.Filter.gt("MEAN_SOLAR_ZENITH_ANGLE", min_sza)
        )
        sza_size = col_sza.size().getInfo()
        if sza_size > 0:
            print(f"  Found {sza_size} images with SZA > {min_sza} deg; "
                  f"preferring those.")
            image = ee.Image(col_sza.sort("CLOUDY_PIXEL_PERCENTAGE").first())
        else:
            print(f"  No images passed SZA > {min_sza} deg; "
                  f"falling back to cloud-only filter.")
            image = ee.Image(collection_base.sort("CLOUDY_PIXEL_PERCENTAGE").first())
    else:
        print("  No SZA filter applied; using cloud-only filtered collection.")
        image = ee.Image(collection_base.sort("CLOUDY_PIXEL_PERCENTAGE").first())

    # --- NEW: Perceptual greyscale from RGB (B4=R, B3=G, B2=B) ---
    # Y ≈ 0.299*R + 0.587*G + 0.114*B
    image_gray = image.expression(
        "0.299 * R + 0.587 * G + 0.114 * B",
        {
            "R": image.select("B4"),
            "G": image.select("B3"),
            "B": image.select("B2"),
        }
    ).rename("luminance")

    return image_gray


def download_ee_image(image, roi, out_path, scale):
    """Download an EE image over ROI to out_path as a ZIP (containing GeoTIFF)."""

    params = {
        "region": roi,
        "scale": scale,
        "filePerBand": False,  # single-band output
        "format": "GEO_TIFF",
    }

    url = image.getDownloadURL(params)
    print("Download URL:\n ", url)
    print(f"Downloading to: {out_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Download complete.")


def main():
    args = parse_args()
    init_ee(args.ee_project)

    roi = ee.Geometry.Rectangle(
        [args.min_lon, args.min_lat, args.max_lon, args.max_lat],
        proj="EPSG:4326",
        geodesic=False,
    )

    print("Building Sentinel-2 image with filters:")
    print(f"  Date range: {args.start_date} to {args.end_date}")
    print(f"  Max cloud: {args.max_cloud}%")
    print(f"  Min SZA:   {args.min_sza} deg")
    print(f"  EE project: {args.ee_project}")

    image = build_s2_image(
        roi=roi,
        start_date=args.start_date,
        end_date=args.end_date,
        max_cloud=args.max_cloud,
        min_sza=args.min_sza,
    )

    download_ee_image(
        image=image,
        roi=roi,
        out_path=args.out_path,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()


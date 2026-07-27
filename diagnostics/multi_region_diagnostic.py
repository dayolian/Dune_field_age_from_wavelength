#!/usr/bin/env python3
"""
multi_region_diagnostic.py

Download one tile per region (from diag_points.csv) and plot each tile's
directional radial spectrum, so we can see whether the "dominant short-wavelength
ripple + weaker longer dune peak" pattern from AlNafud holds elsewhere.

For each region it also prints where the expected dune wavelength is, pulled
from the ground-truth spreadsheet, so you can eyeball whether the FFT peak lines
up with the hand-measured lambda.

Usage:
    python multi_region_diagnostic.py
"""
import os, sys, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fft_full_geotiff_multimode import (
    load_full_image, high_pass_filter, apply_hann_window, compute_fft_power,
    polar_power_histogram_logk, radial_spectrum_directional,
    sigma_px_from_cutoff, MIN_LAMBDA_M, MAX_LAMBDA_M_GLOBAL,
    N_ANGLE_BINS, N_RADIAL_BINS,
)

EE_PY = sys.executable  # use the same (dunes) python we're running under
DOWNLOAD = "download_s2_box_SZA_luminance.py"
EE_PROJECT = "ee-alanaarchbold"
CUTOFF = 3000.0   # the cleaner cutoff identified from the AlNafud diagnostic
HALF = 0.125

def tile_name(region, lat, lon):
    r = region.replace(" ", "_")
    return f"{r}_lat{lat:+05.2f}_lon{lon:+06.2f}".replace(".", "p").replace("+", "")

def ensure_tile(region, lat, lon, tiles_dir="tiles_diag"):
    os.makedirs(tiles_dir, exist_ok=True)
    tif = os.path.join(tiles_dir, tile_name(region, lat, lon) + ".tif")
    if os.path.exists(tif):
        return tif
    cmd = [EE_PY, DOWNLOAD,
           "--min_lat", str(lat-HALF), "--max_lat", str(lat+HALF),
           "--min_lon", str(lon-HALF), "--max_lon", str(lon+HALF),
           "--start_date", "2018-01-01", "--end_date", "2022-12-31",
           "--out_path", tif, "--scale", "20.0",
           "--max_cloud", "30.0", "--min_sza", "30.0",
           "--ee_project", EE_PROJECT]
    print(f"  downloading {region}...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(tif):
        print(f"    FAILED: {r.stderr[-300:]}")
        return None
    return tif

def spectrum(tif):
    img, pix = load_full_image(tif)
    ny, nx = img.shape
    lam_max = min(MAX_LAMBDA_M_GLOBAL, 0.7 * min(ny, nx) * pix)
    sigma = sigma_px_from_cutoff(CUTOFF, pix)
    win = apply_hann_window(high_pass_filter(img, sigma))
    P, kx, ky = compute_fft_power(win, pix)
    tc, kc, H = polar_power_histogram_logk(
        P, kx, ky, n_angle_bins=N_ANGLE_BINS, n_radial_bins=N_RADIAL_BINS,
        min_lambda_m=MIN_LAMBDA_M, max_lambda_m=lam_max)
    S_k, _ = radial_spectrum_directional(tc, kc, H)
    lam = 1.0 / kc
    o = np.argsort(lam)
    return lam[o], S_k[o]

def truth_lambda(region):
    """Median hand-measured lambda for this region, from the spreadsheet."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook("dune_migration_and_lambda.xlsx")
        ws = wb["Sheet1"]
        hdr = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column+1)]
        col = {n: i+1 for i, n in enumerate(hdr) if n}
        vals = []
        for r in range(2, ws.max_row+1):
            if str(ws.cell(row=r, column=col["region"]).value) == region:
                v = ws.cell(row=r, column=col["actual lambda (m)"]).value
                if isinstance(v, (int, float)):
                    vals.append(v)
        return float(np.median(vals)) if vals else None
    except Exception as e:
        return None

def main():
    pts = pd.read_csv("diag_points.csv")
    n = len(pts)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.2*nrow))
    axes = axes.ravel()
    for ax, (_, row) in zip(axes, pts.iterrows()):
        region, lat, lon = row["region"], row["lat"], row["lon"]
        tif = ensure_tile(region, lat, lon)
        if tif is None:
            ax.set_title(f"{region} (download failed)")
            continue
        lam, S = spectrum(tif)
        ax.plot(lam, S, "-")
        ax.set_xscale("log")
        ax.set_xlabel("wavelength (m)"); ax.set_ylabel("dir. power")
        tl = truth_lambda(region)
        title = f"{region}"
        if tl:
            ax.axvline(tl, color="green", ls="--", alpha=0.7)
            title += f"  (measured lambda~{tl:.0f} m, green)"
        # mark the strongest peak
        imax = int(np.argmax(S))
        ax.axvline(lam[imax], color="red", ls=":", alpha=0.6)
        title += f"\nstrongest FFT peak={lam[imax]:.0f} m (red)"
        ax.set_title(title, fontsize=9)
        ax.grid(True, which="both", ls=":", alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Per-region spectra (HP cutoff {CUTOFF:.0f} m). "
                 "green=hand-measured dune lambda, red=strongest FFT peak", fontsize=11)
    fig.tight_layout()
    out = "multi_region_spectrum.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")

if __name__ == "__main__":
    main()

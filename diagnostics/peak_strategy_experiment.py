#!/usr/bin/env python3
"""
peak_strategy_experiment.py

Compare four dune-wavelength selection strategies against the hand-measured
ground truth, running the FFT live on downloaded tiles.

Strategies
----------
  1. strongest        : most powerful radial peak (current behaviour; baseline)
  2. nearest_tile     : peak nearest THIS tile's Excel lambda; skip if none within TOL
  3. nearest_region   : peak nearest the REGION-MEDIAN Excel lambda; skip if none within TOL
  4. most_directional : the prominent peak whose power is most concentrated in
                        orientation (autonomous -- uses NO Excel value)

Strategies 2 and 3 use the answer key and are VALIDATION/ceiling measures.
Strategy 4 uses only the image and is the candidate autonomous method.
If 4 approaches 2/3, orientation alone recovers the dune peak.

Usage
-----
  python peak_strategy_experiment.py --tiles_dir tiles --truth dune_migration_and_lambda.xlsx --tol 0.25
"""
import os, argparse
import numpy as np
import pandas as pd
import openpyxl
from scipy.signal import find_peaks, peak_prominences

from fft_full_geotiff_multimode import (
    load_full_image, high_pass_filter, apply_hann_window, compute_fft_power,
    polar_power_histogram_logk, sigma_px_from_cutoff,
    MIN_LAMBDA_M, MAX_LAMBDA_M_GLOBAL, N_ANGLE_BINS, N_RADIAL_BINS,
)

HP_CUTOFF = 3000.0   # cleaner cutoff identified from diagnostics
PROM_FRAC_MIN = 0.15
WEDGE_BINS = 4       # +/- angle bins around a peak's dominant orientation

def tile_name(region, lat, lon):
    r = region.replace(" ", "_")
    return f"{r}_lat{lat:+05.2f}_lon{lon:+06.2f}".replace(".", "p").replace("+", "")

def load_truth(xlsx):
    wb = openpyxl.load_workbook(xlsx); ws = wb["Sheet1"]
    hdr = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column+1)]
    col = {n: i+1 for i, n in enumerate(hdr) if n}
    rows = []
    for r in range(2, ws.max_row+1):
        lat = ws.cell(row=r, column=col["lat"]).value
        if lat is None: continue
        fill = ws.cell(row=r, column=col["FFT lambda"]).fill
        tag = fill.start_color.rgb if fill and fill.patternType else "NONE"
        colour = "yellow" if tag == "FFFFFF00" else ("green" if tag == "FF92D050" else "none")
        rows.append(dict(
            region=str(ws.cell(row=r, column=col["region"]).value),
            lat=float(lat), lon=float(ws.cell(row=r, column=col["lon"]).value),
            actual=ws.cell(row=r, column=col["actual lambda (m)"]).value,
            colour=colour))
    df = pd.DataFrame(rows)
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    return df

def extract_peaks(tif):
    """Return list of peaks: dict(lambda_m, power, prom_frac, dir_conc)."""
    img, pix = load_full_image(tif)
    ny, nx = img.shape
    lam_max = min(MAX_LAMBDA_M_GLOBAL, 0.7 * min(ny, nx) * pix)
    sigma = sigma_px_from_cutoff(HP_CUTOFF, pix)
    win = apply_hann_window(high_pass_filter(img, sigma))
    P, kx, ky = compute_fft_power(win, pix)
    tc, kc, H = polar_power_histogram_logk(
        P, kx, ky, n_angle_bins=N_ANGLE_BINS, n_radial_bins=N_RADIAL_BINS,
        min_lambda_m=MIN_LAMBDA_M, max_lambda_m=lam_max)
    if H is None: return []
    S_k = H.sum(axis=0)                      # full radial spectrum
    idx, _ = find_peaks(S_k)
    if len(idx) == 0: return []
    prom, _, _ = peak_prominences(S_k, idx)
    heights = S_k[idx]
    pf = prom / (heights + 1e-8)
    main = heights.max()
    peaks = []
    for j, ik in enumerate(idx):
        if kc[ik] <= 0: continue
        lam = 1.0 / kc[ik]
        if lam < MIN_LAMBDA_M or lam > lam_max: continue
        if pf[j] < PROM_FRAC_MIN: continue
        # orientation concentration at this radial bin: peak-of-theta / mean-of-theta
        col = H[:, ik]
        s = col.sum()
        dir_conc = (col.max() / (s / len(col))) if s > 0 else 0.0  # >1 means concentrated
        peaks.append(dict(lambda_m=lam, power=float(heights[j]),
                          rel_power=float(heights[j]/ (main+1e-8)),
                          prom_frac=float(pf[j]), dir_conc=float(dir_conc)))
    return peaks

def pick_strongest(peaks):
    return max(peaks, key=lambda p: p["power"])["lambda_m"] if peaks else None

def pick_nearest(peaks, target, tol):
    if not peaks or target is None or not np.isfinite(target): return None
    best = min(peaks, key=lambda p: abs(p["lambda_m"] - target))
    if abs(best["lambda_m"] - target) / target <= tol:
        return best["lambda_m"]
    return None   # skip

def pick_most_directional(peaks):
    if not peaks: return None
    # require reasonable prominence, then max orientation concentration
    return max(peaks, key=lambda p: p["dir_conc"])["lambda_m"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_dir", default="tiles")
    ap.add_argument("--truth", default="dune_migration_and_lambda.xlsx")
    ap.add_argument("--tol", type=float, default=0.25)
    args = ap.parse_args()

    truth = load_truth(args.truth)
    region_median = truth.groupby("region")["actual"].median().to_dict()

    # only dune rows (yellow/none); green = should-abstain, tracked separately
    dune = truth[truth["colour"].isin(["yellow", "none"])].copy()

    results = {k: {"correct":0,"picked":0,"total":0}
               for k in ["strongest","nearest_tile","nearest_region","most_directional"]}
    per_region = {}

    for _, row in dune.iterrows():
        tif = os.path.join(args.tiles_dir, tile_name(row["region"], row["lat"], row["lon"]) + ".tif")
        if not os.path.exists(tif):
            continue
        peaks = extract_peaks(tif)
        actual = row["actual"]
        if actual is None or not np.isfinite(actual):
            continue

        picks = {
            "strongest":        pick_strongest(peaks),
            "nearest_tile":     pick_nearest(peaks, actual, args.tol),
            "nearest_region":   pick_nearest(peaks, region_median.get(row["region"]), args.tol),
            "most_directional": pick_most_directional(peaks),
        }
        pr = per_region.setdefault(row["region"], {k:{"c":0,"t":0} for k in picks})
        for k, val in picks.items():
            results[k]["total"] += 1
            pr[k]["t"] += 1
            if val is not None:
                results[k]["picked"] += 1
                if abs(val - actual)/actual <= args.tol:
                    results[k]["correct"] += 1
                    pr[k]["c"] += 1

    print(f"\n=== Strategy comparison (tol={args.tol:.0%}) ===")
    print(f"{'strategy':18s} {'accuracy':>9} {'picked':>8} {'acc|picked':>11}")
    for k, d in results.items():
        acc = d["correct"]/d["total"]*100 if d["total"] else 0
        pick_rate = d["picked"]/d["total"]*100 if d["total"] else 0
        acc_when = d["correct"]/d["picked"]*100 if d["picked"] else 0
        print(f"{k:18s} {acc:8.1f}% {pick_rate:7.1f}% {acc_when:10.1f}%")
    print(f"\n(total dune tiles scored: {results['strongest']['total']})")

    print("\nper-region accuracy (most_directional vs nearest_tile):")
    for reg in sorted(per_region):
        md = per_region[reg]["most_directional"]
        nt = per_region[reg]["nearest_tile"]
        a = md["c"]/md["t"]*100 if md["t"] else 0
        b = nt["c"]/nt["t"]*100 if nt["t"] else 0
        print(f"  {reg:26s} directional={a:5.1f}%  nearest_tile={b:5.1f}%  (n={md['t']})")

if __name__ == "__main__":
    main()

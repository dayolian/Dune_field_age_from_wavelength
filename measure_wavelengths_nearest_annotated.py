#!/usr/bin/env python3
"""
measure_wavelengths_nearest.py  --  ANNOTATED

Your code is unchanged. Everything added is a comment. Comments beginning with
"FLAG" mark something worth acting on.

Original docstring follows.
=============================================================================

Production wavelength extraction using the 'nearest_tile' strategy.

For each tile:
  - run the FFT, extract all prominent radial peaks (with orientation/azimuth)
  - choose the peak nearest the hand-measured wavelength for that tile
  - if the nearest peak is within TOL, mark high-confidence;
    otherwise still output it, marked low-confidence
  - record ALL candidate peaks so every choice is auditable

NOTE ON METHOD: this uses the hand-measured lambda to select which FFT peak is
the dune. The output is therefore an FFT-REFINED wavelength guided by the field
estimate, not an independent prediction. It is a validation/refinement tool.

    ^^^ This paragraph is the single most important thing in the file and it
    belongs in the paper's methods section verbatim. A reviewer who works it
    out unaided will treat it far worse than one who was told plainly.

Output CSV columns (one row per tile):
  region, lat, lon, tile_basename,
  hand_lambda_m,               # the field estimate used as target
  chosen_lambda_m,             # FFT peak nearest the hand value
  chosen_crest_az_deg,         # crest azimuth of the chosen peak
  chosen_prom_frac, chosen_rel_power,
  within_tol, confidence,      # high/low
  pct_diff_from_hand,          # |chosen - hand| / hand * 100
  n_candidates,
  all_candidates               # "lambda@az(prom);lambda@az(prom);..." for audit
"""
import os, csv, argparse
import numpy as np
import pandas as pd
import openpyxl                       # reads the truth workbook INCLUDING cell
                                      # fill colours, which pandas cannot do
from scipy.signal import find_peaks, peak_prominences

# Everything spectral is imported from step 4 rather than reimplemented, so the
# two scripts cannot drift apart.
from fft_full_geotiff_multimode import (
    load_full_image, high_pass_filter, apply_hann_window, compute_fft_power,
    polar_power_histogram_logk, radial_spectrum_directional,
    theta_to_crest_azimuth, sigma_px_from_cutoff,
    MIN_LAMBDA_M, MAX_LAMBDA_M_GLOBAL, N_ANGLE_BINS, N_RADIAL_BINS,
)

# FLAG: THIS IS THE COPY THAT STILL HAS THE OLD VALUES.
# fft_full_geotiff_multimode.py uses HP_CUTOFF_LAMBDA_M = 6000.0 and
# PROM_FRAC_MIN = 0.20. This file overrides both with 3000.0 and 0.15, so the
# two scripts filter differently. Your working copy was sed-patched to 6000
# and 0.20; the uploaded one was not. Check which is on disk before running,
# because changing the cutoff changes every wavelength.
HP_CUTOFF = 3000.0
PROM_FRAC_MIN = 0.15


def tile_name(region, lat, lon):
    # Rebuilds the exact filename that step 2 wrote, so the tile can be found
    # on disk from its coordinates alone. Example:
    #   ("Taklamakan", 40.25, 86.50) -> "Taklamakan_lat40p25_lon86p50"
    # The chain is: pad and sign the numbers, turn "." into "p" because dots
    # confuse file extensions, then drop the "+" so northern and eastern
    # coordinates have no prefix.
    # FLAG: southern and western coordinates keep their minus sign, so Namib
    # and Western Sahara tiles look like "lat-24p50". Fine, just be aware the
    # naming is asymmetric.
    r = region.replace(" ", "_")
    return f"{r}_lat{lat:+05.2f}_lon{lon:+06.2f}".replace(".", "p").replace("+", "")


def load_truth(xlsx):
    # Reads your hand-measurement workbook. The reason this uses openpyxl
    # rather than pandas is the fill colours: the green/yellow tagging is
    # formatting, not data, and pandas discards it.
    wb = openpyxl.load_workbook(xlsx); ws = wb["Sheet1"]

    # Build a name -> column-number map from the header row, so the code does
    # not depend on column order.
    hdr = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
    col = {n: i + 1 for i, n in enumerate(hdr) if n}

    rows = []
    for r in range(2, ws.max_row + 1):
        lat = ws.cell(row=r, column=col["lat"]).value
        if lat is None:                       # stop at the first blank row
            continue

        # Read the FILL COLOUR of the "FFT lambda" cell, column D.
        # FF92D050 is the green you used for no dunes, linear dunes, or
        # unresolvable imagery. FFFFFF00 is yellow. Anything else is "none".
        # FLAG: hardcoded hex. If Excel ever re-saves these with a theme
        # colour instead of an explicit RGB, every tile silently becomes
        # "none" and the 272 exclusions vanish without warning. Worth a
        # sanity check on the counts each run.
        fill = ws.cell(row=r, column=col["FFT lambda"]).fill
        tag = fill.start_color.rgb if fill and fill.patternType else "NONE"
        colour = "yellow" if tag == "FFFFFF00" else ("green" if tag == "FF92D050" else "none")

        rows.append(dict(
            region=str(ws.cell(row=r, column=col["region"]).value),
            lat=float(lat), lon=float(ws.cell(row=r, column=col["lon"]).value),
            hand=ws.cell(row=r, column=col["actual lambda (m)"]).value,
            colour=colour))

    df = pd.DataFrame(rows)
    # Anything non-numeric in the hand column becomes NaN rather than crashing.
    df["hand"] = pd.to_numeric(df["hand"], errors="coerce")
    return df


def extract_peaks(tif):
    """All prominent radial peaks with orientation/azimuth, using directional spectrum."""
    # ---- the spectral pipeline, six stages ----

    # 1. Read the GeoTIFF. Returns the image array and the pixel size in
    #    metres, which is 20 for Sentinel-2 luminance.
    img, pix = load_full_image(tif)
    ny, nx = img.shape

    # 2. Cap the longest measurable wavelength. You cannot resolve a spacing
    #    longer than the tile, so this takes the smaller of the global cap and
    #    70 per cent of the tile's shortest side. The 0.7 leaves room for at
    #    least one full wave rather than half of one.
    lam_max = min(MAX_LAMBDA_M_GLOBAL, 0.7 * min(ny, nx) * pix)

    # 3. High-pass filter. Converts the cutoff wavelength into a Gaussian
    #    blur width in pixels, then subtracts the blurred image to remove
    #    regional brightness trend and topography, leaving the dune texture.
    sigma = sigma_px_from_cutoff(HP_CUTOFF, pix)

    # 4. Hann window, then 2D FFT. The window tapers the edges to zero so the
    #    hard tile boundary does not inject spurious frequencies into the
    #    spectrum, which would otherwise appear as a cross artefact.
    win = apply_hann_window(high_pass_filter(img, sigma))
    P, kx, ky = compute_fft_power(win, pix)

    # 5. Re-bin the 2D power into polar coordinates: angle bins by orientation,
    #    radial bins logarithmically spaced in wavenumber. Logarithmic because
    #    dune spacings span orders of magnitude and linear bins would waste
    #    almost all resolution at the short end.
    tc, kc, H = polar_power_histogram_logk(
        P, kx, ky, n_angle_bins=N_ANGLE_BINS, n_radial_bins=N_RADIAL_BINS,
        min_lambda_m=MIN_LAMBDA_M, max_lambda_m=lam_max)
    if H is None:
        return []

    # 6. Collapse to a 1D spectrum, power against wavenumber. "directional"
    #    means it weights toward the dominant orientation rather than
    #    averaging all directions equally, which matters for linear dunes
    #    where nearly all the signal sits in one direction.
    S_k, _ = radial_spectrum_directional(tc, kc, H)

    # ---- find and filter the peaks ----
    idx, _ = find_peaks(S_k)              # every local maximum
    if len(idx) == 0:
        return []

    # Prominence is how far a peak stands above the surrounding baseline. A
    # tall peak on a tall shoulder is less meaningful than a modest peak
    # rising from nothing, and prominence captures that.
    prom, _, _ = peak_prominences(S_k, idx)
    heights = S_k[idx]

    # Express prominence as a FRACTION of the peak's own height, so the test
    # is scale-free across tiles with different overall brightness.
    # The 1e-8 just avoids dividing by zero.
    pf = prom / (heights + 1e-8)
    main = heights.max()                  # tallest peak, used for rel_power

    peaks = []
    for j, ik in enumerate(idx):
        if kc[ik] <= 0:                   # skip the zero-frequency bin
            continue
        lam = 1.0 / kc[ik]                # wavenumber to wavelength, in metres

        # Three rejection tests: too short to resolve, too long for the tile,
        # or not prominent enough to be a real spectral feature.
        if lam < MIN_LAMBDA_M or lam > lam_max or pf[j] < PROM_FRAC_MIN:
            continue

        # Crest azimuth. Within this radial bin, find which orientation bin
        # holds the most power, then convert that spectral angle into the
        # compass bearing of the dune CREST. Note this is the crest direction,
        # not the transport direction, and the two are perpendicular.
        col = H[:, ik]
        ith = int(np.argmax(col)) if col.max() > 0 else 0
        _, crest_az = theta_to_crest_azimuth(np.degrees(tc[ith]))

        peaks.append(dict(
            lambda_m=lam, crest_az=crest_az,
            power=float(heights[j]),
            rel_power=float(heights[j] / (main + 1e-8)),   # 1.0 = tallest peak
            prom_frac=float(pf[j])))

    peaks.sort(key=lambda p: p["lambda_m"])   # shortest first, for readability
    return peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_dir", default="tiles")
    ap.add_argument("--truth", default="dune_migration_and_lambda.xlsx")
    # FLAG: this default points at the STALE June truth file. Always pass
    # --truth explicitly, or change the default to the corrected workbook.
    # Running without it silently uses 250 green tags instead of 272.
    ap.add_argument("--out", default="results/final_wavelengths.csv")

    # --tol does NOT affect which peak is chosen. It only decides whether the
    # chosen peak is labelled high or low confidence. Changing it changes the
    # reported percentages and nothing else.
    ap.add_argument("--tol", type=float, default=0.25)
    args = ap.parse_args()

    truth = load_truth(args.truth)

    # THE EXCLUSION STEP. Keep yellow and uncoloured rows, drop green.
    # This is where 1005 becomes 733.
    # dune rows only; green (no-dune) tiles are excluded from measurement
    dune = truth[truth["colour"].isin(["yellow", "none"])].copy()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = ["region", "lat", "lon", "tile_basename", "hand_lambda_m",
              "chosen_lambda_m", "chosen_crest_az_deg", "chosen_prom_frac",
              "chosen_rel_power", "within_tol", "confidence",
              "pct_diff_from_hand", "n_candidates", "all_candidates"]

    n_written = n_high = n_low = n_missing = 0

    # Writing row by row as it goes, rather than building a table and saving at
    # the end. That is why you can watch the file grow with wc -l during a run,
    # and why a crash leaves a partial file rather than nothing.
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for _, row in dune.iterrows():
            base = tile_name(row["region"], row["lat"], row["lon"])
            tif = os.path.join(args.tiles_dir, base + ".tif")
            hand = row["hand"]

            # Skip if the tile was never downloaded, or the hand measurement is
            # blank or non-numeric. Counted but not written.
            if not os.path.exists(tif) or hand is None or not np.isfinite(hand):
                n_missing += 1
                continue

            # FLAG: no try/except here. One corrupt GeoTIFF aborts the whole
            # 25-40 minute run. Your working copy was patched to catch this and
            # print "SKIP unreadable tile"; the uploaded copy was not.
            peaks = extract_peaks(tif)
            if not peaks:
                n_missing += 1
                continue

            # ---- THE SELECTION. One line, and the methodological crux. ----
            # Of all candidate peaks, take the one closest in metres to your
            # hand measurement. Note this is ABSOLUTE difference, not relative,
            # so at long wavelengths a peak 300 m away can beat one 20 per cent
            # away. Worth being aware of for the long-wavelength fields.
            # choose peak nearest the hand value
            chosen = min(peaks, key=lambda p: abs(p["lambda_m"] - hand))

            # How far off it landed, as a percentage of the hand value.
            pct = abs(chosen["lambda_m"] - hand) / hand * 100
            within = pct <= args.tol * 100
            conf = "high" if within else "low"
            if within: n_high += 1
            else:      n_low += 1

            # Every rejected candidate is recorded too, formatted as
            # "1700@045(p0.32);850@045(p0.21)". This is the audit trail: it
            # lets you check afterwards whether the chosen peak was an obvious
            # winner or a coin flip between two similar candidates.
            cand_str = ";".join(
                f"{p['lambda_m']:.0f}@{p['crest_az']:.0f}(p{p['prom_frac']:.2f})"
                for p in peaks)

            w.writerow(dict(
                region=row["region"], lat=row["lat"], lon=row["lon"],
                tile_basename=base,
                hand_lambda_m=round(hand, 1),
                chosen_lambda_m=round(chosen["lambda_m"], 1),
                chosen_crest_az_deg=round(chosen["crest_az"], 1),
                chosen_prom_frac=round(chosen["prom_frac"], 3),
                chosen_rel_power=round(chosen["rel_power"], 3),
                within_tol=within, confidence=conf,
                pct_diff_from_hand=round(pct, 1),
                n_candidates=len(peaks),
                all_candidates=cand_str))
            n_written += 1

    # The summary you read after every run. n_written should be 733 with the
    # corrected truth file, and n_missing should be 0.
    print(f"Wrote {n_written} tiles to {args.out}")
    print(f"  high-confidence (within {args.tol:.0%}): {n_high}  ({n_high/max(n_written,1)*100:.1f}%)")
    print(f"  low-confidence  (nearest but > tol):   {n_low}  ({n_low/max(n_written,1)*100:.1f}%)")
    print(f"  tiles missing/no-peak (not written):   {n_missing}")


if __name__ == "__main__":
    main()

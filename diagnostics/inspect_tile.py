#!/usr/bin/env python3
"""
inspect_tile.py — look at ONE real tile's radial spectrum to see where the
dune peak actually is and why the wrong mode is winning.

Usage:
    python inspect_tile.py tiles/AlNafudDesert_lat27p50_lon39p25.tif
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from fft_full_geotiff_multimode import (
    load_full_image, high_pass_filter, apply_hann_window, compute_fft_power,
    polar_power_histogram_logk, radial_spectrum_directional,
    sigma_px_from_cutoff, HP_CUTOFF_LAMBDA_M, MIN_LAMBDA_M, MAX_LAMBDA_M_GLOBAL,
    N_ANGLE_BINS, N_RADIAL_BINS,
)

tif = sys.argv[1]
img, pix = load_full_image(tif)
ny, nx = img.shape
window_m = min(ny, nx) * pix
lam_max = min(MAX_LAMBDA_M_GLOBAL, 0.7 * window_m)
print(f"tile: {tif}")
print(f"shape {ny}x{nx}, pixel {pix:.1f} m, lambda search 30-{lam_max:.0f} m")

# Try several high-pass cutoffs and show the radial spectrum for each
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
cutoffs = [6000, 3000, 1500, 800]   # meters; smaller = removes more long-wavelength

for ax, cutoff in zip(axes.ravel(), cutoffs):
    sigma = sigma_px_from_cutoff(cutoff, pix)
    hp = high_pass_filter(img, sigma)
    win = apply_hann_window(hp)
    P, kx, ky = compute_fft_power(win, pix)
    tc, kc, H = polar_power_histogram_logk(
        P, kx, ky, n_angle_bins=N_ANGLE_BINS, n_radial_bins=N_RADIAL_BINS,
        min_lambda_m=MIN_LAMBDA_M, max_lambda_m=lam_max)
    S_k, i_theta = radial_spectrum_directional(tc, kc, H)
    lam = 1.0 / kc
    order = np.argsort(lam)
    ax.plot(lam[order], S_k[order], "-")
    ax.set_xscale("log")
    ax.set_xlabel("wavelength (m)")
    ax.set_ylabel("directional power")
    ax.set_title(f"HP cutoff {cutoff} m (sigma={sigma:.0f}px)")
    ax.axvspan(600, 800, color="green", alpha=0.15)   # rough real-dune band for AlNafud
    ax.axvline(100, color="red", ls="--", alpha=0.5)   # the spike location
    ax.grid(True, which="both", ls=":", alpha=0.4)

fig.suptitle("Where is the dune peak? green=expected dune band, red=100m spike")
fig.tight_layout()
out = "tile_spectrum_diagnostic.png"
fig.savefig(out, dpi=140)
print(f"saved {out}")

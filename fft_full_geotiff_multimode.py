#!/usr/bin/env python3
"""
fft_full_geotiff_multimode.py

Run 2D FFT on an entire GeoTIFF (single band), detect multiple dune modes
(primary + secondary), wavelength-first, with log-spaced radial bins.

This is a full-image version of the earlier patch-based script:
no lat/lon center, no window size; just use the whole raster.

Usage:

  python fft_full_geotiff_multimode.py \
    --tif data/rubalkali_mosaic_2020_30m.tif \
    --out_prefix outputs/rubalkali_full
"""

import argparse
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, peak_prominences
import matplotlib.pyplot as plt

# ----------------- Tunable parameters -----------------

GAUSS_SIGMA_HP = 10.0      # pixels; high-pass filter sigma - Kenzie this deemphasizes the small wavelengths 
N_ANGLE_BINS = 72           # 5 degree bins 
N_RADIAL_BINS = 100       # in log(k) space

MAX_SCALES = 2           # max number of wavelength modes to keep
REL_POWER_MIN = 0.2       # radial peak must be >= this fraction of main radial peak

# Wavelength range of interest (meters); upper bound will also be
# limited by the image size in main()
MIN_LAMBDA_M = 30.0       # ignore ultra-small stuff
MAX_LAMBDA_M_GLOBAL = 10000.0  # global cap; actual max set per-image


# ----------------- Basic helpers -----------------

def high_pass_filter(arr, sigma_px=3.0):
    if sigma_px <= 0:
        return arr
    blurred = gaussian_filter(arr, sigma=sigma_px)
    hp = arr - blurred
    hp = (hp - hp.min()) / (hp.max() - hp.min() + 1e-8)
    return hp


def apply_hann_window(arr):
    ny, nx = arr.shape
    wy = np.hanning(ny)
    wx = np.hanning(nx)
    window = np.outer(wy, wx)
    return arr * window


def compute_fft_power(arr, pixel_size_m):
    ny, nx = arr.shape
    F = np.fft.fft2(arr)
    P = np.abs(F) ** 2
    P_shift = np.fft.fftshift(P)

    fx = np.fft.fftfreq(nx, d=pixel_size_m)
    fy = np.fft.fftfreq(ny, d=pixel_size_m)
    fx_shift = np.fft.fftshift(fx)
    fy_shift = np.fft.fftshift(fy)

    kx, ky = np.meshgrid(fx_shift, fy_shift)  # cycles per meter
    return P_shift, kx, ky


def polar_power_histogram_logk(P, kx, ky,
                               n_angle_bins=180,
                               n_radial_bins=120,
                               min_lambda_m=MIN_LAMBDA_M,
                               max_lambda_m=MAX_LAMBDA_M_GLOBAL):
    """
    Build a 2D histogram H(theta, k) with:
      - theta: orientation angle in [-pi/2, pi/2] (180° symmetry)
      - k: radial frequency (cycles/m) with **log-spaced bins**
    """
    k = np.sqrt(kx**2 + ky**2)

    # image-space crest-normal angle
    theta = np.arctan2(ky, kx)
    theta = np.mod(theta + np.pi/2, np.pi) - np.pi/2

    P_flat = P.ravel()
    k_flat = k.ravel()
    theta_flat = theta.ravel()

    # wavelength limits → k limits
    k_min = 1.0 / max_lambda_m
    k_max = 1.0 / min_lambda_m

    mask = (k_flat > k_min) & (k_flat < k_max) & np.isfinite(P_flat)
    if not np.any(mask):
        return None, None, None

    P_flat = P_flat[mask]
    k_flat = k_flat[mask]
    theta_flat = theta_flat[mask]

    # theta bins (linear)
    theta_edges = np.linspace(-np.pi/2, np.pi/2, n_angle_bins + 1)

    # radial bins in log(k)
    logk_min = np.log10(k_min)
    logk_max = np.log10(k_max)
    k_edges = np.logspace(logk_min, logk_max, n_radial_bins + 1)

    H, theta_edges, k_edges = np.histogram2d(
        theta_flat, k_flat,
        bins=[theta_edges, k_edges],
        weights=P_flat
    )

    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    k_centers = np.sqrt(k_edges[:-1] * k_edges[1:])  # geometric mean
    return theta_centers, k_centers, H


def theta_to_crest_azimuth(theta_deg):
    """
    Convert image-space FFT angle (crest-normal) to geospatial azimuths.

    theta_deg:
        crest-normal angle measured from +x (east), CCW,
        with +y = south (image/FFT convention)

    Returns:
        crest_normal_az : crest-normal azimuth (0°=N, 90°=E, ...)
        crest_az        : crest (ridge) azimuth (0–360°, 0°=N, 90°=E)
    """
    theta_rad = np.radians(theta_deg)

    ux = np.cos(theta_rad)   # east
    uy = np.sin(theta_rad)   # south

    E = ux
    N = -uy   # +south = -north

    crest_normal_az = (np.degrees(np.arctan2(E, N)) + 360.0) % 360.0
    crest_az = (crest_normal_az + 90.0) % 360.0
    return crest_normal_az, crest_az


# ----------------- Multi-mode extraction (wavelength-first) -----------------
def deduplicate_modes_by_wavelength(modes, tol_frac=0.25):
    """
    Merge modes whose wavelengths are very close.

    tol_frac is the *relative* tolerance:
      if |λ2 - λ1| / max(λ1, λ2) < tol_frac,
      we keep only the one with higher rel_power.

    Returns a new, cleaned list of modes.
    """
    if len(modes) <= 1:
        return modes

    # Sort by wavelength
    modes_sorted = sorted(modes, key=lambda m: m["lambda_m"])
    merged = [modes_sorted[0]]

    for m in modes_sorted[1:]:
        last = merged[-1]
        frac_diff = abs(m["lambda_m"] - last["lambda_m"]) / max(m["lambda_m"], last["lambda_m"])
        if frac_diff < tol_frac:
            # Same wavelength family → keep the stronger one
            if m["rel_power"] > last["rel_power"]:
                merged[-1] = m
        else:
            merged.append(m)

    return merged

def extract_modes_wavelength_first(theta_centers, k_centers, H,
                                   max_scales=MAX_SCALES,
                                   rel_power_min=REL_POWER_MIN,
                                   lambda_max_window=None):
    """
    1) Collapse H(theta,k) over theta → S_k(k) (radial spectrum).
    2) Find peaks in S_k(k) (radial modes).
    3) For each radial mode, look at H[:, idx_k] to find best orientation.
    4) Convert k → λ and orientation → crest_az.
    """
    if theta_centers is None or k_centers is None or H is None:
        return [], None, None

    S_theta = H.sum(axis=1)  # orientation spectrum
    S_k = H.sum(axis=0)      # radial spectrum
    
    # --- radial peak detection and filtering ---

    # 1) Find all peaks in the radial spectrum
    peak_idx, _ = find_peaks(S_k)
    if len(peak_idx) == 0:
        return [], None, None
    
        #DEBUG PRINTS 
    #print("Radial peaks (λ, power):")
    #for idx in peak_idx:
    #    k_val = k_centers[idx]
    #    if k_val <= 0:
    #        continue
    #    lam = 1.0 / k_val
    #    print(f"  λ ≈ {lam:.1f} m, S_k = {S_k[idx]:.3f}")

    # 2) Compute peak heights and prominences (absolute, same units as S_k)
    peak_heights = S_k[peak_idx]
    prom, left_bases, right_bases = peak_prominences(S_k, peak_idx)

    # 3) Local prominence fraction (dimensionless, 0–1)
    prom_frac = prom / (peak_heights + 1e-8)

    # 4) For reporting only: main_power for rel_power; not used to filter
    main_power = float(peak_heights.max())

    # 5) Build a list of candidate peaks, assessing *all* peaks
    candidates = []
    ABS_POWER_MIN = 0.0        # or a small floor if you want, e.g. 1e-6
    PROM_FRAC_MIN = 0.20       # tune this: 0.2–0.3 is a good starting point
    # optionally keep REL_POWER_MIN very small or zero if you want to rely mostly on prominence
    # REL_POWER_MIN = 0.0

    for idx_arr, idx_k in enumerate(peak_idx):
        power = float(peak_heights[idx_arr])
        if power <= ABS_POWER_MIN:
            continue

        k_val = k_centers[idx_k]
        if k_val <= 0:
            continue

        lambda_m = 1.0 / k_val

        # respect the global/local wavelength limits
        if lambda_m < MIN_LAMBDA_M:
            continue
        if lambda_max_window is not None and lambda_m > lambda_max_window:
            continue

        # local "peakiness"
        this_prom_frac = float(prom_frac[idx_arr])
        if this_prom_frac < PROM_FRAC_MIN:
            continue

        # optional: super-weak peaks vs main peak, if you still care
        rel_power = power / (main_power + 1e-8)
        # if rel_power < REL_POWER_MIN:
        #     continue

        # store candidate; we will sort later
        candidates.append({
            "idx_k": idx_k,
            "lambda_m": lambda_m,
            "power": power,
            "rel_power": rel_power,
            "prom_frac": this_prom_frac,
        })

    # If nothing survived, bail out
    if not candidates:
        return [], None, None

    # 6) Sort candidates however you like; two good options:
    #    (a) by descending power:
    # candidates.sort(key=lambda c: c["power"], reverse=True)
    #    (b) by descending wavelength (longest scales first):
    candidates.sort(key=lambda c: c["lambda_m"], reverse=True)

    # 7) Now loop over candidates (already filtered), and build modes.
    #    We still respect MAX_SCALES here, *after* filtering.
    modes = []
    for c in candidates:
        if len(modes) >= max_scales:
            break

        idx_k = c["idx_k"]
        lambda_m = c["lambda_m"]
        power = c["power"]
        rel_power = c["rel_power"]
        this_prom_frac = c["prom_frac"]

        # --- your existing orientation / crest azimuth logic here ---
        # choose theta_index for this radial bin, compute theta0_deg, crest_az, etc.
        orient_slice = H[:, idx_k]
        if orient_slice.max() <= 0:
            continue

        idx_theta = np.argmax(orient_slice)
        theta0 = theta_centers[idx_theta]
        theta0_deg = np.degrees(theta0)
        _, crest_az = theta_to_crest_azimuth(theta0_deg)
        # e.g. something like:
        # idx_theta = ...
        # theta0_deg = ...
        # crest_az = ...

        modes.append({
            "theta_image_deg": theta0_deg,
            "crest_az": crest_az,
            "lambda_m": lambda_m,
            "rel_power": rel_power,                 # for reporting
            "spectral_intensity": power,
            "prom_frac": this_prom_frac,           # useful quality metric
            "theta_index": idx_theta,
            "k_index": int(idx_k),
        })

    # done: modes is now filtered by *local* peakiness, not by main-peak comparison

    # Sort by wavelength and merge near-duplicates
    modes.sort(key=lambda m: m["lambda_m"])
    modes = deduplicate_modes_by_wavelength(modes, tol_frac=0.25)
    return modes, S_theta, S_k



# ----------------- GeoTIFF loading (full image) -----------------

def load_full_image(tif_path):
    """
    Load the full first band of a GeoTIFF and normalize it to [0,1]
    Assumes a projected CRS with meter units (e.g., UTM).
    """
    with rasterio.open(tif_path) as src:
        band = src.read(1).astype(np.float32)
        nodata = src.nodata
        pixel_size_x = src.transform.a
        pixel_size_y = -src.transform.e
        pixel_size_m = float((abs(pixel_size_x) + abs(pixel_size_y)) / 2.0)

    if nodata is not None:
        valid = (band != nodata) & np.isfinite(band)
    else:
        valid = np.isfinite(band)

    if not np.any(valid):
        raise RuntimeError("Image contains only nodata.")

    vals = band[valid]
    band[valid] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)

    return band, pixel_size_m


# ----------------- Plotting -----------------

def plot_multimode_results(patch, P_shift, theta_centers, k_centers,
                           S_theta, S_k, modes, out_png):
    """
    2x2 figure:
      - Image patch (here: full image)
      - FFT log power
      - Orientation spectrum S(theta)
      - Radial spectrum S(k) → λ vs power (log λ) with peaks
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # 1) Image
    ax0 = axes[0, 0]
    ax0.imshow(patch, cmap="gray")
    ax0.set_title("Image (full band)")
    ax0.set_xticks([])
    ax0.set_yticks([])

    # 2) FFT magnitude
    ax1 = axes[0, 1]
    im_fft = ax1.imshow(np.log10(P_shift + 1e-8), cmap="inferno") #was "vidris"
    ax1.set_title("FFT log10(power)")
    ax1.set_xticks([])
    ax1.set_yticks([])
    fig.colorbar(im_fft, ax=ax1, shrink=0.8)

    # Overlay crest *orientation* (ridge direction) lines
    #ny, nx = patch.shape
    #cx, cy = nx / 2.0, ny / 2.0
    #R = min(cx, cy) * 0.9

    #for i, m in enumerate(modes):
    #    # theta_image_deg is crest-normal in image coordinates
    #    theta_norm_deg = m["theta_image_deg"]

        # Crest orientation in image coords = crest-normal rotated by 90°
        # (sign doesn't matter for a line; +90° vs -90° just flips direction)
    #    crest_image_deg = theta_norm_deg + 90.0
    #    crest_image_rad = np.radians(crest_image_deg)

     #   dx = R * np.cos(crest_image_rad)
     #   dy = R * np.sin(crest_image_rad)

      #  ax1.plot(
      #      [cx - dx, cx + dx],
      #      [cy - dy, cy + dy],
      #      "-",
      #      linewidth=1.5,
      #      label=f"mode {i+1}"
      #  )

    if len(modes) > 0:
        ax1.legend(loc="lower left", fontsize=8)

    # 3) Orientation spectrum
    ax2 = axes[1, 0]
    if S_theta is not None and theta_centers is not None:
        ax2.plot(np.degrees(theta_centers), S_theta, "-")
        ax2.set_xlabel("θ (image-space crest-normal, deg)")
        ax2.set_ylabel("Power (sum over k)")
        ax2.set_title("Orientation spectrum S(θ)")

        used_theta_idx = sorted({m["theta_index"] for m in modes})
        for idx in used_theta_idx:
            ax2.axvline(
                np.degrees(theta_centers[idx]),
                color="r",
                linestyle="--",
                alpha=0.5,
            )

    # 4) Radial spectrum → λ vs power
    ax3 = axes[1, 1]
    ax3.set_title("Radial spectrum S(k) → λ vs power")

    if S_k is not None and k_centers is not None:
        k_pos = k_centers > 0
        k_vals = k_centers[k_pos]
        S_vals = S_k[k_pos]

        lambda_vals = 1.0 / k_vals
        order = np.argsort(lambda_vals)
        lambda_vals = lambda_vals[order]
        S_vals = S_vals[order]

        ax3.plot(lambda_vals, S_vals, "-")
        ax3.set_xscale("log")
        ax3.set_xlabel("Wavelength λ (m, log scale)")
        ax3.set_ylabel("Power (sum over θ)")

        #for i, m in enumerate(modes):
        #    ax3.axvline(
        #        m["lambda_m"],
        #        color=f"C{i}",
        #        linestyle="--",
        #        alpha=0.7,
        #        label=f"λ={m['lambda_m']:.0f} m, mode {i+1}"
        #    )
        if len(modes) > 0:
            ax3.legend(fontsize=8)

        # keep x-range within physically meaningful dune scales
        ax3.set_xlim(MIN_LAMBDA_M, None)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ----------------- CLI + main -----------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="FFT dune analysis (multiple modes, wavelength-first) on full GeoTIFF."
    )
    ap.add_argument("--tif", required=True, help="Input GeoTIFF (single-band or use band 1)")
    ap.add_argument("--out_prefix", type=str, required=True,
                    help="Prefix for output files (PNG, etc.)")
    return ap.parse_args()


def main():
    args = parse_args()

    print("Loading full image...")
    img, pixel_size_m = load_full_image(args.tif)
    ny, nx = img.shape
    print(f"Image shape: {ny} x {nx} px, pixel size: {pixel_size_m:.2f} m")

    # effective window length in meters (use min dimension to avoid weird λ > window)
    window_m = min(ny, nx) * pixel_size_m
    lambda_max_window = min(MAX_LAMBDA_M_GLOBAL, 0.7 * window_m)
    print(f"Using λ range: {MIN_LAMBDA_M:.1f}–{lambda_max_window:.1f} m")

    print("High-pass filtering + apodization...")
    img_hp = high_pass_filter(img, sigma_px=GAUSS_SIGMA_HP)
    img_win = apply_hann_window(img_hp)

    print("Computing FFT...")
    P_shift, kx, ky = compute_fft_power(img_win, pixel_size_m)

    print("Building polar power histogram (log k bins)...")
    theta_c, k_c, H = polar_power_histogram_logk(
        P_shift, kx, ky,
        n_angle_bins=N_ANGLE_BINS,
        n_radial_bins=N_RADIAL_BINS,
        min_lambda_m=MIN_LAMBDA_M,
        max_lambda_m=lambda_max_window,
    )

    if H is None:
        print("No valid k-range for the specified wavelength limits.")
        return

    print("Extracting modes (wavelength-first)...")
    modes, S_theta, S_k = extract_modes_wavelength_first(
        theta_c, k_c, H,
        max_scales=MAX_SCALES,
        rel_power_min=REL_POWER_MIN,
        lambda_max_window=lambda_max_window,
    )

    if len(modes) == 0:
        print("No significant radial modes found (spectrum may be noisy or non-periodic).")
    else:
        print("\nDetected modes (sorted by wavelength):")
        for i, m in enumerate(modes, start=1):
            print(
                f"  Mode {i}: "
                f"λ = {m['lambda_m']:.1f} m, "
                f"crest_az = {m['crest_az']:.1f} deg, "
                f"rel_power = {m['rel_power']:.2f}, "
                f"theta_image = {m['theta_image_deg']:.1f} deg"
            )

    out_png = f"{args.out_prefix}_multimode.png"
    print(f"\nSaving diagnostic figure to: {out_png}")
    plot_multimode_results(
        img, P_shift, theta_c, k_c,
        S_theta, S_k, modes, out_png
    )
    print("Done.")
    

def analyze_geotiff_multimode(tif_path, out_prefix=None):
    """
    Convenience wrapper: run the full-image multimode FFT on tif_path.
    
    Returns:
        modes: list of dicts with keys:
           'lambda_m', 'crest_az', 'rel_power',
           'theta_image_deg', 'theta_index', 'k_index'
    """
    img, pixel_size_m = load_full_image(tif_path)
    ny, nx = img.shape

    window_m = min(ny, nx) * pixel_size_m
    lambda_max_window = min(MAX_LAMBDA_M_GLOBAL, 0.7 * window_m)

    img_hp = high_pass_filter(img, sigma_px=GAUSS_SIGMA_HP)
    img_win = apply_hann_window(img_hp)
    P_shift, kx, ky = compute_fft_power(img_win, pixel_size_m)

    theta_c, k_c, H = polar_power_histogram_logk(
        P_shift, kx, ky,
        n_angle_bins=N_ANGLE_BINS,
        n_radial_bins=N_RADIAL_BINS,
        min_lambda_m=MIN_LAMBDA_M,
        max_lambda_m=lambda_max_window,
    )

    if H is None:
        return []

    modes, S_theta, S_k = extract_modes_wavelength_first(
        theta_c, k_c, H,
        max_scales=MAX_SCALES,
        rel_power_min=REL_POWER_MIN,
        lambda_max_window=lambda_max_window,
    )
    
    print(modes[0].keys())

    # Optionally save a diagnostic figure if out_prefix is given
    if out_prefix is not None and len(modes) > 0:
        out_png = f"{out_prefix}_multimode.png"
        plot_multimode_results(
            img, P_shift, theta_c, k_c,
            S_theta, S_k, modes, out_png
        )

    return modes



if __name__ == "__main__":
    main()


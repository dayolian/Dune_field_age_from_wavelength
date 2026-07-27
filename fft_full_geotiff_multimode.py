#!/usr/bin/env python3
"""
fft_full_geotiff_multimode.py  (PATCHED)

2D FFT on a full GeoTIFF, detect dune wavelength modes.

WHAT CHANGED vs the original (three fixes, ranked by impact on the
lat/lon spreadsheet ground truth):

  FIX 1 (biggest): High-pass sigma was far too small. GAUSS_SIGMA_HP=10 at
    20 m/pixel suppressed everything above ~200 m, but real dune wavelengths
    are ~700-2300 m -- i.e. the filter was deleting the signal. When the real
    peak is gone, find_peaks locks onto a tiny high-frequency noise spike and
    reports lambda ~ 40 m (the Nyquist floor). This is the "actual is much
    LONGER than FFT" failure cluster. Sigma is now set from a physical cutoff
    wavelength (HP_CUTOFF_LAMBDA_M) instead of a hardcoded pixel count.

  FIX 2: Radial spectrum was S_k = H.sum(axis=0), i.e. summed over ALL
    orientations. That drowns the narrow, oriented dune peak in broadband
    noise from every other direction. We now find the dominant orientation
    first and integrate the radial spectrum only within a wedge around it
    (USE_DIRECTIONAL). Uses only the image, so it's deployment-legal.

  FIX 3: Harmonic collapse. Sharp-crested dunes produce a strong peak at
    lambda/2 (a harmonic). The old dedup only merged peaks within 25% of each
    other, so fundamental (1600) and harmonic (800) both survived and the
    harmonic often won. We now demote a peak if a peak near 2x its wavelength
    exists with comparable power. This is the "actual is ~2x FFT" cluster.

Return contract is UNCHANGED: analyze_geotiff_multimode returns a list of
dicts with keys lambda_m, crest_az, rel_power, spectral_intensity, prom_frac,
theta_image_deg, theta_index, k_index -- so run_pipeline_from_points.py needs
no changes.
"""
import argparse
import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, peak_prominences
import matplotlib.pyplot as plt

# ----------------- Tunable parameters -----------------

# --- FIX 1: high-pass defined by a physical cutoff wavelength, not pixels ---
# The high-pass removes structure LONGER than roughly this wavelength. It must
# sit safely ABOVE the longest dune wavelength you expect, or you delete signal.
# sigma_px is derived from this and the pixel size at runtime.
HP_CUTOFF_LAMBDA_M = 6000.0   # keep dune band (<~3000 m) intact; kill only field-scale trend
HP_SIGMA_PX_MIN    = 20.0     # floor so we always remove *some* low-frequency trend
# (old behaviour was GAUSS_SIGMA_HP = 10.0 -> cutoff ~200 m at 20 m/px: far too aggressive)

# --- FIX 2: directional selection ---
USE_DIRECTIONAL       = True
WEDGE_HALFWIDTH_BINS  = 4      # +/- bins around the dominant orientation (5 deg bins -> +/-~25 deg)

# --- FIX 3: harmonic collapse ---
COLLAPSE_HARMONICS    = True
HARMONIC_TOL          = 0.15   # a peak is a "harmonic" if another peak sits within 15% of 2x its lambda
HARMONIC_POWER_FRAC   = 0.30   # ...and that longer peak carries at least 30% of its power

N_ANGLE_BINS = 72             # 5 degree bins
N_RADIAL_BINS = 100           # in log(k) space
MAX_SCALES = 2                # max number of wavelength modes to keep
REL_POWER_MIN = 0.2
PROM_FRAC_MIN = 0.20          # local peakiness threshold

MIN_LAMBDA_M = 30.0
MAX_LAMBDA_M_GLOBAL = 10000.0


# ----------------- Basic helpers -----------------
def sigma_px_from_cutoff(cutoff_lambda_m, pixel_size_m):
    """
    Convert a desired high-pass cutoff WAVELENGTH into a Gaussian sigma in px.
    A Gaussian blur of sigma_px pixels smooths structure up to ~ (2*pi*sigma)
    pixels in wavelength terms; we invert that and clamp to a sane floor.
    """
    if cutoff_lambda_m <= 0:
        return HP_SIGMA_PX_MIN
    cutoff_px = cutoff_lambda_m / pixel_size_m
    sigma_px = cutoff_px / (2.0 * np.pi)
    return float(max(HP_SIGMA_PX_MIN, sigma_px))


def high_pass_filter(arr, sigma_px):
    if sigma_px <= 0:
        return arr
    blurred = gaussian_filter(arr, sigma=sigma_px)
    hp = arr - blurred
    # NOTE: keep the original min-max renormalization for downstream stability.
    hp = (hp - hp.min()) / (hp.max() - hp.min() + 1e-8)
    return hp


def apply_hann_window(arr):
    ny, nx = arr.shape
    return arr * np.outer(np.hanning(ny), np.hanning(nx))


def compute_fft_power(arr, pixel_size_m):
    ny, nx = arr.shape
    F = np.fft.fft2(arr)
    P = np.abs(F) ** 2
    P_shift = np.fft.fftshift(P)
    fx_shift = np.fft.fftshift(np.fft.fftfreq(nx, d=pixel_size_m))
    fy_shift = np.fft.fftshift(np.fft.fftfreq(ny, d=pixel_size_m))
    kx, ky = np.meshgrid(fx_shift, fy_shift)
    return P_shift, kx, ky


def polar_power_histogram_logk(P, kx, ky,
                               n_angle_bins=180,
                               n_radial_bins=120,
                               min_lambda_m=MIN_LAMBDA_M,
                               max_lambda_m=MAX_LAMBDA_M_GLOBAL):
    k = np.sqrt(kx**2 + ky**2)
    theta = np.arctan2(ky, kx)
    theta = np.mod(theta + np.pi/2, np.pi) - np.pi/2

    P_flat, k_flat, theta_flat = P.ravel(), k.ravel(), theta.ravel()
    k_min = 1.0 / max_lambda_m
    k_max = 1.0 / min_lambda_m
    mask = (k_flat > k_min) & (k_flat < k_max) & np.isfinite(P_flat)
    if not np.any(mask):
        return None, None, None
    P_flat, k_flat, theta_flat = P_flat[mask], k_flat[mask], theta_flat[mask]

    theta_edges = np.linspace(-np.pi/2, np.pi/2, n_angle_bins + 1)
    k_edges = np.logspace(np.log10(k_min), np.log10(k_max), n_radial_bins + 1)
    H, theta_edges, k_edges = np.histogram2d(
        theta_flat, k_flat, bins=[theta_edges, k_edges], weights=P_flat
    )
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    k_centers = np.sqrt(k_edges[:-1] * k_edges[1:])
    return theta_centers, k_centers, H


def theta_to_crest_azimuth(theta_deg):
    theta_rad = np.radians(theta_deg)
    ux = np.cos(theta_rad)
    uy = np.sin(theta_rad)
    E = ux
    N = -uy
    crest_normal_az = (np.degrees(np.arctan2(E, N)) + 360.0) % 360.0
    crest_az = (crest_normal_az + 90.0) % 360.0
    return crest_normal_az, crest_az


# ----------------- FIX 2: directional radial spectrum -----------------
def radial_spectrum_directional(theta_centers, k_centers, H,
                                use_directional=USE_DIRECTIONAL,
                                wedge_halfwidth_bins=WEDGE_HALFWIDTH_BINS):
    """
    Return (S_k, dominant_theta_index).

    If use_directional: find the orientation carrying the most total power,
    then integrate the radial spectrum only within a wedge of +/- N angle bins
    around it. Otherwise fall back to the original sum over all theta.
    """
    S_theta = H.sum(axis=1)
    if not use_directional or S_theta.max() <= 0:
        return H.sum(axis=0), int(np.argmax(S_theta)) if S_theta.max() > 0 else None
    i_theta = int(np.argmax(S_theta))
    lo = max(0, i_theta - wedge_halfwidth_bins)
    hi = min(len(theta_centers), i_theta + wedge_halfwidth_bins + 1)
    S_k = H[lo:hi, :].sum(axis=0)
    return S_k, i_theta


# ----------------- FIX 3: harmonic collapse -----------------
def collapse_harmonics(candidates,
                       tol=HARMONIC_TOL,
                       power_frac=HARMONIC_POWER_FRAC):
    """
    Drop a candidate if another candidate sits near 2x its wavelength (the
    fundamental) with at least `power_frac` of its power. Prefers the longer,
    fundamental wavelength over its shorter harmonic.
    """
    if len(candidates) <= 1:
        return candidates
    kept = []
    for c in candidates:
        lam = c["lambda_m"]
        is_harmonic = any(
            (abs(o["lambda_m"] - 2.0 * lam) / (2.0 * lam) < tol)
            and (o["power"] >= power_frac * c["power"])
            for o in candidates if o is not c
        )
        if not is_harmonic:
            kept.append(c)
    return kept if kept else candidates


def deduplicate_modes_by_wavelength(modes, tol_frac=0.25):
    if len(modes) <= 1:
        return modes
    modes_sorted = sorted(modes, key=lambda m: m["lambda_m"])
    merged = [modes_sorted[0]]
    for m in modes_sorted[1:]:
        last = merged[-1]
        frac_diff = abs(m["lambda_m"] - last["lambda_m"]) / max(m["lambda_m"], last["lambda_m"])
        if frac_diff < tol_frac:
            if m["rel_power"] > last["rel_power"]:
                merged[-1] = m
        else:
            merged.append(m)
    return merged


# ----------------- Multi-mode extraction (wavelength-first) -----------------
def extract_modes_wavelength_first(theta_centers, k_centers, H,
                                   max_scales=MAX_SCALES,
                                   rel_power_min=REL_POWER_MIN,
                                   lambda_max_window=None):
    if theta_centers is None or k_centers is None or H is None:
        return [], None, None

    S_theta = H.sum(axis=1)

    # FIX 2: directional (or fallback) radial spectrum
    S_k, dominant_theta_idx = radial_spectrum_directional(theta_centers, k_centers, H)

    peak_idx, _ = find_peaks(S_k)
    if len(peak_idx) == 0:
        return [], S_theta, S_k

    peak_heights = S_k[peak_idx]
    prom, _, _ = peak_prominences(S_k, peak_idx)
    prom_frac = prom / (peak_heights + 1e-8)
    main_power = float(peak_heights.max())

    candidates = []
    for idx_arr, idx_k in enumerate(peak_idx):
        power = float(peak_heights[idx_arr])
        if power <= 0.0:
            continue
        k_val = k_centers[idx_k]
        if k_val <= 0:
            continue
        lambda_m = 1.0 / k_val
        if lambda_m < MIN_LAMBDA_M:
            continue
        if lambda_max_window is not None and lambda_m > lambda_max_window:
            continue
        this_prom_frac = float(prom_frac[idx_arr])
        if this_prom_frac < PROM_FRAC_MIN:
            continue
        candidates.append({
            "idx_k": int(idx_k),
            "lambda_m": lambda_m,
            "power": power,
            "rel_power": power / (main_power + 1e-8),
            "prom_frac": this_prom_frac,
        })

    if not candidates:
        return [], S_theta, S_k

    # FIX 3: remove harmonics BEFORE we choose which modes to keep
    if COLLAPSE_HARMONICS:
        candidates = collapse_harmonics(candidates)

    # keep the strongest surviving peaks (by power), then report longest-first
    candidates.sort(key=lambda c: c["power"], reverse=True)
    candidates = candidates[:max_scales]
    candidates.sort(key=lambda c: c["lambda_m"], reverse=True)

    modes = []
    for c in candidates:
        idx_k = c["idx_k"]
        # FIX 2: prefer the dominant orientation; fall back to per-bin argmax
        orient_slice = H[:, idx_k]
        if dominant_theta_idx is not None and orient_slice[dominant_theta_idx] > 0:
            idx_theta = dominant_theta_idx
        else:
            if orient_slice.max() <= 0:
                continue
            idx_theta = int(np.argmax(orient_slice))
        theta0_deg = np.degrees(theta_centers[idx_theta])
        _, crest_az = theta_to_crest_azimuth(theta0_deg)
        modes.append({
            "theta_image_deg": theta0_deg,
            "crest_az": crest_az,
            "lambda_m": c["lambda_m"],
            "rel_power": c["rel_power"],
            "spectral_intensity": c["power"],
            "prom_frac": c["prom_frac"],
            "theta_index": int(idx_theta),
            "k_index": int(idx_k),
        })

    modes.sort(key=lambda m: m["lambda_m"])
    modes = deduplicate_modes_by_wavelength(modes, tol_frac=0.25)
    return modes, S_theta, S_k


# ----------------- GeoTIFF loading (full image) -----------------
def load_full_image(tif_path):
    with rasterio.open(tif_path) as src:
        band = src.read(1).astype(np.float32)
        nodata = src.nodata
        pixel_size_x = src.transform.a
        pixel_size_y = -src.transform.e
        pixel_size_m = float((abs(pixel_size_x) + abs(pixel_size_y)) / 2.0)
    valid = (band != nodata) & np.isfinite(band) if nodata is not None else np.isfinite(band)
    if not np.any(valid):
        raise RuntimeError("Image contains only nodata.")
    vals = band[valid]
    band[valid] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
    return band, pixel_size_m


# ----------------- Plotting -----------------
def plot_multimode_results(patch, P_shift, theta_centers, k_centers,
                           S_theta, S_k, modes, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax0 = axes[0, 0]
    ax0.imshow(patch, cmap="gray"); ax0.set_title("Image (full band)")
    ax0.set_xticks([]); ax0.set_yticks([])

    ax1 = axes[0, 1]
    im_fft = ax1.imshow(np.log10(P_shift + 1e-8), cmap="inferno")
    ax1.set_title("FFT log10(power)"); ax1.set_xticks([]); ax1.set_yticks([])
    fig.colorbar(im_fft, ax=ax1, shrink=0.8)

    ax2 = axes[1, 0]
    if S_theta is not None and theta_centers is not None:
        ax2.plot(np.degrees(theta_centers), S_theta, "-")
        ax2.set_xlabel("theta (image-space crest-normal, deg)")
        ax2.set_ylabel("Power (sum over k)")
        ax2.set_title("Orientation spectrum S(theta)")
        for idx in sorted({m["theta_index"] for m in modes}):
            ax2.axvline(np.degrees(theta_centers[idx]), color="r", linestyle="--", alpha=0.5)

    ax3 = axes[1, 1]
    ax3.set_title("Radial spectrum S(k) -> lambda vs power")
    if S_k is not None and k_centers is not None:
        k_pos = k_centers > 0
        lambda_vals = 1.0 / k_centers[k_pos]
        S_vals = S_k[k_pos]
        order = np.argsort(lambda_vals)
        ax3.plot(lambda_vals[order], S_vals[order], "-")
        ax3.set_xscale("log")
        ax3.set_xlabel("Wavelength lambda (m, log scale)")
        ax3.set_ylabel("Power (directional wedge)")
        for i, m in enumerate(modes):
            ax3.axvline(m["lambda_m"], color=f"C{i}", linestyle="--", alpha=0.7,
                        label=f"lambda={m['lambda_m']:.0f} m")
        if modes:
            ax3.legend(fontsize=8)
        ax3.set_xlim(MIN_LAMBDA_M, None)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ----------------- core runner shared by CLI + wrapper -----------------
def _run_core(tif_path):
    img, pixel_size_m = load_full_image(tif_path)
    ny, nx = img.shape
    window_m = min(ny, nx) * pixel_size_m
    lambda_max_window = min(MAX_LAMBDA_M_GLOBAL, 0.7 * window_m)

    sigma_px = sigma_px_from_cutoff(HP_CUTOFF_LAMBDA_M, pixel_size_m)  # FIX 1
    img_hp = high_pass_filter(img, sigma_px=sigma_px)
    img_win = apply_hann_window(img_hp)

    P_shift, kx, ky = compute_fft_power(img_win, pixel_size_m)
    theta_c, k_c, H = polar_power_histogram_logk(
        P_shift, kx, ky,
        n_angle_bins=N_ANGLE_BINS, n_radial_bins=N_RADIAL_BINS,
        min_lambda_m=MIN_LAMBDA_M, max_lambda_m=lambda_max_window,
    )
    modes, S_theta, S_k = ([], None, None) if H is None else \
        extract_modes_wavelength_first(theta_c, k_c, H,
                                       max_scales=MAX_SCALES,
                                       rel_power_min=REL_POWER_MIN,
                                       lambda_max_window=lambda_max_window)
    return {
        "img": img, "pixel_size_m": pixel_size_m, "P_shift": P_shift,
        "theta_c": theta_c, "k_c": k_c, "H": H,
        "modes": modes, "S_theta": S_theta, "S_k": S_k,
        "sigma_px": sigma_px, "lambda_max_window": lambda_max_window,
    }


def analyze_geotiff_multimode(tif_path, out_prefix=None):
    """Convenience wrapper used by run_pipeline_from_points.py."""
    r = _run_core(tif_path)
    modes = r["modes"]
    if out_prefix is not None and modes:
        plot_multimode_results(r["img"], r["P_shift"], r["theta_c"], r["k_c"],
                               r["S_theta"], r["S_k"], modes, f"{out_prefix}_multimode.png")
    return modes


# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser(description="FFT dune analysis (patched) on full GeoTIFF.")
    ap.add_argument("--tif", required=True)
    ap.add_argument("--out_prefix", type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    print("Loading full image...")
    r = _run_core(args.tif)
    ny, nx = r["img"].shape
    print(f"Image shape: {ny} x {nx} px, pixel size: {r['pixel_size_m']:.2f} m")
    print(f"High-pass sigma: {r['sigma_px']:.1f} px "
          f"(cutoff ~{HP_CUTOFF_LAMBDA_M:.0f} m)")
    print(f"Directional selection: {USE_DIRECTIONAL}, harmonic collapse: {COLLAPSE_HARMONICS}")
    print(f"lambda range: {MIN_LAMBDA_M:.1f}-{r['lambda_max_window']:.1f} m")

    modes = r["modes"]
    if not modes:
        print("No significant radial modes found.")
    else:
        print("\nDetected modes (sorted by wavelength):")
        for i, m in enumerate(modes, start=1):
            print(f"  Mode {i}: lambda = {m['lambda_m']:.1f} m, "
                  f"crest_az = {m['crest_az']:.1f} deg, "
                  f"rel_power = {m['rel_power']:.2f}, "
                  f"prom_frac = {m['prom_frac']:.2f}")

    out_png = f"{args.out_prefix}_multimode.png"
    print(f"\nSaving diagnostic figure to: {out_png}")
    plot_multimode_results(r["img"], r["P_shift"], r["theta_c"], r["k_c"],
                           r["S_theta"], r["S_k"], modes, out_png)
    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Synthetic-dune test harness.

Builds fake dune fields with KNOWN wavelength + orientation, then compares:
  (A) current approach:   S_k = H.sum(axis=0)  -> argmax/peaks over ALL angles
  (B) directional approach: find dominant orientation, take S_k along that wedge

Also tests the effect of the high-pass sigma. No real tiles needed.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, peak_prominences

PIXEL_SIZE_M = 20.0          # matches pipeline SCALE
MIN_LAMBDA_M = 30.0
MAX_LAMBDA_M_GLOBAL = 10000.0
N_ANGLE_BINS = 72
N_RADIAL_BINS = 100

# ---------------- synthetic field ----------------
def make_dune_field(ny, nx, lambda_m, orient_deg, pixel_size_m,
                    sharpness=1.0, noise=0.3, seed=0):
    """
    Ridges with wavelength lambda_m along direction perpendicular to crests.
    orient_deg = crest orientation (ridges run along this angle).
    sharpness>1 makes non-sinusoidal (sawtooth-ish) profiles -> harmonics.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    # unit vector perpendicular to crests (the direction wavelength is measured)
    phi = np.radians(orient_deg + 90.0)
    proj = (xx * np.cos(phi) + yy * np.sin(phi)) * pixel_size_m
    phase = 2 * np.pi * proj / lambda_m
    base = np.sin(phase)
    if sharpness != 1.0:
        # add harmonics to mimic sharp dune crests
        base = np.sign(base) * (np.abs(base) ** (1.0 / sharpness))
    field = base + noise * rng.standard_normal((ny, nx))
    # add a big brightness gradient (illumination trend) to mimic real tiles
    field += 0.8 * (xx / nx)
    return field

# ---------------- shared FFT machinery (mirrors the real script) ----------------
def high_pass_filter(arr, sigma_px):
    if sigma_px <= 0:
        return arr
    blurred = gaussian_filter(arr, sigma=sigma_px)
    hp = arr - blurred
    hp = (hp - hp.min()) / (hp.max() - hp.min() + 1e-8)
    return hp

def apply_hann_window(arr):
    ny, nx = arr.shape
    return arr * np.outer(np.hanning(ny), np.hanning(nx))

def compute_fft_power(arr, pixel_size_m):
    ny, nx = arr.shape
    F = np.fft.fft2(arr)
    P = np.abs(F) ** 2
    P = np.fft.fftshift(P)
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=pixel_size_m))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=pixel_size_m))
    kx, ky = np.meshgrid(fx, fy)
    return P, kx, ky

def polar_hist(P, kx, ky, max_lambda_m):
    k = np.sqrt(kx**2 + ky**2)
    theta = np.arctan2(ky, kx)
    theta = np.mod(theta + np.pi/2, np.pi) - np.pi/2
    Pf, kf, tf = P.ravel(), k.ravel(), theta.ravel()
    k_min, k_max = 1.0 / max_lambda_m, 1.0 / MIN_LAMBDA_M
    m = (kf > k_min) & (kf < k_max) & np.isfinite(Pf)
    Pf, kf, tf = Pf[m], kf[m], tf[m]
    theta_edges = np.linspace(-np.pi/2, np.pi/2, N_ANGLE_BINS + 1)
    k_edges = np.logspace(np.log10(k_min), np.log10(k_max), N_RADIAL_BINS + 1)
    H, _, _ = np.histogram2d(tf, kf, bins=[theta_edges, k_edges], weights=Pf)
    theta_c = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    k_c = np.sqrt(k_edges[:-1] * k_edges[1:])
    return theta_c, k_c, H

# ---------------- selection strategies ----------------
def peaks_from_Sk(S_k, k_c, prom_frac_min=0.20):
    idx, _ = find_peaks(S_k)
    if len(idx) == 0:
        return []
    heights = S_k[idx]
    prom, _, _ = peak_prominences(S_k, idx)
    pf = prom / (heights + 1e-8)
    out = []
    for j, i in enumerate(idx):
        if k_c[i] <= 0:
            continue
        lam = 1.0 / k_c[i]
        if lam < MIN_LAMBDA_M:
            continue
        if pf[j] < prom_frac_min:
            continue
        out.append({"lambda_m": lam, "power": float(heights[j]), "prom_frac": float(pf[j])})
    out.sort(key=lambda c: c["power"], reverse=True)
    return out

def strat_current(theta_c, k_c, H):
    """Sum over ALL theta, then peaks."""
    S_k = H.sum(axis=0)
    return peaks_from_Sk(S_k, k_c)

def strat_directional(theta_c, k_c, H, wedge_halfwidth_bins=4):
    """Find dominant orientation, take S_k only in a wedge around it."""
    S_theta = H.sum(axis=1)
    i_theta = int(np.argmax(S_theta))
    lo = max(0, i_theta - wedge_halfwidth_bins)
    hi = min(len(theta_c), i_theta + wedge_halfwidth_bins + 1)
    S_k = H[lo:hi, :].sum(axis=0)
    return peaks_from_Sk(S_k, k_c)

def harmonic_collapse(cands, tol=0.15):
    """Prefer fundamental: if a peak at ~2*lambda exists, demote the short one."""
    if len(cands) <= 1:
        return cands
    kept = []
    for c in cands:
        is_harmonic = any(
            abs(other["lambda_m"] - 2 * c["lambda_m"]) / (2 * c["lambda_m"]) < tol
            and other["power"] > 0.3 * c["power"]
            for other in cands if other is not c
        )
        if not is_harmonic:
            kept.append(c)
    return kept if kept else cands

# ---------------- run experiment ----------------
def pick_top(cands):
    return cands[0]["lambda_m"] if cands else None

def run_case(lambda_true, orient, sharpness, sigma_px, ny=700, nx=700, seed=0):
    field = make_dune_field(ny, nx, lambda_true, orient, PIXEL_SIZE_M,
                            sharpness=sharpness, noise=0.4, seed=seed)
    img = apply_hann_window(high_pass_filter(field, sigma_px))
    P, kx, ky = compute_fft_power(img, PIXEL_SIZE_M)
    window_m = min(ny, nx) * PIXEL_SIZE_M
    max_lam = min(MAX_LAMBDA_M_GLOBAL, 0.7 * window_m)
    theta_c, k_c, H = polar_hist(P, kx, ky, max_lam)

    cur = strat_current(theta_c, k_c, H)
    dir_ = strat_directional(theta_c, k_c, H)
    dir_hc = harmonic_collapse(dir_)

    return {
        "current": pick_top(cur),
        "directional": pick_top(dir_),
        "directional+harmonic": pick_top(dir_hc),
    }

def err(est, true):
    return None if est is None else abs(est - true) / true

if __name__ == "__main__":
    cases = [
        # (lambda_true_m, crest_orient_deg, sharpness)
        (1600, 30, 2.5),   # sharp crests -> strong harmonics, the hard case
        (1200, 70, 2.5),
        (2000, 10, 1.0),   # sinusoidal
        (800,  45, 2.0),
        (2300, 100, 3.0),  # very sharp, long wavelength
    ]
    sigmas = {"sigma=10 (current)": 10.0, "sigma=40": 40.0, "sigma=80": 80.0}

    for label, sigma in sigmas.items():
        print(f"\n===== high-pass {label} =====")
        print(f"{'true λ':>8} {'orient':>6} {'sharp':>6} | "
              f"{'current':>10} {'direction':>10} {'dir+harm':>10}")
        agg = {"current": [], "directional": [], "directional+harmonic": []}
        for lam, ori, sh in cases:
            r = run_case(lam, ori, sh, sigma)
            def fmt(v):
                if v is None: return "  none"
                e = err(v, lam)
                return f"{v:6.0f}({e*100:3.0f}%)"
            print(f"{lam:>8} {ori:>6} {sh:>6.1f} | "
                  f"{fmt(r['current']):>10} {fmt(r['directional']):>10} {fmt(r['directional+harmonic']):>10}")
            for kk in agg:
                e = err(r[kk], lam)
                if e is not None:
                    agg[kk].append(e)
        print("  mean abs error:  " + "  ".join(
            f"{kk}={np.mean(v)*100:.0f}%" if v else f"{kk}=NA"
            for kk, v in agg.items()))

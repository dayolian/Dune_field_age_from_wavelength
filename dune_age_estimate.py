#!/usr/bin/env python3
"""
dune_age_estimate.py

Python port of Mackenzie Day's MATLAB dune-age scripts
(age_estimating_test.m / powerlawtest2.m / dune_age_linear.m /
 average_age_power_and_linear.m).

Computes dune-field ages from wavelengths using the model growth curve,
with three growth laws (power law, linear, and their geometric mean).

INPUTS
------
  --physics new_model_age_analysis.xlsx
        Per-site physics: region (col E), incipient wavelength (F),
        model incipient wavelength (G), real flux Qr in m^2/s (K),
        intermittency in percent (O). Modern wavelength (I) is REPLACED
        by your FFT values if --wavelengths is given.

  --wavelengths final_wavelengths.csv   (optional)
        Your nearest_tile FFT output. Its 'chosen_lambda_m' replaces the
        physics file's modern wavelength, joined by region + within-region
        order. If omitted, the physics file's own col I is used.

PHYSICS (canonical choices, matching age_estimating_test.m)
----------------------------------------------------------
  l0     = incipient_real / model_incipient(43.429)          [m per block]
  t0     = Qs_model / (Qr_yr * hyst)                          [years/timestep]
           Qr_yr = Qr(m^2/s) * 31536000
           Qs_model = 0.125, hyst = 0.2
  W      = modern_lambda / l0                                 [l0 units]
  power law:  W = 11.09 * t^0.2026   -> t = (W/11.09)^(1/0.2026)
  linear:     W = 8.26e-5 * t + 110.2 -> t = (W-110.2)/8.26e-5
  age    = t0 * N_timesteps / intermittency_fraction         [years]

  NOTE ON t0: three MATLAB scripts disagree on this formula.
    - age_estimating_test.m: t0 = Qs/(Qr*hyst)         <-- used here (canonical)
    - powerlawtest2.m:       t0 = Qs/Qr                (no hyst -> ages 5x larger)
    - data_analysis.m:       t0 = (l0^0.2 * Qs)/(Qr*interm*hyst)
    Change HYST or the formula below if the team settles on a different one.
"""
import argparse
import numpy as np
import pandas as pd
import openpyxl

# ---- model constants ----
MODEL_INCIP = 43.429     # l0 units (col G, same for all sites)
QS_MODEL    = 0.125
HYST        = 0.2
SEC_PER_YR  = 31536000.0

# power law fit  W = pow_a * t^pow_b
POW_A, POW_B = 11.09, 0.2026
# linear fit     W = lin_m * t + lin_b
LIN_M, LIN_B = 0.0000826, 110.2
# model boundary (max W actually simulated) -> minimum-age floor
W_BOUNDARY   = 187.5
W_FLAG       = 5000.0    # W above this = highly uncertain extrapolation


def load_physics(xlsx, coords_csv=None):
    """
    Load per-site physics. The xlsx has no lat/lon, so if coords_csv (a modes
    CSV with region+lat+lon for the SAME 1005 tiles in the SAME order) is given,
    attach lat/lon by order -- validated safe because both are the full tile set.
    """
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    ws = wb["Sheet1"]
    rows = []
    for r in range(2, ws.max_row + 1):
        reg = ws.cell(row=r, column=5).value
        if reg is None:
            continue
        def g(c):
            v = ws.cell(row=r, column=c).value
            return v if isinstance(v, (int, float)) else np.nan
        rows.append(dict(
            region=str(reg),
            incip_real=g(6),          # col F  [m]
            model_incip=g(7),         # col G  [l0]
            modern=g(9),              # col I  [m]  (may be replaced)
            Qr_m2s=g(11),             # col K  [m^2/s]
            interm_pct=g(15),         # col O  [percent]
        ))
    phys = pd.DataFrame(rows)

    if coords_csv:
        modes = pd.read_csv(coords_csv)
        tiles = modes[["region", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
        if len(tiles) != len(phys):
            raise ValueError(f"coords tiles ({len(tiles)}) != physics rows ({len(phys)}); "
                             "cannot attach lat/lon by order")
        if not (tiles["region"].values == phys["region"].values).all():
            raise ValueError("region order differs between physics and coords file; "
                             "lat/lon attachment unsafe")
        phys["lat"] = tiles["lat"].values
        phys["lon"] = tiles["lon"].values
    return phys


def attach_fft_wavelengths(phys, wl_csv):
    """Replace physics 'modern' with chosen_lambda_m, joined EXACTLY on lat/lon."""
    if "lat" not in phys.columns:
        raise ValueError("physics has no lat/lon -- pass --coords so the FFT join "
                         "can match on coordinates instead of fragile row order")
    wl = pd.read_csv(wl_csv)
    if "chosen_lambda_m" not in wl.columns:
        raise ValueError("wavelengths CSV missing 'chosen_lambda_m'")

    def key(df):
        return list(zip(df["lat"].round(3), df["lon"].round(3)))
    wl_map = dict(zip(key(wl), wl["chosen_lambda_m"]))
    conf_map = dict(zip(key(wl), wl["confidence"])) if "confidence" in wl.columns else {}

    out = phys.copy()
    ph_keys = key(out)
    out["modern_fft"] = [wl_map.get(k, np.nan) for k in ph_keys]
    out["fft_confidence"] = [conf_map.get(k, "") for k in ph_keys]
    out["modern_used"] = out["modern_fft"].fillna(out["modern"])
    out["wavelength_source"] = np.where(out["modern_fft"].notna(), "fft", "physics_colI")

    matched = out["modern_fft"].notna().sum()
    print(f"  lat/lon join: matched {matched}/{len(wl)} FFT rows to physics tiles")
    if matched < len(wl):
        print(f"  NOTE: {len(wl)-matched} FFT rows had no lat/lon match in physics")
    return out


def compute_ages(df):
    d = df.copy()
    # drop rows lacking essentials
    ok = (d["incip_real"] > 0) & (d["Qr_m2s"] > 0) & \
         (d["interm_pct"] > 0) & (d["modern_used"] > 0)
    d = d[ok].copy()

    l0 = d["incip_real"] / MODEL_INCIP                      # m/block
    Qr_yr = d["Qr_m2s"] * SEC_PER_YR
    t0 = QS_MODEL / (Qr_yr * HYST)                          # years/step
    interm = d["interm_pct"] / 100.0
    W = d["modern_used"] / l0                               # l0 units

    # invert growth laws
    N_pow = (W / POW_A) ** (1.0 / POW_B)
    N_lin = (W - LIN_B) / LIN_M                             # may go negative

    age_pow = t0 * N_pow / interm
    age_lin = t0 * N_lin / interm
    # geometric mean only where both positive
    both_pos = (age_pow > 0) & (age_lin > 0)
    age_geo = np.where(both_pos, np.sqrt(age_pow.clip(lower=0) * age_lin.clip(lower=0)), np.nan)

    # minimum-age floor at model boundary W=187.5
    N_floor = (W_BOUNDARY / POW_A) ** (1.0 / POW_B)
    age_floor = t0 * N_floor / interm

    d["l0_m"] = l0
    d["t0_years"] = t0
    d["intermittency_frac"] = interm
    d["W_modern_l0"] = W
    d["N_timesteps_power"] = N_pow
    d["N_timesteps_linear"] = N_lin
    d["age_power_years"] = age_pow
    d["age_linear_years"] = age_lin
    d["age_geomean_years"] = age_geo
    d["age_floor_years"] = age_floor
    d["flagged_W_gt_5000"] = W > W_FLAG
    d["linear_unphysical"] = N_lin <= 0     # W below linear intercept
    return d


def summarize(d):
    def stats(col, sub=None):
        data = sub if sub is not None else d
        v = data[col][np.isfinite(data[col]) & (data[col] > 0)]
        if len(v) == 0:
            return "     (none)"
        return f"{v.median():.2e}"

    cols = ["age_power_years", "age_linear_years", "age_geomean_years", "age_floor_years"]
    names = ["power", "linear", "geomean", "floor"]

    print("\n=== MEDIAN AGES (years), all tiles ===")
    for c, n in zip(cols, names):
        print(f"  {n:9s} {stats(c)}")
    print(f"\nflagged (W>5000 l0): {int(d['flagged_W_gt_5000'].sum())} / {len(d)}")
    print(f"linear unphysical (W<110.2): {int(d['linear_unphysical'].sum())} / {len(d)}")

    print("\n=== PER-REGION MEDIAN AGES (years) ===")
    print(f"{'region':26s} {'power':>10} {'linear':>10} {'geomean':>10} {'floor':>10}  n")
    for reg, g in d.groupby("region"):
        line = f"{reg:26s}"
        for c in cols:
            line += f" {stats(c, g):>10}"
        line += f"  {len(g)}"
        print(line)

    # ---- MATLAB 'reasonable range' cut: wavelengths < 2000 m ----
    sub = d[d["modern_used"] < 2000]
    print(f"\n=== FILTERED: wavelength < 2000 m only ({len(sub)}/{len(d)} tiles) ===")
    print("  (matches geomean_age_under2000m_lambda.m / plot_age_ranges_test.m cut)")
    for c, n in zip(cols, names):
        print(f"  {n:9s} {stats(c, sub)}")
    print(f"  flagged remaining: {int(sub['flagged_W_gt_5000'].sum())} / {len(sub)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--physics", default="new_model_age_analysis.xlsx")
    ap.add_argument("--coords", default=None,
                    help="modes CSV with region+lat+lon for all 1005 tiles, same "
                         "order as physics; enables exact lat/lon join to FFT")
    ap.add_argument("--wavelengths", default=None,
                    help="final_wavelengths.csv (optional; replaces modern lambda)")
    ap.add_argument("--out", default="results/dune_ages.csv")
    args = ap.parse_args()

    phys = load_physics(args.physics, coords_csv=args.coords)
    print(f"loaded physics: {len(phys)} rows, {phys['region'].nunique()} regions")

    if args.wavelengths:
        phys = attach_fft_wavelengths(phys, args.wavelengths)
        n_fft = (phys["wavelength_source"] == "fft").sum()
        print(f"attached FFT wavelengths to {n_fft} rows "
              f"({(phys['wavelength_source']=='physics_colI').sum()} fell back to col I)")
    else:
        phys["modern_used"] = phys["modern"]
        phys["wavelength_source"] = "physics_colI"
        phys["fft_confidence"] = ""

    d = compute_ages(phys)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    keep = ["region", "wavelength_source", "fft_confidence",
            "incip_real", "l0_m", "Qr_m2s", "t0_years", "intermittency_frac",
            "modern_used", "W_modern_l0",
            "age_power_years", "age_linear_years", "age_geomean_years",
            "age_floor_years", "flagged_W_gt_5000", "linear_unphysical"]
    d[keep].to_csv(args.out, index=False)
    print(f"\nwrote {len(d)} age estimates to {args.out}")
    summarize(d)


if __name__ == "__main__":
    main()

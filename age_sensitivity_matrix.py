#!/usr/bin/env python3
"""
age_sensitivity_matrix.py

Compares every combination of the three open modelling choices against the
independently dated dune fields, so the choices can be made on evidence rather
than convention.

The three axes:

  intermittency definition (4)   raw        : all above-threshold winds
                                 net_algo   : best 180-deg sector minus worst
                                              (what analysis.ipynb computes)
                                 gross_mig  : winds in the migration-anchored
                                              sector, no subtraction
                                 net_mig    : migration sector minus its
                                              opposing sector

  hysteresis factor (2)          with       : t0 = Qs / (Qr_yr * 0.2)
                                 without    : t0 = Qs / Qr_yr
                                 (removing it divides every age by 5)

  growth law (4)                 power      : W = 11.09 * t^0.2026
                                 linear     : W = 8.26e-5 * t + 110.2
                                 geomean    : sqrt(power * linear)
                                 floor      : age at the model boundary
                                              W = 187.5, i.e. a lower bound

That is 32 combinations. Each is scored by mean |log10(modelled / published)|
over the dune fields that have independent luminescence, magnetic-remanence or
cosmogenic dates. 0.0 would be exact; 0.3 is about a factor of 2; 1.0 is an
order of magnitude.

CAUTION ON INTERPRETATION: choosing the combination that best matches six
published ages is CALIBRATION, not independent validation. Report it as such.
With only six dated fields, differences smaller than ~0.2 log units between
combinations are not meaningful.

Usage:
    python age_sensitivity_matrix.py \
        --physics new_model_age_analysis.xlsx \
        --coords modes_multimode_allregions_luminance_20m.csv \
        --intermittency intermittency_by_migration.csv \
        --wavelengths results/final_wavelengths.csv \
        --out results/age_sensitivity.csv
"""
import os
import argparse
import itertools
import numpy as np
import pandas as pd
import openpyxl

# ---- model constants ----
MODEL_INCIP = 43.429
QS_MODEL    = 0.125
HYST        = 0.2
SEC_PER_YR  = 31536000.0
POW_A, POW_B = 11.09, 0.2026
LIN_M, LIN_B = 0.0000826, 110.2
W_BOUNDARY   = 187.5

# published ages in YEARS (Table S2 values are in kyr)
PUBLISHED_AGES_YR = {
    "WhiteSands":  7.0e3,
    "Namib":       1.0e6,
    "Taklamakan":  7.0e5,
    "Tengger":     6.8e5,
    "RubAlKhali":  2.1e5,
    "Thar":        2.0e5,
}

INTERMIT_COLS = {
    "raw":       "intermit_raw_pct",
    "net_algo":  "intermit_net_algo_pct",
    "gross_mig": "intermit_gross_mig_pct",
    "net_mig":   "intermit_net_mig_pct",
}


def load_physics(xlsx, coords_csv):
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
        rows.append(dict(region=str(reg), incip_real=g(6), modern=g(9),
                         Qr_m2s=g(11), interm_orig_pct=g(15)))
    phys = pd.DataFrame(rows)

    modes = pd.read_csv(coords_csv)
    tiles = modes[["region", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
    if len(tiles) != len(phys):
        raise ValueError(f"coords tiles ({len(tiles)}) != physics rows ({len(phys)})")
    if not (tiles["region"].values == phys["region"].values).all():
        raise ValueError("region order differs between physics and coords file")
    phys["lat"] = tiles["lat"].values
    phys["lon"] = tiles["lon"].values
    return phys


def attach(df, csv_path, cols, keyround=3):
    """Left-join extra columns from a lat/lon CSV."""
    src = pd.read_csv(csv_path)
    src = src.copy()
    src["_k"] = list(zip(src["lat"].round(keyround), src["lon"].round(keyround)))
    lut = {c: dict(zip(src["_k"], src[c])) for c in cols if c in src.columns}
    out = df.copy()
    keys = list(zip(out["lat"].round(keyround), out["lon"].round(keyround)))
    for c, m in lut.items():
        out[c] = [m.get(k, np.nan) for k in keys]
    missing = [c for c in cols if c not in lut]
    if missing:
        print(f"  NOTE: {csv_path} has no columns {missing}")
    return out


def ages_for(df, interm_pct, use_hyst):
    """Return a frame with the four growth-law ages for one configuration."""
    d = df.copy()
    d["_interm"] = interm_pct
    ok = ((d["incip_real"] > 0) & (d["Qr_m2s"] > 0) &
          (d["_interm"] > 0) & (d["modern_used"] > 0))
    d = d[ok].copy()
    if len(d) == 0:
        return d

    l0 = d["incip_real"] / MODEL_INCIP
    Qr_yr = d["Qr_m2s"] * SEC_PER_YR
    t0 = QS_MODEL / (Qr_yr * HYST) if use_hyst else QS_MODEL / Qr_yr
    interm = d["_interm"] / 100.0
    W = d["modern_used"] / l0

    N_pow = (W / POW_A) ** (1.0 / POW_B)
    N_lin = (W - LIN_B) / LIN_M
    N_flo = (W_BOUNDARY / POW_A) ** (1.0 / POW_B)

    a_pow = t0 * N_pow / interm
    a_lin = t0 * N_lin / interm
    a_flo = t0 * N_flo / interm
    a_geo = np.where((a_pow > 0) & (a_lin > 0),
                     np.sqrt(a_pow.clip(lower=0) * a_lin.clip(lower=0)), np.nan)

    d["power"] = a_pow
    d["linear"] = a_lin
    d["geomean"] = a_geo
    d["floor"] = a_flo
    d["W_modern_l0"] = W
    return d


def score(d, law):
    """Mean |log10(model/published)| across dated fields, plus per-field detail."""
    errs, detail = [], {}
    for reg, pub in PUBLISHED_AGES_YR.items():
        g = d[d["region"] == reg]
        if len(g) == 0:
            continue
        v = g[law]
        v = v[np.isfinite(v) & (v > 0)]
        if len(v) == 0:
            continue
        lr = np.log10(v.median() / pub)
        errs.append(abs(lr))
        detail[reg] = (v.median(), lr)
    return (np.mean(errs) if errs else np.nan), len(errs), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--physics", default="new_model_age_analysis.xlsx")
    ap.add_argument("--coords", default="modes_multimode_allregions_luminance_20m.csv")
    ap.add_argument("--intermittency", default="intermittency_by_migration.csv")
    ap.add_argument("--wavelengths", default=None,
                    help="final_wavelengths.csv; if omitted the physics file's "
                         "own modern wavelength is used")
    ap.add_argument("--out", default="results/age_sensitivity.csv")
    args = ap.parse_args()

    phys = load_physics(args.physics, args.coords)
    print(f"physics: {len(phys)} rows, {phys['region'].nunique()} regions")

    # intermittency definitions
    phys = attach(phys, args.intermittency, list(INTERMIT_COLS.values()))
    have = [k for k, c in INTERMIT_COLS.items() if c in phys.columns
            and phys[c].notna().any()]
    print(f"intermittency definitions available: {have}")

    # wavelengths
    if args.wavelengths:
        phys = attach(phys, args.wavelengths, ["chosen_lambda_m"])
        phys["modern_used"] = phys["chosen_lambda_m"].fillna(phys["modern"])
        n = phys["chosen_lambda_m"].notna().sum()
        print(f"using FFT wavelengths for {n} points "
              f"({len(phys)-n} fell back to the physics file)")
    else:
        phys["modern_used"] = phys["modern"]
        print("using the physics file's own modern wavelength")

    laws = ["power", "linear", "geomean", "floor"]
    results = []
    cache = {}

    for idef, hy in itertools.product(have, [True, False]):
        col = INTERMIT_COLS[idef]
        d = ages_for(phys, phys[col], hy)
        cache[(idef, hy)] = d
        for law in laws:
            err, n, _ = score(d, law)
            results.append(dict(intermittency=idef,
                                hysteresis="with" if hy else "without",
                                growth_law=law, mean_log10_err=err,
                                n_fields=n, n_points=len(d),
                                median_age_yr=(d[law][np.isfinite(d[law]) &
                                                      (d[law] > 0)].median()
                                               if len(d) else np.nan)))

    res = pd.DataFrame(results).sort_values("mean_log10_err")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    res.to_csv(args.out, index=False)

    print("\n" + "=" * 78)
    print("ALL COMBINATIONS, ranked by agreement with published ages")
    print("(mean |log10(model/published)|; 0.3 ~ factor of 2, 1.0 = 10x off)")
    print("=" * 78)
    print(f"{'intermittency':13s} {'hyst':>8} {'growth':>9} {'err':>7} "
          f"{'median age (yr)':>17}")
    for r in res.itertuples():
        if not np.isfinite(r.mean_log10_err):
            continue
        print(f"{r.intermittency:13s} {r.hysteresis:>8} {r.growth_law:>9} "
              f"{r.mean_log10_err:>7.2f} {r.median_age_yr:>17.2e}")

    # detail for the winner
    best = res[np.isfinite(res["mean_log10_err"])].iloc[0]
    print("\n" + "=" * 78)
    print(f"BEST: intermittency={best.intermittency}, "
          f"hysteresis={best.hysteresis}, growth={best.growth_law} "
          f"(err {best.mean_log10_err:.2f})")
    print("=" * 78)
    d = cache[(best.intermittency, best.hysteresis == "with")]
    _, _, detail = score(d, best.growth_law)
    print(f"{'field':14s} {'published':>12} {'modelled':>12} {'log10 ratio':>12}")
    for reg in sorted(detail):
        med, lr = detail[reg]
        print(f"{reg:14s} {PUBLISHED_AGES_YR[reg]:>12.2e} {med:>12.2e} {lr:>+12.2f}")

    print("\nper-region median age under the best combination:")
    for reg, g in d.groupby("region"):
        v = g[best.growth_law]
        v = v[np.isfinite(v) & (v > 0)]
        if len(v):
            print(f"  {reg:26s} {v.median():.2e} yr   (n={len(v)})")

    print(f"\nfull matrix written to {args.out}")
    print("\nNOTE: this is calibration against six dated fields, not independent")
    print("validation. Differences under ~0.2 log units are not meaningful.")


if __name__ == "__main__":
    main()

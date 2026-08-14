#!/usr/bin/env python3
"""
age_sensitivity_matrix.py

Compares every combination of the three open modelling choices against the
independently dated dune fields, so the choices can be made on evidence rather
than convention.

The three axes:

  intermittency definition (4)   raw        : all above-threshold winds
                                 net_algo   : best 180-deg sector minus worst
                                              (what analysis.ipynb computes,
                                              and what physics col O holds)
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

That is 32 combinations, scored by mean log10 distance from the published age
of each dated field. 0.0 is exact; 0.3 is about a factor of 2; 1.0 is an order
of magnitude.

CHANGES FROM THE PREVIOUS VERSION
---------------------------------
1. lat/lon alignment. load_physics used to attach coordinates from the modes
   CSV in row order. The physics rows actually follow the intermittency CSV's
   per-region file order, and the two only coincide where lat and lon are both
   positive. That silently mispaired 242 rows in Namib, Western Sahara and
   Grand Erg Occidental. Alignment is now done per region against the
   intermittency CSV, and verified by comparing intermittency values.

2. Green-tagged rows are excluded. The matrix used to calibrate on tiles the
   rest of the pipeline drops (no dunes / linear dunes / unresolvable).
   Pass --truth to exclude them.

3. No column I fallback. Physics column I is a superseded pre-fix FFT run (its
   values sit on the old log-k bin grid and run about half the current FFT's).
   Tiles with no FFT wavelength are dropped rather than backfilled.

4. Published ages are ranges, and two target sets are scored side by side.
   LEGACY is what the script previously hardcoded; COMPILATION is the
   literature review. A model age inside a published range scores zero error
   rather than being penalised for missing a midpoint. If both sets pick the
   same winning configuration the disagreement does not matter; if they pick
   different winners, that needs a decision.

CAUTION ON INTERPRETATION: choosing the combination that best matches a handful
of published ages is CALIBRATION, not independent validation. Report it as
such. With only six dated fields, differences smaller than ~0.2 log units
between combinations are not meaningful.

Usage:
    python age_sensitivity_matrix.py \
        --physics new_model_age_analysis.xlsx \
        --intermittency intermittency_by_migration.csv \
        --truth ~/Desktop/era5_wind_ts/dune_migration_and_lambda_corrected.xlsx \
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

GREEN = "FF92D050"

# Published ages in YEARS as (low, high). None for an open upper bound.
# A model age inside the range scores zero error.
TARGET_SETS = {
    "legacy": {                       # what this script hardcoded before
        "WhiteSands":  (7.0e3,  7.0e3),
        "Namib":       (1.0e6,  1.0e6),
        "Taklamakan":  (7.0e5,  7.0e5),
        "Tengger":     (6.8e5,  6.8e5),
        "RubAlKhali":  (2.1e5,  2.1e5),
        "Thar":        (2.0e5,  2.0e5),
    },
    "compilation": {                  # from the literature review
        "WhiteSands":  (6.5e3,  8.77e3),   # Kocurek 2007; Holliday 2023
        "Namib":       (1.0e6,  None),     # Vermeesch 2010, cosmogenic, >1 Ma
        "Taklamakan":  (7.0e6,  2.67e7),   # Sun 2009 in-situ to Zheng 2015 onset
        "Tengger":     (8.0e5,  8.0e5),    # Guan 2011, indirect, low confidence
        "RubAlKhali":  (1.0e4,  1.25e5),   # Atkinson 2011, late Pleistocene
        # Thar omitted: the compilation reports phases, not a single range
    },
}

INTERMIT_COLS = {
    "raw":       "intermit_raw_pct",
    "net_algo":  "intermit_net_algo_pct",
    "gross_mig": "intermit_gross_mig_pct",
    "net_mig":   "intermit_net_mig_pct",
}


def align_latlon_from_intermittency(phys, intermittency_csv):
    """Attach lat/lon using the intermittency CSV's PER-REGION file order.

    The physics spreadsheet orders its region blocks alphabetically while the
    CSV does not, so alignment must happen inside each region and never
    globally. Intermittency values are compared as a check: if they disagree
    the pairing is wrong and we raise rather than produce silently wrong ages.
    """
    im = pd.read_csv(intermittency_csv)
    lat = np.full(len(phys), np.nan)
    lon = np.full(len(phys), np.nan)
    for reg, idx in phys.groupby("region", sort=False).groups.items():
        idx = list(idx)
        sub = im[im.region == reg]
        if len(sub) != len(idx):
            raise ValueError(
                "%s: physics %d rows vs intermittency %d" % (reg, len(idx), len(sub)))
        a = phys.loc[idx, "interm_orig_pct"].values.astype(float)
        b = sub["intermit_net_algo_pct"].values.astype(float)
        off = int((np.abs(a - b) >= 0.01).sum())
        if off:
            raise ValueError(
                "%s: %d rows disagree on intermittency, alignment unsafe" % (reg, off))
        lat[idx] = sub["lat"].values
        lon[idx] = sub["lon"].values
    phys = phys.copy()
    phys["lat"] = lat
    phys["lon"] = lon
    return phys


def load_physics(xlsx, intermittency_csv):
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
        rows.append(dict(region=str(reg), incip_real=g(6), modern_colI=g(9),
                         Qr_m2s=g(11), interm_orig_pct=g(15)))
    phys = pd.DataFrame(rows)
    return align_latlon_from_intermittency(phys, intermittency_csv)


def load_green(truth_xlsx):
    """(lat,lon) set of green-tagged rows: no dunes / linear / unresolvable."""
    ws = openpyxl.load_workbook(os.path.expanduser(truth_xlsx))["Sheet1"]
    green = set()
    for r in range(2, ws.max_row + 1):
        lat = ws.cell(row=r, column=1).value
        if lat is None:
            continue
        f = ws.cell(row=r, column=4).fill
        tag = f.start_color.rgb if f and f.patternType else "NONE"
        if tag == GREEN:
            green.add((round(float(lat), 3),
                       round(float(ws.cell(row=r, column=2).value), 3)))
    return green


def attach(df, csv_path, cols, keyround=3):
    """Left-join extra columns from a lat/lon CSV."""
    src = pd.read_csv(csv_path)
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
    """Four growth-law ages for one configuration."""
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
    # NOTE: N_flo is a CONSTANT. The floor age therefore carries no wavelength
    # information at all -- it is a flux-and-intermittency clock, not a dune
    # measurement. Interpret any agreement it shows with that in mind.
    N_flo = (W_BOUNDARY / POW_A) ** (1.0 / POW_B)

    a_pow = t0 * N_pow / interm
    a_lin = t0 * N_lin / interm
    a_flo = t0 * N_flo / interm
    a_geo = np.where((a_pow > 0) & (a_lin > 0),
                     np.sqrt(a_pow.clip(lower=0) * a_lin.clip(lower=0)), np.nan)

    d["power"], d["linear"] = a_pow, a_lin
    d["geomean"], d["floor"] = a_geo, a_flo
    d["W_modern_l0"] = W
    return d


def log_distance(value, lo, hi):
    """0 if value sits inside [lo, hi]; else log10 distance to the near edge."""
    if hi is None:                      # open upper bound, e.g. ">1 Ma"
        return 0.0 if value >= lo else abs(np.log10(value / lo))
    if value < lo:
        return abs(np.log10(value / lo))
    if value > hi:
        return abs(np.log10(value / hi))
    return 0.0


def score(d, law, targets):
    """Mean log10 distance from the published range, plus per-field detail."""
    errs, detail = [], {}
    for reg, (lo, hi) in targets.items():
        g = d[d["region"] == reg]
        if len(g) == 0:
            continue
        v = g[law]
        v = v[np.isfinite(v) & (v > 0)]
        if len(v) == 0:
            continue
        med = float(v.median())
        e = log_distance(med, lo, hi)
        errs.append(e)
        detail[reg] = (med, e)
    return (np.mean(errs) if errs else np.nan), len(errs), detail


def run_matrix(phys, have, laws, targets):
    results, cache = [], {}
    for idef, hy in itertools.product(have, [True, False]):
        d = ages_for(phys, phys[INTERMIT_COLS[idef]], hy)
        cache[(idef, hy)] = d
        for law in laws:
            err, n, _ = score(d, law, targets)
            results.append(dict(intermittency=idef,
                                hysteresis="with" if hy else "without",
                                growth_law=law, mean_log10_err=err,
                                n_fields=n, n_points=len(d),
                                median_age_yr=(d[law][np.isfinite(d[law]) &
                                                      (d[law] > 0)].median()
                                               if len(d) else np.nan)))
    return pd.DataFrame(results).sort_values("mean_log10_err"), cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--physics", default="new_model_age_analysis.xlsx")
    ap.add_argument("--intermittency", default="intermittency_by_migration.csv")
    ap.add_argument("--truth", default=None,
                    help="corrected truth xlsx; green rows are excluded")
    ap.add_argument("--wavelengths", default=None,
                    help="final_wavelengths.csv; required for meaningful results")
    ap.add_argument("--out", default="results/age_sensitivity.csv")
    args = ap.parse_args()

    phys = load_physics(args.physics, args.intermittency)
    print(f"physics: {len(phys)} rows, {phys['region'].nunique()} regions")

    if args.truth:
        green = load_green(args.truth)
        keys = list(zip(phys["lat"].round(3), phys["lon"].round(3)))
        keep = [k not in green for k in keys]
        print(f"excluding {len(keep) - sum(keep)} green-tagged rows "
              f"(no dunes / linear / unresolvable)")
        phys = phys[keep].copy()
    else:
        print("NOTE: no --truth given, so green-tagged tiles are NOT excluded")

    phys = attach(phys, args.intermittency, list(INTERMIT_COLS.values()))
    have = [k for k, c in INTERMIT_COLS.items()
            if c in phys.columns and phys[c].notna().any()]
    print(f"intermittency definitions available: {have}")

    if not args.wavelengths:
        raise SystemExit("--wavelengths is required; column I is a superseded "
                         "pre-fix FFT run and is no longer used as a fallback")
    phys = attach(phys, args.wavelengths, ["chosen_lambda_m"])
    n_drop = int(phys["chosen_lambda_m"].isna().sum())
    if n_drop:
        print(f"dropping {n_drop} rows with no FFT wavelength")
    phys = phys[phys["chosen_lambda_m"].notna()].copy()
    phys["modern_used"] = phys["chosen_lambda_m"]
    print(f"using FFT wavelengths for {len(phys)} points")

    laws = ["power", "linear", "geomean"]
    all_res = {}
    for name, targets in TARGET_SETS.items():
        res, cache = run_matrix(phys, have, laws, targets)
        all_res[name] = (res, cache, targets)

        print("\n" + "=" * 78)
        print(f"TARGET SET: {name}   ({len(targets)} dated fields)")
        print("mean log10 distance from the published range; 0 = inside it")
        print("=" * 78)
        print(f"{'intermittency':13s} {'hyst':>8} {'growth':>9} {'err':>7} "
              f"{'median age (yr)':>17}")
        for r in res.itertuples():
            if not np.isfinite(r.mean_log10_err):
                continue
            print(f"{r.intermittency:13s} {r.hysteresis:>8} {r.growth_law:>9} "
                  f"{r.mean_log10_err:>7.2f} {r.median_age_yr:>17.2e}")

        best = res[np.isfinite(res["mean_log10_err"])].iloc[0]
        print(f"\nBEST: intermittency={best.intermittency}, "
              f"hysteresis={best.hysteresis}, growth={best.growth_law} "
              f"(err {best.mean_log10_err:.2f})")
        d = cache[(best.intermittency, best.hysteresis == "with")]
        _, _, detail = score(d, best.growth_law, targets)
        print(f"{'field':14s} {'published':>22} {'modelled':>12} {'log10 dist':>11}")
        for reg in sorted(detail):
            med, e = detail[reg]
            lo, hi = targets[reg]
            rng = f"{lo:.2e} to {'open' if hi is None else f'{hi:.2e}'}"
            print(f"{reg:14s} {rng:>22} {med:>12.2e} {e:>+11.2f}")

    # ---- do the two target sets agree on a winner? ----
    print("\n" + "=" * 78)
    winners = {}
    for name, (res, _, _) in all_res.items():
        b = res[np.isfinite(res["mean_log10_err"])].iloc[0]
        winners[name] = (b.intermittency, b.hysteresis, b.growth_law)
        print(f"{name:12s} winner: {b.intermittency}, {b.hysteresis} hysteresis, "
              f"{b.growth_law}")
    if len(set(winners.values())) == 1:
        print("\nBoth target sets pick the SAME configuration. The disagreement "
              "over published values does not affect the choice.")
    else:
        print("\nThe target sets pick DIFFERENT configurations. Which published "
              "ages to calibrate against is now a decision that matters, and it "
              "should be made on the literature rather than on the ranking.")
    print("=" * 78)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    combined = pd.concat([r.assign(target_set=n) for n, (r, _, _) in all_res.items()])
    combined.to_csv(args.out, index=False)
    print(f"\nfull matrix written to {args.out}")
    print("\nNOTE: this is calibration against a handful of dated fields, not")
    print("independent validation. Differences under ~0.2 log units are not")
    print("meaningful. Note also that every field currently sits far beyond the")
    print("model's fitted boundary of W = 187.5, so all four growth laws are")
    print("being read outside the range they were calibrated on.")


if __name__ == "__main__":
    main()

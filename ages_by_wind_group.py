#!/usr/bin/env python3
"""
ages_by_wind_group.py

Averages dune-field ages within groups of shared wind (migration) direction,
rather than lumping a whole region together. A single dune field can contain
sub-populations that migrate different ways; grouping by (region, migration
direction) keeps dunes of one transport regime together, so each group is a
more physically coherent age estimate.

For every (region, migration-direction) group it reports all four growth-law
ages, summarised both as a geometric mean and a median:

  growth laws : power, linear, geomean(of power&linear), floor(min age)
  summaries   : geomean and median across the points in the group

Ages are computed fresh from the physics inputs so the configuration is
explicit and self-contained. Defaults: FFT wavelengths, and the t0 formula
WITHOUT the hysteresis factor (the configuration that best matched published
dates in the sensitivity analysis). Both are switchable.

Points with no recorded migration direction are kept as a separate
"NONE" group rather than dropped. Green (no-dune) rows are excluded.

Usage:
    python ages_by_wind_group.py \
        --physics new_model_age_analysis.xlsx \
        --coords modes_multimode_allregions_luminance_20m.csv \
        --truth dune_migration_and_lambda_corrected.xlsx \
        --wavelengths results/final_wavelengths.csv \
        --out results/ages_by_wind_group.csv
"""
import os
import argparse
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

COMPASS = {'N':0,'NNE':22.5,'NE':45,'ENE':67.5,'E':90,'ESE':112.5,
           'SE':135,'SSE':157.5,'S':180,'SSW':202.5,'SW':225,'WSW':247.5,
           'W':270,'WNW':292.5,'NW':315,'NNW':337.5}


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
                         Qr_m2s=g(11), interm_pct=g(15)))
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


def load_migration(xlsx):
    """(lat,lon) -> (migration label, excluded flag)."""
    wb = openpyxl.load_workbook(xlsx)
    ws = wb["Sheet1"]
    out = {}
    for r in range(2, ws.max_row + 1):
        lat = ws.cell(row=r, column=1).value
        if lat is None:
            continue
        lon = ws.cell(row=r, column=2).value
        mig = ws.cell(row=r, column=5).value
        fill = ws.cell(row=r, column=4).fill
        tag = fill.start_color.rgb if fill and fill.patternType else "NONE"
        label = str(mig).strip().upper() if mig else "NONE"
        out[(round(float(lat), 3), round(float(lon), 3))] = (label, tag == "FF92D050")
    return out


def attach(df, csv_path, cols):
    src = pd.read_csv(csv_path)
    src["_k"] = list(zip(src["lat"].round(3), src["lon"].round(3)))
    out = df.copy()
    keys = list(zip(out["lat"].round(3), out["lon"].round(3)))
    for c in cols:
        if c in src.columns:
            m = dict(zip(src["_k"], src[c]))
            out[c] = [m.get(k, np.nan) for k in keys]
    return out


def compute_ages(df, use_hyst):
    d = df.copy()
    ok = ((d["incip_real"] > 0) & (d["Qr_m2s"] > 0) &
          (d["interm_pct"] > 0) & (d["modern_used"] > 0))
    d = d[ok].copy()
    l0 = d["incip_real"] / MODEL_INCIP
    Qr_yr = d["Qr_m2s"] * SEC_PER_YR
    t0 = QS_MODEL / (Qr_yr * HYST) if use_hyst else QS_MODEL / Qr_yr
    interm = d["interm_pct"] / 100.0
    W = d["modern_used"] / l0
    N_pow = (W / POW_A) ** (1.0 / POW_B)
    N_lin = (W - LIN_B) / LIN_M
    N_flo = (W_BOUNDARY / POW_A) ** (1.0 / POW_B)
    d["power"] = t0 * N_pow / interm
    d["linear"] = t0 * N_lin / interm
    d["floor"] = t0 * N_flo / interm
    d["geomean"] = np.where((d["power"] > 0) & (d["linear"] > 0),
                            np.sqrt(d["power"].clip(lower=0) * d["linear"].clip(lower=0)),
                            np.nan)
    d["W_modern_l0"] = W
    return d


def geomean(v):
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.exp(np.mean(np.log(v)))) if len(v) else np.nan


def median_pos(v):
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.median(v)) if len(v) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--physics", default="new_model_age_analysis.xlsx")
    ap.add_argument("--coords", default="modes_multimode_allregions_luminance_20m.csv")
    ap.add_argument("--truth", default="dune_migration_and_lambda_corrected.xlsx")
    ap.add_argument("--wavelengths", default=None,
                    help="final_wavelengths.csv; if omitted uses physics modern lambda")
    ap.add_argument("--with-hyst", action="store_true",
                    help="use t0 = Qs/(Qr*0.2) instead of the default t0 = Qs/Qr")
    ap.add_argument("--out", default="results/ages_by_wind_group.csv")
    args = ap.parse_args()

    phys = load_physics(args.physics, args.coords)
    mig = load_migration(os.path.expanduser(args.truth))
    keys = list(zip(phys["lat"].round(3), phys["lon"].round(3)))
    phys["mig_label"] = [mig.get(k, ("NONE", False))[0] for k in keys]
    phys["excluded"] = [mig.get(k, ("NONE", False))[1] for k in keys]
    phys = phys[~phys["excluded"]].copy()

    if args.wavelengths:
        phys = attach(phys, args.wavelengths, ["chosen_lambda_m"])
        phys["modern_used"] = phys["chosen_lambda_m"].fillna(phys["modern"])
    else:
        phys["modern_used"] = phys["modern"]

    use_hyst = args.with_hyst
    print(f"config: {'FFT' if args.wavelengths else 'physics'} wavelengths, "
          f"t0 {'WITH' if use_hyst else 'WITHOUT'} hysteresis")

    d = compute_ages(phys, use_hyst)

    laws = ["power", "linear", "geomean", "floor"]
    rows = []
    for (reg, lab), g in d.groupby(["region", "mig_label"]):
        rec = dict(region=reg, migration=lab, n_points=len(g))
        for law in laws:
            rec[f"{law}_geomean_yr"] = geomean(g[law].to_numpy())
            rec[f"{law}_median_yr"] = median_pos(g[law].to_numpy())
        rec["W_median"] = median_pos(g["W_modern_l0"].to_numpy())
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["region", "n_points"],
                                         ascending=[True, False])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {len(out)} (region, direction) groups to {args.out}")

    # readable console table: linear + floor, geomean summary
    print("\n=== AGE BY WIND-DIRECTION GROUP (geomean; linear & floor, yr) ===")
    print(f"{'region':24s} {'dir':>5} {'n':>4} {'linear':>11} {'floor':>11}")
    cur = None
    for r in out.itertuples():
        if r.region != cur:
            print("-" * 60)
            cur = r.region
        print(f"{r.region:24s} {r.migration:>5} {r.n_points:>4} "
              f"{r.linear_geomean_yr:>11.2e} {r.floor_geomean_yr:>11.2e}")

    # one age per field, thin groups excluded
    append_field_summary(d, args.out, min_n=3)

    # flag fields whose sub-groups disagree a lot (possible mixed generations
    # or unreliable migration directions)
    print("\n=== WITHIN-FIELD SPREAD (linear geomean, groups with n>=3) ===")
    for reg, g in out[out["n_points"] >= 3].groupby("region"):
        vals = g["linear_geomean_yr"].dropna()
        if len(vals) >= 2:
            spread = vals.max() / vals.min()
            flag = "  <-- large spread" if spread >= 3 else ""
            print(f"  {reg:24s} {len(vals)} groups, "
                  f"{vals.min():.1e} to {vals.max():.1e}  ({spread:.1f}x){flag}")


def field_summary(d, laws, min_n=3):
    """One age per field, geomean-weighted across groups with n>=min_n.
    Weighting is by group size, in log space (appropriate for
    order-of-magnitude ages). Thin groups (n<min_n) are excluded."""
    rows = []
    for reg, g in d.groupby("region"):
        # per-group geomean age + size, then weight groups by n
        grp = []
        for lab, gg in g.groupby("mig_label"):
            n = len(gg)
            entry = {"n": n}
            for law in laws:
                v = gg[law].to_numpy()
                v = v[np.isfinite(v) & (v > 0)]
                entry[law] = float(np.exp(np.mean(np.log(v)))) if len(v) else np.nan
            grp.append(entry)
        kept = [e for e in grp if e["n"] >= min_n]
        used = kept if kept else grp   # fall back to all groups if none qualify
        rec = dict(region=reg, n_groups_used=len(used),
                   n_points_used=sum(e["n"] for e in used),
                   n_groups_total=len(grp))
        for law in laws:
            vals = np.array([e[law] for e in used if np.isfinite(e[law])])
            wts = np.array([e["n"] for e in used if np.isfinite(e[law])], dtype=float)
            if len(vals):
                rec[f"{law}_yr"] = float(np.exp(np.average(np.log(vals), weights=wts)))
            else:
                rec[f"{law}_yr"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("region")


def append_field_summary(d, out_csv, min_n=3):
    laws = ["power", "linear", "geomean", "floor"]
    fs = field_summary(d, laws, min_n=min_n)
    base, ext = os.path.splitext(out_csv)
    fpath = f"{base}_field_summary{ext}"
    fs.to_csv(fpath, index=False)
    print(f"\n=== ONE AGE PER FIELD (groups with n>={min_n}, size-weighted geomean) ===")
    print(f"{'region':24s} {'grp':>4} {'pts':>4} {'linear':>11} {'geomean':>11} {'floor':>11}")
    for r in fs.itertuples():
        print(f"{r.region:24s} {r.n_groups_used:>4} {r.n_points_used:>4} "
              f"{r.linear_yr:>11.2e} {r.geomean_yr:>11.2e} {r.floor_yr:>11.2e}")
    print(f"\nwrote field summary to {fpath}")


if __name__ == "__main__":
    main()

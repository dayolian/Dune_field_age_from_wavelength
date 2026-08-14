#!/usr/bin/env python3
"""
ages_by_wind_group.py

Averages dune-field ages within groups of shared WIND direction, rather than
lumping a whole region together or relying on hand-measured dune migration
directions.

WHY WIND RATHER THAN MIGRATION
------------------------------
Earlier versions grouped by the migration direction recorded by eye in the
truth spreadsheet. That measurement is resolution-limited: many tiles have no
direction at all, and thin direction groups were being discarded by the min_n
filter. Grouping instead on the ERA5 best-fit transport sector covers every
tile, needs no hand measurement, and is internally consistent with the
intermittency, which is derived from the same sector.

  group label = compass direction the dominant wind PUSHES the dunes,
                i.e. (center_algo_deg + 180) mod 360, binned to 8 or 16 points.
                center_algo_deg is the direction the wind blows FROM.

INTERMITTENCY
-------------
Default is net_algo: 100 * (count in the best 180-deg sector - count in the
worst) / total. This is what physics column O holds and what analysis.ipynb
computes. It requires no migration direction, so it works at every tile. The
migration-anchored alternatives remain selectable via --intermittency-def for
comparison, but they drop tiles wherever migration was unresolvable.

The script also reports how closely the ERA5 wind direction agrees with the
hand-measured migration direction, using only tiles where migration could be
resolved. That is a check of the two against each other and is reported
separately from the ages.

NOTE ON THE FLOOR LAW: N_flo is a constant, so the floor age is a function of
flux and intermittency only and carries no wavelength information at all.

NOTE ON GROUPING: the field summary is a size-weighted geometric mean of the
per-group geometric means, weighted in log space. That is algebraically
identical to the geometric mean over all points whenever every group clears
min_n. Grouping therefore only changes the field-level answer where the min_n
filter drops a group.

Usage:
    python ages_by_wind_group.py \
        --physics new_model_age_analysis.xlsx \
        --intermittency intermittency_by_migration.csv \
        --truth ~/Desktop/era5_wind_ts/dune_migration_and_lambda_corrected.xlsx \
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

GREEN = "FF92D050"

INTERMIT_COLS = {
    "net_algo":  "intermit_net_algo_pct",
    "raw":       "intermit_raw_pct",
    "gross_mig": "intermit_gross_mig_pct",
    "net_mig":   "intermit_net_mig_pct",
}

COMPASS = {'N':0,'NNE':22.5,'NE':45,'ENE':67.5,'E':90,'ESE':112.5,
           'SE':135,'SSE':157.5,'S':180,'SSW':202.5,'SW':225,'WSW':247.5,
           'W':270,'WNW':292.5,'NW':315,'NNW':337.5}
COMPASS_8 = {k: v for k, v in COMPASS.items() if v % 45 == 0}

LAWS = ["power", "linear", "geomean", "floor"]


def deg_to_label(deg, bins=8):
    """Nearest compass label for a bearing in degrees."""
    if deg is None or not np.isfinite(deg):
        return None
    table = COMPASS_8 if bins == 8 else COMPASS
    return min(table.items(),
               key=lambda kv: abs((kv[1] - deg + 180) % 360 - 180))[0]


def ang_diff(a, b):
    """Smallest absolute angle between two bearings, 0-180 deg."""
    return np.abs((np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0)


def align_and_attach(phys, intermittency_csv):
    """Attach lat/lon, every intermittency definition, and the wind sector.

    The physics spreadsheet orders its region blocks alphabetically while the
    intermittency CSV does not, so alignment must happen inside each region and
    never globally. Physics column O is compared against net_algo as a check:
    if they disagree the pairing is wrong and we raise rather than produce
    silently mispaired ages.
    """
    im = pd.read_csv(intermittency_csv)
    n = len(phys)
    grab = list(INTERMIT_COLS.values()) + ["center_algo_deg", "mig_deg"]
    cols = {c: np.full(n, np.nan) for c in grab}
    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)

    for reg, idx in phys.groupby("region", sort=False).groups.items():
        idx = list(idx)
        sub = im[im.region == reg]
        if len(sub) != len(idx):
            raise ValueError(
                "%s: physics %d rows vs intermittency %d" % (reg, len(idx), len(sub)))
        a = phys.loc[idx, "interm_pct"].values.astype(float)
        b = sub["intermit_net_algo_pct"].values.astype(float)
        off = int((np.abs(a - b) >= 0.01).sum())
        if off:
            raise ValueError(
                "%s: %d rows disagree on intermittency, alignment unsafe" % (reg, off))
        lat[idx] = sub["lat"].values
        lon[idx] = sub["lon"].values
        for c in grab:
            if c not in sub.columns:
                raise ValueError("intermittency CSV has no column %s" % c)
            cols[c][idx] = sub[c].values

    phys = phys.copy()
    phys["lat"], phys["lon"] = lat, lon
    for c, v in cols.items():
        phys[c] = v
    return phys


def load_physics(xlsx, intermittency_csv):
    ws = openpyxl.load_workbook(xlsx, data_only=True)["Sheet1"]
    rows = []
    for r in range(2, ws.max_row + 1):
        reg = ws.cell(row=r, column=5).value
        if reg is None:
            continue
        def g(c):
            v = ws.cell(row=r, column=c).value
            return v if isinstance(v, (int, float)) else np.nan
        rows.append(dict(region=str(reg), incip_real=g(6), modern_colI=g(9),
                         Qr_m2s=g(11), interm_pct=g(15)))
    return align_and_attach(pd.DataFrame(rows), intermittency_csv)


def load_truth(xlsx):
    """(lat,lon) -> (hand migration label or None, green flag)."""
    ws = openpyxl.load_workbook(os.path.expanduser(xlsx))["Sheet1"]
    out = {}
    for r in range(2, ws.max_row + 1):
        lat = ws.cell(row=r, column=1).value
        if lat is None:
            continue
        mig = ws.cell(row=r, column=5).value
        f = ws.cell(row=r, column=4).fill
        tag = f.start_color.rgb if f and f.patternType else "NONE"
        label = str(mig).strip().upper() if mig else None
        out[(round(float(lat), 3),
             round(float(ws.cell(row=r, column=2).value), 3))] = (label, tag == GREEN)
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


def compute_ages(df, interm_col, use_hyst):
    d = df.copy()
    d["_interm_pct"] = d[interm_col]
    ok = ((d["incip_real"] > 0) & (d["Qr_m2s"] > 0) &
          (d["_interm_pct"] > 0) & (d["modern_used"] > 0))
    d = d[ok].copy()
    if len(d) == 0:
        return d
    l0 = d["incip_real"] / MODEL_INCIP
    Qr_yr = d["Qr_m2s"] * SEC_PER_YR
    t0 = QS_MODEL / (Qr_yr * HYST) if use_hyst else QS_MODEL / Qr_yr
    interm = d["_interm_pct"] / 100.0
    W = d["modern_used"] / l0
    N_pow = (W / POW_A) ** (1.0 / POW_B)
    N_lin = (W - LIN_B) / LIN_M
    N_flo = (W_BOUNDARY / POW_A) ** (1.0 / POW_B)   # constant, no wavelength
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


def group_table(d, suffix=""):
    rows = []
    for (reg, lab), g in d.groupby(["region", "wind_dir"]):
        rec = {"region": reg, "wind_dir": lab, f"n_points{suffix}": len(g)}
        for law in LAWS:
            rec[f"{law}_geomean_yr{suffix}"] = geomean(g[law].to_numpy())
            rec[f"{law}_median_yr{suffix}"] = median_pos(g[law].to_numpy())
        rec[f"W_median{suffix}"] = median_pos(g["W_modern_l0"].to_numpy())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["region", f"n_points{suffix}"],
                                          ascending=[True, False])


def field_summary(d, min_n=3, suffix=""):
    rows = []
    for reg, g in d.groupby("region"):
        grp = []
        for lab, gg in g.groupby("wind_dir"):
            entry = {"n": len(gg)}
            for law in LAWS:
                entry[law] = geomean(gg[law].to_numpy())
            grp.append(entry)
        kept = [e for e in grp if e["n"] >= min_n]
        used = kept if kept else grp
        rec = {"region": reg, f"n_groups_used{suffix}": len(used),
               f"n_points_used{suffix}": sum(e["n"] for e in used),
               f"n_points_total{suffix}": len(g),
               f"n_groups_total{suffix}": len(grp),
               f"thin_fallback{suffix}": (not kept)}
        for law in LAWS:
            vals = np.array([e[law] for e in used if np.isfinite(e[law])])
            wts = np.array([e["n"] for e in used if np.isfinite(e[law])], float)
            rec[f"{law}_yr{suffix}"] = (float(np.exp(np.average(np.log(vals), weights=wts)))
                                        if len(vals) else np.nan)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("region")


def agreement_report(d, bins):
    """How closely does the ERA5 wind sector agree with the hand-measured dune
    migration direction? Uses only tiles where migration could be resolved."""
    m = d[d["mig_deg"].notna() & d["center_algo_deg"].notna()].copy()
    if not len(m):
        print("\nno tiles with both a measured migration direction and a wind sector")
        return None
    m["wind_push_deg"] = (m["center_algo_deg"] + 180.0) % 360.0
    m["offset_deg"] = ang_diff(m["wind_push_deg"], m["mig_deg"])
    m["same_label"] = [deg_to_label(w, bins) == deg_to_label(g, bins)
                       for w, g in zip(m["wind_push_deg"], m["mig_deg"])]

    print("\n" + "=" * 78)
    print("WIND DIRECTION vs HAND-MEASURED DUNE MIGRATION DIRECTION")
    print("(tiles with an unresolvable migration direction are excluded)")
    print("=" * 78)
    print(f"{'region':24s} {'n':>5} {'median':>8} {'same':>6} {'<=45':>7} "
          f"{'<=90':>7} {'>135':>7}")
    rows = []
    for reg, g in m.groupby("region"):
        o = g["offset_deg"]
        rows.append(dict(region=reg, n=len(g), median_offset_deg=float(o.median()),
                         frac_same_label=float(g["same_label"].mean()),
                         frac_within_45=float((o <= 45).mean()),
                         frac_within_90=float((o <= 90).mean()),
                         frac_over_135=float((o > 135).mean())))
        print(f"{reg:24s} {len(g):>5} {o.median():>7.0f}d "
              f"{g['same_label'].mean()*100:>5.0f}% {(o<=45).mean()*100:>6.0f}% "
              f"{(o<=90).mean()*100:>6.0f}% {(o>135).mean()*100:>6.0f}%")
    o = m["offset_deg"]
    print("-" * 78)
    print(f"{'ALL':24s} {len(m):>5} {o.median():>7.0f}d "
          f"{m['same_label'].mean()*100:>5.0f}% {(o<=45).mean()*100:>6.0f}% "
          f"{(o<=90).mean()*100:>6.0f}% {(o>135).mean()*100:>6.0f}%")
    print("\n0 deg means the wind pushes dunes exactly the way they were measured")
    print("to move; 180 deg means it opposes them entirely. Randomly related")
    print("directions would give a median near 90 deg and about 25% within 45.")
    return pd.DataFrame(rows).sort_values("region")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--physics", default="new_model_age_analysis.xlsx")
    ap.add_argument("--intermittency", default="intermittency_by_migration.csv")
    ap.add_argument("--truth",
                    default="dune_migration_and_lambda_corrected.xlsx")
    ap.add_argument("--wavelengths", default=None, help="final_wavelengths.csv")
    ap.add_argument("--intermittency-def", default="net_algo",
                    choices=list(INTERMIT_COLS),
                    help="default net_algo: wind-based, needs no migration direction")
    ap.add_argument("--wind-bins", type=int, default=8, choices=[8, 16],
                    help="compass resolution for wind-direction grouping")
    ap.add_argument("--with-hyst", action="store_true",
                    help="use t0 = Qs/(Qr*0.2) instead of the default t0 = Qs/Qr")
    ap.add_argument("--max-offset", type=float, default=None,
                    help="keep only tiles where the ERA5 transport direction agrees "
                         "with the hand-measured migration direction to within this "
                         "many degrees. 90 keeps tiles that agree to within half the "
                         "compass, the appropriate bar for loosely recorded directions")
    ap.add_argument("--drop-unresolvable", action="store_true",
                    help="with --max-offset, also drop tiles that have no hand-measured "
                         "migration direction. Default keeps them, since they fail the "
                         "test only for want of image resolution")
    ap.add_argument("--min-n", type=int, default=1,
                    help="1 = no thin-group filter; every measured tile contributes")
    ap.add_argument("--out", default="results/ages_by_wind_group.csv")
    args = ap.parse_args()

    phys = load_physics(args.physics, args.intermittency)
    truth = load_truth(args.truth)
    keys = list(zip(phys["lat"].round(3), phys["lon"].round(3)))
    phys["hand_mig_label"] = [truth.get(k, (None, False))[0] for k in keys]
    phys["excluded"] = [truth.get(k, (None, False))[1] for k in keys]
    n_ex = int(phys["excluded"].sum())
    phys = phys[~phys["excluded"]].copy()
    print(f"loaded {len(phys) + n_ex} rows, excluded {n_ex} green-tagged")

    if not args.wavelengths:
        raise SystemExit("--wavelengths is required; physics column I is a "
                         "superseded pre-fix FFT run and is not used")
    phys = attach(phys, args.wavelengths, ["chosen_lambda_m"])
    n_drop = int(phys["chosen_lambda_m"].isna().sum())
    if n_drop:
        print(f"dropping {n_drop} rows with no FFT wavelength")
    phys = phys[phys["chosen_lambda_m"].notna()].copy()
    phys["modern_used"] = phys["chosen_lambda_m"]

    push = (phys["center_algo_deg"] + 180.0) % 360.0
    phys["wind_push_deg"] = push
    phys["wind_dir"] = [deg_to_label(x, args.wind_bins) for x in push]
    n_nowind = int(phys["wind_dir"].isna().sum())
    if n_nowind:
        print(f"NOTE: {n_nowind} tiles have no wind sector and are dropped")
        phys = phys[phys["wind_dir"].notna()].copy()

    # ---- wind / migration agreement filter ----
    # Both populations are carried through and written to the output, so the
    # effect of the filter is visible rather than hidden. Report the filtered
    # ages; the unfiltered ones appear in _allpts columns for comparison.
    phys_all = phys.copy()
    # The hand-measured migration directions were recorded loosely, so the
    # sensible bar is whether ERA5 pushes the dunes into the same half of the
    # compass, not whether the two match closely.
    if args.max_offset is not None:
        off = ang_diff(phys["wind_push_deg"], phys["mig_deg"])
        has_mig = phys["mig_deg"].notna()
        keep = (off <= args.max_offset) & has_mig
        if not args.drop_unresolvable:
            keep = keep | (~has_mig)
        n_fail = int((has_mig & (off > args.max_offset)).sum())
        n_nomig = int((~has_mig).sum())
        print(f"agreement filter at {args.max_offset:.0f} deg: dropping {n_fail} tiles "
              f"whose ERA5 direction disagrees with the measured migration")
        print(f"        {n_nomig} tiles have no measured migration direction and are "
              f"{'DROPPED' if args.drop_unresolvable else 'kept'}")
        lost = phys[has_mig & (off > args.max_offset)]
        if len(lost):
            for reg, g in lost.groupby("region"):
                print(f"          {reg:26s} {len(g):>3} dropped")
        phys = phys[keep].copy()
        print(f"        {len(phys)} tiles remain; all {len(phys_all)} are also "
              f"reported in the output for comparison")

    icol = INTERMIT_COLS[args.intermittency_def]
    print(f"config: FFT wavelengths, intermittency={args.intermittency_def}, "
          f"t0 {'WITH' if args.with_hyst else 'WITHOUT'} hysteresis, "
          f"grouped by wind direction ({args.wind_bins}-point), {len(phys)} points")
    n_hand = int(phys["hand_mig_label"].notna().sum())
    print(f"        {n_hand} of {len(phys)} tiles have a hand-measured migration "
          f"direction ({len(phys) - n_hand} unresolvable)")

    d = compute_ages(phys, icol, args.with_hyst)
    lost = len(phys) - len(d)
    if lost:
        print(f"        {lost} dropped for missing or non-positive intermittency")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    base, ext = os.path.splitext(args.out)

    gt = group_table(d)
    fs = field_summary(d, min_n=args.min_n)

    if args.max_offset is not None:
        # Same calculation on every tile, so the CSV carries both the filtered
        # result and the unfiltered one it replaces.
        d_all = compute_ages(phys_all, icol, args.with_hyst)
        gt = gt.merge(group_table(d_all, suffix="_allpts"),
                      on=["region", "wind_dir"], how="outer")
        fs = fs.merge(field_summary(d_all, min_n=args.min_n, suffix="_allpts"),
                      on="region", how="outer")
        note = ("plain columns use only tiles whose ERA5 direction agrees with "
                f"the measured migration within {args.max_offset:.0f} deg; "
                "_allpts columns use every tile")
    else:
        d_all = d
        note = "no agreement filter applied"

    gt.to_csv(args.out, index=False)
    fs.to_csv(f"{base}_field_summary{ext}", index=False)

    print("\n" + "=" * 66)
    print("AGE BY WIND-DIRECTION GROUP (geomean; linear & floor, yr)")
    print("=" * 66)
    print(f"{'region':24s} {'wind':>5} {'n':>4} {'linear':>11} {'floor':>11}")
    cur = None
    # Groups that vanished under the filter still sit in the CSV with blank
    # filtered columns. Printing them as nan rows is noise, so skip them here.
    shown = gt[gt["n_points"].notna()] if args.max_offset is not None else gt
    shown = shown.sort_values(["region", "n_points"], ascending=[True, False])
    for r in shown.itertuples():
        if r.region != cur:
            print("-" * 60); cur = r.region
        print(f"{r.region:24s} {r.wind_dir:>5} {int(r.n_points):>4} "
              f"{r.linear_geomean_yr:>11.2e} {r.floor_geomean_yr:>11.2e}")
    print("\n" + "=" * 78)
    print(f"ONE AGE PER FIELD (groups with n>={args.min_n}, size-weighted geomean)")
    print(f"{'region':24s} {'grp':>4} {'used':>5} {'all':>5} {'linear':>11} "
          f"{'geomean':>11} {'floor':>11}")
    for r in fs.itertuples():
        print(f"{r.region:24s} {r.n_groups_used:>4} {r.n_points_used:>5} "
              f"{r.n_points_total:>5} {r.linear_yr:>11.2e} {r.geomean_yr:>11.2e} "
              f"{r.floor_yr:>11.2e}")
    thin = list(fs.loc[fs.thin_fallback, "region"])
    if thin:
        print(f"\nWARNING: no group reached n>={args.min_n}, age may rest on 1-2 "
              f"points: {', '.join(thin)}")

    print("\n=== WITHIN-FIELD SPREAD (linear geomean, groups with n>=3) ===")
    for reg, g in gt[gt["n_points"] >= 3].groupby("region"):
        vals = g["linear_geomean_yr"].dropna()
        if len(vals) >= 2:
            spread = vals.max() / vals.min()
            flag = "  <-- large spread" if spread >= 3 else ""
            print(f"  {reg:24s} {len(vals)} groups, {vals.min():.1e} to "
                  f"{vals.max():.1e}  ({spread:.1f}x){flag}")

    # ALWAYS computed on the unfiltered population. Running it on the filtered
    # set would be circular, since the filter guarantees every tile passes.
    agree = agreement_report(d_all, args.wind_bins)
    if args.max_offset is not None:
        print("\n  NOTE: the table above uses ALL tiles, not the filtered set.")
        print("  Reporting it from the filtered set would be circular, because")
        print("  the filter guarantees 100 percent agreement within the cutoff.")
    if agree is not None:
        agree.to_csv(f"{base}_wind_vs_migration{ext}", index=False)
        print(f"\nwrote agreement table to {base}_wind_vs_migration{ext}")
    print(f"wrote groups to {args.out} and field summary to "
          f"{base}_field_summary{ext}")
    print(f"  {note}")


if __name__ == "__main__":
    main()

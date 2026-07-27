#!/usr/bin/env python3
"""
reanalyze_intermittency_by_migration.py

Recomputes wind intermittency from the ERA5 hourly time series, anchoring the
180-degree transport sector to the OBSERVED dune migration direction at each
point rather than letting the algorithm pick whichever sector happens to
capture the most above-threshold wind.

FLUX IS UNCHANGED. Sediment flux (q) and u_star_mean are still computed from
the algorithm-chosen dominant sector, exactly as analysis.ipynb does. Only
intermittency is re-anchored.

Intermittency is reported four ways per point so they can be compared:

  raw            100 * (# timesteps above threshold) / total
                 no directional weighting at all

  net_algo       100 * (count in BEST 180-deg sector - count in WORST) / total
                 what analysis.ipynb currently computes; maximised by
                 construction, so it is the largest of these values

  gross_mig      100 * (count in the migration-anchored sector) / total
                 all transport winds pushing the dunes the observed way

  net_mig        100 * (count in migration sector
                        - count in the opposing sector) / total
                 the same "minus the reverse direction" logic as net_algo,
                 but anchored to observed migration instead of the best fit

The migration-anchored sector is centred on (migration_direction + 180),
because ERA5 wdir is the direction wind blows FROM while migration direction
is where the dunes move TO.

Rows marked green in the truth spreadsheet (no dunes / linear dunes / too
coarse to resolve) are flagged 'excluded' and dropped from the summary.

Because age = t0 * N / intermittency, and net_mig <= net_algo by construction,
switching to a migration-anchored value can only make ages the same or older.

Usage:
    python reanalyze_intermittency_by_migration.py \
        --base ~/Desktop/era5_wind_ts \
        --truth dune_migration_and_lambda_corrected.xlsx \
        --reference flux_lambda_by_coordinate.csv \
        --out intermittency_by_migration.csv
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
import openpyxl

# ---- constants (identical to analysis.ipynb) ----
Z_MEAS  = 10.0
KAPPA   = 0.4
Z0      = 1e-3
RHO_AIR = 1.225
A_THR   = 0.0123
G       = 9.81
GAMMA   = 0.0003

REGION_PROPS = {                      # (grain density kg/m^3, diameter m)
    "AlNafudDesert":            (2650.0, 0.00038),
    "Chad":                     (2650.0, 0.000125),
    "GrandErgOccidental":       (2650.0, 0.0002),
    "GrandErgOrientalAlgeria":  (2650.0, 0.00016),
    "Karakum":                  (2680.0, 0.000219),
    "LencoisMaranhensesBrazil": (2650.0, 0.000266),
    "LutDesert":                (2650.0, 0.00018),
    "Namib":                    (2650.0, 0.0002),
    "RegistanDesert":           (2640.0, 0.00017),
    "RubAlKhali":               (2650.0, 0.000153),
    "Taklamakan":               (2550.0, 0.000118),
    "Tengger":                  (2670.0, 0.0003),
    "Thar":                     (2660.0, 0.00018),
    "WesternSahara":            (2650.0, 0.00023),
    "WhiteSands":               (2300.0, 0.0004),
}

COMPASS = {'N':0,'NNE':22.5,'NE':45,'ENE':67.5,'E':90,'ESE':112.5,
           'SE':135,'SSE':157.5,'S':180,'SSW':202.5,'SW':225,'WSW':247.5,
           'W':270,'WNW':292.5,'NW':315,'NNW':337.5}


# ---------------- wind helpers ----------------
def read_all_years(folder):
    """Concatenate u10/v10 from every era5_wind_*.csv in a coordinate folder."""
    files = sorted(glob.glob(os.path.join(folder, "era5_wind_*.csv")))
    us, vs = [], []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=["u10", "v10"])
        except Exception:
            continue
        us.append(pd.to_numeric(df["u10"], errors="coerce").to_numpy())
        vs.append(pd.to_numeric(df["v10"], errors="coerce").to_numpy())
    if not us:
        return None, None
    u = np.concatenate(us)
    v = np.concatenate(vs)
    ok = np.isfinite(u) & np.isfinite(v)
    return u[ok], v[ok]


def wind_speed_and_dir(u10, v10):
    """Exactly as analysis.ipynb: wdir is the direction the wind blows FROM."""
    spd = np.sqrt(u10 * u10 + v10 * v10)
    era5_angle = np.degrees(np.arctan2(v10, u10))
    wdir = (270.0 - era5_angle) % 360.0
    return spd, wdir


def count_in_180(wdir, center):
    d = (wdir - center + 180.0) % 360.0 - 180.0
    return int(np.sum(np.abs(d) <= 90.0))


def best_worst_180(wdir_above, step=1.0):
    if wdir_above.size == 0:
        return np.nan, 0, np.nan, 0
    centers = np.arange(0.0, 360.0, step)
    counts = np.array([count_in_180(wdir_above, c) for c in centers])
    i, j = int(np.argmax(counts)), int(np.argmin(counts))
    return float(centers[i]), int(counts[i]), float(centers[j]), int(counts[j])


def sector_mask(wdir, center, half=90.0):
    d = (wdir - center + 180.0) % 360.0 - 180.0
    return np.abs(d) <= half


def sediment_flux(u_star, u_star_cr, rho_s, d):
    excess = u_star ** 2 - u_star_cr ** 2
    pref = 25.0 * (RHO_AIR / rho_s) * np.sqrt(d / G)
    return float(pref * max(excess, 0.0))


# ---------------- migration directions ----------------
def load_migration(xlsx):
    """(lat,lon) -> (region, migration_deg, excluded_flag)."""
    wb = openpyxl.load_workbook(xlsx)
    ws = wb["Sheet1"]
    out = {}
    for r in range(2, ws.max_row + 1):
        lat = ws.cell(row=r, column=1).value
        if lat is None:
            continue
        lon = ws.cell(row=r, column=2).value
        reg = str(ws.cell(row=r, column=3).value)
        mig = ws.cell(row=r, column=5).value
        fill = ws.cell(row=r, column=4).fill
        tag = fill.start_color.rgb if fill and fill.patternType else "NONE"
        excluded = (tag == "FF92D050")
        deg = COMPASS.get(str(mig).strip().upper()) if mig else None
        out[(round(float(lat), 2), round(float(lon), 2))] = (
            reg, deg if deg is not None else np.nan, excluded)
    return out


def parse_lat_lon(name):
    try:
        a, b = name.split("_lon")
        return float(a.replace("lat", "")), float(b)
    except Exception:
        return None, None


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".", help="era5_wind_ts directory")
    ap.add_argument("--truth", default="dune_migration_and_lambda_corrected.xlsx")
    ap.add_argument("--reference", default=None,
                    help="flux_lambda_by_coordinate.csv, to verify this script "
                         "reproduces the original numbers")
    ap.add_argument("--out", default="intermittency_by_migration.csv")
    args = ap.parse_args()

    base = os.path.expanduser(args.base)
    mig_map = load_migration(os.path.expanduser(args.truth))
    print(f"loaded migration directions for {len(mig_map)} points")

    folders = sorted(n for n in os.listdir(base)
                     if os.path.isdir(os.path.join(base, n))
                     and n.startswith("lat") and "_lon" in n)
    print(f"found {len(folders)} coordinate folders")

    rows, no_mig = [], 0
    for i, name in enumerate(folders, 1):
        if i % 100 == 0:
            print(f"  {i}/{len(folders)}")
        lat, lon = parse_lat_lon(name)
        if lat is None:
            continue
        key = (round(lat, 2), round(lon, 2))
        if key not in mig_map:
            continue
        region, mig_deg, excluded = mig_map[key]
        props = REGION_PROPS.get(region)
        if props is None:
            continue
        rho_s, d = props

        u10, v10 = read_all_years(os.path.join(base, name))
        if u10 is None or u10.size == 0:
            continue

        sigma = rho_s / RHO_AIR
        u_star_cr = (A_THR * ((sigma * G * d) + (GAMMA / (RHO_AIR * d)))) ** 0.5
        U_cr = (u_star_cr / KAPPA) * np.log(Z_MEAS / Z0)

        spd, wdir = wind_speed_and_dir(u10, v10)
        above = spd >= U_cr
        n_tot = float(wdir.size)
        n_above = float(np.sum(above))

        rec = dict(lat=lat, lon=lon, region=region, excluded=excluded,
                   mig_deg=mig_deg, u_star_cr=u_star_cr, U_cr=U_cr,
                   n_timesteps=int(n_tot), n_above=int(n_above),
                   intermit_raw_pct=100.0 * n_above / n_tot)

        if n_above == 0:
            rec.update(intermit_net_algo_pct=0.0, intermit_gross_mig_pct=0.0,
                       intermit_net_mig_pct=0.0, center_algo_deg=np.nan,
                       center_mig_deg=np.nan, sector_offset_deg=np.nan,
                       u_star_mean=np.nan, q_m2s=0.0)
            rows.append(rec)
            continue

        wa = wdir[above]

        # --- algorithm-chosen sector (used for net_algo AND for flux) ---
        c_algo, cnt_max, _, cnt_min = best_worst_180(wa)
        net_algo = 100.0 * (cnt_max - cnt_min) / n_tot

        # FLUX: unchanged -- still uses the algorithm's dominant sector
        use_flux = above & sector_mask(wdir, c_algo)
        if np.sum(use_flux) == 0:
            use_flux = above
        u_star_mean = KAPPA * float(np.mean(spd[use_flux])) / np.log(Z_MEAS / Z0)
        q = sediment_flux(u_star_mean, u_star_cr, rho_s, d)

        # --- INTERMITTENCY: anchored to observed migration direction ---
        if np.isfinite(mig_deg):
            c_mig = (mig_deg + 180.0) % 360.0      # wind must blow FROM here
            cnt_mig = count_in_180(wa, c_mig)
            cnt_opp = count_in_180(wa, (c_mig + 180.0) % 360.0)
            gross_mig = 100.0 * cnt_mig / n_tot
            net_mig = 100.0 * (cnt_mig - cnt_opp) / n_tot
            offset = abs((c_algo - c_mig + 180.0) % 360.0 - 180.0)
        else:
            no_mig += 1
            c_mig = gross_mig = net_mig = offset = np.nan

        rec.update(intermit_net_algo_pct=net_algo,
                   intermit_gross_mig_pct=gross_mig,
                   intermit_net_mig_pct=net_mig,
                   center_algo_deg=c_algo, center_mig_deg=c_mig,
                   sector_offset_deg=offset,
                   u_star_mean=u_star_mean, q_m2s=q)
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} rows to {args.out}")
    if no_mig:
        print(f"  ({no_mig} points had no migration direction recorded)")

    # ---- verify this script reproduces the original notebook ----
    if args.reference and os.path.exists(args.reference):
        ref = pd.read_csv(args.reference)
        j = df.merge(ref[["lat", "lon", "intermittency_pct", "u_star_cr", "q_m2s"]],
                     on=["lat", "lon"], how="inner", suffixes=("", "_ref"))
        if len(j):
            print(f"\ncross-check vs {args.reference} ({len(j)} matched points):")
            print(f"  max |u_star_cr diff|          : "
                  f"{(j['u_star_cr'] - j['u_star_cr_ref']).abs().max():.2e}")
            print(f"  median |net_algo - original|  : "
                  f"{(j['intermit_net_algo_pct'] - j['intermittency_pct']).abs().median():.4f} pct pts")
            print(f"  median |q - original q|       : "
                  f"{(j['q_m2s'] - j['q_m2s_ref']).abs().median():.2e} m2/s")
            print("  (all should be ~0 if this reproduces analysis.ipynb)")

    # ---- per-region comparison ----
    d2 = df[~df["excluded"]].copy()
    print("\n=== PER-REGION MEDIAN INTERMITTENCY, % (excluded rows dropped) ===")
    print(f"{'region':26s} {'raw':>7} {'net_algo':>9} {'gross_mig':>10} "
          f"{'net_mig':>8} {'netmig/algo':>12} {'offset':>7}  n")
    for reg, g in d2.groupby("region"):
        ratio = (g["intermit_net_mig_pct"] / g["intermit_net_algo_pct"]).median()
        print(f"{reg:26s} {g['intermit_raw_pct'].median():>7.2f} "
              f"{g['intermit_net_algo_pct'].median():>9.2f} "
              f"{g['intermit_gross_mig_pct'].median():>10.2f} "
              f"{g['intermit_net_mig_pct'].median():>8.2f} "
              f"{ratio:>12.2f} {g['sector_offset_deg'].median():>7.0f}  {len(g)}")

    print("\nnetmig/algo < 1 lowers intermittency and therefore RAISES age by")
    print("the inverse factor. 'offset' is how far the observed migration")
    print("sector sits from the algorithm's best-fit sector.")


if __name__ == "__main__":
    main()

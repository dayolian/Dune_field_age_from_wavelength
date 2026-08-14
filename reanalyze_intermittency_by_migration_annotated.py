#!/usr/bin/env python3
"""
reanalyze_intermittency_by_migration.py  --  ANNOTATED

Your code is unchanged. Everything added is a comment. Comments beginning with
"FLAG" mark something worth acting on.

This is step 6. It is the only script that touches the raw ERA5 wind record,
and it produces the file every later step depends on. Original docstring
follows.
=============================================================================

Recomputes wind intermittency from the ERA5 hourly time series, anchoring the
180-degree transport sector to the OBSERVED dune migration direction at each
point rather than letting the algorithm pick whichever sector happens to
capture the most above-threshold wind.

FLUX IS UNCHANGED. Sediment flux (q) and u_star_mean are still computed from
the algorithm-chosen dominant sector, exactly as analysis.ipynb does. Only
intermittency is re-anchored.

    ^^^ Worth holding onto. Flux and intermittency are anchored to DIFFERENT
    sectors within the same run. Flux always uses the algorithm's best-fit
    sector; only intermittency has a migration-anchored variant. Since you
    settled on net_algo, both now use the same sector and this asymmetry does
    not bite, but it would if you ever switched.

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
import openpyxl                # again, needed for the cell FILL COLOURS

# ---- constants (identical to analysis.ipynb) ----
Z_MEAS  = 10.0      # ERA5 reports wind at 10 m above ground
KAPPA   = 0.4       # von Karman constant, the slope of the log wind profile
Z0      = 1e-3      # aerodynamic roughness length, 1 mm, i.e. smooth sand
                    # FLAG: one value for every field. Real z0 varies with
                    # vegetation and crusting, and the vegetated relict dunes
                    # you flagged would be rougher than 1 mm. u_star scales
                    # as 1/log(z/z0), so this is a modest but systematic term.
RHO_AIR = 1.225     # kg/m3
A_THR   = 0.0123    # empirical coefficient in the Shao & Lu threshold formula
G       = 9.81
GAMMA   = 0.0003    # interparticle cohesion, matters only for fine grains

# Grain density and diameter per field, taken from the literature.
# THESE ARE THE ONLY PER-FIELD PHYSICAL INPUTS in the whole pipeline.
# Everything else about a field comes from imagery or from ERA5.
REGION_PROPS = {                      # (grain density kg/m^3, diameter m)
    "AlNafudDesert":            (2650.0, 0.00038),   # 380 um, coarse
    "Chad":                     (2650.0, 0.000125),  # 125 um, the finest
    # FLAG: Chad is set to QUARTZ at 2650. The Bodele dunes are largely
    # diatomite fragments, which are far less dense. This is the row to
    # revisit if you pursue that. Note the 125 um here does NOT match the
    # ~262 um implied by the u_star_cr in the output CSV, so check which
    # value actually produced the stored numbers.
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
    "WhiteSands":               (2300.0, 0.0004),    # gypsum, correctly lighter
}
# FLAG: one grain size per FIELD, not per tile. Chad spans 97 tiles from the
# Manga sand sheets to the Bodele diatomite and they all get 125 um. That is
# a real limitation worth one sentence in the methods.

# Text compass labels to degrees, for reading your hand-written directions.
COMPASS = {'N':0,'NNE':22.5,'NE':45,'ENE':67.5,'E':90,'ESE':112.5,
           'SE':135,'SSE':157.5,'S':180,'SSW':202.5,'SW':225,'WSW':247.5,
           'W':270,'WNW':292.5,'NW':315,'NNW':337.5}


# ---------------- wind helpers ----------------
def read_all_years(folder):
    """Concatenate u10/v10 from every era5_wind_*.csv in a coordinate folder."""
    # One folder per coordinate, one CSV per year inside it. This glues every
    # year into a single long series of eastward and northward wind components.
    # Sorting matters only for reproducibility, since order is irrelevant to a
    # counting statistic.
    files = sorted(glob.glob(os.path.join(folder, "era5_wind_*.csv")))
    us, vs = [], []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=["u10", "v10"])
        except Exception:
            continue
            # FLAG: silent skip. A corrupt or truncated year is dropped with no
            # message, and the point still gets a result computed from fewer
            # hours. n_timesteps in the output is your only clue. Worth
            # checking that column is uniform across points before publishing.
        us.append(pd.to_numeric(df["u10"], errors="coerce").to_numpy())
        vs.append(pd.to_numeric(df["v10"], errors="coerce").to_numpy())
    if not us:
        return None, None
    u = np.concatenate(us)
    v = np.concatenate(vs)
    ok = np.isfinite(u) & np.isfinite(v)   # drop any gaps
    return u[ok], v[ok]


def wind_speed_and_dir(u10, v10):
    """Exactly as analysis.ipynb: wdir is the direction the wind blows FROM."""
    # Speed is Pythagoras on the two components.
    spd = np.sqrt(u10 * u10 + v10 * v10)

    # Direction needs a convention change. arctan2 gives a MATHEMATICAL angle,
    # anticlockwise from east. Meteorology wants a compass bearing, clockwise
    # from north, and reports where wind comes FROM rather than goes TO.
    # (270 - angle) mod 360 does both flips at once.
    # Example: wind blowing toward the east has era5_angle 0, and this returns
    # 270, meaning it blows FROM the west. Correct.
    era5_angle = np.degrees(np.arctan2(v10, u10))
    wdir = (270.0 - era5_angle) % 360.0
    return spd, wdir


def count_in_180(wdir, center):
    # How many wind records fall within 90 degrees either side of `center`,
    # i.e. inside a half-circle facing that way.
    # The (x + 180) % 360 - 180 pattern wraps the difference into -180..+180
    # so that, say, 350 and 10 degrees are treated as 20 apart rather than 340.
    d = (wdir - center + 180.0) % 360.0 - 180.0
    return int(np.sum(np.abs(d) <= 90.0))


def best_worst_180(wdir_above, step=1.0):
    # Rotate a half-circle window through all 360 orientations, one degree at
    # a time, and record how many transport-competent winds fall inside at each
    # orientation. Return the best and worst.
    #
    # THIS FUNCTION IS THE DEFINITION OF net_algo. The best sector is the
    # dominant transport direction; the worst is its opposite. Their difference
    # is the net directional signal.
    #
    # Note it is maximised by construction, which is why net_algo is always
    # the largest of the four intermittency values.
    if wdir_above.size == 0:
        return np.nan, 0, np.nan, 0
    centers = np.arange(0.0, 360.0, step)
    counts = np.array([count_in_180(wdir_above, c) for c in centers])
    i, j = int(np.argmax(counts)), int(np.argmin(counts))
    return float(centers[i]), int(counts[i]), float(centers[j]), int(counts[j])


def sector_mask(wdir, center, half=90.0):
    # Same wrap trick, but returns a true/false array rather than a count, so
    # it can select which wind records to average for the flux.
    d = (wdir - center + 180.0) % 360.0 - 180.0
    return np.abs(d) <= half


def sediment_flux(u_star, u_star_cr, rho_s, d):
    # Saturated sand flux. Transport scales with how far shear velocity exceeds
    # the threshold, squared, and max(...,0) means no wind below threshold moves
    # any sand at all. The prefactor carries grain density and size.
    # Units are m2/s, a volume per unit width per unit time.
    excess = u_star ** 2 - u_star_cr ** 2
    pref = 25.0 * (RHO_AIR / rho_s) * np.sqrt(d / G)
    return float(pref * max(excess, 0.0))


# ---------------- migration directions ----------------
def load_migration(xlsx):
    """(lat,lon) -> (region, migration_deg, excluded_flag)."""
    # Same workbook as step 5, read the same way and for the same reason:
    # the green tagging is a cell colour, not a value.
    wb = openpyxl.load_workbook(xlsx)
    ws = wb["Sheet1"]
    out = {}
    for r in range(2, ws.max_row + 1):
        lat = ws.cell(row=r, column=1).value
        if lat is None:
            continue
        lon = ws.cell(row=r, column=2).value
        reg = str(ws.cell(row=r, column=3).value)
        mig = ws.cell(row=r, column=5).value        # your hand direction, e.g. "NW"
        fill = ws.cell(row=r, column=4).fill
        tag = fill.start_color.rgb if fill and fill.patternType else "NONE"
        excluded = (tag == "FF92D050")              # green

        # Text to degrees. A blank cell, or anything not in COMPASS, becomes
        # NaN. That is the 33 tiles where the imagery was too coarse to read
        # a migration direction.
        deg = COMPASS.get(str(mig).strip().upper()) if mig else None

        # FLAG: keys rounded to 2 decimal places here, but step 5 and the age
        # scripts round to 3. Harmless on a 0.125 degree grid, where every
        # coordinate is exact to 3 places anyway, but the inconsistency is
        # the kind of thing that silently drops rows if the grid ever changes.
        out[(round(float(lat), 2), round(float(lon), 2))] = (
            reg, deg if deg is not None else np.nan, excluded)
    return out


def parse_lat_lon(name):
    # Turns a folder name like "lat27.50_lon-002.75" back into numbers.
    try:
        a, b = name.split("_lon")
        return float(a.replace("lat", "")), float(b)
    except Exception:
        return None, None


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".", help="era5_wind_ts directory")
    # FLAG: default is the current directory, but the wind folders live in
    # ~/Desktop/era5_wind_ts. Always pass --base explicitly.
    ap.add_argument("--truth", default="dune_migration_and_lambda_corrected.xlsx")
    ap.add_argument("--reference", default=None,
                    help="flux_lambda_by_coordinate.csv, to verify this script "
                         "reproduces the original numbers")
    ap.add_argument("--out", default="intermittency_by_migration.csv")
    args = ap.parse_args()

    base = os.path.expanduser(args.base)
    mig_map = load_migration(os.path.expanduser(args.truth))
    print(f"loaded migration directions for {len(mig_map)} points")

    # Find every coordinate folder. THE ORDER THESE COME OUT IN is what the
    # alignment fix in steps 7 and 8 keys on. Sorting is by folder name, which
    # is lat-then-lon as text, and that is NOT the same ordering as the
    # alphabetical-by-region ordering in the physics spreadsheet. That mismatch
    # was the alignment bug.
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
            continue                    # folder with no matching spreadsheet row
        region, mig_deg, excluded = mig_map[key]
        props = REGION_PROPS.get(region)
        if props is None:
            continue                    # region not in the grain table
        rho_s, d = props

        u10, v10 = read_all_years(os.path.join(base, name))
        if u10 is None or u10.size == 0:
            continue

        # ---- THRESHOLD. The single most consequential calculation here. ----
        sigma = rho_s / RHO_AIR         # grain density relative to air, ~2160

        # Shao & Lu threshold shear velocity. Two competing terms: the first
        # is gravity holding the grain down, which grows with size, and the
        # second is cohesion sticking it to its neighbours, which grows as
        # grains get SMALLER. Their sum has a minimum near 80-100 um, which is
        # why medium sand is the easiest thing on Earth to blow around.
        u_star_cr = (A_THR * ((sigma * G * d) + (GAMMA / (RHO_AIR * d)))) ** 0.5

        # Convert that shear velocity into a 10 m wind speed, by inverting the
        # logarithmic wind profile. This is the number an anemometer would see.
        U_cr = (u_star_cr / KAPPA) * np.log(Z_MEAS / Z0)

        spd, wdir = wind_speed_and_dir(u10, v10)
        above = spd >= U_cr             # the transport-competent hours
        n_tot = float(wdir.size)        # every hour on record
        n_above = float(np.sum(above))

        rec = dict(lat=lat, lon=lon, region=region, excluded=excluded,
                   mig_deg=mig_deg, u_star_cr=u_star_cr, U_cr=U_cr,
                   n_timesteps=int(n_tot), n_above=int(n_above),
                   # DEFINITION 1 of 4: raw, no direction at all
                   intermit_raw_pct=100.0 * n_above / n_tot)

        # A site where the wind never crosses threshold gets zeros rather than
        # a crash. None of your fields hit this.
        if n_above == 0:
            rec.update(intermit_net_algo_pct=0.0, intermit_gross_mig_pct=0.0,
                       intermit_net_mig_pct=0.0, center_algo_deg=np.nan,
                       center_mig_deg=np.nan, sector_offset_deg=np.nan,
                       u_star_mean=np.nan, q_m2s=0.0)
            rows.append(rec)
            continue

        wa = wdir[above]                # directions of the competent hours only

        # --- algorithm-chosen sector (used for net_algo AND for flux) ---
        # DEFINITION 2 of 4: net_algo. THIS IS THE ONE YOU USE.
        # Best half-circle minus worst half-circle, as a percentage of all
        # hours. Needs no hand measurement, which is why it works at every
        # tile including the 33 with no readable migration direction.
        c_algo, cnt_max, _, cnt_min = best_worst_180(wa)
        net_algo = 100.0 * (cnt_max - cnt_min) / n_tot

        # FLUX: unchanged -- still uses the algorithm's dominant sector
        # Average the wind speed over competent hours inside the dominant
        # sector, convert back to shear velocity, and feed the flux law.
        # Note the average is of SPEED, then converted, rather than converting
        # each hour and averaging. Because flux goes as u_star squared, and
        # the mean of squares exceeds the square of the mean, this
        # systematically UNDERESTIMATES flux relative to summing hour by hour.
        # A lower flux means a larger t0 and therefore an OLDER age.
        use_flux = above & sector_mask(wdir, c_algo)
        if np.sum(use_flux) == 0:
            use_flux = above
        u_star_mean = KAPPA * float(np.mean(spd[use_flux])) / np.log(Z_MEAS / Z0)
        q = sediment_flux(u_star_mean, u_star_cr, rho_s, d)

        # --- INTERMITTENCY: anchored to observed migration direction ---
        if np.isfinite(mig_deg):
            # Wind must blow FROM the opposite side to push dunes TOWARD the
            # observed migration bearing. Hence the +180.
            c_mig = (mig_deg + 180.0) % 360.0      # wind must blow FROM here
            cnt_mig = count_in_180(wa, c_mig)
            cnt_opp = count_in_180(wa, (c_mig + 180.0) % 360.0)
            # DEFINITION 3: gross_mig, everything pushing the right way
            gross_mig = 100.0 * cnt_mig / n_tot
            # DEFINITION 4: net_mig, minus everything pushing back
            net_mig = 100.0 * (cnt_mig - cnt_opp) / n_tot
            # How far the algorithm's best sector sits from the one your
            # measurement implies. THIS is the raw material for the whole
            # wind-versus-migration agreement analysis, and the 42 degree
            # median comes from this column.
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

    # One row per coordinate, 1005 of them. Excluded rows are KEPT in the file
    # and only dropped from the printed summary, which is why steps 7 and 8 can
    # use this file's row order for alignment even though they exclude green
    # tiles themselves.
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} rows to {args.out}")
    if no_mig:
        print(f"  ({no_mig} points had no migration direction recorded)")

    # ---- verify this script reproduces the original notebook ----
    # A regression test against the original analysis.ipynb output. If the
    # three printed differences are near zero, this script is a faithful
    # reimplementation and only the added definitions are new. Worth running
    # once and quoting in the methods.
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
    # Green rows dropped here only, for the printed table.
    d2 = df[~df["excluded"]].copy()
    print("\n=== PER-REGION MEDIAN INTERMITTENCY, % (excluded rows dropped) ===")
    print(f"{'region':26s} {'raw':>7} {'net_algo':>9} {'gross_mig':>10} "
          f"{'net_mig':>8} {'netmig/algo':>12} {'offset':>7}  n")
    for reg, g in d2.groupby("region"):
        # Ratio of the two net definitions, the direct lever on age.
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

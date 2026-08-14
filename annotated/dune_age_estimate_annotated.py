#!/usr/bin/env python3
"""
dune_age_estimate.py  --  ANNOTATED

Your code is unchanged. Everything added is a comment. Comments beginning with
"FLAG" mark something worth acting on.

This is step 7. It produces PER-TILE ages. It is a diagnostic, not the source
of the numbers in the paper — those come from step 8, ages_by_wind_group.py,
which does the same physics but groups by wind direction and does not fall back
to the superseded column I. Original docstring follows.
=============================================================================

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

    ^^^ RESOLVED SINCE THIS WAS WRITTEN. The 0.2 is physically motivated, the
    duration of saltation once grain motion has begun, and Mackenzie wants it
    kept. It is now reported as a RANGE rather than chosen between: the low end
    uses Qs/Qr and the high end Qs/(Qr*0.2), a factor of five apart. Note this
    file hardcodes only the WITH-hysteresis end; step 8 has a --with-hyst switch
    and defaults to WITHOUT. So this script's ages are 5x those of step 8.
"""
import argparse
import numpy as np
import pandas as pd
import openpyxl

# FLAG: hardcoded filename rather than an argument, so the intermittency CSV
# must sit in the working directory under exactly this name.
INTERMITTENCY_CSV = "intermittency_by_migration.csv"

# ---- model constants ----
# Every one of these comes from the ReSCAL simulation, not from your data.
MODEL_INCIP = 43.429     # incipient wavelength IN MODEL UNITS. Dividing the
                         # real incipient wavelength by this gives l0, the
                         # metres-per-model-block conversion for that site.
QS_MODEL    = 0.125      # the model's saturated flux, in model units
HYST        = 0.2        # saltation hysteresis; see the docstring note above
SEC_PER_YR  = 31536000.0 # 365 * 24 * 3600

# power law fit  W = pow_a * t^pow_b
# Describes the EARLY part of the ReSCAL run, where spacing grows fast then
# slows. Exponent 0.2026 means growth is strongly decelerating.
POW_A, POW_B = 11.09, 0.2026

# linear fit     W = lin_m * t + lin_b
# Describes the LATER part of the same run, once growth has settled into a
# steady creep. The 110.2 intercept is where the fitted line crosses zero
# time, and it is why anything below W = 110.2 returns a negative age.
LIN_M, LIN_B = 0.0000826, 110.2

# model boundary (max W actually simulated) -> minimum-age floor
# FLAG: THE SINGLE MOST IMPORTANT NUMBER IN THIS FILE. The simulation only ran
# out to W = 187.5. Your fields sit at W of 800 to 4800, so every age here is
# read 4 to 25 times beyond the range the laws were fitted on.
W_BOUNDARY   = 187.5

W_FLAG       = 5000.0    # W above this = highly uncertain extrapolation
                         # FLAG: this threshold flags only ~200 of 1005 tiles,
                         # which understates the problem. Everything above
                         # 187.5 is extrapolated, not just above 5000.



def align_latlon_from_intermittency(phys, intermittency_csv):
    """Attach lat/lon to physics rows using per-region file order of the
    intermittency CSV. Physics blocks are alphabetical; the CSV blocks are not,
    so alignment must be done within each region, never globally."""
    # THIS FUNCTION IS THE BUG FIX. The physics spreadsheet has no coordinates
    # at all, only a region name per row, so coordinates have to be inferred
    # from row order. The original code took them from the modes CSV, which is
    # sorted lat-ascending. The physics rows actually follow the intermittency
    # CSV's per-region file order. The two orderings coincide only where both
    # lat and lon are positive, which is why 242 rows in Namib, Western Sahara
    # and Grand Erg Occidental got another tile's coordinates.
    import numpy as np, pandas as pd
    im = pd.read_csv(intermittency_csv)
    lat = np.full(len(phys), np.nan)
    lon = np.full(len(phys), np.nan)

    # sort=False preserves the physics file's own region order rather than
    # re-sorting alphabetically. Matching happens WITHIN each region.
    for reg, idx in phys.groupby("region", sort=False).groups.items():
        idx = list(idx)
        sub = im[im.region == reg]

        # Guard 1: same number of rows in both files for this region.
        if len(sub) != len(idx):
            raise ValueError(
                "%s: physics %d rows vs intermittency %d" % (reg, len(idx), len(sub)))

        # Guard 2: the real check. Both files carry an intermittency value per
        # row, so if the pairing is correct those values must agree. If they
        # do not, the alignment is wrong and the script REFUSES to continue
        # rather than producing silently mispaired ages. This is what makes
        # the fix trustworthy rather than merely plausible.
        a = phys.loc[idx, "interm_pct"].values.astype(float)
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

def load_physics(xlsx, coords_csv=None):
    """
    Load per-site physics. The xlsx has no lat/lon, so if coords_csv (a modes
    CSV with region+lat+lon for the SAME 1005 tiles in the SAME order) is given,
    attach lat/lon by order -- validated safe because both are the full tile set.
    """
    # ^^^ FLAG: this docstring is now WRONG. It describes the old, buggy
    # behaviour. coords_csv is no longer read at all; its presence merely
    # triggers the intermittency-based alignment below. Worth rewriting.
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    # data_only=True returns computed VALUES rather than formulas, which
    # matters because these columns are calculated in Excel.
    ws = wb["Sheet1"]
    rows = []
    for r in range(2, ws.max_row + 1):
        reg = ws.cell(row=r, column=5).value
        if reg is None:
            continue
        # Small helper: return the cell value if it is a number, else NaN.
        # Catches blanks, text and Excel error strings without crashing.
        def g(c):
            v = ws.cell(row=r, column=c).value
            return v if isinstance(v, (int, float)) else np.nan
        rows.append(dict(
            region=str(reg),
            incip_real=g(6),          # col F  [m]   incipient wavelength, real
            model_incip=g(7),         # col G  [l0]  read but never used; the
                                      #             constant 43.429 is used instead
            modern=g(9),              # col I  [m]  (may be replaced)
                                      # FLAG: this is the SUPERSEDED pre-fix FFT
                                      # run. Its values sit on the old log-k bin
                                      # grid and run about half the current FFT.
                                      # It must never be used as a fallback.
            Qr_m2s=g(11),             # col K  [m^2/s]  sediment flux, from step 6
            interm_pct=g(15),         # col O  [percent] = intermit_net_algo_pct
        ))
    phys = pd.DataFrame(rows)

    # FLAG: the alignment only runs if --coords is passed, even though the
    # coords file is not actually read. Omit --coords and you get no lat/lon
    # at all, and the FFT join below then raises. Always pass it.
    if coords_csv:
        phys = align_latlon_from_intermittency(phys, INTERMITTENCY_CSV)
    return phys


def attach_fft_wavelengths(phys, wl_csv):
    """Replace physics 'modern' with chosen_lambda_m, joined EXACTLY on lat/lon."""
    if "lat" not in phys.columns:
        raise ValueError("physics has no lat/lon -- pass --coords so the FFT join "
                         "can match on coordinates instead of fragile row order")
    wl = pd.read_csv(wl_csv)
    if "chosen_lambda_m" not in wl.columns:
        raise ValueError("wavelengths CSV missing 'chosen_lambda_m'")

    # Build (lat, lon) lookup keys rounded to 3 decimals, so floating-point
    # dust cannot break a match between two files that should agree exactly.
    def key(df):
        return list(zip(df["lat"].round(3), df["lon"].round(3)))
    wl_map = dict(zip(key(wl), wl["chosen_lambda_m"]))
    conf_map = dict(zip(key(wl), wl["confidence"])) if "confidence" in wl.columns else {}

    out = phys.copy()
    ph_keys = key(out)
    out["modern_fft"] = [wl_map.get(k, np.nan) for k in ph_keys]
    out["fft_confidence"] = [conf_map.get(k, "") for k in ph_keys]

    # ================= THE BUG THAT MAKES THIS FILE A DIAGNOSTIC =============
    # A tile with no FFT wavelength is one that step 5 DELIBERATELY skipped,
    # because it was green-tagged as having no dunes, linear dunes, or
    # unresolvable imagery. This line fills those 272 tiles back in from the
    # superseded column I, resurrecting exactly the tiles you excluded, using
    # numbers you know to be wrong.
    #
    # Step 8 drops them instead, which is correct. That is the main reason the
    # paper's numbers come from step 8 and not from here.
    out["modern_used"] = out["modern_fft"].fillna(out["modern"])
    # ========================================================================

    # At least it records which is which, so the damage is visible in the CSV.
    out["wavelength_source"] = np.where(out["modern_fft"].notna(), "fft", "physics_colI")

    matched = out["modern_fft"].notna().sum()
    print(f"  lat/lon join: matched {matched}/{len(wl)} FFT rows to physics tiles")
    if matched < len(wl):
        print(f"  NOTE: {len(wl)-matched} FFT rows had no lat/lon match in physics")
    return out


def compute_ages(df):
    d = df.copy()
    # drop rows lacking essentials
    # Any of these being zero or missing would make the arithmetic meaningless
    # or divide by zero, so those tiles are simply removed.
    ok = (d["incip_real"] > 0) & (d["Qr_m2s"] > 0) & \
         (d["interm_pct"] > 0) & (d["modern_used"] > 0)
    d = d[ok].copy()

    # ---- STEP A: how many metres is one model block at this site? ----
    # The simulation is dimensionless. l0 is the scale factor that turns model
    # blocks into metres, and it differs per site because incipient dune
    # spacing depends on grain size and air density.
    l0 = d["incip_real"] / MODEL_INCIP                      # m/block

    # ---- STEP B: how many years is one model timestep at this site? ----
    # Flux per second becomes flux per year, then t0 is the model's saturated
    # flux divided by the real one. A site where sand moves fast has a large
    # Qr, so a small t0: each timestep represents less real time.
    Qr_yr = d["Qr_m2s"] * SEC_PER_YR
    t0 = QS_MODEL / (Qr_yr * HYST)                          # years/step
    # FLAG: HYST is applied unconditionally here. Step 8 makes it optional and
    # defaults to OFF, so this file's ages are five times step 8's.

    interm = d["interm_pct"] / 100.0                        # percent -> fraction

    # ---- STEP C: measured spacing in model units ----
    W = d["modern_used"] / l0                               # l0 units

    # ---- STEP D: invert the growth laws to get elapsed timesteps ----
    # Both laws are rearranged from "W as a function of t" into "t as a
    # function of W", which is why the exponent is 1/0.2026 rather than 0.2026.
    N_pow = (W / POW_A) ** (1.0 / POW_B)
    N_lin = (W - LIN_B) / LIN_M                             # may go negative
    # ^^^ This is the White Sands problem in one line. W = 71 there, below the
    # 110.2 intercept, so the subtraction is negative and so is the age. Not a
    # wrong answer, an undefined one.

    # ---- STEP E: timesteps to years ----
    # Multiply by years-per-timestep, then DIVIDE by intermittency because sand
    # only moves during that fraction of hours. A site where wind is competent
    # 10 per cent of the time needs ten times the elapsed years to accumulate
    # the same transport.
    age_pow = t0 * N_pow / interm
    age_lin = t0 * N_lin / interm

    # geometric mean only where both positive
    # A hedge between the two laws. In practice the power law returns 1e10 to
    # 1e12 years at these W values, so the geometric mean is dragged far too
    # old and is not used in the paper.
    both_pos = (age_pow > 0) & (age_lin > 0)
    age_geo = np.where(both_pos, np.sqrt(age_pow.clip(lower=0) * age_lin.clip(lower=0)), np.nan)

    # minimum-age floor at model boundary W=187.5
    # FLAG: N_floor is a CONSTANT, identical for every tile, so the floor age
    # carries no wavelength information at all. It is a flux-and-intermittency
    # clock rather than a dune measurement, and it describes a dune field an
    # order of magnitude smaller in spacing than any of yours. Already dropped
    # from the published comparison for that reason.
    N_floor = (W_BOUNDARY / POW_A) ** (1.0 / POW_B)
    age_floor = t0 * N_floor / interm

    # Everything intermediate is kept in the output, which is what makes this
    # file useful as a diagnostic: you can see l0, t0, W and N per tile and
    # work out which term is driving an odd result.
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
    # MEDIAN rather than mean, and only over positive finite values, so a
    # single negative or infinite tile cannot swamp a region's summary.
    # Note step 8 uses the geometric mean instead, which is the better choice
    # for quantities spanning orders of magnitude.
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
    # Carried over from the MATLAB scripts. FLAG: this is an outcome-based
    # filter, keeping tiles because their wavelength looks reasonable rather
    # than for any stated physical reason. Fine as a diagnostic comparison,
    # but it should not appear in the paper without a justification.
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
    # FLAG: help text describes the old behaviour. The file is not read; the
    # flag now just switches the alignment on. Misleading, worth rewording.
    ap.add_argument("--wavelengths", default=None,
                    help="final_wavelengths.csv (optional; replaces modern lambda)")
    ap.add_argument("--out", default="results/dune_ages.csv")
    args = ap.parse_args()

    phys = load_physics(args.physics, coords_csv=args.coords)
    print(f"loaded physics: {len(phys)} rows, {phys['region'].nunique()} regions")

    if args.wavelengths:
        phys = attach_fft_wavelengths(phys, args.wavelengths)
        n_fft = (phys["wavelength_source"] == "fft").sum()
        # THIS PRINTED LINE IS THE WARNING. When it says "272 fell back to
        # col I" it is telling you 272 excluded tiles have been resurrected
        # with superseded wavelengths. Step 8 prints "dropping 272" instead.
        print(f"attached FFT wavelengths to {n_fft} rows "
              f"({(phys['wavelength_source']=='physics_colI').sum()} fell back to col I)")
    else:
        # No FFT file at all: run entirely on the superseded column I.
        phys["modern_used"] = phys["modern"]
        phys["wavelength_source"] = "physics_colI"
        phys["fft_confidence"] = ""

    d = compute_ages(phys)

    import os                       # FLAG: import buried mid-function
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    keep = ["region", "wavelength_source", "fft_confidence",
            "incip_real", "l0_m", "Qr_m2s", "t0_years", "intermittency_frac",
            "modern_used", "W_modern_l0",
            "age_power_years", "age_linear_years", "age_geomean_years",
            "age_floor_years", "flagged_W_gt_5000", "linear_unphysical"]
    # FLAG: lat and lon are computed but NOT written out, so the per-tile CSV
    # cannot be mapped or joined to anything without rerunning the alignment.
    # Adding them to this list would cost nothing and help a lot.
    d[keep].to_csv(args.out, index=False)
    print(f"\nwrote {len(d)} age estimates to {args.out}")
    summarize(d)


if __name__ == "__main__":
    main()

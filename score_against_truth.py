#!/usr/bin/env python3
"""
score_against_truth.py

Evaluate the FFT wavelength estimates against the hand-measured ground truth in
dune_migration_and_lambda.xlsx.

IMPORTANT (methodology): the ground-truth spreadsheet is used ONLY to SCORE the
method here. It is never fed into the selection logic in
fft_full_geotiff_multimode.py. That separation is what keeps the "answer key"
from leaking into the method, so the accuracy you measure is honest.

Two ways to run:

  (A) Score an existing pipeline output CSV (the one run_pipeline_from_points.py
      writes) against the spreadsheet -- no tiles needed:

        python score_against_truth.py \
            --truth dune_migration_and_lambda.xlsx \
            --pred  pipeline_modes.csv

  (B) Run the patched FFT live on a tiles directory and score that:

        python score_against_truth.py \
            --truth dune_migration_and_lambda.xlsx \
            --tiles_dir tiles

Reports overall accuracy, per-region accuracy, and a region-holdout split so you
can see whether the method generalizes to regions it wasn't inspected on.

Scoring rules:
  - A row COUNTS AS CORRECT if any predicted mode's lambda is within TOL_FRAC of
    the actual lambda (default 15%).
  - Rows the spreadsheet marks GREEN (no dunes / linear dunes / unresolvable) are
    scored separately as "should abstain": success there means the method returns
    NO confident mode, not a number.
  - YELLOW rows (originally correct) are tracked as a regression guard.
"""
import argparse
import numpy as np
import pandas as pd
import openpyxl

TOL_FRAC = 0.15
YELLOW = "FFFFFF00"
GREEN = "FF92D050"


def load_truth(xlsx_path):
    """Read lat, lon, region, actual lambda, and colour tag per row."""
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb["Sheet1"]
    header = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
    col = {name: i + 1 for i, name in enumerate(header) if name}
    rows = []
    for r in range(2, ws.max_row + 1):
        lat = ws.cell(row=r, column=col["lat"]).value
        if lat is None:
            continue
        fill = ws.cell(row=r, column=col["FFT lambda"]).fill
        tag = fill.start_color.rgb if fill and fill.patternType else "NONE"
        colour = "yellow" if tag == YELLOW else ("green" if tag == GREEN else "none")
        rows.append({
            "lat": float(lat),
            "lon": float(ws.cell(row=r, column=col["lon"]).value),
            "region": str(ws.cell(row=r, column=col["region"]).value),
            "actual_lambda": ws.cell(row=r, column=col["actual lambda (m)"]).value,
            "orig_fft": ws.cell(row=r, column=col["FFT lambda"]).value,
            "colour": colour,
        })
    return pd.DataFrame(rows)


def key(lat, lon):
    return (round(float(lat), 3), round(float(lon), 3))


def preds_from_csv(pred_csv):
    """Map (lat,lon) -> list of predicted lambdas from a pipeline output CSV."""
    df = pd.read_csv(pred_csv)
    out = {}
    for _, row in df.iterrows():
        k = key(row["lat"], row["lon"])
        out.setdefault(k, []).append(float(row["lambda_m"]))
    return out


def preds_from_tiles(truth_df, tiles_dir):
    """Run the patched FFT on each tile named after its lat/lon; return preds map."""
    import os
    from fft_full_geotiff_multimode import analyze_geotiff_multimode

    def make_tile_basename(region, lat, lon):
        r = region.replace(" ", "_")
        return f"{r}_lat{lat:+05.2f}_lon{lon:+06.2f}".replace(".", "p").replace("+", "")

    out = {}
    for _, row in truth_df.iterrows():
        base = make_tile_basename(row["region"], row["lat"], row["lon"])
        tif = os.path.join(tiles_dir, base + ".tif")
        if not os.path.exists(tif):
            continue
        try:
            modes = analyze_geotiff_multimode(tif, out_prefix=None)
        except Exception as e:
            print(f"  FFT failed on {base}: {e}")
            continue
        out[key(row["lat"], row["lon"])] = [m["lambda_m"] for m in modes]
    return out


def row_correct(actual, pred_lambdas, tol=TOL_FRAC):
    if not pred_lambdas or actual is None:
        return False
    return any(abs(p - actual) / actual <= tol for p in pred_lambdas)


def score(truth_df, preds):
    dune = truth_df[truth_df["colour"].isin(["yellow", "none"])].copy()
    green = truth_df[truth_df["colour"] == "green"].copy()

    dune["pred"] = dune.apply(lambda r: preds.get(key(r["lat"], r["lon"]), []), axis=1)
    dune["has_pred"] = dune["pred"].apply(len) > 0
    dune["correct"] = dune.apply(
        lambda r: row_correct(r["actual_lambda"], r["pred"]), axis=1)

    green["pred"] = green.apply(lambda r: preds.get(key(r["lat"], r["lon"]), []), axis=1)
    green["abstained"] = green["pred"].apply(len) == 0

    return dune, green


def report(dune, green, label=""):
    scored = dune[dune["has_pred"]]
    n = len(dune)
    n_scored = len(scored)
    acc = dune["correct"].mean() if n else float("nan")
    acc_scored = scored["correct"].mean() if n_scored else float("nan")
    yellow = dune[dune["colour"] == "yellow"]
    yellow_acc = yellow["correct"].mean() if len(yellow) else float("nan")

    print(f"\n===== {label} =====")
    print(f"Dune rows: {n}  (with a prediction: {n_scored})")
    print(f"  overall accuracy (no-pred counts as miss): {acc*100:5.1f}%")
    print(f"  accuracy among rows that got a prediction: {acc_scored*100:5.1f}%")
    print(f"  YELLOW regression guard ({len(yellow)} rows):  {yellow_acc*100:5.1f}%  (want ~100%)")
    if len(green):
        print(f"  GREEN abstain rate ({len(green)} rows):        "
              f"{green['abstained'].mean()*100:5.1f}%  (higher = better)")

    print("\n  per-region accuracy:")
    for reg, g in dune.groupby("region"):
        gs = g[g["has_pred"]]
        a = g["correct"].mean() * 100
        print(f"    {reg:28s} n={len(g):3d}  acc={a:5.1f}%  "
              f"(scored {len(gs):3d})")


def region_holdout(truth_df, preds, holdout_regions):
    dune, green = score(truth_df, preds)
    train = dune[~dune["region"].isin(holdout_regions)]
    test = dune[dune["region"].isin(holdout_regions)]
    print("\n########## REGION HOLDOUT ##########")
    print(f"Holdout regions: {sorted(holdout_regions)}")
    for name, subset in [("TRAIN (inspected)", train), ("TEST (held out)", test)]:
        if len(subset):
            a = subset["correct"].mean() * 100
            ascored = subset[subset["has_pred"]]["correct"].mean() * 100 \
                if subset["has_pred"].any() else float("nan")
            print(f"  {name:22s} n={len(subset):3d}  "
                  f"overall={a:5.1f}%  among-scored={ascored:5.1f}%")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="dune_migration_and_lambda.xlsx")
    ap.add_argument("--pred", help="pipeline output CSV (lat, lon, lambda_m, ...)")
    ap.add_argument("--tiles_dir", help="run patched FFT live on this tiles dir")
    ap.add_argument("--tol", type=float, default=TOL_FRAC)
    ap.add_argument("--holdout", nargs="*", default=None,
                    help="region names to hold out (default: last 5 alphabetically)")
    return ap.parse_args()


def main():
    args = parse_args()
    global TOL_FRAC
    TOL_FRAC = args.tol

    truth = load_truth(args.truth)
    print(f"Loaded {len(truth)} ground-truth rows across "
          f"{truth['region'].nunique()} regions.")

    if args.pred:
        preds = preds_from_csv(args.pred)
    elif args.tiles_dir:
        preds = preds_from_tiles(truth, args.tiles_dir)
    else:
        raise SystemExit("Provide either --pred CSV or --tiles_dir.")

    dune, green = score(truth, preds)
    report(dune, green, label=f"OVERALL (tol={TOL_FRAC:.0%})")

    regions = sorted(truth["region"].unique())
    holdout = args.holdout if args.holdout else regions[-5:]
    region_holdout(truth, preds, set(holdout))


if __name__ == "__main__":
    main()

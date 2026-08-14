#!/usr/bin/env python3
"""
plot_age_maps.py

Maps dune-field ages with arrows showing the direction the ERA5 wind pushes the
dunes. Arrow colour is the modelled age of that tile on a log scale.

Two outputs:
  1. A global overview with every tile.
  2. A grid of per-region panels, so individual dune fields are readable.

Optionally a third mode plots one arrow per wind-direction group at the group
centroid, which is less cluttered but hides within-group scatter.

The age physics is imported from ages_by_wind_group.py rather than duplicated,
so the numbers here always match the field summary. No constants are redefined.

Arrow direction is (center_algo_deg + 180) mod 360, the compass bearing the
dominant transport sector pushes toward. center_algo_deg is the direction the
wind blows FROM.

Colour is log10(age in years). The default is the linear growth law with
t0 = Qs/Qr. Pass --with-hyst for the t0 = Qs/(Qr*0.2) end of the bracket,
which multiplies every age by 5 and shifts the whole colour scale.

Cartopy is used for coastlines if it is installed, and quietly skipped if not.
The maps are readable either way.

Usage:
    python plot_age_maps.py \
        --physics new_model_age_analysis.xlsx \
        --intermittency intermittency_by_migration.csv \
        --truth ~/Desktop/era5_wind_ts/dune_migration_and_lambda_corrected.xlsx \
        --wavelengths results/final_wavelengths.csv \
        --outdir figures
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from ages_by_wind_group import (
    load_physics, load_truth, attach, compute_ages, deg_to_label, INTERMIT_COLS,
)

from matplotlib.ticker import FuncFormatter
from matplotlib.collections import LineCollection

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:
    HAVE_CARTOPY = False

CMAP = "viridis"
LAND = "#f4f1ea"
OCEAN = "#dce7ef"
COAST = "#5a5a5a"

_WORLD = None
COASTLINES_PATH = None


def world_outlines():
    """Continent/country outlines as a list of Nx2 lon/lat arrays, or None.

    Only used when cartopy is absent. Reads a GeoJSON with the plain json
    module, so it needs no geospatial libraries at all -- geopandas and fiona
    are bypassed entirely. Point the COASTLINES env var (or --coastlines) at a
    world GeoJSON; get one with:

      curl -L -o ~/Desktop/world.geojson \
        https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson
      export COASTLINES=~/Desktop/world.geojson
    """
    global _WORLD
    if _WORLD is not None:
        return _WORLD if len(_WORLD) else None

    path = COASTLINES_PATH or os.environ.get("COASTLINES")
    if path:
        path = os.path.expanduser(path)
    if not path or not os.path.exists(path):
        _WORLD = []
        return None

    import json
    try:
        with open(path) as fh:
            gj = json.load(fh)
    except Exception as e:
        print(f"  could not read {path}: {e}")
        _WORLD = []
        return None

    rings = []
    feats = gj.get("features", [gj]) if isinstance(gj, dict) else []
    for feat in feats:
        geom = feat.get("geometry") or feat
        t, co = geom.get("type"), geom.get("coordinates")
        if t == "Polygon":
            polys = [co]
        elif t == "MultiPolygon":
            polys = co
        else:
            continue
        for poly in polys:
            for ring in poly:
                a = np.asarray(ring, dtype=float)
                if a.ndim == 2 and a.shape[1] >= 2 and len(a) > 2:
                    rings.append(a[:, :2])
    _WORLD = rings
    return rings if rings else None


def deg_formatter(axis):
    """Tick labels like 30°N / 15°W."""
    pos, neg = ("N", "S") if axis == "y" else ("E", "W")
    def f(v, _):
        h = pos if v >= 0 else neg
        return f"{abs(v):g}\u00b0{h}"
    return FuncFormatter(f)


def build_tiles(args):
    """Per-tile ages plus lat/lon and the wind push bearing."""
    phys = load_physics(args.physics, args.intermittency)
    truth = load_truth(args.truth)
    keys = list(zip(phys["lat"].round(3), phys["lon"].round(3)))
    phys["hand_mig_label"] = [truth.get(k, (None, False))[0] for k in keys]
    phys["excluded"] = [truth.get(k, (None, False))[1] for k in keys]
    phys = phys[~phys["excluded"]].copy()

    phys = attach(phys, args.wavelengths, ["chosen_lambda_m"])
    phys = phys[phys["chosen_lambda_m"].notna()].copy()
    phys["modern_used"] = phys["chosen_lambda_m"]

    push = (phys["center_algo_deg"] + 180.0) % 360.0
    phys["wind_push_deg"] = push
    phys["wind_dir"] = [deg_to_label(x, args.wind_bins) for x in push]
    phys = phys[phys["wind_dir"].notna()].copy()

    d = compute_ages(phys, INTERMIT_COLS[args.intermittency_def], args.with_hyst)
    d = d[np.isfinite(d["linear"]) & (d["linear"] > 0)].copy()
    d["logage"] = np.log10(d["linear"])
    return d


def group_centroids(d):
    """One row per (region, wind_dir): mean position, geomean age, count."""
    rows = []
    for (reg, w), g in d.groupby(["region", "wind_dir"]):
        v = g["linear"].to_numpy()
        v = v[np.isfinite(v) & (v > 0)]
        if not len(v):
            continue
        rows.append(dict(region=reg, wind_dir=w, n=len(g),
                         lat=g["lat"].mean(), lon=g["lon"].mean(),
                         # circular mean of the bearings
                         wind_push_deg=np.degrees(np.arctan2(
                             np.mean(np.sin(np.radians(g["wind_push_deg"]))),
                             np.mean(np.cos(np.radians(g["wind_push_deg"]))))) % 360,
                         linear=float(np.exp(np.mean(np.log(v))))))
    out = pd.DataFrame(rows)
    out["logage"] = np.log10(out["linear"])
    return out


def draw_arrows(ax, df, norm, scale, width, edge=False):
    """Quiver of compass-bearing arrows coloured by log10 age."""
    ang = np.radians(df["wind_push_deg"].to_numpy())
    u, v = np.sin(ang), np.cos(ang)      # bearing: 0=N, 90=E
    kw = dict(angles="uv", scale=scale, scale_units="width",
              width=width, cmap=CMAP, norm=norm, pivot="middle", zorder=3)
    if edge:
        kw.update(edgecolor="k", linewidth=0.3)
    if HAVE_CARTOPY and hasattr(ax, "projection"):
        kw["transform"] = ccrs.PlateCarree()
    return ax.quiver(df["lon"].to_numpy(), df["lat"].to_numpy(), u, v,
                     df["logage"].to_numpy(), **kw)


def style_axes(ax, lons, lats, pad_frac=0.12):
    """Lat/lon axes with a physically sensible aspect ratio."""
    lo0, lo1 = float(np.min(lons)), float(np.max(lons))
    la0, la1 = float(np.min(lats)), float(np.max(lats))
    dx = max(lo1 - lo0, 0.5); dy = max(la1 - la0, 0.5)
    px, py = dx * pad_frac, dy * pad_frac
    ax.set_xlim(lo0 - px, lo1 + px)
    ax.set_ylim(la0 - py, la1 + py)
    mid = np.radians((la0 + la1) / 2.0)
    # 1 deg lat is ~1/cos(lat) times longer than 1 deg lon on the ground
    ax.set_aspect(1.0 / max(np.cos(mid), 0.2))
    rings = world_outlines()
    if rings:
        ax.add_collection(LineCollection(rings, colors=COAST, linewidths=0.5,
                                         zorder=0))
    ax.xaxis.set_major_formatter(deg_formatter("x"))
    ax.yaxis.set_major_formatter(deg_formatter("y"))
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.tick_params(labelsize=7)


def add_colorbar(fig, norm, label, cax=None, ax=None):
    sm = ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, ax=ax, fraction=0.025, pad=0.02,
                      extend="both")
    lo, hi = int(np.floor(norm.vmin)), int(np.ceil(norm.vmax))
    ticks = list(range(lo, hi + 1))
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"$10^{{{t}}}$" for t in ticks])
    cb.set_label(label, fontsize=9)
    return cb


def cartopy_axes(ax, lons, lats, pad, label_size=7):
    """Coastlines, land/ocean fill and labelled lat/lon gridlines."""
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=LAND, zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor=COAST, linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS, edgecolor=COAST, linewidth=0.25,
                   alpha=0.5, zorder=2)
    ax.add_feature(cfeature.LAKES, facecolor=OCEAN, edgecolor=COAST,
                   linewidth=0.3, zorder=1)
    lo0, lo1 = float(np.min(lons)), float(np.max(lons))
    la0, la1 = float(np.min(lats)), float(np.max(lats))
    dx = max(lo1 - lo0, 0.5) * pad
    dy = max(la1 - la0, 0.5) * pad
    ax.set_extent([lo0 - dx, lo1 + dx, la0 - dy, la1 + dy],
                  crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4,
                      color="gray", linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": label_size}
    gl.ylabel_style = {"size": label_size}
    return gl


def plot_global(d, norm, outpath, title):
    if HAVE_CARTOPY:
        fig = plt.figure(figsize=(16, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        cartopy_axes(ax, d.lon, d.lat, pad=0.10, label_size=8)
    else:
        fig = plt.figure(figsize=(16, 8))
        ax = plt.axes()
        style_axes(ax, d.lon, d.lat, pad_frac=0.08)

    draw_arrows(ax, d, norm, scale=45, width=0.0022)
    tkw = {"transform": ccrs.PlateCarree()} if HAVE_CARTOPY else {}
    for reg, g in d.groupby("region"):
        ax.text(g.lon.mean(), g.lat.max() + 0.6, reg, fontsize=7,
                ha="center", va="bottom", zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.75), **tkw)
    add_colorbar(fig, norm, "dune field age (yr)", ax=ax)
    ax.set_title(title, fontsize=11)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


def plot_panels(d, norm, outpath, title, ncols=4):
    regs = sorted(d.region.unique())
    nrows = int(np.ceil(len(regs) / ncols))
    kw = {"subplot_kw": {"projection": ccrs.PlateCarree()}} if HAVE_CARTOPY else {}
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), **kw)
    axes = np.atleast_1d(axes).ravel()
    for ax, reg in zip(axes, regs):
        g = d[d.region == reg]
        if HAVE_CARTOPY:
            cartopy_axes(ax, g.lon, g.lat, pad=0.18, label_size=6)
        else:
            style_axes(ax, g.lon, g.lat, pad_frac=0.18)
        draw_arrows(ax, g, norm, scale=26, width=0.007, edge=True)
        gm = float(np.exp(np.mean(np.log(g["linear"]))))
        ax.set_title(f"{reg}\nn={len(g)}   geomean {gm:.2e} yr", fontsize=8)
    for ax in axes[len(regs):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0.02, 0.93, 0.97])
    cax = fig.add_axes([0.945, 0.12, 0.012, 0.72])
    add_colorbar(fig, norm, "age (yr)", cax=cax)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--physics", default="new_model_age_analysis.xlsx")
    ap.add_argument("--intermittency", default="intermittency_by_migration.csv")
    ap.add_argument("--truth", default="dune_migration_and_lambda_corrected.xlsx")
    ap.add_argument("--wavelengths", default="results/final_wavelengths.csv")
    ap.add_argument("--intermittency-def", default="net_algo",
                    choices=list(INTERMIT_COLS))
    ap.add_argument("--wind-bins", type=int, default=8, choices=[8, 16])
    ap.add_argument("--with-hyst", action="store_true",
                    help="t0 = Qs/(Qr*0.2); multiplies every age by 5")
    ap.add_argument("--groups", action="store_true",
                    help="one arrow per wind group at its centroid, not per tile")
    ap.add_argument("--coastlines", default=None,
                    help="world GeoJSON for continent outlines (only needed "
                         "if cartopy is not installed)")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    global COASTLINES_PATH
    COASTLINES_PATH = args.coastlines

    d = build_tiles(args)
    print(f"{len(d)} tiles with ages, {d.region.nunique()} regions")
    if HAVE_CARTOPY:
        print("  cartopy found: coastlines and labelled gridlines enabled")
    elif world_outlines() is not None:
        print(f"  drawing outlines from {COASTLINES_PATH or os.environ.get('COASTLINES')}")
    else:
        print("  no coastline data: axes will have degree labels but no")
        print("  continent outlines. Fix with either:")
        print("    conda install -c conda-forge cartopy")
        print("  or download a world GeoJSON and pass --coastlines PATH")

    plotted = group_centroids(d) if args.groups else d
    if args.groups:
        print(f"collapsed to {len(plotted)} wind-direction groups")

    # Shared colour scale across both figures so they are comparable.
    # Clipped to the 2nd-98th percentile: a handful of tiles sit within a few
    # units of the linear intercept (W = 110.2) where the age collapses toward
    # zero, and letting them set the scale washes out everything else. The
    # colourbar is drawn with arrows to show values fall outside both ends.
    lo, hi = np.percentile(plotted.logage, [2, 98])
    norm = Normalize(vmin=np.floor(lo), vmax=np.ceil(hi))
    print(f"colour scale (2nd-98th pct): {10**norm.vmin:.0e} to "
          f"{10**norm.vmax:.0e} yr")

    near = d[d["W_modern_l0"] < 130]
    if len(near):
        print(f"NOTE: {len(near)} tiles have W < 130, close to the linear "
              f"intercept of 110.2 where the age collapses toward zero:")
        for reg, g in near.groupby("region"):
            print(f"      {reg:24s} {len(g):3d} tiles, "
                  f"ages {g.linear.min():.1e} to {g.linear.max():.1e} yr")

    tag = ("groups" if args.groups else "tiles")
    tag += "_hyst" if args.with_hyst else ""
    hy = "t0 = Qs/(Qr x 0.2)" if args.with_hyst else "t0 = Qs/Qr"
    sub = (f"arrows point the way ERA5 wind pushes the dunes | "
           f"{args.intermittency_def} intermittency | linear growth law | {hy}")

    os.makedirs(args.outdir, exist_ok=True)
    plot_global(plotted, norm, os.path.join(args.outdir, f"age_map_global_{tag}.png"),
                "Dune field ages and transport directions\n" + sub)
    plot_panels(plotted, norm, os.path.join(args.outdir, f"age_map_panels_{tag}.png"),
                "Dune field ages by region\n" + sub)


if __name__ == "__main__":
    main()

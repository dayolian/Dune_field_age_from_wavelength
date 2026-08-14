# Methods, in run order

Eight scripts in the main path, plus validation and plotting off to the side. Steps 1 to 4 build the imagery and measure spacing. Step 6 builds the wind record. Steps 7 and 8 combine them into ages.

---

## Step 1. `make_sampling_points_from_kmz_folder.py`

Takes hand-drawn KMZ polygons, one per dune field, and lays a regular grid of sampling points inside each. Constants are hardcoded at the top rather than passed as arguments, so grid spacing and the KMZ folder are edited in the file.

**Outputs** a list of 1,005 lat/lon sampling points tagged by region.

**Used by** step 3, which downloads one image tile per point.

---

## Step 2. `download_s2_box_SZA_luminance.py`

Pulls a Sentinel-2 image box centred on a single point from Google Earth Engine, selecting the least-cloudy scene in the date window and converting to a luminance band at 20 m per pixel. Solar zenith angle is recorded so that illumination geometry can be checked.

**Outputs** one GeoTIFF per point into `tiles/`.

**Not run directly.** Step 3 calls it in a loop.

---

## Step 3. `run_pipeline_from_points.py`

The download and FFT driver. For each sampling point it calls step 2 to fetch the tile if it is not already on disk, then calls step 4 to extract the spectrum. Existing tiles are skipped, which is why `tiles/` behaves as data rather than as a cache.

**Outputs** `modes_multimode_allregions_luminance_20m.csv`, holding every candidate spectral peak for every tile.

**Used by** step 5, which chooses among those peaks.

---

## Step 4. `fft_full_geotiff_multimode.py`

The spectral engine, imported by step 3 rather than run alone. Reads a tile, applies a high-pass filter at 6000 m to remove regional topographic trend, takes a two-dimensional FFT, radially averages the power spectrum into 100 logarithmic wavenumber bins between 30 and 10,000 m, and returns every peak that clears a prominence threshold of 0.20 of the maximum.

**Outputs** a list of candidate wavelengths per tile, with prominence, feeding step 3's CSV.

---

## Step 5. `measure_wavelengths_nearest.py`

Chooses one wavelength per tile. It reads the truth spreadsheet, skips every green-tagged row (no dunes, linear dunes, or unresolvable imagery), and for each remaining tile picks the FFT peak nearest the hand-measured wavelength. The `--tol 0.25` argument does not affect that choice; it only labels a tile high or low confidence according to whether the chosen peak lands within 25 per cent of the hand measurement.

**Outputs** `results/final_wavelengths.csv`, 733 tiles with `chosen_lambda_m` and a confidence flag.

**Used by** steps 7 and 8 as the wavelength input. This is the slow step, 25 to 40 minutes.

**Note.** Because selection is anchored to the hand measurement, this is a refinement of a manual measurement rather than an independent prediction, and should be described that way.

---

## Step 6. `reanalyze_intermittency_by_migration.py`

Independent of steps 1 to 5 and skippable if its output already exists. For each sampling point it reads the hourly ERA5 wind time series, computes shear velocity, compares it against a threshold set by grain size and grain density, and counts the fraction of hours above threshold under four different definitions.

**Outputs** `intermittency_by_migration.csv`, carrying `intermit_raw_pct`, `intermit_net_algo_pct`, `intermit_gross_mig_pct`, `intermit_net_mig_pct`, the sediment flux `q_m2s`, the best-fit wind sector centre `center_algo_deg`, and the hand-measured migration direction in degrees.

**Used by** steps 7 and 8. It also supplies the per-region row order that fixes the lat/lon alignment.

**Which definition is used.** `net_algo`, the best 180 degree sector minus the worst, chosen because it needs no hand-measured migration direction and therefore works at every tile.

---

## Step 7. `dune_age_estimate.py`

Per-tile ages. Reads the physics spreadsheet for incipient wavelength, sediment flux and intermittency, attaches coordinates through `align_latlon_from_intermittency()`, joins the FFT wavelengths, and converts spacing to years.

**Outputs** `results/dune_ages_final.csv`, one row per tile.

**Status.** A diagnostic rather than a source of reported numbers, since it still carries the superseded column I fallback. Field ages come from step 8.

---

## Step 8. `ages_by_wind_group.py`

The script that produces the reported ages. It groups tiles by the compass direction the ERA5 wind pushes them, `center_algo_deg` plus 180 binned to eight points, rather than by hand-measured migration. It optionally filters to tiles where the two agree within a chosen angle, and it computes the age of every group and every field.

**Arguments that matter.** `--max-offset 90` keeps only tiles whose ERA5 direction agrees with the measured migration to within half the compass. `--min-n 1` disables the thin-group filter so every tile contributes. `--with-hyst` switches t0 from Qs/Qr to Qs/(Qr x 0.2).

**Outputs** three files. `results/ages_by_wind_group.csv` for per-group ages, `..._field_summary.csv` for one age per field, and `..._wind_vs_migration.csv` for the agreement statistics. Filtered results sit in the plain columns and unfiltered results in parallel `_allpts` columns.

***DOES NOT DEPEND ON THE OUTPUTS FROM STEP 7, STEP 7 IS MAINLY A SANITY CHECK OF THE AGES PER TILE***

**Used for** the published-age comparison and the maps.

---

## The age calculation itself

Four steps inside `compute_ages()`.

Dimensionless spacing comes from dividing the measured wavelength by a site-specific length scale, `l0` equals incipient wavelength over 43.429, giving `W`. Every field lands between roughly 800 and 4800.

Model timesteps come from the linear growth law, `N = (W - 110.2) / 8.26e-5`, fitted to the ReSCAL simulation. The power-law fit describes the early part of the same run and returns ages of 1e10 to 1e12 years when extrapolated this far, so it is not used.

A timestep becomes a year through `t0 = Qs / Qr`, where Qs is the model saturated flux of 0.125 and Qr the site flux in metres squared per year. Multiplying by the hysteresis factor of 0.2 gives the other end of the reported range, reflecting that grains keep saltating below the wind speed needed to start them moving.

Dividing by intermittency converts continuous transport time into elapsed time, since sand only moves during the fraction of hours above threshold.

Field ages are a size-weighted geometric mean of the per-group geometric means, weighted in log space. That is algebraically the geometric mean over all tiles whenever no group is dropped, so with `--min-n 1` the grouping does not alter the field number and exists to show within-field structure.

---

## Off the main path

`age_sensitivity_matrix.py` runs every combination of intermittency definition, hysteresis setting and growth law against published ages, scoring by mean log10 distance. **It still carries an unpatched copy of `load_physics` with the lat/lon alignment bug.**

`score_against_truth.py` scores step 5 output against the hand measurements.

`plot_age_maps.py` draws the maps, with arrows in the ERA5 transport direction coloured by age. It imports the age physics from step 8 rather than reimplementing it, so the maps cannot drift from the tables.

Diagnostics not in the flow: `synth_test.py`, `inspect_tile.py`, `multi_region_diagnostic.py`, `peak_strategy_experiment.py`.

---

## Reproducing the reported ages

Steps 1 to 4 are not rerun. Re-downloading imagery can change which scene Earth Engine selects and therefore the wavelengths. Step 6 is not rerun either, since its numbers do not depend on the truth spreadsheet.

```
TRUTH=~/Desktop/era5_wind_ts/dune_migration_and_lambda_corrected.xlsx

python measure_wavelengths_nearest.py \
  --tiles_dir tiles --truth "$TRUTH" \
  --out results/final_wavelengths.csv --tol 0.25

python ages_by_wind_group.py \
  --physics new_model_age_analysis.xlsx \
  --intermittency intermittency_by_migration.csv \
  --truth "$TRUTH" \
  --wavelengths results/final_wavelengths.csv \
  --max-offset 90 \
  --out results/ages_by_wind_group.csv
```

---

## Numbers a methods section will need

1,005 sampling points, 272 green-tagged and excluded, 733 measured. Sentinel-2 luminance at 20 m per pixel. High-pass cutoff 6000 m, prominence threshold 0.20, 100 logarithmic bins from 30 to 10,000 m. 90.9 per cent of tiles high-confidence at 25 per cent tolerance. The agreement filter at 90 degrees drops 91 tiles, leaving 642. Fifteen fields across four continents.

## Caveats that belong in the methods

The growth law is fitted to W up to 187.5 and the fields sit at 800 to 4800, so it is read four to twenty-five times beyond its calibrated range. Below W of 110.2 it returns negative ages, which is why White Sands has none.

Wavelength selection is anchored to hand measurements, so it refines rather than independently predicts.

Agreement statistics must come from the unfiltered run. Taking them from a `--max-offset` run is circular, and the script now enforces this by always computing them on all tiles.

# Dune field ages from wavelength

Estimates the age of a dune field from the spacing of its dunes, measured by
FFT on Sentinel-2 imagery and converted to years using ERA5 wind records and a
growth curve fitted to ReSCAL simulations.

Eight steps. Steps 1 to 4 build the imagery and measure spacing. Step 5 picks
one wavelength per tile. Step 6 builds the wind record. Steps 7 and 8 turn
those into ages.

---

## IMPORTANT

**Steps 7 and 8 do not feed into each other.** The numbering makes it look like
step 8 takes step 7's output and carries on. It does not. Both read the same
raw inputs and each works out ages from scratch. They then report differently,
so their numbers will not match, and that is correct rather than a sign
something broke. Step 7 is for looking at individual tiles. Step 8 is where
field ages come from.

**Never rerun step 3 once you have results.** It downloads satellite imagery,
and each run asks Earth Engine for the least cloudy scene available. If a
better scene has been added since, you get different imagery, and different
imagery means different wavelengths for every tile in that field. Treat the
tiles folder as data you collected once, not as a cache you can rebuild.

**The --tol setting does not change any measurement.** Step 5 picks whichever
spectral peak is closest to your hand measurement, and that happens regardless.
All --tol does is decide the label: within 25 percent gets called high
confidence, further out gets called low. 

**Agreement statistics must come from a run without the filter.** The
--max-offset 90 filter throws out tiles where wind and dune direction disagree
by more than 90 degrees. If you then measure agreement on what is left, of
course everything agrees, because you removed the disagreements. That figure
would be meaningless. The script always computes agreement on all tiles and
prints a note explaining why.

---

## Before you start

Draw a polygon around each dune field in Google Earth, export each as a KMZ,
and name the file after the region. The filename becomes the region name
everywhere downstream, so `Taklamakan.kmz` produces rows labelled `Taklamakan`.
Put them all in one folder.

Python 3.8+. Needs numpy, pandas, scipy, openpyxl, geopandas, rasterio,
earthengine-api, matplotlib. Cartopy is optional and only affects the maps.

Note on this machine: `pip` and `python` resolve to different installations,
so install with `python -m pip install ...` rather than `pip install ...`.

---

## Step 1. Lay a grid of sampling points inside each polygon

Edit `KMZ_FOLDER` at the top of the script to point at your KMZ folder, then:

    python make_sampling_points_from_kmz_folder.py

No command-line arguments; the settings are constants at the top of the file.

Writes `sampling_points_all_regions_8thdeg.csv`, one row per point with region,
lat and lon. Currently 1005 points across 15 fields.

Points sit on a 0.125 degree grid aligned to the ERA5 lattice. Note that is
HALF the ERA5 spacing, so roughly four sampling points share each wind cell.

---

## Step 2 and 3. Download imagery and run the FFT

Step 2, `download_s2_box_SZA_luminance.py`, is called by step 3. You do not run
it yourself.

    python run_pipeline_from_points.py \
      --points_csv sampling_points_all_regions_8thdeg.csv \
      --out_csv results/modes_multimode_allregions_luminance_20m.csv \
      --tiles_dir tiles \
      --diag_dir outputs

For every point this downloads a 0.25 by 0.25 degree Sentinel-2 tile at 20 m
per pixel, converts it to luminance, and extracts every candidate wavelength
from its spectrum.

    tiles/    the GeoTIFFs, one per point
    outputs/  diagnostic figures
    results/  the CSV of all candidate spectral peaks

Tiles already on disk are skipped, so a repeat run only fetches what is
missing.

DO NOT RERUN THIS ONCE YOU HAVE RESULTS. Earth Engine may select a different
scene on a later run, which changes the imagery and therefore every wavelength
downstream. Treat `tiles/` as data, not as a cache.

`fft_full_geotiff_multimode.py` is the spectral engine. Step 3 imports it; you
never run it directly. Its constants are the ones that matter: high-pass cutoff
6000 m, prominence threshold 0.20, 100 log-spaced bins between 30 and 10000 m.

---

## Step 4. Hand-measure a wavelength for every tile

Not a script. Open each tile and measure the dune spacing by eye, into
`dune_migration_and_lambda_corrected.xlsx`.

    column A, B    lat, lon
    column C       region
    column D       FFT lambda. THE FILL COLOR OF THIS CELL IS THE FLAG:
                   green means exclude this tile (no dunes, linear dunes,
                   or imagery too coarse to resolve). Yellow or no fill
                   means measure it.
    column E       migration direction as a compass label, e.g. NW.
                   Leave blank if the imagery cannot resolve it.
    column F       your hand-measured wavelength in metres

Currently 272 green rows and 733 measured.

The green flag is read from the cell color, not from a value. If Excel ever
re-saves this with theme colors instead of explicit RGB, every exclusion
silently disappears. The tile count printed by step 5 is your only guard: it
should say 733.

---

## Step 5. Pick one wavelength per tile

    TRUTH=~/Desktop/era5_wind_ts/dune_migration_and_lambda_corrected.xlsx

    python measure_wavelengths_nearest.py \
      --tiles_dir tiles \
      --truth "$TRUTH" \
      --out results/final_wavelengths.csv \
      --tol 0.25

Reruns the FFT on each non-green tile and picks the candidate peak nearest your
hand measurement. Takes 25 to 40 minutes.

`--tol` does NOT affect which peak is chosen. It only decides whether a tile is
labelled high or low confidence, based on whether the chosen peak lands within
25 percent of your measurement.

Expect: 733 tiles written, about 91 percent high confidence, 0 missing. If the
count is not 733, you are pointing at the wrong spreadsheet.

Because selection is anchored to your hand measurement, the output is an
FFT-refined version of a manual measurement, not an independent prediction.

---

## Step 6. Build the wind record

    python reanalyze_intermittency_by_migration.py \
      --base ~/Desktop/era5_wind_ts \
      --truth "$TRUTH" \
      --out intermittency_by_migration.csv

Reads the hourly ERA5 series for each point, works out the wind speed needed to
move sand there from grain size and density, and counts what fraction of hours
exceed it. Also computes sediment flux and the dominant wind sector.

Independent of steps 1 to 5. Skip it if `intermittency_by_migration.csv`
already exists; nothing in it depends on the spreadsheet.

Grain size and density per field live in the `REGION_PROPS` table at the top of
this script. That table is the only place per-field physical properties enter
the pipeline.

Intermittency is reported four ways. We use `net_algo`, the best 180 degree
wind sector minus the worst, because it needs no migration direction and so
works at every tile.

---

## Step 7. Per-tile ages, for debugging

    python dune_age_estimate.py \
      --physics new_model_age_analysis.xlsx \
      --coords any \
      --wavelengths results/final_wavelengths.csv \
      --out results/dune_ages_final.csv

One row per tile with each tile's value, so you can see whether an odd
result came from the wavelength, the flux or the intermittency. 

Two reasons its numbers differ from step 8, both expected. It always applies
the hysteresis factor, making its ages five times larger. And it fills the 272
excluded tiles back in using an old superseded wavelength column. Use it to
inspect individual tiles, not to report field ages.

`--coords` takes any string. The file is not read; the flag only switches on
coordinate recovery, which the FFT join needs.

---

## Step 8. Field ages. This is the one you report.

    python ages_by_wind_group.py \
      --physics new_model_age_analysis.xlsx \
      --intermittency intermittency_by_migration.csv \
      --truth "$TRUTH" \
      --wavelengths results/final_wavelengths.csv \
      --max-offset 90 \
      --out results/ages_by_wind_group.csv

Groups tiles by the compass direction ERA5 wind pushes them, ages each group,
then combines groups into one age per field.

    results/ages_by_wind_group.csv                 per wind group
    results/ages_by_wind_group_field_summary.csv   ONE AGE PER FIELD, cite this
    results/ages_by_wind_group_wind_vs_migration.csv  agreement statistics

Useful flags:

    --max-offset 90   keep only tiles where the ERA5 direction agrees with your
                      measured migration to within half the compass. Drops 91
                      of 733, leaving 642.
    --with-hyst       use t0 = Qs/(Qr*0.2) instead of t0 = Qs/Qr. Every age
                      comes out 5 times older.
    --min-n 1         no thin-group filter, every tile contributes. Default.

Ages are reported as a RANGE, so run it twice:

    python ages_by_wind_group.py ... --out results/ages_no_hyst.csv
    python ages_by_wind_group.py ... --with-hyst --out results/ages_with_hyst.csv

The first is the low end, the second the high end. They differ by exactly 5x.

With `--max-offset`, the CSV carries both populations: plain columns are the
filtered result, parallel `_allpts` columns are every tile.

Report the wind versus migration agreement statistics from the UNFILTERED run.
Taking them from a filtered run is circular, because the filter guarantees
100 percent agreement within the cutoff. The script always computes them on all
tiles and prints a note saying so.

---

## Optional

    python plot_age_maps.py \
      --physics new_model_age_analysis.xlsx \
      --intermittency intermittency_by_migration.csv \
      --truth "$TRUTH" \
      --wavelengths results/final_wavelengths.csv \
      --coastlines ~/Desktop/world.geojson \
      --outdir figures

Maps of dune ages, arrows pointing the way the wind pushes, colored by age.
Imports its physics from step 8 so the maps cannot drift from the tables. Add
`--groups` for one arrow per wind group instead of per tile.

Coastlines need either cartopy installed or a world GeoJSON passed with
`--coastlines`. Without either you still get labelled lat/lon axes.

    python score_against_truth.py     scores step 5 against the hand measurements
    python age_sensitivity_matrix.py  ranks every combination of intermittency
                                      definition, hysteresis setting and growth law

---

## To reproduce the published ages from scratch

Steps 1 to 4 are already done and must not be repeated. Step 6 does not need
rerunning either. Only steps 5 and 8:

    TRUTH=~/Desktop/era5_wind_ts/dune_migration_and_lambda_corrected.xlsx

    python measure_wavelengths_nearest.py --tiles_dir tiles --truth "$TRUTH" \
      --out results/final_wavelengths.csv --tol 0.25

    python ages_by_wind_group.py --physics new_model_age_analysis.xlsx \
      --intermittency intermittency_by_migration.csv --truth "$TRUTH" \
      --wavelengths results/final_wavelengths.csv --max-offset 90 \
      --out results/ages_by_wind_group.csv

Check as you go: step 5 should write 733 tiles at about 91 percent high
confidence, and step 8 should exclude 272 green-tagged rows and drop 91 tiles
to the agreement filter.

---

## Known limitations

The growth law was fitted to simulated dune fields up to W = 187.5 l0 (ReSCAL model length unit). Real fields
sit at W = 800 l0 to 4800 l0, so every age is read 4 to 25 times beyond the fitted
range. This is the largest uncertainty in the whole method.

Below W = 110.2 the linear law returns a negative age. White Sands sits at
W = 71 and therefore has no age. That is undefined rather than wrong.

Wavelength selection is anchored to a hand measurement, so the FFT refines a
manual estimate rather than predicting independently.

Sampling points are twice as dense as the ERA5 grid, so tile counts overstate
the number of independent wind records by about four.

One grain size and density per field, not per tile.

---

## Files

Scripts run in order:

    make_sampling_points_from_kmz_folder.py    step 1
    download_s2_box_SZA_luminance.py           step 2, called by step 3
    run_pipeline_from_points.py                step 3
    fft_full_geotiff_multimode.py              step 4, imported by step 3
    measure_wavelengths_nearest.py             step 5
    reanalyze_intermittency_by_migration.py    step 6
    dune_age_estimate.py                       step 7, diagnostic
    ages_by_wind_group.py                      step 8, reported ages

Other scripts:

    plot_age_maps.py             maps
    score_against_truth.py       scores step 5
    age_sensitivity_matrix.py    parameter sensitivity
    global_fft_superplot.py      figures
    model_fft_analysis.py        ReSCAL model analysis
    make_gif.py                  animations from tiles

Inputs:

    dune_migration_and_lambda_corrected.xlsx      hand measurements, THE key file
    new_model_age_analysis.xlsx                   per-site physics
    intermittency_by_migration.csv                wind record, from step 6
    results/final_wavelengths.csv                 chosen wavelengths, from step 5
    dunefields_grainsizedensity.xlsx              grain properties
    dune_field_locations_kenzie.csv               field locations
    modes_multimode_allregions_luminance_20m.csv  all candidate peaks, from step 3

`annotated/` holds annotated copies of the main scripts with line-by-line comments

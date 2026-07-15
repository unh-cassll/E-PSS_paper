# build_from_scratch

Scripts that build the **published production data files** — the Zenodo data
products consumed by the paper — from the **raw Stokes-parameter archive**.

These are **not** part of paper reproduction. To reproduce the paper, download
the finished products with `../grab_observational_data.py` and run the figure
pipeline (`make figures`); nothing here is needed for that. They are kept for
provenance and for rebuilding the products when the source data change.

The raw Stokes archive (per-run S0/S1/S2 `.nc` files, one per acquisition run)
is **not yet published** — archiving it is the next major to-do. Until then
these scripts only run where that archive is mounted (`EPSS_STOKES_DIR`).

## Scripts

| script | builds |
|---|---|
| `compute_short_wave_spectra_stats.py` | the `ASIT2019_wave_spectra_stats_timeseries_{no,lab,empirical}_gain.nc` files (per-run directional slope/elevation spectra + slope statistics) from the 512² Stokes archive, via `pss` (polarimetric slope sensing) + `slopespectra` |
| `build_sws_timeseries.py` | assembles those timeseries `.nc` files from the per-run npz cache written by the script above |
| `build_canonical_from_stokes.py` | one-command canonical rebuild: the three timeseries files **and** `ASIT2019_slope_fields_reduced.nc`, straight from Stokes |
| `compute_s0_exposure_qc.py` | root-level S0 under/over-exposure QC scan of the raw Stokes archive |

## Running

Run from the `_codes/` directory (data paths resolve relative to it), with
`slopespectra` and `pss`/`epss` importable and the Stokes archive mounted:

```bash
cd _codes
export EPSS_STOKES_DIR=/path/to/stokes_archive
PYTHONPATH=/path/to/wave-slope-spectral-analysis \
    python build_from_scratch/build_canonical_from_stokes.py
```

Outputs are written to `../_data/`. Relevant environment variables:

- `EPSS_STOKES_DIR` — directory (or colon list) of source Stokes `.nc` files
- `EPSS_SWS_OUTDIR` — per-run npz cache (default `../_data/_sws_intermediate/`)

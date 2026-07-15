"""
Root-level exposure QC scan of the raw Stokes archive: per-run fraction of
pixels near the sensor's overexposure ceiling and underexposure floor, from
raw S0 counts only (no gain, no inversion, no lidar/seapol reference; see
project rule epss-qc-root-level-only).

S0 = 0.5*(I0+I45+I90+I135), each channel a 12-bit count (0-4095) from the DoFP
micropolarizer mosaic (pss.stokes docstring) -> hard ceiling S0=8190 when all
four channels saturate. OVEREXP_CEIL is set just below that (any channel
saturating drags S0 close to it even if not all four do). UNDEREXP_FLOOR is
set from the observed structural-defect floor (see below), not tuned against
any external truth.

Also reports the fraction of frames affected by the single dead readout
column found at raw-array index -1 (last column of the on-disk 512x512
frame, BEFORE any transpose) during the 2026-07 recompute campaign -- a fixed
hardware defect, not an exposure signal. Counted separately so it does not
inflate the underexposure metric.

Env:
  EPSS_STOKES_DIR   colon list of raw Stokes directories
  EPSS_S0_STRIDE    frames between samples (default 60 -> 300 frames/run)
  EPSS_S0_OUT       output npz (default ../_data/_sws_intermediate/s0_exposure_qc.npz)

@author: nathanlaxague
"""
import os
import glob

import numpy as np
import netCDF4 as nc

import compute_short_wave_spectra_stats as builder

STRIDE = int(os.environ.get('EPSS_S0_STRIDE', '60'))
OVEREXP_CEIL = 8180.0     # within 10 counts of the hard 4-channel-saturated ceiling (8190)
DEAD_COL_IDX = -1         # last column, raw (frames, columns, rows) on-disk order
UNDEREXP_FLOOR = 20.0     # near-zero floor, excluding the dead-column defect
OUT = os.environ.get('EPSS_S0_OUT',
                     '../_data/_sws_intermediate/s0_exposure_qc.npz')


def scan_run(path):
    ds = nc.Dataset(path)
    ds.set_auto_mask(False)
    n = ds.dimensions['frames'].size
    idx = np.arange(0, n, STRIDE)
    over = under = dead = total = 0
    s0_vals = []
    for i in idx:
        s0 = ds['S0'][i].astype(np.float64)      # raw on-disk (columns, rows)
        core = np.delete(s0, DEAD_COL_IDX, axis=1)   # exclude the known dead column
        over += int(np.sum(core >= OVEREXP_CEIL))
        under += int(np.sum((core < UNDEREXP_FLOOR) & (core > 0)))  # exclude exact 0 too (dead-pixel-like)
        under += int(np.sum(core == 0))
        dead += int(np.sum(s0[:, DEAD_COL_IDX] == 0))
        total += core.size
        s0_vals.append(s0.mean())
    ds.close()
    return dict(over_frac=over / total, under_frac=under / total,
               dead_col_zero_frac=dead / (len(idx) * s0.shape[0]),
               s0_mean=float(np.mean(s0_vals)), n_frames_sampled=len(idx))


def main():
    tbl = builder.build_run_table()
    rows = []
    for run in sorted(tbl):
        path, wdir = tbl[run]
        r = scan_run(path)
        r['run'] = run
        rows.append(r)
        print('run %3d  over %.4f  under %.4f  dead-col-zero %.3f  S0mean %6.0f  (%s)'
              % (run, r['over_frac'], r['under_frac'], r['dead_col_zero_frac'],
                 r['s0_mean'], os.path.basename(path)), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print('wrote', OUT)


if __name__ == '__main__':
    main()

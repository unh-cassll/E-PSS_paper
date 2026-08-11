"""
Compute the single-triplet ("pseudo-buoy") directional estimator suite into
one ASIT2019 product on the E-PSS frequency/direction grid: per run, the
center-pixel (eta, sx, sy) triplet of the E-PSS field (native polarimeter
slopes) with the disc (jinc) response divided out of S(f), all estimators
steering on h(theta) = [1, i k cos, i k sin] (`subroutines.pseudo_buoy`).
Rows outside the disc support band (H^2 >= 0.1, f <= ~0.65 Hz) render blank.
@author: nathanlaxague
"""

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import netCDF4 as nc
import xarray as xr
import warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor, as_completed

from subroutines.utils import (DX_M, WATER_DEPTH_M, FS_HZ, NUM_RUNS,
                               NUM_SAMPLES, epss_ewdm_grids)
from subroutines.pseudo_buoy import BUOY_DIAM_M, H2_MIN, WELCH_NPERSEG, WELCH_OVERLAP

path = '../_data/'
slope_field_file = path + os.environ.get('EPSS_FLD', 'ASIT2019_slope_fields_reduced.nc')
env_file = path + 'ASIT2019_supporting_environmental_observations.nc'
outfile = path + 'ASIT2019_pseudo_buoy_directional_spectra.nc'

# estimator tag -> name used in this project's variable suffixes
tags = [('triplet', 'EWDM_triplet'),
        ('csm', 'EMEP'),
        ('bdm', 'BDM'),
        ('imlm', 'IMLM'),
        ('mem', 'MEM'),
        ('mlm', 'MLM')]

DD, KAPPA = 4.0, 36.0
F_ASSERT = 0.7                 # onshore/downwind assertion band edge [Hz]

_DS = {}


def _ds():
    if not _DS:
        _DS['fld'] = nc.Dataset(slope_field_file)
        _DS['env'] = nc.Dataset(env_file)
    return _DS


def _wind_dir_to(run_ind):
    """Going-to wind direction [deg CW from N, wrapped to (-180, 180]];
    COARE_Wdir is meteorological (coming-from)."""
    wd = float(np.ma.filled(_ds()['env']['COARE_Wdir'][run_ind], np.nan))
    if not np.isfinite(wd):
        return float('nan')
    return ((wd + 180.0) + 180.0) % 360.0 - 180.0


def work(run_ind):
    from multiaperture import build_eta_field
    from subroutines.pseudo_buoy import asit_triplet, estimator_dataset
    d = _ds()
    se = np.ma.filled(d['fld']['slope_east'][run_ind][..., :NUM_SAMPLES], np.nan)
    sn = np.ma.filled(d['fld']['slope_north'][run_ind][..., :NUM_SAMPLES], np.nan)
    if not np.isfinite(se).any():                        # NaN-flagged (corrupt) run
        return run_ind, None
    se, sn = np.nan_to_num(se).astype(float), np.nan_to_num(sn).astype(float)
    eta, _, _, _, _ = build_eta_field(se, sn, WATER_DEPTH_M, FS_HZ,
                                      depiston_n=1.5, return_components=True,
                                      longwave_method='fourier')
    eta_b, sx_b, sy_b, npx = asit_triplet(eta, se, sn, DX_M)

    freqs, _, _ = epss_ewdm_grids(DX_M)
    wind_to = _wind_dir_to(run_ind)
    est = estimator_dataset(
        eta_b, sx_b, sy_b, fs=FS_HZ, depth=WATER_DEPTH_M, freqs=freqs,
        dd=DD, kappa=KAPPA, r_m=BUOY_DIAM_M / 2.0,
        onshore=0.0, f_on=F_ASSERT,
        downwind=float(wind_to) if np.isfinite(wind_to) else None,
        f_dw=F_ASSERT if np.isfinite(wind_to) else None)
    out = {'S_f': np.asarray(est['frequency_spectrum'].values, float),
           'attrs': dict(est.attrs)}
    for tag, name in tags:
        out['F_' + name] = np.asarray(
            est['directional_spectrum_f_' + tag].values, float)
        out['mwd_' + name] = np.asarray(
            est['mean_direction_' + tag].values, float)
        out['spr_' + name] = np.asarray(
            est['directional_spread_' + tag].values, float)
    est.close()
    return run_ind, out


def main():
    freqs, _, _ = epss_ewdm_grids(DX_M)
    freqs = freqs.astype(np.float32)
    dirs_deg = np.arange(-180.0, 180.0, DD)
    nf, nd = len(freqs), len(dirs_deg)

    nw = int(os.environ.get('EPSS_NWORKERS', min(8, (os.cpu_count() or 4) - 1)))
    print(f"Computing pseudo-buoy directional spectra ({NUM_RUNS} runs, {nw} workers)...")
    results = {}
    with ProcessPoolExecutor(max_workers=nw) as ex:
        futs = {ex.submit(work, r): r for r in range(NUM_RUNS)}
        done = 0
        for fu in as_completed(futs):
            r, out = fu.result()
            done += 1
            if out is not None:
                results[r] = out
            if done % 25 == 0:
                print(f"  {done}/{NUM_RUNS}", flush=True)

    def nan_f4(*shape):
        return np.full(shape, np.nan, np.float32)

    F = {name: nan_f4(nf, nd, NUM_RUNS) for _, name in tags}
    mwd = {name: nan_f4(nf, NUM_RUNS) for _, name in tags}
    spr = {name: nan_f4(nf, NUM_RUNS) for _, name in tags}
    S_f = nan_f4(nf, NUM_RUNS)
    for r, o in results.items():
        # unsupported rows: zero-filled spectra but NaN S_f; carry the S_f mask
        band = np.isfinite(o['S_f'])
        S_f[:, r] = o['S_f']
        for _, name in tags:
            F[name][:, :, r] = np.where(band[:, None], o['F_' + name], np.nan)
            mwd[name][:, r] = np.where(band, np.radians(o['mwd_' + name]), np.nan)
            spr[name][:, r] = np.where(band, np.radians(o['spr_' + name]), np.nan)
    found = sorted(results)
    attrs0 = results[found[0]]['attrs'] if found else {}
    print(f"computed {len(found)}/{NUM_RUNS} runs")

    # write on the units and direction convention used by the other E-PSS
    # directional products; the estimators label per-degree but integrate to
    # S_f over radians, so only the label changes here
    present = np.zeros(NUM_RUNS, np.int8)
    present[found] = 1

    data_vars = {'S_f': (('frequency', 'run'), S_f),
                 'run_present': (('run',), present)}
    for _, name in tags:
        data_vars['F_f_d_' + name] = (('frequency', 'direction', 'run'), F[name])
        data_vars['mean_direction_' + name] = (('frequency', 'run'), mwd[name])
        data_vars['directional_spread_' + name] = (('frequency', 'run'), spr[name])

    ds = xr.Dataset(data_vars,
                    coords={'frequency': freqs,
                            'direction': np.radians(dirs_deg).astype(np.float32),
                            'run': np.arange(NUM_RUNS, dtype=np.int32)})

    ds['frequency'].attrs = {'units': 'Hz'}
    ds['direction'].attrs = {'units': 'radians clockwise from true North'}
    ds['S_f'].attrs = {'units': 'm^2/Hz',
                       'long_name': 'disc-response-corrected elevation spectrum'}
    ds['run_present'].attrs = {'units': '1',
                               'long_name': '1 where the run computed'}
    for _, name in tags:
        ds['F_f_d_' + name].attrs = {'units': 'm^2/Hz/rad',
                                     'long_name': 'frequency-directional spectrum, '
                                                  + name + ' on the 3 m triplet'}
        ds['mean_direction_' + name].attrs = {'units': 'radians clockwise from true North'}
        ds['directional_spread_' + name].attrs = {'units': 'radians'}

    ds.attrs.update(
        title='ASIT2019 single-triplet (pseudo-buoy) directional wave spectra',
        buoy_support_note='rows outside the disc-response support band are empty',
        direction_convention='cw_from_N',
        source='compile_pseudo_buoy_directional_spectra.py on the in-repo '
               'slope fields (subroutines.pseudo_buoy)',
        runs_present=len(found))
    for key in ('buoy_diameter_m', 'buoy_h2_min', 'buoy_support_f_hz',
                'welch_nperseg', 'welch_overlap', 'csm_conjugated'):
        if key in attrs0:
            ds.attrs[key] = attrs0[key]

    enc = {v: {'zlib': True, 'complevel': 4} for v in ds.data_vars}
    ds.to_netcdf(outfile, encoding=enc)
    ds.close()
    print('wrote ' + outfile)


if __name__ == '__main__':
    main()

"""
Precompute, for every run, the multi-aperture omnidirectional F(k) -- both each
aperture's contribution and the stitched composite -- plus the per-aperture trusted
bands and the inferred long-wave elevation eta(t). Saved to _dustbin so figures like
aperture_field_stitch / longwave_shortwave_demo can be built on the fly without
re-running the per-run EWDM. Estimator config matches the validated generator
(compute_all_directional_spectra): full-frame Krogstad disc, gated de-piston solve
field, 3-D-FFT sign anchor.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import netCDF4 as nc
import xarray as xr
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

from multiaperture import (build_eta_field, default_apertures, multiaperture_spectra,
                           sftheta_sign_anchor)
from subroutines.utils import (slope_to_elev_wavelet, DX_M, WATER_DEPTH_M, FS_HZ,
                               NUM_RUNS, NUM_SAMPLES, epss_ewdm_grids)

path = '../_data/'
out = '../_dustbin/multiaperture_Fk_longwave.nc'
fs, depth, num_samples, num_runs = FS_HZ, WATER_DEPTH_M, NUM_SAMPLES, NUM_RUNS
dx = DX_M
krog_disc, depiston_n = 32, 2.0
n_workers = 16

freqs, k_grid, nu_grid = epss_ewdm_grids(dx)
ap = default_apertures()
ap_names = [n for n, e in ap]

# centered disc (32x32 frame) for the long-wave slope average
yy, xx = np.ogrid[:32, :32]
disc = (yy-15.5)**2 + (xx-15.5)**2 <= (krog_disc/2.0)**2

_DS = {}
def _ds():
    if not _DS:           # opened per worker process (after fork), never in the parent
        _DS['fld'] = nc.Dataset(path+'ASIT2019_slope_fields_reduced.nc')
        _DS['ref'] = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc')
    return _DS

def work(r):
    d = _ds()
    se = np.ma.filled(d['fld']['slope_east'][r][..., :num_samples], np.nan)
    sn = np.ma.filled(d['fld']['slope_north'][r][..., :num_samples], np.nan)
    if not np.isfinite(se).any():        # NaN-flagged (corrupt) run
        return r, None
    se = np.nan_to_num(se).astype(float); sn = np.nan_to_num(sn).astype(float)
    eta, _, eta_solve = build_eta_field(se, sn, depth, fs, krog_disc=krog_disc, depiston_n=depiston_n)
    el = slope_to_elev_wavelet(se[disc].mean(0), sn[disc].mean(0), depth, fs)
    M = multiaperture_spectra(eta, dx, freqs, k_grid, nu_grid, depth, fs,
                              apertures=ap, n_staff=16, solve_eta=eta_solve,
                              sign_anchor=sftheta_sign_anchor(d['ref'], r))
    ap_Fk = np.array([np.asarray(ok) for ok, ink in M['ap_ok_omni']])
    return r, (np.asarray(M['Fk']), ap_Fk, np.asarray(M['ap_bands']), np.asarray(el))

if __name__ == '__main__':
    nk, nap = len(k_grid), len(ap)
    Fk = np.full((num_runs, nk), np.nan)
    ap_Fk = np.full((num_runs, nap, nk), np.nan)
    ap_bands = np.full((num_runs, nap, 2), np.nan)
    eta_long = np.full((num_runs, num_samples), np.nan)
    done = 0
    with Pool(n_workers) as pool:
        for r, res in pool.imap_unordered(work, range(num_runs)):
            done += 1
            if res is None:
                print(f'[{done}/{num_runs}] run {r}: NaN-flagged, skipped', flush=True); continue
            Fk[r], ap_Fk[r], ap_bands[r], eta_long[r] = res
            print(f'[{done}/{num_runs}] run {r}: done', flush=True)

    t = np.arange(num_samples)/fs
    xr.Dataset(
        {'Fk': (('run', 'k'), Fk, {'units': 'm^3', 'long_name': 'stitched multi-aperture omnidirectional F(k)'}),
         'ap_Fk': (('run', 'aperture', 'k'), ap_Fk, {'units': 'm^3', 'long_name': 'per-aperture omnidirectional F(k)'}),
         'ap_bands': (('run', 'aperture', 'edge'), ap_bands, {'units': 'rad/m', 'long_name': 'per-aperture trusted band [klo, khi]'}),
         'eta_long': (('run', 'time'), eta_long, {'units': 'm', 'long_name': 'inferred long-wave elevation eta(t)'})},
        coords={'run': np.arange(num_runs), 'k': ('k', k_grid, {'units': 'rad/m'}),
                'aperture': ap_names, 'time': ('time', t, {'units': 's'})},
        attrs={'config': 'matches compute_all_directional_spectra',
               'dx_m': dx, 'depth_m': depth, 'krog_disc_px': krog_disc,
               'depiston_n': depiston_n, 'fs_Hz': fs, 'n_staff': 16},
    ).to_netcdf(out)
    print('saved', out)

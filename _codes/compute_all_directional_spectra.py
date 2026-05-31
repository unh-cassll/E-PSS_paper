"""
Compute per-run frequency-directional wave spectra from earth-referenced PSS
slopes via the Extended Wavelet Directional Method (EWDM, Triplets with
use="slopes"), preceded by a wavelet-based slope -> elevation reconstruction.

Replaces the previous Welch + custom Maximum Entropy Method pipeline.

Created: 2025-09-17
Refactored to EWDM/upstream: 2026-05-29

@author: nathanlaxague
"""

from pathlib import Path

import numpy as np
import xarray as xr

import netCDF4 as nc

import ewdm

from subroutines.utils import slope_to_elev_wavelet, trim_EPSS_dirspec

import warnings
warnings.filterwarnings("ignore")


def _ewdm_dirs_to_cw_from_N(dirs_ccw_from_E):
    """Convert EWDM's CCW-from-east bins to CW-from-North ("coming from", deg) in
    [-180, 180). The +180 flips the slope-gradient direction that EWDM
    Triplets(use='slopes') reports into the wave coming-from convention used by
    the ADCP reference and the EWDM Arrays (displacement) method -- without it the
    E-PSS directional spectrum is 180 deg reversed (e.g. run 116 peak reads ~150
    deg instead of ~330, vs ADCP 339)."""
    cw = (90.0 - dirs_ccw_from_E + 540.0 + 180.0) % 360.0 - 180.0
    order = np.argsort(cw)
    return cw[order], order


output_file_name = '../_data/ASIT2019_EPSS_directional_spectra.nc'
pathname = Path(output_file_name)

if pathname.exists():
    print(f"File already exists: {pathname}")

else:

    print("Computing frequency-directional spectra via E-PSS/EWDM (Triplets)...")

    g = 9.81

    path = '../_data/'

    fn = path + 'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc'
    ds = nc.Dataset(fn)

    slope_north = ds['slope_north'][:]
    slope_east = ds['slope_east'][:]

    num_samples = slope_north.shape[1]
    num_runs = 190

    # PSS slope timeseries on disk are at 10 Hz (downsampled from 30 Hz).
    sampling_rate_PSS = 10.0
    water_depth_m = 15.0

    # EWDM analysis band (integer octaves: 2**omin..2**omax Hz).
    # omin=-5 -> ~0.031 Hz; omax=0 -> 1 Hz.
    ewdm_omin = -5
    ewdm_omax = 0
    ewdm_nvoice = 16

    # trim_EPSS_dirspec ambiguity-resolution band
    fmin = 5e-2
    fmax = 6e-1
    theta_halfwidth = 90
    smoothnum = 3

    t_s = np.arange(num_samples) / sampling_rate_PSS

    # Probe the output frequency / direction grids on the first run, then
    # allocate the per-run stack.
    F_EPSS_stack = None
    Ff_stack = None
    D_stack = None
    freq_axis = None
    dir_axis = None
    dir_reorder = None

    for run_ind in np.arange(0, num_runs):

        sE = slope_east[run_ind, :]
        sN = slope_north[run_ind, :]
        sE = np.where(np.isfinite(sE), sE, 0.0)
        sN = np.where(np.isfinite(sN), sN, 0.0)

        # 0.08 Hz gentle high-pass on the elevation inference (sets eta_var,
        # the directional-spectrum energy normalization below).
        elev_m = slope_to_elev_wavelet(
            sE, sN, water_depth_m, sampling_rate_PSS, fmin_Hz=0.08,
        )
        # slope_to_elev_wavelet may return one fewer sample if N is odd; pad
        # back to keep the time coord aligned with the raw slopes.
        if elev_m.size != num_samples:
            elev_m = np.concatenate([elev_m, [elev_m[-1]]])[:num_samples]

        ds_triplet = xr.Dataset(
            data_vars={
                "eastward_slope": ("time", sE.astype(float)),
                "northward_slope": ("time", sN.astype(float)),
                "surface_elevation": ("time", elev_m.astype(float)),
            },
            coords={"time": t_s},
            attrs={"sampling_rate": sampling_rate_PSS},
        )

        # Time coord is float seconds, not datetime64 - skip the upstream
        # interpolate_na step (its max_gap="10s" requires datetime time).
        # NaNs are already filled with 0 above.
        spec = ewdm.Triplets(ds_triplet, fs=sampling_rate_PSS, interpolate=False)
        out = spec.compute(
            omin=ewdm_omin, omax=ewdm_omax, nvoice=ewdm_nvoice,
            dd=5.0, use="slopes",
        )

        if dir_reorder is None:
            dir_axis, dir_reorder = _ewdm_dirs_to_cw_from_N(
                out["direction"].data
            )
            freq_axis = out["frequency"].data
            num_f = freq_axis.size
            num_dirs = dir_axis.size
            F_EPSS_stack = np.full((num_f, num_dirs, num_runs), np.nan)
            Ff_stack = np.full((num_f, num_runs), np.nan)
            D_stack = np.full((num_f, num_dirs, num_runs), np.nan)

        # Reindex direction onto CW-from-N convention.
        E_arr = out["directional_spectrum"].data[:, dir_reorder]
        D_arr = out["directional_distribution"].data[:, dir_reorder]
        S_arr = out["frequency_spectrum"].data

        F_EPSS = xr.DataArray(
            E_arr,
            coords={"frequency": freq_axis, "direction": dir_axis},
            dims=("frequency", "direction"),
        )

        F_EPSS = trim_EPSS_dirspec(F_EPSS, theta_halfwidth, fmin, fmax, smoothnum)

        # Normalize so the total elevation energy equals var(eta_long_band).
        # The wavelet eta reconstruction already targets that variance; here we
        # just renormalize the trimmed spectrum to match var(eta) over the
        # analysis band, matching the previous pipeline's convention.
        eta_var = float(np.var(elev_m))
        spec_total = float(
            F_EPSS.integrate('frequency').integrate('direction')
        )
        if spec_total > 0 and np.isfinite(spec_total):
            F_EPSS = F_EPSS * (eta_var / spec_total)

        F_EPSS_stack[:, :, run_ind] = F_EPSS.data
        Ff_stack[:, run_ind] = S_arr
        D_stack[:, :, run_ind] = D_arr

    F_EPSS_ds = xr.Dataset(
        coords={
            "frequency": freq_axis,
            "direction": dir_axis,
            "run number": np.arange(num_runs),
        },
        data_vars={
            "F_f_d": (["frequency", "direction", "run"], F_EPSS_stack),
            "F_f": (["frequency", "run"], Ff_stack),
            "D_f_d": (["frequency", "direction", "run"], D_stack),
        },
        attrs={
            "units": "m^2/Hz/deg",
            "method": "E-PSS slope -> EWDM Triplets(use='slopes')",
        },
    )

    F_EPSS_ds['frequency'].attrs = {'units': 'Hz'}
    F_EPSS_ds['direction'].attrs = {'units': 'degrees clockwise from true North'}
    F_EPSS_ds['run number'].attrs = {'units': 'sequential run number'}
    F_EPSS_ds['F_f_d'].attrs = {
        'units': 'm^2/Hz/deg',
        'description': 'Frequency-directional spectrum (post ambiguity trim)',
    }
    F_EPSS_ds['F_f'].attrs = {
        'units': 'm^2/Hz',
        'description': 'EWDM frequency spectrum S(f) before trim',
    }
    F_EPSS_ds['D_f_d'].attrs = {
        'units': '1/deg',
        'description': 'EWDM directional distribution D(f, theta)',
    }

    F_EPSS_ds.to_netcdf(output_file_name)

    print("Done computing frequency-directional spectra via E-PSS/EWDM!")

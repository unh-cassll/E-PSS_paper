"""
Compute per-run omnidirectional elevation spectra for E-PSS gain treatments
(no/lab/empirical) and the Riegl lidar reference. Units: m^2/Hz.

Created: 2025-11-15; refactored: 2026-05-29
@author: nathanlaxague
"""

import os

import numpy as np
import xarray as xr

import netCDF4 as nc

from scipy import signal

from subroutines.utils import (omni_complete_spectrum, L_FOV_M, WATER_DEPTH_M,
                               FS_HZ)

from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

# output overridable via env (write to temp -> verify -> swap)
output_file_name = '../_data/' + os.environ.get('EPSS_OMNI_OUT',
                                                 'ASIT2019_omnidirectional_spectra.nc')
pathname = Path(output_file_name)

if pathname.exists():
    print(f"File already exists: {pathname}")

else:

    print("Computing omnidirectional frequency spectra...")

    path = '../_data/'

    gain_names = ['no_gain', 'lab_gain', 'empirical_gain']
    # masked entries filled with NaN (omni_complete_spectrum zeros non-finite)
    slopes = {}
    for name in gain_names:
        ds = nc.Dataset(path + f'ASIT2019_wave_spectra_stats_timeseries_{name}.nc')
        slopes[name] = (np.ma.filled(ds['slope_east'][:], np.nan),
                        np.ma.filled(ds['slope_north'][:], np.nan))

    ds_other = nc.Dataset(path + 'ASIT2019_supporting_environmental_observations.nc')

    elev_m_lidar = np.ma.filled(ds_other["wse_m_Riegl"][:], np.nan)

    nfft = 3000
    nperseg = 1500
    num_freqs = nfft // 2 + 1
    num_runs = np.size(elev_m_lidar, axis=2)
    # PSS slopes (originally 30 Hz) and Riegl lidar stored at 10 Hz; slopes downsampled 3x upstream.
    sampling_rate_PSS = FS_HZ
    sampling_rate_lidar = FS_HZ
    num_lidars = 3

    run_number = np.arange(num_runs)

    water_depth_m = WATER_DEPTH_M
    # per-gain slopes are full-inscribed-disc averages; divide out the disc's
    # jinc transfer to recover the suppressed high-f (see omni_complete_spectrum)
    aperture_diameter_m = L_FOV_M

    F_f_m2_Hz_lidar = np.nan * np.ones((num_freqs, num_runs, num_lidars))
    F_f_m2_Hz = {name: np.nan * np.ones((num_freqs, num_runs)) for name in gain_names}

    for run_ind in range(num_runs):

        for name, (slope_east, slope_north) in slopes.items():
            f_Hz, S = omni_complete_spectrum(slope_east[run_ind, :], slope_north[run_ind, :],
                water_depth_m, sampling_rate_PSS, highpass_peak_fraction=0.5, nfft=nfft,
                nperseg=nperseg, aperture_diameter_m=aperture_diameter_m)
            F_f_m2_Hz[name][:, run_ind] = S

        for lidar_ind in range(num_lidars):

            f_Hz, Pxx_den = signal.welch(elev_m_lidar[lidar_ind, :, run_ind], sampling_rate_lidar, nfft=nfft, nperseg=nperseg)
            F_f_m2_Hz_lidar[:, run_ind, lidar_ind] = Pxx_den

    F_f_m2_Hz_lidar = np.median(F_f_m2_Hz_lidar, axis=2)

    data_vars = {f'F_f_m2_Hz_{name}': (['frequency', 'run number'], F_f_m2_Hz[name])
                 for name in gain_names}
    data_vars['F_f_m2_Hz_lidar'] = (['frequency', 'run number'], F_f_m2_Hz_lidar)

    # time coordinate carried from the supporting-env (0-189, chronological);
    # omni is built by iterating that source in index order, so it aligns by index
    src_time = np.asarray(ds_other['time'][:], 'f8')

    F_f_m2_Hz_all_ds = xr.Dataset(
        data_vars,
        coords={
            'frequency': f_Hz,
            'run number': run_number,
            'time': (['run number'], src_time),
        },
        attrs={'units': 'm^2/Hz', 'Conventions': 'CF-1.10',
               'title': 'ASIT 2019 omnidirectional elevation frequency spectra '
                        '(E-PSS no/lab/empirical gain and Riegl lidar reference)',
               'institution': 'University of New Hampshire',
               'source': 'compute_all_omnidirectional_spectra.py; renumbered 0-189',
               'history': 'omni F(f) from renumbered slope-timeseries + lidar'},
    )

    F_f_m2_Hz_all_ds['frequency'].attrs = {'units': 'Hz'}
    F_f_m2_Hz_all_ds['run number'].attrs = {'units': 'sequential run number 0-189 (chronological)'}
    F_f_m2_Hz_all_ds['time'].attrs = {'standard_name': 'time',
                                      'units': 'seconds since 1970-01-01 00:00:00',
                                      'calendar': 'standard', 'axis': 'T'}

    F_f_m2_Hz_all_ds.to_netcdf(output_file_name)

    print("Done computing omnidirectional frequency spectra!")

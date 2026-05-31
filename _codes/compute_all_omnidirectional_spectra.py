"""
Compute per-run omnidirectional elevation spectra for the three E-PSS gain
treatments (no/lab/empirical) and for the Riegl lidar reference. The E-PSS
estimator is the directionally-complete (S_sx+S_sy)/k^2 omnidirectional spectrum
(omni_complete_spectrum): summing both slope-variance components carries the full
directional spread, recovering the ~25% that the single-direction Krogstad
wavelet projection loses in 0.2-0.7 Hz. Same aperture-MTF deconvolution and
high-pass as before; frequency spectrum only (no timeseries/direction).

Created: 2025-11-15
Refactored to upstream: 2026-05-29
Switched E-PSS omni estimator to directionally-complete (S_sx+S_sy)/k^2: 2026-05-31

@author: nathanlaxague
"""

import numpy as np
import xarray as xr

import netCDF4 as nc

from scipy import signal

from subroutines.utils import omni_complete_spectrum, highpass_squared

from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

output_file_name = '../_data/ASIT2019_omnidirectional_spectra.nc'
pathname = Path(output_file_name)

if pathname.exists():
    print(f"File already exists: {pathname}")

else:

    print("Computing omnidirectional frequency spectra...")

    path = '../_data/'

    ds_no = nc.Dataset(path + 'ASIT2019_wave_spectra_stats_timeseries_no_gain.nc')
    ds_lab = nc.Dataset(path + 'ASIT2019_wave_spectra_stats_timeseries_lab_gain.nc')
    ds_emp = nc.Dataset(path + 'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc')

    slope_east_no = ds_no['slope_east'][:]
    slope_north_no = ds_no['slope_north'][:]

    slope_east_lab = ds_lab['slope_east'][:]
    slope_north_lab = ds_lab['slope_north'][:]

    slope_east_emp = ds_emp['slope_east'][:]
    slope_north_emp = ds_emp['slope_north'][:]

    ds_other = nc.Dataset(path + 'ASIT2019_supporting_environmental_observations.nc')

    elev_m_lidar = ds_other["wse_m_Riegl"][:]

    nfft = 3000
    nperseg = 1500
    num_freqs = np.int16(nfft / 2 + 1)
    num_runs = np.size(elev_m_lidar, axis=2)
    # Both the PSS slope timeseries (originally 30 Hz) and the Riegl lidar
    # are stored on disk at 10 Hz; the slopes were downsampled by 3 upstream.
    sampling_rate_PSS = 10.0
    sampling_rate_lidar = 10.0
    num_lidars = 3

    run_number = np.arange(num_runs)

    water_depth_m = 15.0

    # Aperture-MTF deconvolution of the footprint-averaged (spatial-mean) slope,
    # using the empirical gain curve calibrated from the field's center-vs-mean
    # slope spectra (calibrate_aperture_mtf). Recovers the 0.3-0.7 Hz attenuation
    # the aperture causes. (Effect inside the 0.08-0.3 Hz Hm0/T_E band is small --
    # the aperture is ~transparent there -- but it corrects the 0.3-0.7 Hz part of
    # the spectrum used by full-band plots.)
    _mtf = np.load(path + 'ASIT2019_aperture_mtf_gain.npz')
    mtf_curve = (_mtf['freqs_Hz'], _mtf['gain'])

    # Adaptive high-pass: corner = 0.5 * fp, with fp the peak of the directionally-
    # complete (S_sx+S_sy)/k^2 spectrum (floored at 0.08, corner floored at 0.06).
    # Tracks the sea-state peak so long-period swell is not clipped while the
    # 1/k^2-amplified low-f drift is still suppressed. The SAME per-run corner is
    # applied to the lidar reference (F_f_m2_Hz_lidar_passband) so Hm0/T_E are
    # compared through an identical passband.
    hp_fraction = 0.5

    F_f_m2_Hz_lidar = np.nan * np.ones((num_freqs, num_runs, num_lidars))
    F_f_m2_Hz_lidar_passband = np.nan * np.ones((num_freqs, num_runs))

    F_f_m2_Hz_no_gain = np.nan * np.ones((num_freqs, num_runs))
    F_f_m2_Hz_lab_gain = np.nan * np.ones((num_freqs, num_runs))
    F_f_m2_Hz_empirical_gain = np.nan * np.ones((num_freqs, num_runs))

    for run_ind in range(num_runs):

        # Directionally-complete E-PSS omni spectrum per gain treatment, adaptive
        # high-pass corner from each spectrum's own (S_sx+S_sy)/k^2 peak.
        f_Hz, S, _ = omni_complete_spectrum(
            slope_east_no[run_ind, :], slope_north_no[run_ind, :],
            water_depth_m, sampling_rate_PSS, aperture_mtf_curve=mtf_curve,
            highpass_peak_fraction=hp_fraction, nfft=nfft, nperseg=nperseg)
        F_f_m2_Hz_no_gain[:, run_ind] = S

        f_Hz, S, _ = omni_complete_spectrum(
            slope_east_lab[run_ind, :], slope_north_lab[run_ind, :],
            water_depth_m, sampling_rate_PSS, aperture_mtf_curve=mtf_curve,
            highpass_peak_fraction=hp_fraction, nfft=nfft, nperseg=nperseg)
        F_f_m2_Hz_lab_gain[:, run_ind] = S

        f_Hz, S, corner = omni_complete_spectrum(
            slope_east_emp[run_ind, :], slope_north_emp[run_ind, :],
            water_depth_m, sampling_rate_PSS, aperture_mtf_curve=mtf_curve,
            highpass_peak_fraction=hp_fraction, nfft=nfft, nperseg=nperseg)
        F_f_m2_Hz_empirical_gain[:, run_ind] = S

        # lidar through the SAME E-PSS passband (emp-gain corner; the peak, hence
        # the corner, is gain-independent). Raw lidar kept in F_f_m2_Hz_lidar.
        hp2 = highpass_squared(f_Hz, corner)
        lid_pb = np.nan * np.ones((num_freqs, num_lidars))
        for lidar_ind in range(num_lidars):

            f_Hz, Pxx_den = signal.welch(elev_m_lidar[lidar_ind, :, run_ind], sampling_rate_lidar, nfft=nfft, nperseg=nperseg)
            F_f_m2_Hz_lidar[:, run_ind, lidar_ind] = Pxx_den
            lid_pb[:, lidar_ind] = Pxx_den * hp2
        F_f_m2_Hz_lidar_passband[:, run_ind] = np.median(lid_pb, axis=1)

    F_f_m2_Hz_lidar = np.median(F_f_m2_Hz_lidar, axis=2)

    F_f_m2_Hz_all_ds = xr.Dataset(
        {
            'F_f_m2_Hz_no_gain': (['frequency', 'run number'], F_f_m2_Hz_no_gain),
            'F_f_m2_Hz_lab_gain': (['frequency', 'run number'], F_f_m2_Hz_lab_gain),
            'F_f_m2_Hz_empirical_gain': (['frequency', 'run number'], F_f_m2_Hz_empirical_gain),
            'F_f_m2_Hz_lidar': (['frequency', 'run number'], F_f_m2_Hz_lidar),
            'F_f_m2_Hz_lidar_passband': (['frequency', 'run number'], F_f_m2_Hz_lidar_passband),
        },
        coords={
            'frequency': f_Hz,
            'run number': run_number,
        },
        attrs={'units': 'm^2/Hz'},
    )

    F_f_m2_Hz_all_ds['frequency'].attrs = {'units': 'Hz'}
    F_f_m2_Hz_all_ds['run number'].attrs = {'units': 'sequential run number'}

    F_f_m2_Hz_all_ds.to_netcdf(output_file_name)

    print("Done computing omnidirectional frequency spectra!")

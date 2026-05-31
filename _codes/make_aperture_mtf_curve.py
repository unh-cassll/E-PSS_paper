"""Generate the aperture-MTF deconvolution gain curve used by the E-PSS
slope->elevation inversion -- from CAMERA DATA ONLY.

The footprint-averaged (spatial-mean) slope is aperture low-passed relative to a
point measurement. We calibrate that aperture transfer empirically from the
camera's own 2-D slope FIELD: the ratio of the spatial-mean slope spectrum to
the center-pixel (aperture-free) slope spectrum. NO lidar / Riegl / external
reference is used at any point -- this is essential so the downstream
camera-vs-lidar comparison stays an independent validation.

The empirical gain is bounded by the analytic 1/|sinc(k*L/2)| (the most a square
aperture can attenuate) so it (a) stays ~1 in the swell band where the aperture
is transparent and the center-vs-mean difference is just center-pixel noise, and
(b) keeps the gentler, directionally-correct empirical shape near 0.5-0.7 Hz.

Output: ../_data/ASIT2019_aperture_mtf_gain.npz  (freqs_Hz, gain, footprint_m, n_runs)

Run:  ../.venv/bin/python make_aperture_mtf_curve.py
"""
import numpy as np
import netCDF4 as nc
from subroutines.utils import calibrate_aperture_mtf

DATA = '../_data/'
fs = 10.0                 # Hz
water_depth_m = 15.0      # m
FOOTPRINT_M = 2.915       # field side length (m), from the directional k-grid
CALIB_RUNS = list(range(0, 190, 5))     # 38 runs spread across the deployment

fld = nc.Dataset(DATA + 'ASIT2019_slope_fields_reduced.nc')   # CAMERA slope field only
east = [np.where(np.isfinite(fld['slope_east'][r]), fld['slope_east'][r], 0.0).astype(float)
        for r in CALIB_RUNS]
north = [np.where(np.isfinite(fld['slope_north'][r]), fld['slope_north'][r], 0.0).astype(float)
         for r in CALIB_RUNS]

freqs_Hz, gain = calibrate_aperture_mtf(east, north, fs, water_depth_m, FOOTPRINT_M)

np.savez(DATA + 'ASIT2019_aperture_mtf_gain.npz',
         freqs_Hz=freqs_Hz, gain=gain, footprint_m=FOOTPRINT_M, n_runs=len(CALIB_RUNS))

print(f"Saved aperture-MTF gain (camera-only) from {len(CALIB_RUNS)} field runs.")
print(f"  max boost {gain.max():.2f} at f={freqs_Hz[np.argmax(gain)]:.2f} Hz")
for ff in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
    j = int(np.argmin(np.abs(freqs_Hz - ff)))
    print(f"  f={ff:.1f} Hz: gain={gain[j]:.3f}")

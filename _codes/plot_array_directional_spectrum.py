"""Step 3: virtual wave staffs + multi-pass EWDM Arrays on the CAMERA-derived
elevation field, validated against the independent ADCP directional spectrum.

For a run with ADCP data, reconstruct the camera elevation field
  vdisp(x,y,t) = [g2s(slope field) - <.>_t] + eta_long(t)         (camera only)
seed virtual wave staffs at field pixels, run EWDM Arrays over several staff
subsets (multi-pass) and average. Compare mean wave direction vs frequency to
the ADCP (independent) and to the existing E-PSS Triplets (point-slope) spectrum.

@author: nathanlaxague
"""
import numpy as np
import netCDF4 as nc
import xarray as xr
from scipy.signal import detrend
from matplotlib import pyplot as plt
from pyGrad2Surf.g2s import g2s
import ewdm
from subroutines.utils import slope_to_elev_wavelet, figure_style
import warnings
warnings.filterwarnings("ignore")

color_list, fullwidth, fullheight, fsize = figure_style()
path = '../_data/'
fs = 10.0
depth = 15.0
L = 2.915
run_ind = 116                # has ADCP data + full camera field
NPASS, NSTAFF = 6, 7


def _ewdm_dirs_to_cw_from_N(dirs_ccw_from_E):
    cw = (90.0 - dirs_ccw_from_E + 540.0) % 360.0 - 180.0
    order = np.argsort(cw)
    return cw[order], order


def wmean_dir(E, ang_deg):
    """Energy-weighted circular mean direction (deg, same convention as ang)."""
    a = np.deg2rad(ang_deg)
    c = np.nansum(E * np.cos(a)); s = np.nansum(E * np.sin(a))
    return np.rad2deg(np.arctan2(s, c)) % 360.0


# --- camera elevation field ---------------------------------------------------
emp = nc.Dataset(path + 'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc')
fld = nc.Dataset(path + 'ASIT2019_slope_fields_reduced.nc')
_m = np.load(path + 'ASIT2019_aperture_mtf_gain.npz')
mtf_curve = (_m['freqs_Hz'], _m['gain'])

sE = np.where(np.isfinite(emp['slope_east'][run_ind]), emp['slope_east'][run_ind], 0.)
sN = np.where(np.isfinite(emp['slope_north'][run_ind]), emp['slope_north'][run_ind], 0.)
eta_long = slope_to_elev_wavelet(sE, sN, depth, fs, aperture_mtf_curve=mtf_curve)

SxF = np.where(np.isfinite(fld['slope_east'][run_ind]), fld['slope_east'][run_ind], 0.).astype(float)
SyF = np.where(np.isfinite(fld['slope_north'][run_ind]), fld['slope_north'][run_ind], 0.).astype(float)
ny, nx, T = SxF.shape
dx = L / nx
xg = np.arange(nx) * dx; yg = np.arange(ny) * dx
Sx0 = SxF - SxF.mean(axis=2, keepdims=True)
Sy0 = SyF - SyF.mean(axis=2, keepdims=True)
Z = np.empty((ny, nx, T))
for i in range(T):
    Z[:, :, i] = g2s(xg, yg, Sx0[:, :, i], Sy0[:, :, i])
Z -= Z.mean(axis=2, keepdims=True)
vdisp = Z + eta_long[None, None, :]

# --- multi-pass EWDM Arrays over virtual wave staffs --------------------------
ii, jj = np.meshgrid(np.arange(4, ny-4, 5), np.arange(4, nx-4, 5), indexing='ij')
cand = list(zip(ii.ravel(), jj.ravel()))
nc_ = len(cand)
cxp, cyp = (nx-1)/2, (ny-1)/2


def arrays_pass(staffs):
    el = detrend(np.stack([vdisp[i, j, :] for (i, j) in staffs], axis=1), axis=0, type='constant')
    # Camera column axis is West-positive (image mirroring), so East = (cxp - j);
    # rows are North-positive. Orientation fixed by requiring the full-field array
    # direction to match the camera's own point-slope Triplets estimate (no lidar/
    # ADCP); the ADCP then independently confirms it (~329 vs ~335 deg, run 116).
    px = np.array([(cxp - j) * dx for (i, j) in staffs])
    py = np.array([(cyp - i) * dx for (i, j) in staffs])
    ds = xr.Dataset(
        {"surface_elevation": (["time", "element"], el),
         "position_x": (["element"], px), "position_y": (["element"], py)},
        coords={"time": np.arange(T)/fs, "element": np.arange(len(staffs))},
        attrs={"sampling_rate": fs})
    return ewdm.Arrays(ds, fs=fs, interpolate=False, normalise=True).compute(
        omin=-4, omax=1, nvoice=8, use="displacements", dd=5.0)

E_acc = S_acc = None
seeds = [0, 2, 4, 1, 3, 5]
for p in range(NPASS):
    sel = [cand[(seeds[p] + q*7) % nc_] for q in range(NSTAFF)]
    out = arrays_pass(sel)
    if E_acc is None:
        f_arr = out['frequency'].values
        th_e = out['direction'].values                 # CCW-from-E (deg)
        E_acc = np.zeros_like(out['directional_spectrum'].values)
        S_acc = np.zeros_like(out['frequency_spectrum'].values)
    E_acc += out['directional_spectrum'].values
    S_acc += out['frequency_spectrum'].values
E_arr = E_acc / NPASS; S_arr = S_acc / NPASS
th_cw, order = _ewdm_dirs_to_cw_from_N(th_e)            # -> CW-from-N
E_arr = E_arr[:, order]
arr_md = np.array([wmean_dir(E_arr[i], th_cw) for i in range(len(f_arr))])

# --- references: ADCP (independent) + E-PSS Triplets (point slope) -------------
env = nc.Dataset(path + 'ASIT2019_supporting_environmental_observations.nc')
fa = env['f_Hz_ADCP'][:]; tha_deg = np.rad2deg(env['theta_rad'][:])   # CW-from-N
Fa = env['F_f_theta_m2_Hz_rad_ADCP'][:, :, run_ind]                   # (theta, f)
adcp_md = np.array([wmean_dir(Fa[:, j], tha_deg) for j in range(len(fa))])

trip = nc.Dataset(path + 'ASIT2019_EPSS_directional_spectra.nc')
fd = trip['frequency'][:]; thd = trip['direction'][:]
Ft = trip['F_f_d'][:, :, run_ind]                                    # (f, dir)
trip_md = np.array([wmean_dir(Ft[j, :], thd) for j in range(len(fd))])

# --- figure -------------------------------------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(fullwidth, fullwidth*0.45))

# (A) array directional spectrum E(f, theta), with reference mean directions
TH, F = np.meshgrid(th_cw, f_arr)
pc = axA.pcolormesh(TH, F, np.log10(np.clip(E_arr, 1e-6, None)), shading='auto', cmap='turbo')
axA.plot(adcp_md, fa, 'w-', lw=2, label='ADCP mean dir')
axA.plot(((trip_md + 180) % 360) - 180 if False else trip_md, fd, 'k--', lw=1.5, label='Triplets mean dir')
axA.set_ylim(0.08, 1.0); axA.set_xlim(-180, 180)
axA.set_xlabel('direction [deg CW from N]'); axA.set_ylabel('f [Hz]')
axA.set_title('camera array $E(f,\\theta)$ (log)', fontsize=fsize)
axA.legend(loc='upper right', fontsize=fsize*0.8)
fig.colorbar(pc, ax=axA, pad=0.02, label='log$_{10}E$')

# (B) mean wave direction vs frequency
axB.plot(adcp_md, fa, 'o-', color='black', ms=3, lw=2, label='ADCP (independent)')
axB.plot(trip_md, fd, 's--', color=color_list[0], ms=3, lw=1.5, label='E-PSS Triplets')
axB.plot(arr_md, f_arr, '^-', color=color_list[2], ms=3, lw=2, label='camera array (step 3)')
axB.set_ylim(0.08, 1.0); axB.set_xlim(0, 360)
axB.set_xlabel('mean wave direction [deg CW from N]'); axB.set_ylabel('f [Hz]')
axB.grid(which='major', linestyle='-', linewidth=0.6)
axB.legend(loc='upper right', fontsize=fsize*0.8)
axB.set_title(f'run {run_ind}: mean direction vs frequency', fontsize=fsize)

plt.tight_layout()
plt.savefig('../_figures/array_directional_spectrum.pdf', bbox_inches='tight')
print(f'[run {run_ind}] array peak f={f_arr[np.nanargmax(S_arr)]:.3f} Hz; '
      f'array vs ADCP mean-dir at peak: '
      f'{arr_md[np.nanargmax(S_arr)]:.0f} vs {adcp_md[np.nanargmax(np.nansum(np.nan_to_num(Fa),axis=0))]:.0f} deg')

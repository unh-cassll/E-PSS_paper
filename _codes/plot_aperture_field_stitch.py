"""
Multi-aperture g2s elevation field and stitched omnidirectional saturation spectrum.
Left 2x2 quad: eta(x,y) through four aperture sizes (A0 broadest -> A7 tightest)
with virtual staff seeds; right panel: per-aperture k^3 F(k), stitched EWDM
composite, and direct full-frame k^3 F(k) reference.

Results cached to aperture_field_stitch.nc (~100 s to recompute); set recompute=True
to regenerate. Estimator config matches compute_all_directional_spectra.
"""

import os
import numpy as np
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from subroutines.utils import (figure_style, DX_M, WATER_DEPTH_M, FS_HZ,
                               NUM_SAMPLES, epss_ewdm_grids)
# slope->elevation front end + sign anchor from this repo; multi-aperture estimator from ewdm.
from multiaperture import build_eta_field, fourier_slope_projection, sftheta_sign_anchor
from ewdm import MultiApertureArrays
from ewdm.multiaperture import default_apertures, seed_aperture
color_list, fullwidth, fullheight, fsize = figure_style()

import warnings
warnings.filterwarnings("ignore")

path = '../_data/'
fs, depth, num_samples = FS_HZ, WATER_DEPTH_M, NUM_SAMPLES
dx = DX_M
runs = [60, 95, 130, 165]              # ensemble for the per-aperture + stitched spectra
snap_run = 130                         # field-snapshot run (Hm0 ~ 1 m)
i_snap = 4600                          # snapshot frame
slope_aperture = 32                    # full-frame disc
depiston_n = 2.0                       # gated de-piston cut (matches generator)

freqs, k_grid, nu_grid = epss_ewdm_grids(dx)

# cached per-aperture + stitched spectra and field snapshot (~100 s to recompute)
CACHE = path + 'aperture_field_stitch.nc'
recompute = False
if recompute or not os.path.exists(CACHE):
    fld = nc.Dataset(path+'ASIT2019_slope_fields_reduced.nc')
    # sign-anchor ref indexed to slope_fields_reduced run numbering; override via EPSS_REF
    ref = nc.Dataset(os.environ.get(
        'EPSS_REF', path+'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc'))
    ap = default_apertures()
    # ensemble per-aperture + stitched spectra (validated de-piston + sign-anchor config)
    ap_sum = [np.zeros(len(k_grid)) for _ in ap]
    Fk_sum = np.zeros(len(k_grid))
    n = 0
    for r in runs:
        se = np.nan_to_num(np.ma.filled(fld['slope_east'][r][..., :num_samples], np.nan)).astype(float)
        sn = np.nan_to_num(np.ma.filled(fld['slope_north'][r][..., :num_samples], np.nan)).astype(float)
        eta, _, eta_solve = build_eta_field(se, sn, depth, fs, slope_aperture=slope_aperture, depiston_n=depiston_n)
        M = MultiApertureArrays.from_field(eta, dx, depth, fs).compute(
            freqs=freqs, k_grid=k_grid, nu_grid=nu_grid, apertures=ap, n_staff=16,
            seed=20, solve_eta=eta_solve, reliability_gate=None,
            sign_anchor=sftheta_sign_anchor(ref, r), return_apertures=True)
        ap_fk = np.asarray(M['aperture_Fk'].values, float)        # (aperture, k) per-aperture omni F(k)
        for a in range(len(ap)):
            ap_sum[a] += np.nan_to_num(ap_fk[a])
        Fk_sum += np.nan_to_num(np.asarray(M['wavenumber_spectrum'].values, float))
        n += 1
    ap_mean = np.array([s/n for s in ap_sum])
    Fk = Fk_sum/n
    # g2s short-wave field snapshot for snap_run
    se = np.nan_to_num(np.ma.filled(fld['slope_east'][snap_run][..., :num_samples], np.nan)).astype(float)
    sn = np.nan_to_num(np.ma.filled(fld['slope_north'][snap_run][..., :num_samples], np.nan)).astype(float)
    eta, _ = build_eta_field(se, sn, depth, fs, slope_aperture=slope_aperture)
    # subtract the long wave to isolate the g2s short-wave field
    el = fourier_slope_projection(se, sn, depth, fs)
    Zsnap = (eta - el[None, None, :])[:, :, i_snap]      # g2s short-wave field (zero spatial mean)
    xr.Dataset(
        {'ap_mean': (('aperture', 'k'), ap_mean),
         'Fk': (('k',), Fk),
         'ap_bands': (('aperture', 'edge'),
                      np.stack([np.asarray(M['aperture_klo'].values, float),
                                np.asarray(M['aperture_khi'].values, float)], axis=1)),
         'ap_ext': (('aperture',), np.array([e for _, e in ap])),
         'Zsnap': (('y', 'x'), Zsnap)},
        coords={'k': k_grid, 'aperture': [str(a) for a in M['aperture_name'].values]},
        attrs={'dx': dx, 'snap_run': snap_run},
    ).to_netcdf(CACHE)

dc = xr.open_dataset(CACHE)
k_grid = dc['k'].values
ap_mean = dc['ap_mean'].values
Fk = dc['Fk'].values
ap_names = [str(a) for a in dc['aperture'].values]
ap_bands = dc['ap_bands'].values
ap_ext = dc['ap_ext'].values
Zsnap = dc['Zsnap'].values
dx = float(dc.attrs['dx'])

# direct full-frame wavenumber saturation k^3 F(k) = k * S_k(k) over the same runs
ds_emp = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc')
dth = np.median(np.diff(ds_emp['theta_rad'][:].data))
k_sl = ds_emp['k_rad_m'][:].data
S_k_theta = np.ma.filled(ds_emp['S_k_theta'][runs], np.nan)   # (nruns, theta, k)
S_k_theta[:, :, -1] = np.nan                                  # drop crashing top bin
S_k_slope = np.nansum(k_sl[None, None, :]*S_k_theta, axis=1)*dth
Bk_direct = np.nanmean(k_sl[None, :]*S_k_slope, axis=0)

# aperture palette: violet, teal, goldenrod, blue; crimson = stitched EWDM; black = direct
show = ['A0', 'A2', 'A4', 'A7']                  # broadest -> tightest
show_idx = [ap_names.index(s) for s in show]
show_cols = [color_list[0], color_list[1], color_list[3], color_list[4]]   # violet, teal, goldenrod, blue

ny, nx = Zsnap.shape
xg = (np.arange(nx)-(nx-1)/2.0)*dx
yg = (np.arange(ny)-(ny-1)/2.0)*dx
vZ = np.percentile(np.abs(Zsnap), 99)            # symmetric coolwarm limit about eta = 0

gauges = []                                      # virtual-staff pixels per aperture
for a in range(len(ap_names)):
    ii, jj, px, py, bmax = seed_aperture(ny, nx, dx, int(ap_ext[a]), 16, a)
    gauges.append((ii, jj))

fig = plt.figure(figsize=(fullwidth, fullwidth*0.6), constrained_layout=True)
# left 50%: 2x2 aperture quad; right 50%: k^3 F(k) spectrum spanning both rows
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 2])

def panel_tag(ax, tag):
    ax.text(0.08, 0.92, tag, transform=ax.transAxes, fontsize=fsize, va='center', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=1.0))

# 2x2 aperture quad (broad -> tight, reading order); y-label on left column,
# x-label on bottom row
quad_cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
tags = ['(a)', '(b)', '(c)', '(d)']
fax = []
for (r, c), idx, col, tag in zip(quad_cells, show_idx, show_cols, tags):
    ax = fig.add_subplot(gs[r, c])
    fax.append(ax)
    im = ax.pcolormesh(xg, yg, Zsnap, cmap='coolwarm', vmin=-vZ, vmax=vZ, shading='auto', rasterized=True)
    ax.set_aspect('equal')
    w = ap_ext[idx]*dx                  # aperture diameter [m]
    ax.add_patch(Circle((0, 0), w/2, fill=False, edgecolor=col, lw=2.5))
    ii, jj = gauges[idx]
    gx = (jj-(nx-1)/2.0)*dx
    gy = (ii-(ny-1)/2.0)*dx
    inside = np.hypot(gx, gy) <= w/2.0
    ax.scatter(gx[inside], gy[inside], s=14, facecolor='white', edgecolor='black', linewidth=0.5, zorder=5)
    # aperture name inset by its ring, in the ring color
    ax.text(w/2*0.70, w/2*0.70, ap_names[idx], color=col, fontsize=fsize, fontweight='bold',
            ha='left', va='bottom', zorder=6,
            bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.65))
    ax.set_ylabel('y [m]') if c == 0 else ax.set_yticklabels([])
    ax.set_xlabel('x [m]') if r == 1 else ax.set_xticklabels([])
    panel_tag(ax, tag)
fig.colorbar(im, ax=fax, location='top', orientation='horizontal',
             label=r'short wave $\eta$ [m]', shrink=0.9, aspect=40, pad=0.02)

ax = fig.add_subplot(gs[:, 2])                       # panel (e): k^3 F(k), right 50%
k3 = k_grid**3
# per-aperture k^3 F(k): solid within the aperture's deposited band, dotted beyond
for idx, col in zip(show_idx, show_cols):
    klo, khi = ap_bands[idx]
    inb = (k_grid >= klo) & (k_grid <= khi)
    ax.loglog(k_grid, k3*ap_mean[idx], ':', color=col, lw=1.2, alpha=0.7)
    ax.loglog(k_grid[inb], (k3*ap_mean[idx])[inb], '-', color=col, lw=2.0, label=ap_names[idx])
# stitched EWDM (crimson), then the direct full-frame reference on top (dashed black)
m = np.isfinite(Fk) & (k_grid <= ap_bands[:, 1].max())
ax.loglog(k_grid[m], (k3*Fk)[m], '-', color=color_list[2], lw=2.5, label='multi-aperture')
ax.loglog(k_sl, Bk_direct, 'k--', lw=2.5, label=r'direct')
ax.set_xlim(5e-2, 5e1)
ax.set_ylim(1e-4, 1e-1)
ax.set_xlabel(r'k [rad m$^{-1}$]'); ax.set_ylabel(r'k$^3$F(k) [rad]')
ax.yaxis.set_label_position('right')
ax.yaxis.tick_right()
ax.grid(which='major', ls='-', lw=0.75); ax.grid(which='minor', ls=':', lw=0.75)
ax.legend(fontsize=fsize, loc='lower right', ncol=1)
panel_tag(ax, '(e)')

# match spectrum panel height to quad block; freeze layout so the override survives savefig
fig.draw_without_rendering()
quad_top = max(a.get_position().y1 for a in fax)
quad_bot = min(a.get_position().y0 for a in fax)
sp = ax.get_position()
fig.set_layout_engine('none')
ax.set_position([sp.x0, quad_bot, sp.width, quad_top - quad_bot])

plt.savefig('../_figures/aperture_field_stitch.pdf', bbox_inches='tight', dpi=300)

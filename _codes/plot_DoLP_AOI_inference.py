"""
Four-panel dual-camera DoLP/AOI figure (Piermont 2025), overcast vs cloudless columns.

Top row    : measured DoLP vs AOI -- wide-FOV curve, narrow-imager scan, Mueller fit, ideal Fresnel.
Bottom row : observed vs inferred AOI for four DoLP->AOI strategies (no gain / lab gain /
             empirical gain / wide-FOV LUT).

Data from _data/Piermont2025_DoLP_AoI_observations.nc. Fresnel inversion helpers are inlined
(mirrors pss.fresnel); no raw-frame stacks or pss dependency needed.
"""

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import netCDF4 as nc

from scipy.interpolate import PchipInterpolator

from subroutines.utils import *
color_list,fullwidth,fullheight,fsize = figure_style()

import warnings

warnings.filterwarnings("ignore")

panel_labels = ['(a)','(b)','(c)','(d)']

path = '../_data/'

N_WATER = 1.33               # matches the ideal-Fresnel curve drawn in the top row
LAB_GAIN = 1.0/0.81          # fixed lab-calibrated DoLP gain (Piermont campaign value)


# Fresnel DoLP <-> AOI helpers (inlined from pss.fresnel)
def fresnel_dolp(theta_deg, n_water=N_WATER):
    """DoLP vs incidence, ideal Fresnel (unpolarized sky)."""
    th = np.deg2rad(np.asarray(theta_deg, float))
    s = np.sin(th)
    c = np.cos(th)
    s2 = s*s
    n2 = n_water*n_water
    return (2.0*s2*c*np.sqrt(n2-s2)) / (n2 - s2 - n2*s2 + 2.0*s2*s2)


def _lut_from_rising(dl, th, n_points=10000):
    """(DOLP_full, theta_full) inverse LUT from a rising DoLP(theta) branch via PCHIP."""
    DOLP_full = np.linspace(0.0, 1.0, n_points)
    tf = PchipInterpolator(dl, th, extrapolate=False)(DOLP_full)
    tf = np.where(DOLP_full > dl[-1], th[-1], tf)
    tf = np.where(DOLP_full < dl[0], th[0], tf)
    tf[0] = 0.0
    return DOLP_full, np.nan_to_num(tf, nan=th[-1])


def build_lookup_table(n_water=N_WATER, n_points=10000):
    """Ideal-Fresnel (DoLP, theta) lookup on the rising (sub-Brewster) branch."""
    th = np.linspace(0.0, 89.99, 200000)
    dl = fresnel_dolp(th, n_water)
    ip = int(np.argmax(dl))
    th, dl = th[:ip+1], dl[:ip+1]
    keep = np.concatenate([[True], np.diff(dl) > 0])
    dl, th = dl[keep], th[keep]
    return _lut_from_rising(dl, th, n_points)


def lut_from_curve(theta, curve, n_points=10000):
    """Empirical (DoLP, theta) lookup from a measured DoLP(theta) curve (rising branch)."""
    theta = np.asarray(theta, float)
    curve = np.asarray(curve, float)
    m = np.isfinite(theta) & np.isfinite(curve)
    th, dl = theta[m], curve[m]
    o = np.argsort(th)
    th, dl = th[o], dl[o]
    ipk = int(np.argmax(dl))
    th, dl = th[:ipk+1], dl[:ipk+1]
    runmax = np.maximum.accumulate(np.concatenate([[-np.inf], dl[:-1]]))
    keep = dl > runmax + 1e-6
    keep[np.argmax(dl > 0)] = True
    th, dl = th[keep], dl[keep]
    return _lut_from_rising(dl, th, n_points)


def dolp_to_aoi(dolp, DOLP_full, theta_full):
    """Map DoLP -> incidence (deg) by index lookup into the inverse LUT."""
    n = len(DOLP_full)
    dolp = np.asarray(dolp, float)
    finite = np.isfinite(dolp)
    idx = np.clip(np.floor(np.where(finite, dolp, 0.0)*n).astype(np.int64), 0, n-1)
    return np.where(finite, np.asarray(theta_full[idx], float), np.nan)


ds = nc.Dataset(path+'Piermont2025_DoLP_AoI_observations.nc')

# (title, wide AoI, wide DoLP, scan theta, scan RAW DoLP, n_refrac, Ssky, Sup, sky DoLP)
cases = [
    ('overcast', ds['AoI_overcast'][:], ds['DoLP_overcast'][:],
     ds['theta_narrow_scan_overcast'][:], ds['DoLP_narrow_scan_overcast'][:],
     ds['n_refrac_overcast'][:], ds['Ssky_overcast'][:], ds['Sup_overcast'][:],
     float(ds['DoLP_overcast_sky'][:])),
    ('cloudless', ds['AoI_sunny'][:], ds['DoLP_sunny'][:],
     ds['theta_narrow_scan_sunny'][:], ds['DoLP_narrow_scan_sunny'][:],
     ds['n_refrac_sunny'][:], ds['Ssky_sunny'][:], ds['Sup_sunny'][:],
     float(ds['DoLP_sunny_sky'][:])),
]

# Export Mueller-fit parameters as LaTeX macros so the paper hardcodes nothing:
# refractive index, fitted-sky S1/S0 & S2/S0 and its DoLP (caption), the upwelling
# fraction Sup_S0/Ssky_S0 (caption), and the directly-measured sky DoLP (panel title).
_abbr = {'overcast': 'Oc', 'cloudless': 'Cl'}
_fit_macros = {}
for (_title, _wa, _wd, _ts, _dl, _nref, _Ssky, _Sup, _skyd) in cases:
    _Ssky = np.asarray(_Ssky, float)
    _Sup = np.asarray(_Sup, float)
    _s0 = _Ssky[0] if _Ssky[0] else 1.0
    p = _abbr[_title]
    _fit_macros[f'Pierm{p}Nref']        = f'{float(_nref):.2f}'
    _fit_macros[f'Pierm{p}SkySone']     = f'{_Ssky[1]/_s0:.3f}'
    _fit_macros[f'Pierm{p}SkyStwo']     = f'{_Ssky[2]/_s0:.3f}'
    _fit_macros[f'Pierm{p}SkyDoLP']     = f'{np.hypot(_Ssky[1], _Ssky[2])/_s0:.3f}'
    _fit_macros[f'Pierm{p}UpFrac']      = f'{_Sup[0]/_s0:.3f}'
    _fit_macros[f'Pierm{p}SkyDoLPmeas'] = f'{float(_skyd):.3f}'
write_tex_macros('Piermont_DoLP_AOI_fit_values.tex', _fit_macros,
                 source='plot_DoLP_AOI_inference.py')

# ideal (unpolarized-sky) Fresnel curve for the top row
ideal_AoI, ideal_DoLP = mueller_calc_full(N_WATER, np.float64([1,0,0,0]), np.float64([0,0,0,0]))

AX_MAX = 90.0                 # shared incidence axis range (deg)

# shared axes: x per column, y per row
fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(fullwidth, fullwidth))

for col, (title, w_aoi, w_dolp, th_scan, dolp_scan, n_refrac, Ssky, Sup, sky_dolp) in enumerate(cases):

    th_scan = np.asarray(th_scan, float)
    dolp_scan = np.asarray(dolp_scan, float)  # RAW DoLP

    # top row: DoLP vs theta_i
    fit_AoI, fit_DoLP = mueller_calc_full(n_refrac, Ssky, Sup)
    ax = axs[0, col]
    ax.plot(w_aoi, w_dolp, linewidth=4, color=color_list[4], label='5 mm lens')
    ax.plot(th_scan, dolp_scan*LAB_GAIN, '*', ms=11, color=color_list[5], label='75 mm lens')
    ax.plot(fit_AoI, fit_DoLP, '--', color='black', label='fit')
    ax.plot(ideal_AoI, ideal_DoLP, '-', color=(0.5,0.5,0.5), label='ideal')
    ax.set_ylim(0,1.1)
    if col == 0:
        ax.set_ylabel('DoLP')
    # title shows the directly-measured sky DoLP (DoLP_*_sky, from the sky-pointing
    # frames) -- distinct from the fitted-Stokes DoLP reported in the Fig. caption
    ax.set_title(title + ' (sky DoLP = ' + f"{sky_dolp:.3f}" + ')')

    # bottom row: observed vs inferred theta_i (4 strategies)
    lut_fresnel = build_lookup_table(n_water=N_WATER)
    lut_wide = lut_from_curve(np.asarray(w_aoi,float), np.asarray(w_dolp,float))

    # empirical gain: scale median narrow DoLP to ideal Fresnel at mean incidence
    theta_ref = float(np.nanmean(th_scan))
    g_emp = fresnel_dolp(theta_ref, n_water=N_WATER) / float(np.nanmedian(dolp_scan))

    # violet / teal / crimson / goldenrod = no gain / lab gain / emp. gain / wide-FOV LUT
    strategies = [
        ('no gain',      dolp_scan*1.0,      lut_fresnel, color_list[0]),
        ('lab gain',     dolp_scan*LAB_GAIN, lut_fresnel, color_list[1]),
        ('emp. gain',    dolp_scan*g_emp,    lut_fresnel, color_list[2]),
        ('dual cam', dolp_scan*LAB_GAIN, lut_wide,    color_list[3]),
    ]
    ax2 = axs[1, col]
    ax2.plot([0,AX_MAX],[0,AX_MAX],'k:',linewidth=1)
    for name, dolp_in, lut, c in strategies:
        inferred = dolp_to_aoi(dolp_in, *lut)
        mae = float(np.nanmean(np.abs(inferred - th_scan)))
        ax2.plot(th_scan, inferred, 's-', ms=5, color=c, linewidth=1.5,
                 label=f'{name} ({mae:.1f}'+r'$^\circ$)')
    ax2.set_xlim(0, AX_MAX)
    ax2.set_ylim(0, AX_MAX)
    ax2.set_xticks(np.arange(0, AX_MAX+1, 15))
    ax2.set_yticks(np.arange(0, AX_MAX+1, 15))
    ax2.set_xlabel(r'true $\theta_i$ [$\circ$]')
    if col == 0:
        ax2.set_ylabel(r'inferred $\theta_i$ [$\circ$]')
    ax2.legend(loc='upper left', fontsize=fsize)

axs[0,0].legend(loc='lower right', bbox_to_anchor=(0.83, 0.0))

for i, ax in enumerate(axs.ravel()):
    ax.text(0.93,0.95,panel_labels[i],fontsize=fsize,ha='left',va='center',transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

plt.tight_layout()

plt.savefig('../_figures/DoLP_AOI_inference.pdf',bbox_inches='tight')
print('saved ../_figures/DoLP_AOI_inference.pdf')

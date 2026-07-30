"""
Project-specific helpers for the E-PSS paper.

    L_FOV_M, DX_M, WATER_DEPTH_M, ...     canonical ASIT2019 deployment constants
    figure_style                          paper plot styling
    wind_speed_bins                       canonical fixed-width U10 bins
    write_tex_macros                      LaTeX \newcommand value file for paper.tex
    scatter_metrics                       R^2/RMSE/slope/bias of a scatter comparison
    draw_metrics_box                      inset metrics table for scatter figures
    york_fit, york_band                   ML fit with errors in x and y, + conf band
    lidar_member_spectra                  per-instrument Riegl spectra (empirical sigma)
    lidar_consistency_flag                reject records where the Riegl units disagree
    ewdm_low_cutoff                       EWDM low-scale trust cutoff (k_low, f_low)
    epss_ewdm_grids                       EWDM f/k/nu grids (generator config)
    mueller_calc_full                     4-Stokes sky+upwelling Mueller calc
    compute_gram_charlier_slope_pdf       Cox-Munk Gram-Charlier slope PDF
    fit_gram_charlier_slope_pdf           least-squares Gram-Charlier fit
    omni_complete_spectrum                directionally-complete (S_sx+S_sy)/k^2 spectrum
    spreading                             row-normalized D(theta|scale) [per-degree]
    lobe_spread, lobe_sigma               single-lobe sigma_theta(scale) [deg]
    compute_mean_wave_direction_and_spreading
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import seaborn as sns

from typing import Union

import scipy.signal as signal
from scipy.optimize import minimize
from scipy.signal import detrend
from scipy.signal.windows import tukey

# Internal imports; underscore-aliased to avoid leaking via `import *`.
from eta_field_recon import lindisp_with_current as _lindisp_with_current
from eta_field_recon import aperture_transfer_function as _aperture_transfer_function

__all__ = [
    'GRAV', 'L_FOV_M', 'N_PX', 'DX_M', 'WATER_DEPTH_M', 'FS_HZ', 'NUM_RUNS',
    'NUM_SAMPLES',
    'figure_style', 'wind_speed_bins', 'binned_center_spread', 'write_tex_macros',
    'scatter_metrics', 'draw_metrics_box', 'york_fit', 'york_band',
    'lidar_member_spectra', 'lidar_consistency_flag',
    'MEDIAN3_SIGMA_FACTOR', 'ewdm_low_cutoff', 'epss_ewdm_grids',
    'mueller_calc_full', 'compute_gram_charlier_slope_pdf',
    'fit_gram_charlier_slope_pdf',
    'omni_complete_spectrum', 'SPREAD_SMOOTHNUM', 'SPREAD_SMOOTHNUM_DIRECT',
    'spreading', 'norm_smooth', 'wrap_deg', 'panel_vmax',
    'lobe_spread', 'lobe_sigma',
    'compute_mean_wave_direction_and_spreading',
]

# %%

# Canonical ASIT2019 / E-PSS deployment constants

GRAV = 9.81             # gravitational acceleration [m/s^2]
L_FOV_M = 2.915         # imaged-patch side length [m]
N_PX = 32               # reduced slope-field pixels per side
DX_M = L_FOV_M / N_PX   # reduced slope-field pixel size [m]
WATER_DEPTH_M = 15.0    # water depth at ASIT [m]
FS_HZ = 10.0            # slope-field / lidar sampling rate [Hz]
NUM_RUNS = 190          # ASIT2019 runs
NUM_SAMPLES = 6000      # samples per run (600 s at 10 Hz)


def ewdm_low_cutoff(n_frame_low=73, L_FOV_m=L_FOV_M, depth_m=WATER_DEPTH_M):
    """EWDM low-scale trust cutoff (k_low [rad/m], f_low [Hz]): lambda =
    n_frame_low*L_FOV (energy SNR = 0.5 vs Riegl lidar), finite-depth dispersion."""
    k_low = 2*np.pi/(n_frame_low*L_FOV_m)
    f_low = np.sqrt(GRAV*k_low*np.tanh(k_low*depth_m))/(2*np.pi)
    return k_low, f_low


def epss_ewdm_grids(dx=DX_M, nf=64, nk=80, nnu=80):
    """Log-spaced EWDM frequency [Hz], wavenumber [rad/m] (to the pixel Nyquist)
    and inverse phase speed [s/m] grids (multi-aperture generator config)."""
    freqs = np.logspace(np.log10(0.035), np.log10(3.5), nf)
    k_grid = 2.0**np.linspace(np.log2(0.01), np.log2(np.pi/dx), nk)
    nu_grid = 2.0**np.linspace(np.log2(0.005), np.log2(2.0), nnu)
    return freqs, k_grid, nu_grid

# %%

# Figure style function

def figure_style(title_fontsize=10, label_fontsize=10, tick_fontsize=10):

    fsize = 10
    lw = 1.0

    # set_theme resets the context; font sizes/linewidths applied via rcParams below
    sns.set_theme(style="ticks",palette="deep",font="Fira Sans")

    color_list = ['#4C2882', '#367588', '#A52A2A', '#C39953', '#2A52BE', '#006611']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

    plt.rcParams.update({
        'axes.grid': True,
        'font.size': fsize,
        'axes.titlesize': title_fontsize,
        'axes.labelsize': label_fontsize,
        'xtick.labelsize': tick_fontsize,
        'ytick.labelsize': tick_fontsize,
        'legend.fontsize': label_fontsize,
        'grid.linewidth': lw,
        'xtick.major.width': lw,
        'ytick.major.width': lw,
    })

    # Full page figure size (assuming letter paper with 0.5 inch margins
    fullwidth = 7.5
    fullheight = 10

    return color_list, fullwidth, fullheight, fsize

# %%

# Canonical fixed-width U10 bins; Umin/Umax are the outer edges [m/s]

# N. Laxague 2026

def wind_speed_bins(Umin=1.0, Umax=13.0, dU=2.0):

    edges = np.arange(Umin, Umax + dU / 2, dU)
    centers = edges[:-1] + dU / 2

    return centers, edges, dU

# %%

# Robust binned aggregate: median center, MAD band, robust SE of the median.

# N. Laxague 2026

def binned_center_spread(x, values, bin_edges):
    """Per bin: (median, MAD half-width for the shaded band, robust SE of the
    median 1.4826*MAD/sqrt(N), count). Non-finite pairs are dropped; the last
    bin includes its right edge."""
    x = np.asarray(x, float)
    values = np.asarray(values, float)
    m = np.isfinite(x) & np.isfinite(values)
    x, values = x[m], values[m]
    nb = len(bin_edges) - 1
    idx = np.digitize(x, bin_edges) - 1
    idx[x == bin_edges[-1]] = nb - 1
    mid = np.full(nb, np.nan)
    mad = np.full(nb, np.nan)
    counts = np.zeros(nb)
    for i in range(nb):
        v = values[idx == i]
        counts[i] = v.size
        if v.size:
            mid[i] = np.median(v)
            mad[i] = np.median(np.abs(v - mid[i]))
    se = 1.4826 * mad / np.sqrt(np.maximum(counts, 1))
    return mid, mad, se, counts

# %%

# Write computed values as LaTeX macros for \input into paper.tex.

# N. Laxague 2026

def write_tex_macros(filename, macros, source=None, directory='../_tex'):
    r"""Write a LaTeX macro file (one \newcommand per value) for \input into paper.tex.

    filename  : output filename, e.g. 'Hm0_values.tex'; written under `directory`
                (default '../_tex').
    macros    : dict {name: value}. `name` must be letters only; `value` is
                stringified (pass a pre-formatted string for explicit precision).
    source    : optional producing-script name for the file header.

    Each macro is emitted as \providecommand{\name}{}\renewcommand{\name}{value}
    (safe on re-\input). File is written atomically."""
    import os
    import re
    bad = []
    for name in macros:
        if not re.fullmatch('[A-Za-z]+', name):
            bad.append(name)
    bad = sorted(bad)
    if bad:
        raise ValueError('LaTeX macro names must be letters only; invalid: %s' % bad)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    header = '%% auto-generated%s; do not edit by hand\n' % (' by ' + source if source else '')
    body = ''
    for name, value in macros.items():
        body += '\\providecommand{\\%s}{}\\renewcommand{\\%s}{%s}\n' % (name, name, value)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        fh.write(header + body)
    os.replace(tmp, path)
    return path

# %%

# Scatter-comparison metrics and inset metrics table for Hm0/Tm02 figures


def scatter_metrics(x, y):
    """(R^2, RMSE, slope, bias) of y vs x over finite pairs. bias = mean(y - x),
    not the regression intercept."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    rmse = float(np.sqrt(np.mean((y[keep] - x[keep])**2)))
    r = np.corrcoef(x[keep], y[keep])[0, 1]
    slope, _ = np.polyfit(x[keep], y[keep], 1)
    bias = float(np.mean(y[keep] - x[keep]))
    return r**2, rmse, float(slope), bias


def york_fit(x, y, sigma_x, sigma_y, r=0.0, tol=1e-12, maxiter=500,
             calibrate=False, calib_iter=20):
    """Maximum-likelihood straight-line fit with uncertainty in BOTH variables,
    by the unified equations of York et al. (2004).

    Ordinary least squares treats x as exact, so error in x attenuates the slope
    toward zero -- reporting a slope below unity that is partly an artifact of the
    reference's own noise. This solves the errors-in-variables problem properly.

    With r = 0 the per-point weight reduces to W = 1/(sigma_y^2 + b^2 sigma_x^2),
    i.e. the effective-variance weight of Orear (1982); York's formulation is the
    exact ML solution and also supplies the standard errors.

    Args:
        x, y: paired observations; non-finite pairs are dropped.
        sigma_x, sigma_y: 1-sigma uncertainties, scalar or per point.
        r: correlation between the x and y errors, scalar or per point.
        calibrate: rescale sigma_y by a single constant until the reduced
            chi-square is unity, holding sigma_x and the shape of sigma_y fixed.
            Use when sigma_x is known independently but sigma_y is not: the
            excess scatter is then attributed to y, which is the conservative
            assignment (it drives lambda up and the slope toward OLS, rather
            than inflating the slope by blaming the reference). The applied
            factor is returned as 'sy_inflation'.

    Returns:
        dict with 'slope', 'intercept', 'se_slope', 'se_intercept',
        'chi2_reduced' (S/(n-2)), 'se_slope_expanded' and 'n'.

    A reduced chi-square far above unity means the supplied uncertainties do not
    account for the observed scatter -- unmodeled disagreement between the two
    quantities rather than measurement noise. The slope is still the ML estimate
    under the given weights, but its nominal standard error is then far too small,
    so 'se_slope_expanded' scales it by sqrt(chi2_reduced). Quote the expanded
    error whenever chi2_reduced >> 1, and say so.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    sx = np.broadcast_to(np.asarray(sigma_x, float), x.shape).astype(float)
    sy = np.broadcast_to(np.asarray(sigma_y, float), y.shape).astype(float)
    rr = np.broadcast_to(np.asarray(r, float), x.shape).astype(float)
    g = (np.isfinite(x) & np.isfinite(y) & np.isfinite(sx) & np.isfinite(sy)
         & (sx > 0) & (sy > 0))
    x, y, sx, sy, rr = x[g], y[g], sx[g], sy[g], rr[g]
    n = x.size
    if n < 3:
        return dict(slope=np.nan, intercept=np.nan, se_slope=np.nan,
                    se_intercept=np.nan, chi2_reduced=np.nan,
                    se_slope_expanded=np.nan, n=n)

    if calibrate:
        # rescale sigma_y until chi2_reduced -> 1; the weights depend on the
        # slope, so the factor and the fit are found together
        k = 1.0
        for _ in range(calib_iter):
            f = york_fit(x, y, sx, sy * k, rr, tol, maxiter)
            c2 = f['chi2_reduced']
            if not np.isfinite(c2) or c2 <= 0:
                break
            k *= np.sqrt(c2)
            if abs(c2 - 1.0) < 1e-6:
                break
        out = york_fit(x, y, sx, sy * k, rr, tol, maxiter)
        out['sy_inflation'] = float(k)
        return out

    wx, wy = 1.0 / sx ** 2, 1.0 / sy ** 2
    alpha = np.sqrt(wx * wy)
    # OLS start; the iteration is on the slope only
    b = np.polyfit(x, y, 1)[0]
    for _ in range(maxiter):
        W = wx * wy / (wx + b ** 2 * wy - 2.0 * b * rr * alpha)
        sW = W.sum()
        xb, yb = (W * x).sum() / sW, (W * y).sum() / sW
        U, V = x - xb, y - yb
        beta = W * (U / wy + b * V / wx - (b * U + V) * rr / alpha)
        denom = (W * beta * U).sum()
        if denom == 0:
            break
        b_new = (W * beta * V).sum() / denom
        if abs(b_new - b) < tol * max(abs(b), 1.0):
            b = b_new
            break
        b = b_new

    W = wx * wy / (wx + b ** 2 * wy - 2.0 * b * rr * alpha)
    sW = W.sum()
    xb, yb = (W * x).sum() / sW, (W * y).sum() / sW
    U, V = x - xb, y - yb
    beta = W * (U / wy + b * V / wx - (b * U + V) * rr / alpha)
    a = yb - b * xb
    # adjusted points, then the slope variance about their weighted centroid
    x_adj = xb + beta
    u = x_adj - (W * x_adj).sum() / sW
    var_b = 1.0 / (W * u ** 2).sum()
    S = (W * (y - b * x - a) ** 2).sum()
    chi2_red = S / (n - 2)
    return dict(slope=float(b), intercept=float(a),
                se_slope=float(np.sqrt(var_b)),
                se_intercept=float(np.sqrt(1.0 / sW + xb ** 2 * var_b)),
                chi2_reduced=float(chi2_red),
                se_slope_expanded=float(np.sqrt(var_b * max(chi2_red, 1.0))),
                var_slope=float(var_b), sum_weights=float(sW),
                x_centroid=float(xb), n=int(n))


def lidar_consistency_flag(member_spectra, freqs, f_lo=0.10, f_hi=0.70,
                           max_dev=0.10, min_live=2, dead_hm0=1e-3):
    """Reject records where one Riegl unit reports a different sea state.

    The three units sit on an equilateral triangle of 1.6 m sides, so at the
    scales resolved here they sample the same water. A time-varying relative
    time-base drift between the units destroys their phase relationship -- which
    is why no directional spectrum can be formed from the triad -- but leaves
    each unit's total variance intact. Hm0 is therefore the drift-immune
    discriminator: units that disagree on it are not merely out of step, one of
    them is measuring something else, and there is no way to tell which.

    A unit reporting essentially zero has no data rather than a small sea; that
    is tolerated as long as `min_live` units remain.

    Args:
        member_spectra: (n_freq, n_run, n_unit) per-unit spectra [m^2/Hz].
        freqs: frequency axis [Hz].
        max_dev: reject above this fractional departure of any live unit from
            the across-unit median Hm0.

    Returns:
        (reject, max_deviation, n_live) -- reject is True where the record should
        be dropped, either for unit disagreement or for too few live units.
    """
    freqs = np.asarray(freqs, float)
    band = np.where((freqs >= f_lo) & (freqs <= f_hi), 1.0, np.nan)
    df = np.median(np.diff(freqs))
    with warnings_suppressed():
        Hm0 = 4 * np.sqrt(np.nansum(band[:, None, None] * member_spectra,
                                    axis=0) * df)
        Hm0 = np.where(Hm0 > dead_hm0, Hm0, np.nan)
        n_live = np.isfinite(Hm0).sum(1)
        med = np.nanmedian(Hm0, axis=1)
        dev = (np.nanmax(np.abs(Hm0 - med[:, None]), axis=1)
               / np.where(med > 0, med, np.nan))
    reject = (n_live < min_live) | (np.isfinite(dev) & (dev > max_dev))
    return reject, dev, n_live


class warnings_suppressed:
    """All-NaN slices are expected here (records with no lidar at all)."""

    def __enter__(self):
        self._c = np.errstate(invalid='ignore', divide='ignore')
        self._c.__enter__()
        import warnings as _w
        self._w = _w.catch_warnings()
        self._w.__enter__()
        _w.simplefilter('ignore', RuntimeWarning)
        return self

    def __exit__(self, *a):
        self._w.__exit__(*a)
        self._c.__exit__(*a)
        return False


def york_band(fit, x, level=0.95):
    """Confidence band on a `york_fit` line: (lo, hi) at abscissae `x`.

    The fit pivots about its weighted centroid, where the slope and the line
    height are uncorrelated, so the variance of the fitted ordinate is
    1/sum(W) + (x - x_centroid)^2 var(slope) -- the familiar hyperbolic band,
    narrowest at the centroid. Uses Student t with n-2 degrees of freedom.

    This is a band on the fitted relationship, not a prediction interval for
    individual records. When the fit was calibrated (chi2_reduced driven to
    unity) the band reflects the observed scatter rather than a priori errors."""
    from scipy import stats as _stats
    x = np.asarray(x, float)
    yhat = fit['intercept'] + fit['slope'] * x
    var = 1.0 / fit['sum_weights'] + (x - fit['x_centroid']) ** 2 * fit['var_slope']
    t = _stats.t.ppf(0.5 + level / 2.0, max(fit['n'] - 2, 1))
    half = t * np.sqrt(var)
    return yhat - half, yhat + half


# Combining the three Riegl spectra by median rather than mean: for 3 normal
# samples var(median) = 1.5 var(mean), so the published lidar spectrum carries
# sqrt(1.5/3) of a single instrument's sampling error.
MEDIAN3_SIGMA_FACTOR = np.sqrt(1.5 / 3.0)


def lidar_member_spectra(path='../_data/', nfft=3000, nperseg=1500, fs=None):
    """Per-instrument Riegl elevation spectra, (n_freq, n_run, 3) [m^2/Hz].

    The published `F_f_m2_Hz_lidar` is the across-instrument median of these, so
    the spread across the three is an empirical, per-record uncertainty on any
    quantity derived from the lidar reference -- no assumed error model needed.
    Welch parameters match compute_all_omnidirectional_spectra.py."""
    import netCDF4 as _nc
    fs = FS_HZ if fs is None else fs
    ds = _nc.Dataset(path + 'ASIT2019_supporting_environmental_observations.nc')
    wse = np.ma.filled(ds['wse_m_Riegl'][:], np.nan)      # (3, n_samp, n_run)
    ds.close()
    n_l, _, n_r = wse.shape
    out = np.full((nfft // 2 + 1, n_r, n_l), np.nan)
    for ri in range(n_r):
        for li in range(n_l):
            xi = wse[li, :, ri]
            if np.isfinite(xi).sum() < nperseg:
                continue
            f, P = signal.welch(np.nan_to_num(xi), fs, nfft=nfft, nperseg=nperseg)
            out[:, ri, li] = P
    return out


def draw_metrics_box(ax, metrics, labels, colors, units, box_xy, box_w, box_h,
                     col_step, unit_dx, fsize, delta_x=(0.03, 0.03, 0.04)):
    """Inset table of per-category (R^2, RMSE, slope, bias) in axes-fraction
    coordinates. metrics: list of scatter_metrics tuples; units: (RMSE, bias)
    unit strings; col_step/unit_dx set the column pitch and units offset."""
    ax.add_patch(plt.Rectangle(box_xy, box_w, box_h, transform=ax.transAxes,
                               color='k', alpha=0.95, edgecolor='k', linewidth=2))
    ax.add_patch(plt.Rectangle(box_xy, box_w, box_h, transform=ax.transAxes,
                               color='w', alpha=0.95, edgecolor='k', linewidth=0.5))
    # Text origin anchored at box_xy lower-left corner
    x0, y0 = box_xy[0] + 0.008, box_xy[1] + box_h - 0.06
    ax.text(x0, y0, 'R²\nRMSE\nslope\nbias', transform=ax.transAxes,
            fontsize=fsize, verticalalignment='top')
    ax.text(x0 + 0.12, y0, ' = \n = \n = \n = ', transform=ax.transAxes,
            fontsize=fsize, verticalalignment='top')
    x = x0 + 0.05
    for (r2, rmse, slope, bias), label, color, dxv in zip(metrics, labels, colors, delta_x):
        x += col_step
        ax.text(x + dxv + 0.01, y0 + 0.05, label, color=color, transform=ax.transAxes,
                fontsize=fsize, verticalalignment='top', horizontalalignment='center')
        ax.text(x, y0, f'{r2:.2f}\n{rmse:.2f}\n{slope:.2f}\n{bias:.2f}', color=color,
                transform=ax.transAxes, fontsize=fsize, verticalalignment='top')
    # units on the dimensional rows (RMSE, bias); blank lines keep row spacing
    ax.text(x + unit_dx, y0, '\n%s\n\n%s' % units, color='k', transform=ax.transAxes,
            fontsize=fsize, verticalalignment='top')

# %%

# Mueller calculus
# N. J. M. Laxague 2019
# From Zappa et al. 2008 (Z08)
# Following Kattawar & Adams 1989 (KA89)

def mueller_calc_full(n,Ssky,Sup):

    theta_i = np.linspace(0,np.pi/2,10000)
    theta_t = np.asin(np.sin(theta_i)/n)

    # Mueller matrix element equations, KA89
    # taken from Eq. 3 of Z08
    alpha = 1/2*(np.tan(theta_i-theta_t)/np.tan(theta_i+theta_t))**2;
    eta = 1/2*(np.sin(theta_i-theta_t)/np.sin(theta_i+theta_t))**2;

    alpha_prime = 1/2*(2*np.sin(theta_t)*np.sin(theta_i)/(np.sin(theta_i+theta_t)*np.cos(theta_i-theta_t)))**2;
    eta_prime = 1/2*(2*np.sin(theta_t)*np.sin(theta_i)/np.sin(theta_t+theta_i))**2;

    gamma_Re = (np.tan(theta_i-theta_t)*np.sin(theta_i-theta_t))/(np.tan(theta_i+theta_t)*np.sin(theta_i+theta_t));
    gamma_Re_prime = 4*(np.sin(theta_t)**2*np.sin(theta_i)**2)/(np.sin(theta_t+theta_i)**2*np.cos(theta_t-theta_i)**2);

    # Stokes parameter components, reflected and transmitted radiance
    # taken from Eq. 4 of Z08
    S0_Re = Ssky[0]*(alpha+eta) + Ssky[1]*(alpha-eta);
    S1_Re = Ssky[0]*(alpha-eta) + Ssky[1]*(alpha+eta);
    S2_Re = Ssky[2]*gamma_Re;
    S3_Re = Ssky[3]*gamma_Re;

    S0_Tr = Sup[0]*(alpha_prime+eta_prime) + Sup[1]*(alpha_prime-eta_prime);
    S1_Tr = Sup[0]*(alpha_prime-eta_prime) + Sup[1]*(alpha_prime+eta_prime);
    S2_Tr = Sup[2]*gamma_Re_prime;
    S3_Tr = Sup[3]*gamma_Re_prime;

    S0 = S0_Re + S0_Tr;
    S1 = S1_Re + S1_Tr;
    S2 = S2_Re + S2_Tr;
    S3 = S3_Re + S3_Tr;

    # DOLP calculation
    DoLP = np.sqrt(S1**2+S2**2+S3**2)/S0;
    DoLP[0] = DoLP[1] - (DoLP[2]-DoLP[1]);

    out_theta = 180/np.pi*theta_i;
    out_DOLP = DoLP;

    return(out_theta,out_DOLP)

# %%

# Given ten-meter wind speed in m/s, returns wave slope
# joint probability density function and upwind/crosswind mean square slope

# Procedure following Cox & Munk [1954]

# N. Laxague 2023-2025

def compute_gram_charlier_slope_pdf(U10_m_s):

    slope_centers = np.linspace(-1,1,num=200)

    mss_up = 1e-3 + 3.16*1e-3*U10_m_s
    mss_cross = 3*1e-3 + 1.85*1e-3*U10_m_s

    c21 = -9.1e-4*U10_m_s**2
    c03 = -0.45*(1+np.exp(7-U10_m_s))**-1
    c40 = 0.3
    c04 = 0.4
    c22 = 0.12

    # xi (crosswind) on axis 0, zeta (upwind) on axis 1 via indexing='ij'
    xi, zeta = np.meshgrid(slope_centers / np.sqrt(mss_cross), slope_centers / np.sqrt(mss_up), indexing='ij')

    coeff = (2 * np.pi * np.sqrt(mss_up) * np.sqrt(mss_cross)) ** -1
    PDF_cross_along = coeff * np.exp(-(xi**2 + zeta**2) / 2) * (
            1 +
            -0.5 * c21 * (xi**2 - 1) * zeta +
            -1/6 * c03 * (zeta**3 - 3 * zeta) +
            1/24 * c40 * (xi**4 - 6 * xi**2 + 3) +
            1/24 * c04 * (zeta**4 - 6 * zeta**2 + 3) +
            1/4 * c22 * (xi**2 - 1) * (zeta**2 - 1)
        )

    wave_slope_PDF = xr.DataArray(
        PDF_cross_along,
        coords = {
            'slope_cross': slope_centers,
            'slope_up': slope_centers
            },
        dims = ['slope_cross', 'slope_up']
        )

    # Normalize to unit integral over the tabulated slope range
    wave_slope_PDF = wave_slope_PDF/wave_slope_PDF.integrate('slope_cross').integrate('slope_up')

    return wave_slope_PDF, mss_cross, mss_up

# %%

# Given water surface slope joint probability density function, compute
# least-squares Gram-Charlier fit and skewness/kurtosis coefficients

# Procedure following Cox & Munk [1954]

# N. Laxague 2023-2025

def fit_gram_charlier_slope_pdf(slope_centers, P_slope_c_u, mss_u, mss_c):

    # xi (crosswind) on axis 0, zeta (upwind) on axis 1, indexing='ij' to match P_slope_c_u
    xi, zeta = np.meshgrid(slope_centers / np.sqrt(mss_c), slope_centers / np.sqrt(mss_u), indexing='ij')

    def fit(b, x, y):
        coeff = (2 * np.pi * np.sqrt(mss_u) * np.sqrt(mss_c)) ** -1
        return coeff * np.exp(-(x**2 + y**2) / 2) * (
            1 +
            -0.5 * b[0] * (x**2 - 1) * y +
            -1/6 * b[1] * (y**3 - 3 * y) +
            1/24 * b[2] * (x**4 - 6 * x**2 + 3) +
            1/24 * b[3] * (y**4 - 6 * y**2 + 3) +
            1/4 * b[4] * (x**2 - 1) * (y**2 - 1)
        )

    def cost_function(b):
        return np.sum((fit(b, xi, zeta) - P_slope_c_u) ** 2)

    initial_guess = np.zeros(5)
    result = minimize(cost_function, initial_guess)

    P_fit = fit(result.x, xi, zeta)
    residuals = P_fit - P_slope_c_u

    ss_res = np.sum(residuals ** 2)  # Residual sum of squares
    ss_tot = np.sum((P_slope_c_u - np.mean(P_slope_c_u)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)  # R-squared
    rmse = np.sqrt(ss_res / np.size(P_slope_c_u))  # RMSE

    # G-C fit and skewness/kurtosis coefficients
    out_struc = {
        'P_fit': P_fit,                    # Gram-Charlier expansion fit
        'c21': result.x[0],                # skewness, upwind
        'c03': result.x[1],                # skewness, upwind
        'c40': result.x[2],                # kurtosis
        'c04': result.x[3],                # kurtosis
        'c22': result.x[4],                # kurtosis
        'R_squared': r_squared,            # R-squared value
        'RMSE': rmse                       # Root Mean Square Error
    }

    return out_struc

# %%

def omni_complete_spectrum(slope_east, slope_north, water_depth_m, fs_Hz,
                           fmin_Hz=0.08, transition_octaves=0.25,
                           nfft=3000, nperseg=1500, highpass_peak_fraction=None,
                           highpass_peak_floor_Hz=0.08, highpass_corner_floor_Hz=0.06,
                           aperture_diameter_m=None, aperture_min_transfer=0.5):
    """Directionally-complete omnidirectional elevation spectrum (S_sx+S_sy)/k^2
    with a squared logistic high-pass (corner fmin_Hz, or adaptive at
    highpass_peak_fraction * fp when set).

    aperture_diameter_m: if set, divide out the circular-disc aperture MTF H(k)^2
    (jinc), capped at 1/aperture_min_transfer^2 to avoid noise amplification near
    nulls."""
    sE = np.asarray(slope_east, dtype=float).reshape(-1)
    sN = np.asarray(slope_north, dtype=float).reshape(-1)
    sE = np.where(np.isfinite(sE), sE, 0.0)
    sN = np.where(np.isfinite(sN), sN, 0.0)
    f, P_sx = signal.welch(sE, fs_Hz, nfft=nfft, nperseg=nperseg)
    _, P_sy = signal.welch(sN, fs_Hz, nfft=nfft, nperseg=nperseg)
    _, k = _lindisp_with_current(2 * np.pi * np.maximum(f, 1e-6), water_depth_m, 0.0)
    k = np.nan_to_num(np.asarray(k, dtype=float), nan=np.inf, posinf=np.inf)
    S = (P_sx + P_sy) / np.maximum(k ** 2, 1e-12)
    if aperture_diameter_m is not None:
        H = _aperture_transfer_function(k, aperture_diameter_m, shape="circular")
        gain = 1.0 / np.maximum(np.abs(H), aperture_min_transfer) ** 2
        S = S * np.where(np.isfinite(gain), gain, 1.0)   # f=0 (k=inf): no correction
    corner = fmin_Hz
    if highpass_peak_fraction is not None:
        sel = (f >= highpass_peak_floor_Hz) & (f <= 0.40)
        if sel.any():
            fp = f[sel][int(np.argmax(S[sel]))]
            corner = max(highpass_corner_floor_Hz, highpass_peak_fraction * fp)
    if corner is not None:
        hp = 1.0 / (1.0 + np.exp(
            -(np.log2(np.maximum(f, 1e-9)) - np.log2(corner)) / transition_octaves))
        S = S * hp ** 2
    return f, S


# %%

# Directional spreading helpers. SPREAD_SMOOTHNUM is the default centered
# rolling-mean window (over the scale/frequency axis) applied to spreading
# densities; set it to 0 to turn that smoothing off everywhere at once.
#
# The direct 3-D-FFT spectrum carries ~37 points per octave against ~7-10 for
# the ADCP and EWDM grids, so it takes a wider window to span the same
# fractional bandwidth (9/37 vs 3/10 of an octave).

SPREAD_SMOOTHNUM = 3
SPREAD_SMOOTHNUM_DIRECT = 9


def _rolling_mean(a, n):
    """Centered n-point rolling mean over axis 0 (NaN-aware, edge-shrinking).
    n <= 1 (or None) returns the input unchanged, i.e. smoothing off.

    Note that `compute_mean_wave_direction_and_spreading` smooths with a true
    rolling median instead; the two are not interchangeable on a ragged row."""
    a = np.asarray(a, float)
    if n is None or n <= 1:
        return a
    m = a.shape[0]
    h = n // 2
    out = a.copy()
    for i in range(m):
        out[i] = np.nanmean(a[max(0, i - h):min(m, i + h + 1)], axis=0)
    return out


def spreading(F, dtheta, smooth=None):
    """Row-normalized directional spreading D(theta|scale), per-degree density.

    F is (scale, direction); dtheta is the direction step [deg]. Each scale row is
    normalized to unit integral over direction and returned per-degree, so a
    per-radian input differs from this by 180/pi. Rows with no energy stay NaN so
    an unsupported band renders blank rather than as zero density. `smooth` is the
    centered rolling-mean window over the scale axis: None uses SPREAD_SMOOTHNUM,
    0 or 1 disables it.
    """
    tot = np.nansum(F, axis=1, keepdims=True) * np.radians(dtheta)
    D = np.divide(F, tot, out=np.full_like(F, np.nan), where=tot > 0) * np.pi / 180
    return _rolling_mean(D, SPREAD_SMOOTHNUM if smooth is None else smooth)


def norm_smooth(F, smooth=None):
    """`spreading` for an xarray (scale, direction) carrying a degree direction
    coordinate. Returns a DataArray on the same coords; `smooth` behaves as in
    `spreading`."""
    dth = float(np.median(np.diff(F['direction'].data)))
    D = F.copy()
    D.data = spreading(F.data, dth, smooth=smooth)
    return D


def wrap_deg(x):
    """Directions wrapped onto [-180, 180) deg."""
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def panel_vmax(panel_vals, pct=98):
    """Upper color limit shared across panels, weighting each panel equally.

    A percentile pooled over the panels is dominated by whichever one retains
    the most in-window rows, so a low-support panel (the ADCP, on its coarse
    grid) would otherwise set the scale for everything. Take each panel's own
    percentile, then the median across panels."""
    per = [np.nanpercentile(v, pct) for v in panel_vals
           if np.size(v) and np.isfinite(v).any()]
    return float(np.nanmedian(per)) if per else 1.0


def lobe_spread(dirs_deg, dens, halfwidth=90.0):
    """Single-lobe RMS directional spread sigma_theta(scale) [deg]. +-halfwidth
    deg isolates the dominant lobe, excluding the direct mirror lobe."""
    th = np.radians(np.asarray(dirs_deg))
    E = np.nan_to_num(dens)
    nsc = E.shape[0]
    sig = np.full(nsc, np.nan)
    h = np.radians(halfwidth)
    for j in range(nsc):
        D = E[j]
        s = D.sum()
        if s <= 0:
            continue
        a = (D * np.cos(th)).sum()
        b = (D * np.sin(th)).sum()
        d = np.angle(np.exp(1j * (th - np.arctan2(b, a))))
        Dk = D * (np.abs(d) <= h)
        sk = Dk.sum()
        if sk <= 0:
            continue
        thc = np.arctan2((Dk * np.sin(th)).sum(), (Dk * np.cos(th)).sum())
        d2 = np.angle(np.exp(1j * (th - thc)))
        sig[j] = np.degrees(np.sqrt((Dk * d2 ** 2).sum() / sk))
    return sig


def lobe_sigma(F, halfwidth=90.0, smooth=5):
    """Single-lobe sigma_theta(scale) from an xarray (scale, direction) with a
    degree direction coord, smoothed by a centered rolling mean (smooth<=1 off)."""
    return _rolling_mean(
        lobe_spread(F['direction'].data, np.nan_to_num(F.data), halfwidth), smooth)


# %%

# Compute mean wave direction and directional spreading width from a frequency-
# or wavenumber-directional spectrum. theta_halfwidth masks the +/-180 deg ambiguity.
#
# N. Laxague 2025

def compute_mean_wave_direction_and_spreading(F_dirspec,theta_halfwidth,smoothnum=3):

    # Work on a copy; the caller's spectrum is left untouched
    F_dirspec = F_dirspec.copy(deep=True)
    F_dirspec.data = np.nan_to_num(F_dirspec.data, nan=0.0)
    spec_energy_density = F_dirspec.data

    wavedir = F_dirspec["direction"].copy()
    dtheta = np.median(np.diff(wavedir.data))

    if 'frequency' in F_dirspec.coords:

        fourier_scale = F_dirspec["frequency"].data
        fourier_scale_name = 'frequency'

    if 'wavenumber' in F_dirspec.coords:

        fourier_scale = F_dirspec["wavenumber"].data
        fourier_scale_name = 'wavenumber'
        spec_energy_density = spec_energy_density*np.reshape(fourier_scale,(len(fourier_scale),1))

    D_array = ((F_dirspec.T / F_dirspec.integrate("direction")).rolling({fourier_scale_name: smoothnum}, center=True).median()).T
    D_array.data = np.nan_to_num(D_array.data, nan=0.0)

    Dtheta = D_array.integrate(fourier_scale_name)
    ind_p = np.argmax(Dtheta.data)

    theta_super = np.concatenate((wavedir-360,wavedir,wavedir+360),axis=0)
    # round to avoid float noise duplicating bins at +/-180 deg boundary
    theta_rel = np.round(theta_super - wavedir.data[ind_p], 3)
    D_array_super = np.concatenate((D_array.data,D_array.data,D_array.data),axis=1)
    F_array_super = np.concatenate((spec_energy_density,spec_energy_density,spec_energy_density),axis=1)

    inds_keep = (theta_rel >= -180) & (theta_rel < 180)
    theta_rel = theta_rel[inds_keep]
    D_array_super = D_array_super[:,inds_keep]
    F_array_super = F_array_super[:,inds_keep]

    D_array["direction"] = theta_rel
    D_array.data = D_array_super

    MWD = np.sum(np.reshape(D_array.direction.data,(1,len(D_array.direction.data)))*F_array_super)/np.sum(F_array_super) + wavedir.data[ind_p]

    wavedir_diff = np.abs(wavedir.data)
    inds_trim = (wavedir_diff > theta_halfwidth) & (wavedir_diff < 360-theta_halfwidth)

    D_array_downwave = D_array.copy()
    upwave_val = np.sum(D_array.data[:,inds_trim])
    downwave_val = np.sum(D_array.data[:,~inds_trim])
    D_array_downwave.data = D_array_downwave.data*(upwave_val+downwave_val)/downwave_val
    D_array_downwave.data[:,inds_trim] = 1e-10

    theta_nought_array = np.sum(np.reshape(D_array_downwave.direction.data,(1,len(D_array_downwave.direction.data)))*D_array_downwave.data,axis=1)*dtheta
    theta_array_mat = np.tile(np.reshape(D_array_downwave.direction.data,(1,len(D_array_downwave.direction.data))),(len(theta_nought_array),1))
    theta_nought_array_mat = np.tile(np.reshape(theta_nought_array,(len(theta_nought_array),1)),(1,len(D_array_downwave.direction.data)))
    d_theta2_array = (theta_array_mat-theta_nought_array_mat)**2

    sigma_theta_array = np.sqrt(np.sum(d_theta2_array*D_array_downwave.data,axis=1)*dtheta)

    sigma_theta_array[sigma_theta_array<1.0] = np.nan

    spread = xr.DataArray(
        sigma_theta_array,
        name = 'sigma_theta',
        coords = {fourier_scale_name: fourier_scale},
        dims = fourier_scale_name,
        attrs = {"units": 'degrees'},
        )

    return MWD, spread

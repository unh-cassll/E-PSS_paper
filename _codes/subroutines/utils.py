"""
Project-specific helpers for the E-PSS paper.

    L_FOV_M, DX_M, WATER_DEPTH_M, ...     canonical ASIT2019 deployment constants
    figure_style                          paper plot styling
    wind_speed_bins                       canonical fixed-width U10 bins
    write_tex_macros                      LaTeX \newcommand value file for paper.tex
    scatter_metrics                       R^2/RMSE/slope/bias of a scatter comparison
    draw_metrics_box                      inset metrics table for scatter figures
    ewdm_low_cutoff                       EWDM low-scale trust cutoff (k_low, f_low)
    epss_ewdm_grids                       EWDM f/k/nu grids (generator config)
    mueller_calc_full                     4-Stokes sky+upwelling Mueller calc
    compute_gram_charlier_slope_pdf       Cox-Munk Gram-Charlier slope PDF
    fit_gram_charlier_slope_pdf           least-squares Gram-Charlier fit
    slope_to_elev_wavelet                 1-D wavelet slope-to-elevation inversion
    omni_complete_spectrum                directionally-complete (S_sx+S_sy)/k^2 spectrum
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

# Internal-only upstream imports (underscore-aliased so they do not leak
# through `from subroutines.utils import *`).
from eta_field_recon import lindisp_with_current as _lindisp_with_current
from eta_field_recon import aperture_transfer_function as _aperture_transfer_function
from eta_field_recon.wavelet_core import (
    _cwt as _ewdm_cwt,
    _inverse_cwt as _ewdm_icwt,
    krogstad_eta_coeffs as _krogstad_eta_coeffs,
    skirt_correction as _skirt_correction,
)

__all__ = [
    'GRAV', 'L_FOV_M', 'N_PX', 'DX_M', 'WATER_DEPTH_M', 'FS_HZ', 'NUM_RUNS',
    'NUM_SAMPLES',
    'figure_style', 'wind_speed_bins', 'write_tex_macros',
    'scatter_metrics', 'draw_metrics_box', 'ewdm_low_cutoff', 'epss_ewdm_grids',
    'mueller_calc_full', 'compute_gram_charlier_slope_pdf',
    'fit_gram_charlier_slope_pdf', 'slope_to_elev_wavelet',
    'omni_complete_spectrum', 'compute_mean_wave_direction_and_spreading',
]

# %%

# Canonical ASIT2019 / E-PSS deployment constants (single source of truth for
# values repeated across the compute and plot scripts)

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

# Canonical fixed-width U10 bins for wind-binned figures; Umin/Umax are the outer edges

# N. Laxague 2026

def wind_speed_bins(Umin=0.0, Umax=14.0, dU=2.0):

    edges = np.arange(Umin, Umax + dU / 2, dU)
    centers = edges[:-1] + dU / 2

    return centers, edges, dU

# %%

# Write computed figure values as LaTeX macros for \input into paper.tex, so the
# numbers in captions/text/tables stay in sync with the figures that produce them.

# N. Laxague 2026

def write_tex_macros(filename, macros, source=None, directory='../_tex'):
    r"""Write a LaTeX macro file (one value per \newcommand) for \input into paper.tex.

    filename  : output name, e.g. 'Hm0_values.tex'; written under `directory`
                (default '../_tex' -> repo-root/_tex when run from _codes).
    macros    : dict {name: value}. `name` must be letters only (a valid LaTeX
                control sequence); prefix per figure to avoid clashes across files
                (e.g. 'HmRMSEemp'). `value` is stringified -- pass a pre-formatted
                string for explicit precision, e.g. f'{x:.2f}'.
    source    : optional producing-script name, recorded in the file header.

    Each macro is emitted as
        \providecommand{\name}{}\renewcommand{\name}{value}
    so re-running the figure (or re-\input) never raises 'already defined'. The file
    is written atomically. Use inline in paper.tex as \name (e.g. \HmRMSEemp)."""
    import os, re
    bad = sorted(k for k in macros if not re.fullmatch('[A-Za-z]+', k))
    if bad:
        raise ValueError('LaTeX macro names must be letters only; invalid: %s' % bad)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    header = '%% auto-generated%s; do not edit by hand\n' % (' by ' + source if source else '')
    body = ''.join('\\providecommand{\\%s}{}\\renewcommand{\\%s}{%s}\n' % (k, k, v)
                   for k, v in macros.items())
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        fh.write(header + body)
    os.replace(tmp, path)
    return path

# %%

# Scatter-comparison metrics and the inset metrics table shared by the
# Hm0/Tm02 lidar-vs-E-PSS figures


def scatter_metrics(x, y):
    """(R^2, RMSE, slope, intercept) of y vs x over finite pairs."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    rmse = float(np.sqrt(np.mean((y[keep] - x[keep])**2)))
    r = np.corrcoef(x[keep], y[keep])[0, 1]
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    return r**2, rmse, float(slope), float(intercept)


def draw_metrics_box(ax, metrics, labels, colors, units, box_xy, box_w, box_h,
                     col_step, unit_dx, fsize, delta_x=(0.03, 0.03, 0.04)):
    """Inset table of per-category (R^2, RMSE, slope, bias) in axes-fraction
    coordinates. metrics: list of scatter_metrics tuples; units: (RMSE, bias)
    unit strings; col_step/unit_dx set the column pitch and units offset."""
    ax.add_patch(plt.Rectangle(box_xy, box_w, box_h, transform=ax.transAxes,
                               color='k', alpha=0.95, edgecolor='k', linewidth=2))
    ax.add_patch(plt.Rectangle(box_xy, box_w, box_h, transform=ax.transAxes,
                               color='w', alpha=0.95, edgecolor='k', linewidth=0.5))
    x0, y0 = 0.02, 0.93
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

    # Assign output
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

    # xi (crosswind) on axis 0, zeta (upwind) on axis 1 to match P_slope_c_u; 'xy' would transpose the model
    xi, zeta = np.meshgrid(slope_centers / np.sqrt(mss_c), slope_centers / np.sqrt(mss_u), indexing='ij')

    # Function to fit
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

    # Least-Squares cost function
    def cost_function(b):
        return np.sum((fit(b, xi, zeta) - P_slope_c_u) ** 2)

    # Minimize Least-Squares
    initial_guess = np.zeros(5)
    result = minimize(cost_function, initial_guess)

    # Calculate fitted values and residuals
    P_fit = fit(result.x, xi, zeta)
    residuals = P_fit - P_slope_c_u

    # Calculate goodness of fit metrics
    ss_res = np.sum(residuals ** 2)  # Residual sum of squares
    ss_tot = np.sum((P_slope_c_u - np.mean(P_slope_c_u)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)  # R-squared
    rmse = np.sqrt(ss_res / np.size(P_slope_c_u))  # Root Mean Square Error

    # Send G-C fit and skewness/kurtosis coefficients to output structure
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

# 1-D wavelet inversion of earth-referenced slope (s_east, s_north) to surface elevation.
# Composed from eta_field_recon.wavelet_core primitives (CWT, Krogstad projection, iCWT).
#
# N. Laxague 2026

def slope_to_elev_wavelet(
    slope_east: np.ndarray,
    slope_north: np.ndarray,
    water_depth_m: float,
    fs_Hz: float,
    fmin_Hz: float = 0.08,
    fmax_Hz: float = None,
    transition_octaves: float = 0.25,
    tukey_alpha: float = 0.25,
    per_scale: bool = True,
    skirt_correct: bool = True,
    window_power_correct: bool = False,
    lf_noise_suppress: bool = False,
    lf_noise_band_Hz: tuple = (2.0, 4.5),
    lf_noise_oversub: float = 1.0,
    highpass_peak_fraction: float = None,
    highpass_peak_floor_Hz: float = 0.08,
    highpass_corner_floor_Hz: float = 0.06,
) -> np.ndarray:
    """1-D wavelet slope-to-elevation inversion (Krogstad signed projection).

    Composes eta_field_recon CWT and Krogstad-projection primitives.
    CWT grid: linspace(0.05, 2.0, 80) Hz. Sigmoidal bandpass suppresses
    the low-frequency 1/k blow-up. per_scale and skirt_correct use upstream
    per-frequency calibration and Krogstad 1/k(omega) skirt reshaping.

    Args:
        slope_east, slope_north : (T,) earth-referenced slope time series
        water_depth_m           : water depth (m)
        fs_Hz                   : sampling frequency (Hz)
        fmin_Hz, fmax_Hz        : -6 dB corners of the sigmoidal bandpass (Hz);
                                  pass None to disable that side.
        transition_octaves      : sigmoid transition width (octaves)
        tukey_alpha             : Tukey window cosine fraction (edge taper)
        lf_noise_suppress       : if True, apply Wiener spectral subtraction of
                                  the white slope noise amplified by 1/k^2
        lf_noise_band_Hz        : (lo, hi) band for slope noise floor estimate (Hz)
        lf_noise_oversub        : Wiener over-subtraction factor (1 = exact)
        highpass_peak_fraction  : if set, place the HP corner adaptively at
                                  this fraction of the spectral peak fp, searched
                                  over [highpass_peak_floor_Hz, 0.40 Hz]
        highpass_peak_floor_Hz  : lower bound of the fp search (Hz)
        highpass_corner_floor_Hz: absolute lower clamp on the adaptive HP corner (Hz)
        window_power_correct    : if True, divide eta by sqrt(mean(w^2)) to
                                  debias variance over the full tapered record

    Returns:
        eta_m : (T,) reconstructed surface elevation (m)
    """
    sE = np.asarray(slope_east, dtype=float).reshape(-1)
    sN = np.asarray(slope_north, dtype=float).reshape(-1)
    if sE.size != sN.size:
        raise ValueError("slope_east and slope_north must have the same length")

    freqs = np.linspace(0.05, 2.0, 80)

    # Linear detrend removes static viewing-angle tilt and slow drift
    sE = detrend(sE, type="linear")
    sN = detrend(sN, type="linear")

    # Adaptive highpass corner from slope-implied eta spectral peak
    fmin_eff = fmin_Hz
    if highpass_peak_fraction is not None and fmin_Hz is not None:
        nps = int(min(512, len(sE)))
        fW, PsE_ = signal.welch(sE, fs_Hz, nperseg=nps)
        _, PsN_ = signal.welch(sN, fs_Hz, nperseg=nps)
        _, kW = _lindisp_with_current(2 * np.pi * np.maximum(fW, 1e-6),
                                      water_depth_m, 0.0)
        kW = np.nan_to_num(np.asarray(kW, dtype=float), nan=np.inf, posinf=np.inf)
        eta_ideal = (PsE_ + PsN_) / np.maximum(kW ** 2, 1e-12)
        sel = (fW >= highpass_peak_floor_Hz) & (fW <= 0.40)
        if sel.any():
            fp = fW[sel][int(np.argmax(eta_ideal[sel]))]
            # corner tied to fp; floored at CWT-edge safety floor
            fmin_eff = max(highpass_corner_floor_Hz, highpass_peak_fraction * fp)

    w = tukey(len(sE), alpha=tukey_alpha)
    sE_w = sE * w
    sN_w = sN * w

    Wsx = _ewdm_cwt(sE_w, freqs=freqs, fs=fs_Hz).values
    Wsy = _ewdm_cwt(sN_w, freqs=freqs, fs=fs_Hz).values

    _, k_disp = _lindisp_with_current(2 * np.pi * freqs, water_depth_m, 0.0)

    if skirt_correct:
        skirt_gain = _skirt_correction(
            freqs, fs_Hz, k_disp, len(sE),
            per_scale=per_scale, temporal_alpha=tukey_alpha,
        )
    else:
        skirt_gain = None
    W_eta, _, _ = _krogstad_eta_coeffs(Wsx, Wsy, k_disp, skirt_gain=skirt_gain)

    # Sigmoidal bandpass on elevation wavelet coefficients
    bandpass = np.ones_like(freqs)
    log2f = np.log2(freqs)
    if fmin_Hz is not None:
        bandpass *= 1.0 / (1.0 + np.exp(
            -(log2f - np.log2(fmin_eff)) / transition_octaves
        ))
    if fmax_Hz is not None:
        bandpass *= 1.0 / (1.0 + np.exp(
            (log2f - np.log2(fmax_Hz)) / transition_octaves
        ))
    W_eta = W_eta * bandpass[:, None]

    eta = _ewdm_icwt(W_eta, freqs=freqs, fs=fs_Hz, per_scale=per_scale)
    eta = np.asarray(eta, dtype=float)

    # Up-positive sign set upstream in krogstad_eta_coeffs (+1j convention).
    # Spectra/Hm0 are quadratic and blind to the sign; only a waveform check
    # (synthetic unit cosine, ASIT2019 lidar) constrains it.

    if window_power_correct:
        eta = eta / np.sqrt(np.mean(w ** 2))

    if lf_noise_suppress:
        n = len(eta)
        nps = int(min(512, n))
        # White slope noise floor (median over high-f band)
        fS, PsE = signal.welch(sE, fs_Hz, nperseg=nps)
        _, PsN = signal.welch(sN, fs_Hz, nperseg=nps)
        mb = (fS >= lf_noise_band_Hz[0]) & (fS <= lf_noise_band_Hz[1])
        Ns = 0.5 * (np.median(PsE[mb]) + np.median(PsN[mb]))
        # Elevation noise PSD per unit slope noise, through this inversion
        fT, T_noise = _lf_noise_transfer(
            fs_Hz, n, water_depth_m, fmin_Hz, fmax_Hz, transition_octaves,
            tukey_alpha, per_scale, skirt_correct, nps)
        _, P_eta = signal.welch(eta, fs_Hz, nperseg=nps)
        G = 1.0 - lf_noise_oversub * (Ns * T_noise) / np.maximum(P_eta, 1e-30)
        G = np.clip(G, 0.0, 1.0)
        G = np.convolve(G, np.ones(5) / 5, mode="same")
        F = np.fft.rfft(eta); ff = np.fft.rfftfreq(n, d=1.0 / fs_Hz)
        gain = np.sqrt(np.clip(np.interp(ff, fT, G, left=G[0], right=1.0), 0.0, 1.0))
        eta = np.fft.irfft(F * gain, n)

    return eta


def omni_complete_spectrum(slope_east, slope_north, water_depth_m, fs_Hz,
                           fmin_Hz=0.08, transition_octaves=0.25,
                           nfft=3000, nperseg=1500, highpass_peak_fraction=None,
                           highpass_peak_floor_Hz=0.08, highpass_corner_floor_Hz=0.06,
                           aperture_diameter_m=None, aperture_min_transfer=0.5):
    """Directionally-complete omnidirectional elevation spectrum (S_sx+S_sy)/k^2
    with a squared logistic high-pass (corner fixed at fmin_Hz, or adaptive at
    highpass_peak_fraction * spectral peak when set).

    aperture_diameter_m: if set, divide out the circular-disc aperture transfer
    H(k)^2 (jinc), undoing the spatial low-pass of averaging slope over the disc.
    The gain is capped at 1/aperture_min_transfer^2 (frozen where H<min_transfer)
    so the near-null region is not noise-amplified; this recovers the high-f
    spectrum up to ~H=0.5 (a single disc cannot recover its nulled band)."""
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


# Cache: elevation noise PSD per unit white-slope-noise PSD, through the exact
# inversion. Run-independent (depends only on config), so computed once.
_LF_NOISE_TRANSFER_CACHE = {}


def _lf_noise_transfer(fs_Hz, n, water_depth_m, fmin_Hz, fmax_Hz,
                       transition_octaves, tukey_alpha, per_scale, skirt_correct,
                       nperseg, nrel=4):
    """Welch PSD of the reconstructed elevation when the input slope is unit-PSD
    white noise, averaged over `nrel` realizations and cached on the config."""
    key = (round(float(fs_Hz), 6), int(n), round(float(water_depth_m), 6),
           None if fmin_Hz is None else round(float(fmin_Hz), 6),
           None if fmax_Hz is None else round(float(fmax_Hz), 6),
           round(float(transition_octaves), 6), round(float(tukey_alpha), 6),
           bool(per_scale), bool(skirt_correct), int(nperseg))
    cached = _LF_NOISE_TRANSFER_CACHE.get(key)
    if cached is not None:
        return cached
    sigma = np.sqrt(fs_Hz / 2.0)          # one-sided white PSD == 1
    Pacc = None; fref = None
    for r in range(nrel):
        rng = np.random.default_rng(1234 + r)
        nE = sigma * rng.standard_normal(n)
        nN = sigma * rng.standard_normal(n)
        en = slope_to_elev_wavelet(
            nE, nN, water_depth_m, fs_Hz, fmin_Hz=fmin_Hz, fmax_Hz=fmax_Hz,
            transition_octaves=transition_octaves, tukey_alpha=tukey_alpha,
            per_scale=per_scale, skirt_correct=skirt_correct,
            lf_noise_suppress=False)
        fref, Pen = signal.welch(en, fs_Hz, nperseg=nperseg)
        Pacc = Pen if Pacc is None else Pacc + Pen
    Pacc /= nrel
    _LF_NOISE_TRANSFER_CACHE[key] = (fref, Pacc)
    return fref, Pacc


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
    # round off float noise so the periodic window keeps exactly one image of
    # each bin (else a bin 180 deg from the peak can appear twice)
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

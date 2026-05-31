"""
Project-specific helpers for the E-PSS paper.

Routines kept here have no direct upstream equivalent:
    - figure_style                          paper plot styling
    - mueller_calc_full                     4-Stokes sky+upwelling Mueller calc
    - compute_gram_charlier_slope_pdf       Cox-Munk Gram-Charlier expansion
    - fit_gram_charlier_slope_pdf           least-squares fit of the above
    - slope_to_elev_wavelet                 1D wavelet inversion of slope -> eta
    - trim_EPSS_dirspec                     +/-180 deg ambiguity resolution
    - compute_mean_wave_direction_and_spreading
    - plot_directional_spectrum and friends N-up polar variant

Upstream entry points (lindisp_with_current, ewdm.Triplets, etc.) are
imported directly in the scripts that need them.
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
from eta_field_recon.wavelet_core import (
    _cwt as _ewdm_cwt,
    _inverse_cwt as _ewdm_icwt,
    krogstad_eta_coeffs as _krogstad_eta_coeffs,
    skirt_correction as _skirt_correction,
)

# %%

# Figure style function

def figure_style(title_fontsize=12, label_fontsize=10, tick_fontsize=10):

    fsize = 10
    lw = 1.0

    sns.set_context("paper", rc={
        "axes.grid": True,
        "font.size": fsize,
        "axes.titlesize": title_fontsize,
        "axes.labelsize": label_fontsize,
        "xtick.labelsize": tick_fontsize,
        "ytick.labelsize": tick_fontsize,
        "grid.linewidth": lw,
        "xtick.major.width": lw,
        "ytick.major.width": lw,
    })

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

    # Create meshgrid for xi and zeta (normalized cross and upwind mss)
    xi, zeta = np.meshgrid(slope_centers / np.sqrt(mss_cross), slope_centers / np.sqrt(mss_up))

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
        dims = {'slope_cross','slope_up'}
        )

    wave_slope_PDF['PDF_cross_along'] = wave_slope_PDF/wave_slope_PDF.integrate('slope_cross').integrate('slope_up')

    return wave_slope_PDF, mss_cross, mss_up

# %%

# Given water surface slope joint probability density function, compute
# least-squares Gram-Charlier fit and skewness/kurtosis coefficients

# Procedure following Cox & Munk [1954]

# N. Laxague 2023-2025

def fit_gram_charlier_slope_pdf(slope_centers, P_slope_c_u, mss_u, mss_c):

    # Create meshgrid for xi and zeta
    xi, zeta = np.meshgrid(slope_centers / np.sqrt(mss_c), slope_centers / np.sqrt(mss_u))

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

# Wavelet inversion of 1-D earth-referenced wave slope (s_east, s_north) to
# water-surface elevation eta(t). This is the 1-D analogue of the eta_long(t)
# computation inside eta_field_recon.reconstruct_eta_field: continuous wavelet
# transform of each slope component, Krogstad signed projection onto the
# elevation wavelet coefficients via the linear-dispersion wavenumber, and
# inverse CWT. The 1-D path is not exposed as a top-level upstream entry point
# but is composed here from the same wavelet_core primitives.
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

    Composes eta_field_recon's CWT and Krogstad-projection primitives. The
    CWT grid is fixed at linspace(0.05, 2.0, 80) Hz to match the upstream
    reconstruct_eta_field default. A smooth sigmoidal high-pass is applied
    to the elevation wavelet coefficients before the inverse CWT to
    suppress the low-frequency blow-up artifact: at the CWT's minimum-
    frequency edge (0.05 Hz) deep-water k ~ 1e-2 rad/m, so the 1/k factor
    on W_eta = i*proj/k amplifies tiny slope noise by ~100x; without a HP
    this dominates the inversion. The default corner (0.08 Hz) is the gentle
    high-pass of the legacy FFT-based elevation pipeline. A low-pass cutoff
    is not enabled by default: 1/k naturally attenuates HF content (1/k^2
    in spectral density), so high-frequency noise contributes negligibly to
    the recovered elevation.

    per_scale=True (default) uses the upstream per-frequency inverse-CWT
    calibration (replaces the universal 1.4383 constant). skirt_correct=
    True (default) applies the Krogstad 1/k(omega) skirt-reshaping
    correction. Together these close most of the wavelet path's ~13-15%
    amplitude under-shoot vs. clean monochromatic round-trips.

    Args:
        slope_east, slope_north : (T,) earth-referenced slope time series
        water_depth_m           : water depth (m)
        fs_Hz                   : sampling frequency (Hz)
        fmin_Hz, fmax_Hz        : -6 dB corners of the sigmoidal bandpass on
                                  W_eta. Pass None on either to disable that
                                  side of the bandpass.
        transition_octaves      : width of each sigmoid transition in octaves.
        tukey_alpha             : Tukey window cosine fraction applied before
                                  the CWT to suppress edge ringing
        lf_noise_suppress       : if True, apply a noise-aware (Wiener) low-
                                  frequency rolloff. The slope->elevation step
                                  divides by k, so the 1/k^2 factor blows up
                                  white slope-measurement noise at low f (~12x
                                  at 0.07 vs 0.18 Hz), overestimating the sub-
                                  0.1 Hz spectrum and biasing T_E high for noisy
                                  runs. This estimates the white slope noise
                                  floor from the input slope at high f, propa-
                                  gates it through this exact inversion (cached)
                                  to get the elevation noise spectrum, and
                                  spectral-subtracts it via a Wiener gain
                                  G(f)=1-P_noise/P_eta. Phase-preserving; only
                                  suppresses where the slope SNR is low (real
                                  swell, where P_eta>>P_noise, is untouched).
        lf_noise_band_Hz        : (lo, hi) high-frequency band over which the
                                  white slope noise floor is estimated (median).
        lf_noise_oversub        : spectral over-subtraction factor (1 = exact
                                  Wiener; >1 suppresses more aggressively).
        highpass_peak_fraction  : if set, place the highpass corner adaptively
                                  at this fraction of the spectral peak fp
                                  (corner = clip(fraction*fp, corner_floor, fmin_Hz
                                  ... see below)), instead of a fixed corner. fp
                                  is the peak of the slope-implied elevation
                                  spectrum (PsE+PsN)/k^2 -- which IS the omni eta
                                  spectrum, since PsE+PsN = k^2 S_eta -- searched
                                  over [highpass_peak_floor_Hz, 0.40] (camera-
                                  only). This targets the REAL low-f excess on the
                                  averaged data: low-f spatially-coherent
                                  contamination (platform / illumination drift,
                                  NOT white noise -- that is averaged away over the
                                  footprint) amplified by 1/k^2. A fixed steeper
                                  corner over-corrects for long-period seas; tying
                                  it to fp adapts to the sea state (~0.5 works; the
                                  wind-wave spectrum has little real energy below
                                  ~0.5-0.6 fp). Unlike the old max(fmin_Hz, .)
                                  form, the corner is ALLOWED below fmin_Hz so a
                                  long-period swell peak (e.g. 0.10 Hz) is not
                                  clipped; it is only floored at the CWT-safe
                                  highpass_corner_floor_Hz.
        highpass_peak_floor_Hz  : lower edge of the fp peak search (default 0.08).
                                  The 1/k^2 weighting makes (PsE+PsN)/k^2 blow up
                                  at the lowest frequencies (the contamination
                                  band), so without a floor the "peak" would lock
                                  onto that artifact; this forbids picking a peak
                                  below it. Only used when highpass_peak_fraction
                                  is set.
        highpass_corner_floor_Hz: absolute lower clamp on the adaptive corner
                                  (default 0.06). The CWT grid starts at 0.05 Hz,
                                  so a corner at/below it does no high-passing and
                                  the 1/k 1/k^2 blow-up returns; this keeps the
                                  corner just above the grid edge for low-peak
                                  (long-swell) seas. Only used when
                                  highpass_peak_fraction is set.
        window_power_correct    : if True, divide the returned eta by
                                  sqrt(mean(w^2)) of the Tukey window so that
                                  the variance integrated over the FULL record
                                  is unbiased. The internal taper otherwise
                                  amplitude-attenuates the record ends, and a
                                  downstream var()/Welch over the full record
                                  inherits a ~mean(w^2) (~16% at alpha=0.25)
                                  variance deficit. This is a single global
                                  scalar; it makes the integrated variance
                                  unbiased on average but does not restore the
                                  per-sample edge amplitudes -- for per-sample
                                  fidelity, restrict statistics to the
                                  taper-free core via wavelet_eta_valid_slice().

    Returns:
        eta_m : (T,) reconstructed surface elevation (m)
    """
    sE = np.asarray(slope_east, dtype=float).reshape(-1)
    sN = np.asarray(slope_north, dtype=float).reshape(-1)
    if sE.size != sN.size:
        raise ValueError("slope_east and slope_north must have the same length")

    freqs = np.linspace(0.05, 2.0, 80)

    # Linear detrend (not just mean subtraction): the slope time series carries
    # a static viewing-angle tilt and can also drift slowly, both of which
    # would leak into the low-frequency end of the inversion and dominate the
    # reconstructed eta via the 1/k amplification at small wavenumbers.
    sE = detrend(sE, type="linear")
    sN = detrend(sN, type="linear")

    # Adaptive highpass corner: place it just below the spectral peak so the
    # 1/k^2-amplified low-frequency contamination (platform/illumination drift,
    # which survives footprint averaging) is suppressed without cutting real
    # swell for long-period seas. fp from the slope-implied eta spectrum.
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
            # corner tied to fp; allowed below fmin_Hz (preserve long swell) but
            # never below the CWT-edge safety floor (avoid 1/k^2 blow-up).
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

    # Smooth sigmoidal bandpass on the recovered elevation wavelet
    # coefficients. Sigmoid is centered at fmin_Hz/fmax_Hz with a logistic
    # half-width set by transition_octaves (so the same fractional taper
    # applies in dB-per-octave terms across the grid).
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

    # Up-positive sign convention is handled upstream: eta_field_recon's
    # krogstad_eta_coeffs uses W_eta = -i*proj/k (eta_hat = s_hat/(i*k)), fixed
    # in epss commit 852ee94. No sign correction needed here.

    if window_power_correct:
        eta = eta / np.sqrt(np.mean(w ** 2))

    if lf_noise_suppress:
        n = len(eta)
        nps = int(min(512, n))
        # white slope noise floor (robust median over the high-f band)
        fS, PsE = signal.welch(sE, fs_Hz, nperseg=nps)
        _, PsN = signal.welch(sN, fs_Hz, nperseg=nps)
        mb = (fS >= lf_noise_band_Hz[0]) & (fS <= lf_noise_band_Hz[1])
        Ns = 0.5 * (np.median(PsE[mb]) + np.median(PsN[mb]))
        # elevation-noise spectrum per unit slope-noise PSD, through THIS inversion
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
                           highpass_peak_floor_Hz=0.08, highpass_corner_floor_Hz=0.06):
    """Directionally-complete omnidirectional elevation spectrum (S_sx+S_sy)/k^2
    with a squared logistic high-pass (corner fixed at fmin_Hz, or adaptive at
    highpass_peak_fraction * spectral peak when set)."""
    sE = np.asarray(slope_east, dtype=float).reshape(-1)
    sN = np.asarray(slope_north, dtype=float).reshape(-1)
    sE = np.where(np.isfinite(sE), sE, 0.0)
    sN = np.where(np.isfinite(sN), sN, 0.0)
    f, P_sx = signal.welch(sE, fs_Hz, nfft=nfft, nperseg=nperseg)
    _, P_sy = signal.welch(sN, fs_Hz, nfft=nfft, nperseg=nperseg)
    _, k = _lindisp_with_current(2 * np.pi * np.maximum(f, 1e-6), water_depth_m, 0.0)
    k = np.nan_to_num(np.asarray(k, dtype=float), nan=np.inf, posinf=np.inf)
    S = (P_sx + P_sy) / np.maximum(k ** 2, 1e-12)
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


def wavelet_eta_valid_slice(n_samples: int, tukey_alpha: float = 0.25) -> slice:
    """Index slice of the taper-free core of a slope_to_elev_wavelet record.

    slope_to_elev_wavelet applies a Tukey(tukey_alpha) taper to the slope
    before the CWT, so the first/last tukey_alpha/2 of the returned eta are
    amplitude-attenuated. Statistics (variance, Hm0, Welch) over the full
    record inherit that deficit. This returns the slice spanning only the flat
    (w == 1) interior, where the reconstruction is unbiased per-sample, so the
    caller can compute taper-free statistics:

        sl = wavelet_eta_valid_slice(len(eta), tukey_alpha)
        Hm0 = 4 * np.std(eta[sl])

    Args:
        n_samples   : record length (samples)
        tukey_alpha : Tukey alpha used in the inversion (default 0.25)

    Returns:
        slice selecting the central taper-free samples.
    """
    edge = int(np.ceil(0.5 * tukey_alpha * n_samples))
    return slice(edge, n_samples - edge)


# %%

# Short-wave elevation from the 2-D slope FIELD via pyGrad2Surf (Harker &
# O'Leary global least-squares gradient integration). This is the 1-D wavelet
# inversion's complement: slope_to_elev_wavelet recovers the footprint-mean
# (long-wave) elevation from the spatially-averaged slope, while this routine
# recovers the WITHIN-footprint (short-wave) structure that spatial averaging
# discards. Summing the two extends the usable frequency range upward toward
# the footprint cutoff (wavelength ~ footprint size). Mirrors the canonical
# MATLAB path in PSS_source/compute_elevation_fields_from_slope_fields.m.
#
# N. Laxague 2026

def slope_field_to_short_wave_elev(
    slope_east_field: np.ndarray,
    slope_north_field: np.ndarray,
    footprint_m: float,
    at: str = "center",
    g2s_N: int = 3,
) -> np.ndarray:
    """Per-frame gradient integration of a slope field -> short-wave eta(t).

    For each time frame, the (temporal-mean-tilt-removed) east/north slope maps
    are integrated to a surface Z(x, y) by pyGrad2Surf.g2s, and the local
    deviation from the frame-mean surface is extracted. Because a surface plane
    (the instantaneous long-wave tilt) has Z_center == Z_mean, the center-minus-
    mean operation naturally rejects the long-wave tilt and returns only the
    within-footprint short-wave displacement, which is the part the spatial-mean
    (wavelet) inversion cannot see. Add the result to slope_to_elev_wavelet's
    output to obtain the total local elevation.

    The temporal-mean tilt is removed first (static viewing-angle bias and slow
    drift), matching both slope_to_elev_wavelet's detrend and the MATLAB driver.

    Args:
        slope_east_field, slope_north_field : (ny, nx, T) earth-referenced slope
            fields (d eta/dx_east, d eta/dx_north). Non-finite entries are
            zero-filled after tilt removal.
        footprint_m : physical side length of the (square) field footprint (m).
            Sets the integration grid dx = footprint_m / nx; the short-wave
            amplitude scales linearly with this.
        at          : "center" (default) returns the center-pixel series co-
            located with a point sensor; "rms" returns the spatial-RMS of the
            frame's short-wave surface (a footprint-integrated short-wave height).
        g2s_N       : stencil width passed to g2s (default 3).

    Returns:
        short_m : (T,) short-wave elevation time series (m), demeaned.
    """
    from pyGrad2Surf.g2s import g2s

    Sx = np.asarray(slope_east_field, dtype=float)
    Sy = np.asarray(slope_north_field, dtype=float)
    if Sx.shape != Sy.shape or Sx.ndim != 3:
        raise ValueError("slope fields must both be (ny, nx, T)")
    ny, nx, T = Sx.shape

    # remove the temporal-mean tilt per pixel, then zero-fill any gaps
    Sx = Sx - np.nanmean(Sx, axis=2, keepdims=True)
    Sy = Sy - np.nanmean(Sy, axis=2, keepdims=True)
    Sx = np.where(np.isfinite(Sx), Sx, 0.0)
    Sy = np.where(np.isfinite(Sy), Sy, 0.0)

    dx = footprint_m / nx
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    cy, cx = ny // 2, nx // 2

    short = np.empty(T, dtype=float)
    for i in range(T):
        # g2s Python arg order is (x, y, Zx, Zy); Zx is the gradient along x
        # (columns -> east), Zy along y (rows -> north).
        Z = g2s(x, y, Sx[:, :, i], Sy[:, :, i], N=g2s_N)
        Z = np.asarray(Z, dtype=float)
        if at == "rms":
            short[i] = np.sqrt(np.mean((Z - Z.mean()) ** 2))
        else:
            short[i] = Z[cy, cx] - Z.mean()
    return short - short.mean()


def crosscorr_lag(a: np.ndarray, b: np.ndarray, fs_Hz: float,
                  band_Hz: tuple = None, max_lag_s: float = 15.0,
                  return_curve: bool = False):
    """Lag (s) and peak correlation that best align series `a` to series `b`.

    Positive lag means `a` lags `b` (a is delayed): a[t] ~ b[t - lag]. Both
    series are mean-removed, optionally band-pass filtered (zero-phase) before
    correlation, then the cross-correlation peak within +/- max_lag_s is found.
    The peak lag is refined to sub-sample precision by a parabolic fit through
    the three samples straddling the discrete maximum (so the lag is no longer
    quantised to 1/fs), and the reported peak correlation is the fitted vertex.

    Args:
        a, b         : equal-length real time series.
        fs_Hz        : sampling rate (Hz).
        band_Hz      : optional (lo, hi) Butterworth band-pass before
                       correlating; None compares the raw (mean-removed) series.
        max_lag_s    : search window half-width (s).
        return_curve : if True, also return (lags_s, r_curve) -- the normalized
                       cross-correlation over the search window, for plotting.

    Returns:
        (lag_s, peak_r)                       if return_curve is False, or
        (lag_s, peak_r, lags_s, r_curve)      if return_curve is True.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    if band_Hz is not None:
        lo, hi = band_Hz
        bb, aa = signal.butter(4, [lo / (fs_Hz / 2), hi / (fs_Hz / 2)], btype="band")
        a = signal.filtfilt(bb, aa, a - a.mean())
        b = signal.filtfilt(bb, aa, b - b.mean())
    else:
        a = a - a.mean(); b = b - b.mean()
    nan_out = (np.nan, np.nan, np.array([]), np.array([])) if return_curve else (np.nan, np.nan)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return nan_out
    xc = signal.correlate(a, b, "full") / (np.std(a) * np.std(b) * len(a))
    lags = signal.correlation_lags(len(a), len(b), "full")
    keep = np.abs(lags) <= int(max_lag_s * fs_Hz)
    xc, lags = xc[keep], lags[keep]
    i = int(np.argmax(xc))
    lag_samp, peak_r = float(lags[i]), float(xc[i])
    # parabolic sub-sample refinement around the discrete peak
    if 0 < i < len(xc) - 1:
        y0, y1, y2 = xc[i - 1], xc[i], xc[i + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom            # in (-1, 1) samples
            lag_samp = lags[i] + delta
            peak_r = y1 - 0.25 * (y0 - y2) * delta
    lag_s = lag_samp / fs_Hz
    if return_curve:
        return lag_s, peak_r, lags / fs_Hz, xc
    return lag_s, peak_r


# %%

# Given frequency-directional spectra computed directly from wave slope fields
# and inferred from long wave surface elevation and tilt, attempts to resolve
# +/-180 degree directional ambiguity in E-PSS spectra by using the 'direct'
# spectrum as the tiebreaker

# N. Laxague 2025

def trim_EPSS_dirspec(F_EPSS,theta_halfwidth,fmin,fmax,smoothnum=3):

    theta_halfwidth = 90

    f = F_EPSS["frequency"].data
    D = F_EPSS["direction"].data
    Ff_EPSS = F_EPSS.integrate("direction")

    D_EPSS = ((F_EPSS.T / F_EPSS.integrate("direction")).rolling(frequency=smoothnum, center=True).median()).T

    D_EPSS.data = np.nan_to_num(D_EPSS.data,0)

    Sm = np.nanmean(np.sin(np.pi/180*D)*F_EPSS.data)

    Cm = np.nanmean(np.cos(np.pi/180*D)*F_EPSS.data)

    Vm = 180/np.pi*np.atan2(Sm,Cm)

    direction_mega = np.hstack([D-720,D-360,D,D+360,D+720])
    spect_mega = Ff_EPSS.data[:,np.newaxis] * np.hstack([D_EPSS.data,D_EPSS.data,D_EPSS.data,D_EPSS.data,D_EPSS.data])

    spect_mega_copy = spect_mega.copy()

    f_ind_start = np.argmax(f > fmin)-1
    f_ind_end = np.argmax(f > fmax)-1

    dir_diff = np.abs(Vm-direction_mega)
    d_ind_start = np.argmin(dir_diff)

    inds_base = np.arange(0,len(D))-np.int8(len(D)/2)

    lower_ind_halfwidth = np.int8(len(D)/2-theta_halfwidth/5.0)
    upper_ind_halfwidth = np.int8(lower_ind_halfwidth+len(D)/2)

    lower_ind_quarterwidth = np.int8(len(D)/2-theta_halfwidth/5.0/4)
    upper_ind_quarterwidth = np.int8(lower_ind_quarterwidth+theta_halfwidth/5.0/2)

    inds_inside = d_ind_start + inds_base
    inds_outside = inds_inside + len(D)
    inds_lower = direction_mega < direction_mega[inds_inside[lower_ind_halfwidth]]
    inds_higher = direction_mega > direction_mega[inds_inside[upper_ind_halfwidth]]
    inds_exclude = inds_lower | inds_higher

    inds_low_f = np.arange(0,f_ind_start)
    spect_mega_copy[inds_low_f,:] = 0

    spect_mega_copy[f_ind_start,inds_inside] = spect_mega_copy[f_ind_start,inds_inside] + spect_mega_copy[f_ind_start,inds_outside]
    spect_mega_copy[f_ind_start,inds_exclude] = 0

    for i in np.arange(f_ind_start+1,f_ind_end):

        inds_lower = direction_mega < direction_mega[inds_inside[lower_ind_quarterwidth]]
        inds_higher = direction_mega > direction_mega[inds_inside[upper_ind_quarterwidth]]
        inds_exclude = inds_lower | inds_higher

        spect_slice = spect_mega_copy[i,:].copy()
        spect_slice[inds_exclude] = 0
        ind_p = np.argmax(spect_slice)

        inds_inside = ind_p + inds_base
        inds_outside = inds_inside + 72
        inds_lower = direction_mega < direction_mega[inds_inside[lower_ind_halfwidth]]
        inds_higher = direction_mega > direction_mega[inds_inside[upper_ind_halfwidth]]
        inds_exclude = inds_lower | inds_higher

        spect_mega_copy[i,inds_inside] = spect_mega_copy[i,inds_inside] + spect_mega_copy[i,inds_outside]
        spect_mega_copy[i,inds_exclude] = 0

    inds_high_f = np.arange(f_ind_end,len(f))
    spect_mega_copy[inds_high_f,:] = 0

    inds_center = np.arange(0,len(D)) + 2*len(D)
    spect_trimmed = spect_mega_copy[:,inds_center]

    F_EPSS.data = spect_trimmed

    return F_EPSS

# %%

# Given frequency-directional or wavenumber-directional spectrum, computes the
# mean wave direction and spectral spreading width

# Also takes as an input 'theta_halfwidth', a frequency (or wavenumber) dependent
# mask which is designed to mitigate the effect of +/-180 degree ambiguity

# N. Laxague 2025

def compute_mean_wave_direction_and_spreading(F_dirspec,theta_halfwidth,smoothnum=3):

    F_dirspec.data = np.nan_to_num(F_dirspec.data,0)
    spec_energy_density = F_dirspec.data

    wavedir = F_dirspec["direction"].copy()
    dtheta = np.median(np.diff(wavedir.data))

    if 'frequency' in F_dirspec.coords:

        fourier_scale = F_dirspec["frequency"].data
        fourier_scale_name = 'frequency'

    if 'wavenumber' in F_dirspec.coords:

        fourier_scale = F_dirspec["wavenumber"].data
        fourier_scale_name = 'wavenumber'
        spec_energy_density = spec_energy_density*np.reshape(fourier_scale,(1,len(wavedir)))

    D_array = ((F_dirspec.T / F_dirspec.integrate("direction")).rolling(frequency=smoothnum, center=True).median()).T
    D_array.data = np.nan_to_num(D_array.data,0)

    Dtheta = D_array.integrate("frequency")
    ind_p = np.argmax(Dtheta.data)

    theta_super = np.concatenate((wavedir-360,wavedir,wavedir+360),axis=0)
    theta_rel = theta_super - wavedir.data[ind_p]
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
    theta_nought_array_mat = np.tile(np.reshape(theta_nought_array,(len(theta_nought_array),1)),(1,72))
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

# %%
# N-up polar variant of the directional-spectrum plotter. The upstream
# ewdm.plots.plot_directional_spectrum uses CCW-from-east convention (math);
# this paper uses CW-from-North (compass) so the local copy swaps sin/cos to
# put North at the top of the polar plot.
# Adapted from D. Pelaez-Zapata's EWDM plotting helpers.

# plot directional wave spectrum {{{
def _smooth(E, ws=(5, 2)):
    """Applies a simple circular smoothing to 2D array.

    Args:
        E (ndarray): Input function.
        ws (tuple): Window size. For example, for a directional spectrum
            size E(360,64) and ws=(10,2), the filter acts averaging
            10 directions and 2 frequencies.

    Returns:
        ndarray: Function smoothed.
    """

    # define window
    nd, nf = ws
    if nf == nd:
        frqwin = signal.hamming(nf)
    else:
        frqwin = np.ones(nf)

    dirwin = signal.hamming(nd)
    window = frqwin[None,:] * dirwin[:,None]
    window = window / window.sum()

    # permorm convolution and return output
    return signal.convolve2d(E, window, mode='same', boundary='wrap')

def _get_axes(
        ax=None, rmin=0.1, rmax=0.5, rstep=0.1, angle=-135,
        color="0.8", as_period=False
    ):
    """Draw polar grid on specific axes.

    Args:
        ax (matplotlib.axes, optional): Axes object to draw the grid on.
            If None, a new figure and axes are created.
        rmin (float, optional): Minimum radius for the grid. Defaults to 0.1.
        rmax (float, optional): Maximum radius for the grid. Defaults to 0.5.
        rstep (float, optional): Step size for the radius. Defaults to 0.1.
        angle (int, optional): Angle for the radius labels. Defaults to -135.
        color (str, optional): Color of the grid lines and labels. Defaults to "0.8".
        as_period (bool, optional): If True, labels are formatted as wave periods.
            Defaults to False.

    Returns:
        tuple: A tuple containing the figure and axes objects if a new figure is created. Otherwise, returns None.
    """

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5,5))

    ax.set_aspect("equal")

    for radii in np.arange(rmin, rmax+rstep, rstep):
        circle = plt.Circle(
            (0,0), radii, color=color, alpha=0.5,
            linestyle="dashed", fill=False, zorder=2
        )
        ax.add_artist(circle)
        if radii <= rmax:
            if as_period:
                radii_label = f"{1/radii:.1f}"
            else:
                radii_label = f"{radii:.2f}"
            ax.text(
                radii*np.cos(np.radians(angle)),
                radii*np.sin(np.radians(angle)),
                radii_label, fontsize="small",
                ha="center", va="center", zorder=3
            )

    ax.axhline(0, color=color, ls="dashed", alpha=0.5)
    ax.axvline(0, color=color, ls="dashed", alpha=0.5)
    ax.plot([0,1], [0,1], "--", c=color, alpha=0.5, transform=ax.transAxes)
    ax.plot([0,1], [1,0], "--", c=color, alpha=0.5, transform=ax.transAxes)

    _label_args = {"fontsize": "small", "ha": "center", "va": "center"}
    ax.text(0.50, 0.95, "N", transform=ax.transAxes, **_label_args)
    ax.text(0.95, 0.50, "E", transform=ax.transAxes, **_label_args)
    ax.text(0.50, 0.05, "S", transform=ax.transAxes, **_label_args)
    ax.text(0.05, 0.50, "W", transform=ax.transAxes, **_label_args)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if as_period:
        ax.set_ylabel("Wave period [s]")
    else:
        ax.set_ylabel("$f$ [Hz]")

    ax.set_xlim([-rmax+rstep, rmax-rstep])
    ax.set_ylim([-rmax+rstep, rmax-rstep])

    try:
        return fig, ax
    except NameError:
        pass


def _get_cmap(colors=None, N=256):
    """Return colormap for a given color list.

    Args:
        colors (list): List of colours.
        N (int): Number of colours

    Returns:
        Colormap object. If not colors passed, it returns viridis with white bottom.

    """

    if colors is None:
        colors = [(1, 1, 1), *plt.cm.viridis(np.linspace(0, 1, 8))]
    return mcolors.LinearSegmentedColormap.from_list('cmap', colors, N=N)

def _add_cbar(
        pc, ax, cax=None, style="outside", ticks=None, orientation="horizontal",
        label="$\\log_{10} E \\; \\mathrm{[m^2 Hz^{-1} deg^{-1}]}$"
    ):
    """Return a colorbar object"""

    if cax is None:
        if style == "inside":
            ticks = mticker.LinearLocator(ticks)
            orientation = "horizontal"
            cax = ax.inset_axes(
                [0.07, 0.92, 0.3, 0.035], transform=ax.transAxes
            )
        if style == "outside":
            ticks = mticker.AutoLocator()
            orientation = "vertical"
            cax = ax.inset_axes(
                [1.02, 0.0, 0.04, 1.0], transform=ax.transAxes
            )
    else:
        pass

    return plt.colorbar(
        pc, cax=cax, orientation=orientation, ticks=ticks, label=label
    )

def _add_wind_info(ax, wspd, wdir, color="k", wind_sea_radius=True):
    """Add some info relative to wind speed"""

    fwind = 9.8 / (2*np.pi * wspd)
    uwnd, vwnd = (
        fwind * np.sin(np.radians(wdir)), fwind * np.cos(np.radians(wdir))
    )

    ax.arrow(
        uwnd, vwnd, -uwnd, -vwnd, color=color, head_width=0.01,
        length_includes_head=True
    )

    if wind_sea_radius:
        circle = plt.Circle(
            (0,0), fwind, color=color, alpha=0.5,
            linestyle="dashed", fill=False
        )
        ax.add_artist(circle)

def _add_current_info(ax, curspd, curdir, color="red", wind_sea_radius=True):
    """Add some info relative to wind speed"""

    fcur = 9.8 / (2*np.pi * curspd)
    ucur, vcur = (
        fcur * np.sin(np.radians(curdir)), fcur * np.cos(np.radians(curdir))
    )

    ax.arrow(
        0, 0, ucur, vcur, color=color, head_width=0.01,
        length_includes_head=True
    )

    if wind_sea_radius:
        circle = plt.Circle(
            (0,0), fcur, color=color, alpha=0.5,
            linestyle="dashed", fill=False
        )
        ax.add_artist(circle)


def plot_directional_spectrum(
        da: Union[xr.DataArray, np.ndarray],
        frqs: Union[str, np.ndarray] = "frequency",
        dirs: Union[str, np.ndarray] = "direction",
        ax = None,
        smooth = None,
        cmap = None,
        levels: int = 30,
        vmin: Union[float, int] = None,
        vmax: Union[float, int] = None,
        contours: Union[float, int] = None,
        colorbar: bool = False,
        wspd: Union[float, int] =None,
        wdir: Union[float, int] =None,
        wind_sea_radius: Union[float, int] = None,
        curspd: Union[float, int] = None,
        curdir: Union[float, int] = None,
        cbar_kw={},
        axes_kw={}
    ) -> None:
    """
    Make a simple plot of a directional wave spectrum.

    Args:
        da: Directional spectrum data.
        frqs: Frequency label name or numpy array.
        dirs: Direction label name or numpy array.
        ax (optional): Matplotlib axis object.
        smooth (tuple): Smoothing factor for visualisation. Defaults to None.
        cmap (optional): Colormap for the plot. Defaults to None.
        levels (int): Number of contour levels. Defaults to 30. If None, pseudo-color plot is made.
        vmin (int or float): Minimum value for colormap.
        vmax (int or float]): Maximum value for colormap.
        contours (int or float): Specific contour levels to plot on top.
        colorbar (bool): Whether to display colorbar. Defaults to False.
        wspd (int or float): Wind speed value to draw arrow. Defaults to None.
        wdir (int or float): Wind direction value in cartesian convention.
        wind_sea_radius (int or float):Whether to display wind sea separation radius.
        curspd (int or float): Current speed value.
        curdir (int or float): Current direction value in cartesian convention.
        cbar_kw (dict): Additional arguments for colorbar.
        axes_kw (dict): Additional arguments for plot axes.

    Returns:
        fig, ax
    """
    # get axis if not given
    if ax is None:
        fig, ax = _get_axes(**axes_kw)
    else:
        _get_axes(ax=ax, **axes_kw)

    # wrap spectra around the circle
    if da[dirs][0] != da[dirs][-1]:
        padded = da.pad({dirs: (1,0)}, mode="wrap")
    else:
        padded = da.copy()

    # calculate cartesian fx,fy to mimic polar coordinates
    _deg2rad = np.pi/180
    frqx = padded[frqs].data[:,None]*np.sin(padded[dirs].data[None,:]*_deg2rad)
    frqy = padded[frqs].data[:,None]*np.cos(padded[dirs].data[None,:]*_deg2rad)

    # smooth spectra
    if smooth:
        smoothed = xr.apply_ufunc(lambda x: _smooth(x, ws=smooth), padded)
    else:
        smoothed = padded.copy()

    # get colormap
    if cmap is None:
        cmap = _get_cmap()
    else:
        if isinstance(cmap, list):
            cmap = _get_cmap(cmap)

    # plot pcolormesh if levels is None, otherwise go for contourf
    if levels is None:
        pc = ax.pcolormesh(
            frqx, frqy, smoothed[1:,1:], cmap=cmap,
            shading="flat",rasterized='true', vmin=vmin, vmax=vmax
        )
    else:
        pc = ax.contourf(
            frqx, frqy, smoothed, levels=levels, cmap=cmap,
            vmin=vmin, vmax=vmax
        )

    # plot contour lines if contours is not None
    if contours is not None:
        cts = ax.contour(
            frqx, frqy, smoothed, levels=contours, colors="k"
        )

    # add colobar if True
    if colorbar:
        cbar = _add_cbar(pc, ax, **cbar_kw)

    # add current data
    if (curspd is None) or (curspd <= 0):
        curspd = 100
    if curdir is not None:
        fmax = axes_kw["rmax"]
        curspd = 9.8/(2*np.pi*fmax)*1.25
        _add_current_info(
            ax, curspd=curspd, curdir=curdir+180, color="red",
            wind_sea_radius=wind_sea_radius
        )


    # add wind data
    if (wspd is None) or (wspd <= 0):
        wspd = 10
    if wdir is not None:
        fmax = axes_kw["rmax"]
        wspd = 9.8/(2*np.pi*fmax)*1.25
        _add_wind_info(
            ax, wspd=wspd, wdir=wdir, color="k",
            wind_sea_radius=wind_sea_radius
        )

    try:
        return fig, ax, pc
    except NameError:
        pc

# }}}

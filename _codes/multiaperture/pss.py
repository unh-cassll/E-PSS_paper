"""E-PSS project glue (not portable).

Builds the combined camera elevation field from polarimetric slope fields and
provides the 3-D-FFT sign anchor. Depends on pyGrad2Surf and the paper's
slope_to_elev_wavelet.
"""
import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
from pyGrad2Surf.g2s import g2s
from subroutines.utils import slope_to_elev_wavelet

L_FOV = 2.915                       # E-PSS imaged-patch side length [m]
GRAV = 9.81


def _highpass_1d(x, fs, fc, width_oct=0.5):
    """Smooth (log-tanh) temporal high-pass of a 1-D series above corner fc [Hz]."""
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    with np.errstate(divide='ignore'):
        lr = np.log2(np.maximum(f, 1e-9) / fc) / width_oct
    return np.fft.irfft(X * np.clip(0.5 * (1.0 + np.tanh(lr)), 0.0, 1.0), n=len(x))


def build_eta_field(SxF, SyF, depth, fs, L=L_FOV, krog_disc=8, depiston_n=None):
    """Combined camera elevation field eta(y,x,t): Krogstad wavelet long wave
    (slope averaged over a centered circular disc of diameter krog_disc pixels)
    + per-frame g2s short-wave field. SxF, SyF are (ny,nx,T) earth-referenced
    slope fields. Returns (eta, dx). A larger krog_disc averages more short-wave
    energy out of the long-wave reconstruction.

    depiston_n (if set): also return a de-pistoned solve field eta_solve for
    multiaperture_spectra(solve_eta=...). The spatially-uniform eta_long is kept
    where it is real long-wave signal (low f -> low k) but its leakage above the
    frequency of a wave whose wavelength is depiston_n frame-sizes (wavenumber
    k_n = 2*pi/(depiston_n*L), finite-depth dispersion) is removed from eta_solve
    so it cannot bias the FOV-scale |k| solve low. Larger depiston_n = longer
    cut wavelength = lower corner = more aggressive de-piston. Returns
    (eta, dx, eta_solve)."""
    ny, nx, T = SxF.shape
    dx = L / nx
    xg = np.arange(nx) * dx
    yg = np.arange(ny) * dx
    # centered circular disc of diameter krog_disc px for the long-wave slope avg
    yy, xx = np.ogrid[:ny, :nx]
    disc = (yy - (ny - 1)/2.0)**2 + (xx - (nx - 1)/2.0)**2 <= (krog_disc/2.0)**2
    eta_long = slope_to_elev_wavelet(SxF[disc].mean(0), SyF[disc].mean(0), depth, fs)
    Sx0 = SxF - SxF.mean(axis=2, keepdims=True)
    Sy0 = SyF - SyF.mean(axis=2, keepdims=True)
    Z = np.empty((ny, nx, T))
    for i in range(T):
        Z[:, :, i] = g2s(xg, yg, Sx0[:, :, i], Sy0[:, :, i])
    Z -= Z.mean(axis=2, keepdims=True)
    eta = Z + eta_long[None, None, :]
    if depiston_n is None:
        return eta, dx
    k_n = 2 * np.pi / (depiston_n * L)
    f_n = np.sqrt(GRAV * k_n * np.tanh(k_n * depth)) / (2 * np.pi)
    eta_solve = eta - _highpass_1d(eta_long, fs, f_n)[None, None, :]
    return eta, dx, eta_solve


def sftheta_sign_anchor(ref, run_ind, rmin=0.15, f_max=1.0):
    """Per-frequency sign-resolved direction anchor from the full-resolution
    3-D-FFT directional spectrum S_f_theta (ASIT2019 ..._empirical_gain dataset),
    which resolves the propagation sign and breaks the array's 180-deg ambiguity.

    Returns (freqs_Hz, dir_cwN_deg, R): energy-weighted mean direction (CW-from-N)
    and resultant length R per frequency, for multiaperture_spectra(sign_anchor=).
    R<rmin bins get NaN direction (unresolved, e.g. long swell).
    """
    Sft = np.nan_to_num(np.asarray(ref['S_f_theta'][run_ind]))      # (ntheta, nf)
    fR = np.asarray(ref['f_Hz'][:])
    thR = np.degrees(np.asarray(ref['theta_rad'][:]))
    a = np.radians(thR)
    C = (Sft * np.cos(a)[:, None]).sum(0)
    S = (Sft * np.sin(a)[:, None]).sum(0)
    tot = Sft.sum(0)
    R = np.where(tot > 0, np.hypot(C, S) / np.where(tot > 0, tot, 1), 0.0)
    dirs = np.degrees(np.arctan2(S, C)) % 360.0
    # restrict to the physical wave band (above ~1 Hz the slope field is sub-FOV
    # noise/aliasing carrying a spurious camera-fixed direction)
    dirs = np.where((R >= rmin) & (fR <= f_max), dirs, np.nan)
    return fR, dirs, R

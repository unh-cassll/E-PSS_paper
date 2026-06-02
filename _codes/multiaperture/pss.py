"""E-PSS project glue (not portable).

Builds the combined camera elevation field from polarimetric slope fields and
reads the ADCP directional reference. Depends on pyGrad2Surf and the paper's
slope_to_elev_wavelet.
"""
import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
from pyGrad2Surf.g2s import g2s
from subroutines.utils import slope_to_elev_wavelet
from .core import circ_stats

L_FOV = 2.915                       # E-PSS imaged-patch side length [m]


def build_eta_field(SxF, SyF, depth, fs, L=L_FOV, aperture=8):
    """Combined camera elevation field eta(y,x,t): center-aperture Krogstad
    wavelet long wave + per-frame g2s short-wave field. SxF, SyF are (ny,nx,T)
    earth-referenced slope fields. Returns (eta, dx)."""
    ny, nx, T = SxF.shape
    dx = L / nx
    xg = np.arange(nx) * dx
    yg = np.arange(ny) * dx
    a = aperture
    i0 = (ny - a) // 2
    j0 = (nx - a) // 2
    eta_long = slope_to_elev_wavelet(
        SxF[i0:i0+a, j0:j0+a].mean((0, 1)),
        SyF[i0:i0+a, j0:j0+a].mean((0, 1)), depth, fs)
    Sx0 = SxF - SxF.mean(axis=2, keepdims=True)
    Sy0 = SyF - SyF.mean(axis=2, keepdims=True)
    Z = np.empty((ny, nx, T))
    for i in range(T):
        Z[:, :, i] = g2s(xg, yg, Sx0[:, :, i], Sy0[:, :, i])
    Z -= Z.mean(axis=2, keepdims=True)
    return Z + eta_long[None, None, :], dx


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


def adcp_dir_spread(env, run_ind):
    """ADCP mean direction & spread vs frequency (CW-from-N) for `run_ind`, from
    an open ASIT2019 supporting-observations netCDF dataset. Returns
    (f, theta, F(theta,f), mean_dir, spread, omni S(f))."""
    fa = np.asarray(env['f_Hz_ADCP'][:])
    tha = np.degrees(np.asarray(env['theta_rad'][:]))
    Fa = np.asarray(env['F_f_theta_m2_Hz_rad_ADCP'][:, :, run_ind])
    Fa = np.where(np.isfinite(Fa), Fa, 0.0)
    mean = np.full(len(fa), np.nan); spread = np.full(len(fa), np.nan)
    for j in range(len(fa)):
        if Fa[:, j].sum() > 0:
            mean[j], spread[j] = circ_stats(Fa[:, j], tha)
    Sf = np.trapz(Fa, np.radians(tha), axis=0)
    return fa, tha, Fa, mean, spread, Sf

"""E-PSS project glue (not portable).

Builds the combined camera elevation field from polarimetric slope fields and
provides the 3-D-FFT sign anchor. Depends on pyGrad2Surf and eta_field_recon.
"""
import warnings
import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
from scipy import signal
from scipy.signal.windows import tukey
from pyGrad2Surf.g2s import g2s
from subroutines.utils import L_FOV_M
from eta_field_recon.wavelet_core import (lindisp_with_current, aperture_transfer_gain,
                                          _cwt as _ewdm_cwt, _inverse_cwt as _ewdm_icwt)

L_FOV = L_FOV_M                     # E-PSS imaged-patch side length [m]
GRAV = 9.81


def _highpass_1d(x, fs, fc, width_oct=0.5):
    """Smooth (log-tanh) temporal high-pass of a 1-D series above corner fc [Hz]."""
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    with np.errstate(divide='ignore'):
        lr = np.log2(np.maximum(f, 1e-9) / fc) / width_oct
    return np.fft.irfft(X * np.clip(0.5 * (1.0 + np.tanh(lr)), 0.0, 1.0), n=len(x))


def _disc_mask(ny, nx, d_px):
    """Centered circular disc mask of diameter d_px pixels."""
    yy, xx = np.ogrid[:ny, :nx]
    return (yy - (ny - 1)/2.0)**2 + (xx - (nx - 1)/2.0)**2 <= (d_px/2.0)**2


def _direct_complete_amplitude(sE, sN, depth, fs, diam_m, jinc=True,
                               hp_fmin=0.08, hp_width_oct=0.25, tukey_alpha=0.25):
    """rfft-grid directionally-complete long-wave amplitude A(f)=sqrt(|Sx|^2+|Sy|^2)/k
    (Phillips 1977), jinc aperture-corrected and logistic high-passed. Returns
    (A, Sx, Sy, T); Sx, Sy are the windowed disc-mean slope rffts. Shared by the
    fourier and wavelet long-wave projections."""
    sE = signal.detrend(np.asarray(sE, float))
    sN = signal.detrend(np.asarray(sN, float))
    T = sE.size
    win = tukey(T, tukey_alpha)
    wn = np.sqrt(np.mean(win ** 2))
    f = np.fft.rfftfreq(T, 1.0 / fs)
    _, k = lindisp_with_current(2 * np.pi * f, depth, 0.0)
    k = np.asarray(k, float)
    Sx = np.fft.rfft(sE * win) / wn
    Sy = np.fft.rfft(sN * win) / wn
    m = np.sqrt(np.abs(Sx) ** 2 + np.abs(Sy) ** 2) + 1e-30
    with np.errstate(divide='ignore', invalid='ignore'):
        A = np.where(np.isfinite(m / k), m / k, 0.0)
    if jinc:
        with warnings.catch_warnings():                  # null bands expected
            warnings.simplefilter('ignore', UserWarning)
            g = aperture_transfer_gain(f, k, diam_m, shape='circular', min_transfer=0.3)
        A = A * np.where(np.isfinite(g), g, 0.0)
    with np.errstate(divide='ignore'):
        lr = (np.log2(np.maximum(f, 1e-12)) - np.log2(hp_fmin)) / hp_width_oct
    A = A * np.clip(1.0 / (1.0 + np.exp(-lr)), 0.0, 1.0)
    return A, Sx, Sy, T


def wavelet_slope_projection(SxF, SyF, depth, fs, L=L_FOV, slope_aperture=None, jinc=True,
                             hp_fmin=0.08, hp_width_oct=0.25, tukey_alpha=0.25):
    """Long-wave eta(t): wavelet (CWT) signed slope projection for the phase, with the
    same directionally-complete direct amplitude as fourier_slope_projection.

    Disc-mean slopes -> Morlet CWT. Per (f, t): direction cos=|Wsx|/m, sin=(|Wsy|/m)*
    sign(Re(Wsy conj Wsx)); elevation coeffs Weta = +1j*(cos*Wsx + sin*Wsy)/k(f),
    logistic high-passed; eta_krog = Re(iCWT). The amplitude is then imposed from the
    direct slope spectrum and the wavelet carries only the phase. The directional
    estimator of Krogstad, Magnusson & Donelan (2006), reduced to the per-(f,t)
    projection (no skirt correction, no aperture blend)."""
    ny, nx = SxF.shape[:2]
    d_px = nx if slope_aperture is None else min(slope_aperture, nx)
    disc = _disc_mask(ny, nx, d_px)
    sE, sN = SxF[disc].mean(0), SyF[disc].mean(0)
    A, _, _, T = _direct_complete_amplitude(sE, sN, depth, fs, L * d_px / nx,
                                            jinc, hp_fmin, hp_width_oct, tukey_alpha)
    fcwt = np.linspace(0.05, 2.0, 80)
    win = tukey(T, tukey_alpha)
    Wsx = _ewdm_cwt(signal.detrend(sE) * win, freqs=fcwt, fs=fs).values
    Wsy = _ewdm_cwt(signal.detrend(sN) * win, freqs=fcwt, fs=fs).values
    _, kc = lindisp_with_current(2 * np.pi * fcwt, depth, 0.0)
    kc = np.asarray(kc, float)
    m = np.sqrt(np.abs(Wsx) ** 2 + np.abs(Wsy) ** 2) + 1e-30
    rel = np.sign(np.real(Wsy * np.conj(Wsx)))
    rel = np.where(rel == 0, 1.0, rel)
    with np.errstate(divide='ignore', invalid='ignore'):
        Weta = 1j * ((np.abs(Wsx) / m) * Wsx + (np.abs(Wsy) / m) * rel * Wsy) / kc[:, None]
    Weta = np.where(np.isfinite(Weta), Weta, 0.0)
    bp = 1.0 / (1.0 + np.exp(-(np.log2(fcwt) - np.log2(hp_fmin)) / hp_width_oct))
    eta_krog = np.real(_ewdm_icwt(Weta * bp[:, None], freqs=fcwt, fs=fs, per_scale=True))
    phase = np.angle(np.fft.rfft(eta_krog - eta_krog.mean()))
    eta = np.fft.irfft(A * np.exp(1j * phase), n=T)
    return eta - eta.mean()


def fourier_slope_projection(SxF, SyF, depth, fs, L=L_FOV, slope_aperture=None, jinc=True,
                             hp_fmin=0.08, hp_width_oct=0.25, tukey_alpha=0.25):
    """Long-wave eta(t) by per-frequency signed slope projection (default long wave).

    Disc-mean slope rffts. Per frequency: direction cos=|Sx|/m, sin=(|Sy|/m)*
    sign(Re(Sy conj Sx)) (180-deg ambiguity from the channels' relative phase); the
    projection carries only the phase, the directionally-complete direct amplitude the
    magnitude: eta = irfft(A * exp(i*angle(+1j*(cos*Sx + sin*Sy)))). The Fourier-
    amplitude form of the directional estimator of Krogstad, Magnusson & Donelan (2006)
    (their wavelet method puts wavelet amplitudes in place of these Fourier amplitudes)."""
    ny, nx = SxF.shape[:2]
    d_px = nx if slope_aperture is None else min(slope_aperture, nx)
    disc = _disc_mask(ny, nx, d_px)
    A, Sx, Sy, T = _direct_complete_amplitude(SxF[disc].mean(0), SyF[disc].mean(0),
                                              depth, fs, L * d_px / nx, jinc,
                                              hp_fmin, hp_width_oct, tukey_alpha)
    m = np.sqrt(np.abs(Sx) ** 2 + np.abs(Sy) ** 2) + 1e-30
    rel = np.sign(np.real(Sy * np.conj(Sx)))
    rel = np.where(rel == 0, 1.0, rel)
    carrier = 1j * ((np.abs(Sx) / m) * Sx + (np.abs(Sy) / m) * rel * Sy)
    eta = np.fft.irfft(A * np.exp(1j * np.angle(carrier)), n=T)
    return eta - eta.mean()


def anchored_freq_recolor(eta_long, Z, fs, freqs, fc=0.55, band=(0.5, 0.6),
                          nperseg=1024):
    """Per-frequency recolor ratio mapping the EWDM omni spectrum to the
    anchored brick-wall splice omni, evaluated on the EWDM frequency grid `freqs`.

    Computes R = mean(F_long/F_short) over `band`, then:
        F_anch(f) = F_long(f)   for f <= fc
                    R*F_short(f) for f >  fc
    Returns ratio_on_freqs = F_anch / (F_long + F_short) and R.
    Ratio is a scalar per frequency; D(f,theta) is unchanged."""
    nseg = int(min(nperseg, len(eta_long)))
    fw, Pl = signal.welch(eta_long, fs, nperseg=nseg)
    _, Ps = signal.welch(Z.reshape(-1, Z.shape[-1]), fs, nperseg=nseg, axis=1)
    Ps = Ps.mean(0)
    m = (fw >= band[0]) & (fw <= band[1]) & (Ps > 0)
    R = float(np.nanmean(Pl[m] / Ps[m])) if m.any() else 1.0
    Fsum = Pl + Ps
    Fanch = np.where(fw <= fc, Pl, R * Ps)
    ratio_w = np.divide(Fanch, Fsum, out=np.ones_like(Fsum), where=Fsum > 0)
    # extrapolate: ratio -> 1 below long-wave band, -> R above Welch range
    return np.interp(freqs, fw, ratio_w, left=1.0, right=R), R


def build_eta_field(SxF, SyF, depth, fs, L=L_FOV, slope_aperture=None, depiston_n=None,
                    return_components=False, longwave_method='fourier'):
    """Combined camera elevation field eta(y,x,t) [m]: long wave (Fourier
    slope-projection by default, or wavelet) + per-frame g2s short-wave field.

    Args:
        SxF, SyF: (ny, nx, T) earth-referenced slope fields.
        depth: water depth [m].
        fs: sampling rate [Hz].
        L: FOV side length [m].
        slope_aperture: disc diameter [px] for the long-wave FOV-mean tilt
            (None = full frame).
        depiston_n: if set, also return eta_solve with long-wave piston high-passed
            above f(k_n = 2*pi/(depiston_n*L)) for the |k| solve. Returns
            (eta, dx, eta_solve).
        return_components: if True, append (eta_long, Z) to the return tuple.
        longwave_method: 'fourier' (default, fourier_slope_projection) or 'wavelet'
            (wavelet_slope_projection); both share the direct amplitude and reproduce
            the same S(f)/Hm0 to <1%, differing only in the phase source.

    Returns:
        (eta, dx) by default; extended by eta_solve and/or (eta_long, Z) as above."""
    ny, nx, T = SxF.shape
    dx = L / nx
    xg = np.arange(nx) * dx
    yg = np.arange(ny) * dx
    proj = wavelet_slope_projection if longwave_method == 'wavelet' else fourier_slope_projection
    eta_long = proj(SxF, SyF, depth, fs, L, slope_aperture)
    Sx0 = SxF - SxF.mean(axis=2, keepdims=True)
    Sy0 = SyF - SyF.mean(axis=2, keepdims=True)
    Z = np.empty((ny, nx, T))
    for i in range(T):
        Z[:, :, i] = g2s(xg, yg, Sx0[:, :, i], Sy0[:, :, i])
    Z -= Z.mean(axis=2, keepdims=True)
    eta = Z + eta_long[None, None, :]
    extra = (eta_long, Z) if return_components else ()
    if depiston_n is None:
        return (eta, dx) + extra
    k_n = 2 * np.pi / (depiston_n * L)
    f_n = np.sqrt(GRAV * k_n * np.tanh(k_n * depth)) / (2 * np.pi)
    eta_solve = eta - _highpass_1d(eta_long, fs, f_n)[None, None, :]
    return (eta, dx, eta_solve) + extra


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
    # mask low-R and above f_max (sub-FOV aliasing)
    dirs = np.where((R >= rmin) & (fR <= f_max), dirs, np.nan)
    return fR, dirs, R

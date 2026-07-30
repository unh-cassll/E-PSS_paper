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

# Rotation from the reduced field's array axes onto the compass. The reduced
# fields are stored on the CAMERA IMAGE grid (`_earth_components` rotates the
# polarimetric components into (east, north) but leaves the array indices
# alone), so index-differentiating operators work in the image frame. The row
# axis lies along the camera look azimuth and the column axis 90 deg clockwise:
# a proper rotation of CAMERA_AZ from (east, north) onto (column, row). This is
# the physical mounting azimuth, not a fitted parameter.
#
# `_curvature_trace` needs it too and does not apply it: the slope-gradient
# tensor trace is rotation-invariant only when components and axes rotate
# together, which they do not here. Components rotated by phi but differentiated
# along the image axes give cos(phi) div(s) - sin(phi) curl(s); cos(190 deg) =
# -0.98, so it returns nearly MINUS the Laplacian and its sign reference is pi
# out.
CAMERA_AZ_DEG = 190.0           # ASIT deployment look azimuth [deg true]


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


def _image_axis_components(SxF, SyF, cam_az_deg=CAMERA_AZ_DEG):
    """Earth-referenced slope components resolved onto the camera image axes.

    The stored components are (east, north) but the array indices run along the
    camera row and column axes, so a derivative taken with respect to an index
    only means what it says once the components are expressed in that same
    frame. Returns (s_column, s_row)."""
    col = np.radians(cam_az_deg + 90.0)
    row = np.radians(cam_az_deg)
    SxF = np.asarray(SxF, float)
    SyF = np.asarray(SyF, float)
    return (SxF * np.sin(col) + SyF * np.cos(col),
            SxF * np.sin(row) + SyF * np.cos(row))


def _curvature_trace(SxF, SyF, L, cam_az_deg=CAMERA_AZ_DEG):
    """Frame-mean surface curvature trace lap(t) = div(s) [1/m].

    Least-squares slope of each component against its own coordinate over the
    whole frame (all pixels, unlike an edge difference). For a wave of
    wavenumber k the surface obeys lap(eta) = -k^2 eta, so this trace is exactly
    anti-phase with the elevation -- a sign-unambiguous phase reference, from
    the in-frame slope variation that the disc-mean time series discards.

    The components are resolved onto the image axes first. A divergence is
    rotation-invariant only when the components and the axes rotate TOGETHER;
    this expression pairs a component subscript with an axis subscript, so it
    needs both frames to agree. Differentiating the EARTH components along the
    IMAGE axes returns cos(phi) div(s) - sin(phi) curl(s) instead, and with
    phi = CAMERA_AZ the leading factor is -0.98 -- very nearly MINUS the
    Laplacian, which puts the sign reference pi out and inverts the piston."""
    ny, nx = SxF.shape[:2]
    dx = L / nx
    s_col, s_row = _image_axis_components(SxF, SyF, cam_az_deg)
    ex = (np.arange(nx) - (nx - 1) / 2.0) * dx
    ey = (np.arange(ny) - (ny - 1) / 2.0) * dx
    wx = ex / (nx * np.sum(ex ** 2))              # LS ramp projector, per row
    wy = ey / (ny * np.sum(ey ** 2))              # ... per column
    return (np.einsum('j,ijt->t', wx, s_col) * nx
            + np.einsum('i,ijt->t', wy, s_row) * ny)


def _curvature_phase(SxF, SyF, Sx, Sy, L, T, fs, tukey_alpha=0.25,
                     smooth_hz=0.05, cam_az_deg=CAMERA_AZ_DEG):
    """Per-frequency elevation phase: the slope projection, with its sign fixed
    by the in-frame surface curvature.

    The projection phase arg{i (cos th_f Sx + sin th_f Sy)} is well determined
    -- the disc-mean slope carries most of the long-wave signal -- but it is
    ambiguous by 180 deg, because cos(th_f) = |Sx|/m is non-negative by
    construction and so th_f can never point west. Only the SIGN is in doubt,
    and only the sign is taken from the curvature.

    grad^2 eta = -k^2 eta puts the curvature exactly out of phase with the
    elevation, so arg{-F[lap]} settles that sign. Taking the curvature phase
    wholesale instead would import its noise: curvature scales as k^2 eta, and
    across a 2.9 m frame at 0.1 Hz it is second-order small, so a per-bin
    curvature phase wanders and drags the swell into the wrong hemisphere.
    Here the curvature only votes.

    The vote is amplitude-weighted and smoothed over `smooth_hz` in frequency
    before its sign is taken, so bins with no curvature signal inherit the
    decision of neighbors that have one. That is legitimate because the sign is
    not random per bin: it flips only where the true direction crosses the
    north-south axis, which a veering sea does a few times at most.

    `cam_az_deg` reaches `_curvature_trace`, which needs it to resolve the
    stored earth components onto the image axes. Pass 0 for synthetic fields
    laid out in the earth frame."""
    win = tukey(T, tukey_alpha)
    Lap = np.fft.rfft(signal.detrend(
        _curvature_trace(SxF, SyF, L, cam_az_deg)) * win)

    m = np.sqrt(np.abs(Sx) ** 2 + np.abs(Sy) ** 2) + 1e-30
    rel = np.sign(np.real(Sy * np.conj(Sx)))
    rel = np.where(rel == 0, 1.0, rel)
    phi = np.angle(1j * ((np.abs(Sx) / m) * Sx + (np.abs(Sy) / m) * rel * Sy))

    vote = np.abs(Lap) * np.exp(1j * (np.angle(-Lap) - phi))
    nb = max(3, int(round(smooth_hz * T / fs)) | 1)     # odd window
    kern = np.ones(nb) / nb
    smooth = np.convolve(vote, kern, mode='same')
    return phi + np.pi * (np.real(smooth) < 0.0)


def wavelet_slope_projection(SxF, SyF, depth, fs, L=L_FOV, slope_aperture=None, jinc=True,
                             hp_fmin=0.08, hp_width_oct=0.25, tukey_alpha=0.25,
                             phase_source='curvature',
                             cam_az_deg=CAMERA_AZ_DEG):
    """Long-wave eta(t): wavelet (CWT) signed slope projection for the phase, with the
    same directionally-complete direct amplitude as fourier_slope_projection.

    Carries the same 180-deg ambiguity as the Fourier projection -- per (f, t) the
    cosine is |Wsx|/m, again non-negative -- so phase_source defaults to
    'curvature' here too; 'projection' restores the wavelet phase.

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
    A, Sx, Sy, T = _direct_complete_amplitude(sE, sN, depth, fs, L * d_px / nx,
                                              jinc, hp_fmin, hp_width_oct,
                                              tukey_alpha)
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
    if phase_source == 'projection':
        phase = np.angle(np.fft.rfft(eta_krog - eta_krog.mean()))
    else:
        phase = _curvature_phase(SxF, SyF, Sx, Sy, L, T, fs, tukey_alpha,
                                 cam_az_deg=cam_az_deg)
    eta = np.fft.irfft(A * np.exp(1j * phase), n=T)
    return eta - eta.mean()


def fourier_slope_projection(SxF, SyF, depth, fs, L=L_FOV, slope_aperture=None, jinc=True,
                             hp_fmin=0.08, hp_width_oct=0.25, tukey_alpha=0.25,
                             phase_source='curvature',
                             cam_az_deg=CAMERA_AZ_DEG):
    """Long-wave eta(t): directionally-complete direct amplitude with a phase from
    the in-frame surface curvature (default long wave).

    Disc-mean slope rffts set the magnitude, A(f) = sqrt(|Sx|^2+|Sy|^2)/k, the
    Fourier-amplitude form of the directional estimator of Krogstad, Magnusson &
    Donelan (2006). The phase comes from the frame-mean curvature trace,
    arg{-F[lap]}, which is sign-unambiguous (see `_curvature_phase`).

    phase_source='projection' restores the original per-frequency signed slope
    projection, eta = irfft(A * exp(i*angle(+1j*(cos*Sx + sin*Sy)))) with
    cos=|Sx|/m and sin=(|Sy|/m)*sign(Re(Sy conj Sx)). That form is ambiguous by
    180 deg -- it is retained only to reproduce published results."""
    ny, nx = SxF.shape[:2]
    d_px = nx if slope_aperture is None else min(slope_aperture, nx)
    disc = _disc_mask(ny, nx, d_px)
    A, Sx, Sy, T = _direct_complete_amplitude(SxF[disc].mean(0), SyF[disc].mean(0),
                                              depth, fs, L * d_px / nx, jinc,
                                              hp_fmin, hp_width_oct, tukey_alpha)
    if phase_source == 'projection':
        m = np.sqrt(np.abs(Sx) ** 2 + np.abs(Sy) ** 2) + 1e-30
        rel = np.sign(np.real(Sy * np.conj(Sx)))
        rel = np.where(rel == 0, 1.0, rel)
        carrier = 1j * ((np.abs(Sx) / m) * Sx + (np.abs(Sy) / m) * rel * Sy)
        phase = np.angle(carrier)
    else:
        phase = _curvature_phase(SxF, SyF, Sx, Sy, L, T, fs, tukey_alpha,
                                 cam_az_deg=cam_az_deg)
    eta = np.fft.irfft(A * np.exp(1j * phase), n=T)
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
                    return_components=False, longwave_method='fourier',
                    cam_az_deg=CAMERA_AZ_DEG):
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
    eta_long = proj(SxF, SyF, depth, fs, L, slope_aperture,
                    cam_az_deg=cam_az_deg)
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

"""Portable multi-aperture directional wave spectra from a virtual-staff
elevation array on a small field of view.

Each wavenumber octave is resolved with a matched aperture (coarse/sparse staffs
for low k, tight clusters for high k), each trusted within its anti-alias band
k < pi/baseline_max, then stitched. Depends only on numpy, xarray, ewdm.

multiaperture_spectra: recommended estimator (von-Mises kernel deposit onto
measured k/nu/f, energy-conserving).
"""
import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
import xarray as xr
import ewdm
from ewdm.wavelets import cwt, Morlet
from ewdm.density import (estimate_radial_distribution,
                          estimate_directional_distribution)

GRAV = 9.81


def _nu_of_k(k, depth):
    """Inverse phase speed nu = k/omega along the linear dispersion relation."""
    return np.sqrt(k / (GRAV * np.tanh(np.clip(k * depth, 1e-9, 50.0))))


# --- dispersion & circular statistics ----------------------------------------
def k_dispersion(f, depth):
    """Linear gravity-wave wavenumber k(f) [rad/m] at finite depth."""
    w = 2 * np.pi * np.asarray(f, float)
    k = w**2 / GRAV
    for _ in range(100):
        k = w**2 / (GRAV * np.tanh(np.clip(k * depth, 1e-9, 50.0)))
    return k


def circ_stats(F, theta_deg, axis=-1):
    """Energy-weighted circular mean direction and spread (deg) of F over theta."""
    a = np.radians(theta_deg)
    C = np.sum(F * np.cos(a), axis=axis)
    S = np.sum(F * np.sin(a), axis=axis)
    tot = np.sum(F, axis=axis)
    tot = np.where(tot == 0, np.nan, tot)
    R = np.hypot(C, S) / tot
    mean = np.degrees(np.arctan2(S, C)) % 360.0
    spread = np.degrees(np.sqrt(np.clip(2.0 * (1.0 - R), 0, None)))
    return mean, spread


def _math_angle_to_cw_from_N(theta_en_rad, flip=False):
    deg = (90.0 - np.degrees(theta_en_rad)) % 360.0
    if flip:
        deg = (deg + 180.0) % 360.0
    return ((deg + 180.0) % 360.0) - 180.0


def _wrap180(x):
    return ((x + 180.0) % 360.0) - 180.0


# --- array geometry -----------------------------------------------------------
def _poisson_disk_pixels(rng, i0, j0, e, n_staff, dx, cxp, cyp, min_sep,
                         tries=20000):
    """Poisson-disk sampling of integer-pixel staff positions with pairwise
    separation >= min_sep (m); threshold relaxed x0.85 if n_staff cannot fit.
    Returns (ii, jj) int arrays."""
    target = float(min_sep)
    while True:
        ii, jj, xs, ys = [], [], [], []
        for _ in range(tries):
            if len(ii) >= n_staff:
                break
            i = int(rng.integers(i0, i0 + e + 1))
            j = int(rng.integers(j0, j0 + e + 1))
            x, y = (cxp - j) * dx, (cyp - i) * dx
            t2 = target * target
            if all((x - xx)**2 + (y - yy)**2 >= t2 for xx, yy in zip(xs, ys)):
                ii.append(i); jj.append(j); xs.append(x); ys.append(y)
        if len(ii) >= n_staff:
            return np.asarray(ii), np.asarray(jj)
        target *= 0.85
        if target < 0.5 * dx:        # below half a pixel: accept any draw
            target = 0.0


def seed_aperture(ny, nx, dx, extent_px, n_staff, seed, min_sep=0.0):
    """Random virtual staffs in a centered square window of side extent_px pixels.
    Column axis West-positive: East=(cxp-j)dx, North=(cyp-i)dx. min_sep>0 (m)
    enables Poisson-disk spacing; min_sep<=0 is a plain uniform draw.
    Returns (ii, jj, px, py, b_max)."""
    rng = np.random.default_rng(seed)
    e = int(min(extent_px, min(ny, nx) - 1))
    i0 = (ny - e) // 2
    j0 = (nx - e) // 2
    cxp, cyp = (nx - 1) / 2, (ny - 1) / 2
    if min_sep and min_sep > 0:
        ii, jj = _poisson_disk_pixels(rng, i0, j0, e, n_staff, dx, cxp, cyp, min_sep)
    else:
        ii = rng.integers(i0, i0 + e + 1, n_staff)
        jj = rng.integers(j0, j0 + e + 1, n_staff)
    px = (cxp - jj) * dx
    py = (cyp - ii) * dx
    b = np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :])
    return ii, jj, px, py, float(b.max())


def aperture_band(b_max, lo_frac=0.30, hi_frac=1.0):
    """Trusted wavenumber band [lo_frac/b_max, hi_frac*pi/b_max] for an aperture
    of longest baseline b_max. hi = anti-alias ceiling, lo = minimum resolvable
    phase over the longest baseline."""
    return lo_frac / b_max, hi_frac * np.pi / b_max


def default_apertures():
    """Coarse->tight apertures (name, extent_px). On a 32-pixel / ~2.9 m grid the
    anti-alias ceilings k_hi=pi/b_max are ~1.2, 2.5, 4.9, 8.6 rad/m."""
    return [('A0', 28), ('A1', 14), ('A2', 7), ('A3', 4)]


def default_grids(dx, fmin=0.06, fmax=3.5, nfreq=52,
                  kmin=0.08, numin=0.01, numax=2.0, nrad=64):
    """Log-spaced frequency, wavenumber (to the pixel Nyquist pi/dx) and inverse
    phase speed grids."""
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), nfreq)
    kmax = np.pi / dx
    k_grid = 2.0**np.linspace(np.log2(kmin), np.log2(kmax), nrad)
    nu_grid = 2.0**np.linspace(np.log2(numin), np.log2(numax), nrad)
    return freqs, k_grid, nu_grid


# --- wavelet wavevectors & sign ----------------------------------------------
def cwt_stack(staff_series, freqs, fs):
    """CWT each staff time series -> (n_staff, n_freq, T) complex array."""
    mother = Morlet(6.0)
    return np.stack([cwt(s.astype(float), freqs=freqs, fs=fs, mother=mother).values
                     for s in staff_series])


def solve_wavevectors(W, px, py):
    """Per-(freq,time) least-squares wavevector from cross-staff phase
    differences. W: (n_staff, n_freq, T). Returns (kx, ky, resid)."""
    nstaff, nf, T = W.shape
    pairs = [(m, n) for m in range(nstaff) for n in range(m + 1, nstaff)]
    A = np.array([[px[n] - px[m], py[n] - py[m]] for m, n in pairs])
    kx = np.empty((nf, T)); ky = np.empty((nf, T)); resid = np.empty((nf, T))
    for fi in range(nf):
        dphi = np.empty((len(pairs), T))
        for q, (m, n) in enumerate(pairs):
            dphi[q] = np.angle(W[n, fi] * np.conj(W[m, fi]))
        sol, *_ = np.linalg.lstsq(A, dphi, rcond=None)
        kx[fi], ky[fi] = sol[0], sol[1]
        resid[fi] = np.sqrt(np.mean((A @ sol - dphi)**2, axis=0))
    return kx, ky, resid


def lh_direction(We, Wsx, Wsy, flip=False):
    """Longuet-Higgins first-moment mean direction (unambiguous 0-360) from
    co-located heave (We) and slope (Wsx, Wsy = grad eta) wavelet coefficients.
    Resolves the array's 180-deg sign ambiguity. Returns CW-from-N deg."""
    cross_x = np.mean(We * np.conj(Wsx), axis=(0, 2))
    cross_y = np.mean(We * np.conj(Wsy), axis=(0, 2))
    Cee = np.mean(np.abs(We)**2, axis=(0, 2))
    Cxx = np.mean(np.abs(Wsx)**2, axis=(0, 2))
    Cyy = np.mean(np.abs(Wsy)**2, axis=(0, 2))
    den = np.sqrt(np.clip(Cee * (Cxx + Cyy), 1e-30, None))
    return _math_angle_to_cw_from_N(np.arctan2(np.imag(cross_y) / den,
                                               np.imag(cross_x) / den), flip=flip)


# --- recommended estimator ----------------------------------------------------
def multiaperture_spectra(eta, dx, freqs, k_grid, nu_grid, depth, fs,
                          apertures=None, n_staff=16, seed=20, dd=4.0, kappa=36.0,
                          lo_frac=0.30, lo_frac_broad=0.05, flip_dir=False,
                          bandwidth=0.04, bandwidth_floor=1e-3, nu_lo_broad=0.04,
                          min_sep=0.0, min_sep_broad=None,
                          sign_anchor=None, sign_anchor_rmin=0.15, sign_coh_min=0.6):
    """Composite multi-aperture elevation directional spectra on the array-
    measured (WDM) wavenumber, binned with ewdm.density and stitched over each
    aperture's trusted band. Power pre-scaled so int S(f) df = var(eta).

    Returns omni S(f), F(k), Q(nu) and polar F(f,theta), F(k,theta), Q(nu,theta);
    theta is deg CW-from-N.
    """
    ny, nx, T = eta.shape
    if apertures is None:
        apertures = default_apertures()
    var_eta = eta.var(axis=2).mean()
    gy, gx = np.gradient(eta, dx, axis=(0, 1))
    time = np.arange(T) / fs
    coords = {'frequency': freqs, 'time': time}

    ap = []
    for ai, (name, ext) in enumerate(apertures):
        # widest aperture optionally uses a larger minimum staff separation
        ms = min_sep_broad if (ai == 0 and min_sep_broad is not None) else min_sep
        ii, jj, px, py, bmax = seed_aperture(ny, nx, dx, ext, n_staff, seed + ai,
                                             min_sep=ms)
        es = np.stack([eta[i, j, :] for i, j in zip(ii, jj)])
        es = es - es.mean(1, keepdims=True)
        We = cwt_stack(es, freqs, fs)
        kx, ky, resid = solve_wavevectors(We, px, py)
        klo, khi = aperture_band(bmax, lo_frac=(lo_frac_broad if ai == 0 else lo_frac))
        ap.append(dict(name=name, bmax=bmax, klo=klo, khi=khi, ii=ii, jj=jj,
                       kmag=np.hypot(kx, ky), resid=resid, We=We,
                       dir=_math_angle_to_cw_from_N(np.arctan2(ky, kx), flip=flip_dir),
                       P=(np.abs(We)**2).mean(0)))

    # 180-deg sign from the coarsest aperture's LH first moment (reuse its staffs)
    a0 = ap[0]
    sxs = np.stack([gx[i, j, :] for i, j in zip(a0['ii'], a0['jj'])])
    sys = np.stack([gy[i, j, :] for i, j in zip(a0['ii'], a0['jj'])])
    Wsx = cwt_stack(sxs - sxs.mean(1, keepdims=True), freqs, fs)
    Wsy = cwt_stack(sys - sys.mean(1, keepdims=True), freqs, fs)
    lh = lh_direction(a0['We'], Wsx, Wsy, flip=flip_dir)

    # Sign reference theta(f): (1) per-frequency axis from the energy-weighted
    # measured wavevector (double-angle mean, smoothed); (2) unwrap the axis
    # across frequency into a signed direction; (3) set the global sign from the
    # anchor (fallback: energy-weighted LH hemisphere).
    P0 = ap[0]['P']
    dir0 = ap[0]['dir']
    wf = P0.mean(1)
    ca = (P0 * np.cos(np.radians(2 * dir0))).sum(1)
    sa = (P0 * np.sin(np.radians(2 * dir0))).sum(1)
    R2 = np.hypot(ca, sa) / np.maximum(P0.sum(1), 1e-30)      # axis coherence per f
    sm = np.array([0.25, 0.5, 0.25])
    axis = np.degrees(np.arctan2(np.convolve(sa, sm, mode='same'),
                                 np.convolve(ca, sm, mode='same'))) / 2.0

    def _branch(a, prev):
        c1, c2 = _wrap180(a), _wrap180(a + 180.0)
        return c1 if abs(_wrap180(c1 - prev)) <= abs(_wrap180(c2 - prev)) else c2
    # seed the unwrap at the spectral peak and propagate outward: above the peak
    # always unwrap; below the peak only coherent frequencies (R2 >= sign_coh_min)
    # update the running direction, incoherent ones hold it
    pk = int(np.argmax(wf))
    theta = np.empty(len(freqs))
    cur = axis[pk]
    theta[pk] = cur
    for i in range(pk + 1, len(freqs)):                  # above peak: always unwrap
        cur = _branch(axis[i], cur)
        theta[i] = cur
    cur = theta[pk]
    for i in range(pk - 1, -1, -1):                      # below peak: coherence-gated
        if R2[i] >= sign_coh_min:
            cur = _branch(axis[i], cur)
        theta[i] = cur

    def _ewmean(deg, w):
        return np.degrees(np.arctan2((w * np.sin(np.radians(deg))).sum(),
                                     (w * np.cos(np.radians(deg))).sum()))
    if sign_anchor is not None and (np.isfinite(np.asarray(sign_anchor[1], float))
                                    & (np.asarray(sign_anchor[2], float) >= sign_anchor_rmin)).any():
        af, ad, aR = (np.asarray(x, float) for x in sign_anchor)
        ok = np.isfinite(ad) & (aR >= sign_anchor_rmin)
        jc = np.argmin(np.abs(af[ok, None] - freqs[None, :]), axis=1)
        w = aR[ok] * wf[jc]
        flip = np.cos(np.radians(_ewmean(theta[jc], w) - _ewmean(ad[ok], w))) < 0
    else:
        flip = np.cos(np.radians(_ewmean(theta, wf) - _ewmean(lh, wf))) < 0
    ref = _wrap180(theta + 180.0) if flip else theta

    def _fold(dir_cwN):
        mis = np.cos(np.radians(dir_cwN - ref[:, None])) < 0
        return np.where(mis, _wrap180(dir_cwN + 180.0), dir_cwN)
    for d in ap:
        d['dirf'] = _fold(d['dir'])

    # one variance calibration (full-FOV power) so int S(f) df = var_eta
    cal = var_eta / np.trapz(ap[0]['P'].mean(1), freqs)

    bins_dir = np.arange(-180.0, 180.0, dd)
    Fk = np.zeros(len(k_grid)); ck = np.zeros(len(k_grid))
    Qn = np.zeros(len(nu_grid)); cn = np.zeros(len(nu_grid))
    Fkd = np.zeros((len(k_grid), len(bins_dir)))
    Qnd = np.zeros((len(nu_grid), len(bins_dir)))
    for d in ap:
        power = xr.DataArray(d['P'] * cal, dims=['frequency', 'time'], coords=coords)
        theta = xr.DataArray(d['dirf'], dims=['frequency', 'time'], coords=coords)
        kk = xr.DataArray(d['kmag'], dims=['frequency', 'time'], coords=coords)
        nu = xr.DataArray(d['kmag'] / (2 * np.pi * freqs[:, None]),
                          dims=['frequency', 'time'], coords=coords)
        ok = estimate_radial_distribution(power, theta, kk, 'wavenumber', k_grid,
                                          dd, kappa, bandwidth, bandwidth_floor)
        on = estimate_radial_distribution(power, theta, nu, 'nu', nu_grid,
                                          dd, kappa, bandwidth, bandwidth_floor)
        ink = (k_grid >= d['klo']) & (k_grid <= d['khi'])
        # nu band from the aperture's k-band along dispersion; broadest aperture
        # extended down to nu_lo_broad to show the measured-nu tail
        nlo, nhi = _nu_of_k(d['khi'], depth), _nu_of_k(d['klo'], depth)
        lo = min(nlo, nhi)
        if d is ap[0]:
            lo = min(lo, nu_lo_broad)
        inn = (nu_grid >= lo) & (nu_grid <= max(nlo, nhi))
        Fk[ink] += ok['wavenumber_spectrum'].values[ink]; ck[ink] += 1
        Fkd[ink] += ok['directional_spectrum'].values[ink]
        Qn[inn] += on['nu_spectrum'].values[inn]; cn[inn] += 1
        Qnd[inn] += on['directional_spectrum'].values[inn]
    Fk = np.where(ck > 0, Fk / np.maximum(ck, 1), np.nan); Fkd /= np.maximum(ck, 1)[:, None]
    Qn = np.where(cn > 0, Qn / np.maximum(cn, 1), np.nan); Qnd /= np.maximum(cn, 1)[:, None]

    # S(f) and F(f,theta) from the coarsest aperture (EWDM frequency-direction path)
    power0 = xr.DataArray(ap[0]['P'] * cal, dims=['frequency', 'time'], coords=coords)
    theta0 = xr.DataArray(ap[0]['dirf'], dims=['frequency', 'time'], coords=coords)
    of = estimate_directional_distribution(power0, theta0, dd, kappa)
    Sf = of['frequency_spectrum'].values
    Fft = of['directional_spectrum'].values
    thg = of['direction'].values
    thbar, sigma = circ_stats(Fft, thg)
    return dict(freqs=freqs, theta=thg, k=k_grid, nu=nu_grid, var_eta=var_eta,
                Sf=Sf, Fft=Fft, Fk=Fk, Qn=Qn, Fkd=Fkd, Qnd=Qnd,
                thbar=thbar, sigma=sigma, lh=lh, sign_ref=ref, apertures=ap,
                ap_names=[d['name'] for d in ap],
                ap_bands=[(d['klo'], d['khi']) for d in ap])

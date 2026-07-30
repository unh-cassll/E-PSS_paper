"""
Traditional cross-spectral-matrix (CSM) directional wave spectra for E-PSS.

    mlm, imlm, emep, bdm             fixed-wavenumber CSM directional estimators
    mlm_kscan                        dispersion-free wavevector scan (Capon)
    traditional_directional_spectra  driver: elevation field -> D(theta|f), D(theta|k)

The wavelet directional method estimates one wavevector per (frequency, time)
sample and pools them, so two wave trains crossing at the same frequency are
seen only through their compromise -- a first-order mixture bias. These
estimators instead invert the SECOND-order statistics: the cross-spectral
matrix C(f) of a virtual-gauge array, in which interfering trains decorrelate.
MLM is the Capon (1969) maximum-likelihood spatial spectrum, the elevation-only
case of the Isobe, Kondo & Horikawa (1984) EMLM; IMLM deconvolves the Capon
smearing kernel from that estimate by iterated forward modeling (Pawka 1983;
Oltman-Shay & Guza 1984) and is the default; EMEP is the extended maximum
entropy principle with harmonic order chosen by AIC (Hashimoto, Nagai & Asai
1994); BDM is the Bayesian Directional Method with a circular second-difference
smoothness prior (Hashimoto & Kobune 1988; Hashimoto 1997).

The f-domain estimators pin the wavenumber to the linear-dispersion shell k(f),
so they are band-limited by the array baseline. The wavevector scan does not:
it steers over an independent (k, theta) grid and localizes k by the fit, which
makes it a check on dispersion rather than an assumption of it.

CONVENTION: the steering vector is exp(i k (x cos th + y sin th)), so C must be
in the math/FFT cross-phase sign. The Morlet wavelet cross-spectral matrix
C[m, n] = <W[m] conj(W[n])> uses the opposite convention and is CONJUGATED
before inversion -- getting this backwards lands every lobe 180 degrees off.
Internally theta is in radians CCW from East; the driver takes and returns the
compass grid (degrees CW from North) used everywhere else in this project.

N. Laxague 2026
"""

import numpy as np

from scipy.optimize import least_squares

# Internal imports; underscore-aliased to avoid leaking via `import *`.
from ewdm.multiaperture import (seed_aperture as _seed_aperture,
                                _bilinear_stack as _bilinear_stack,
                                cwt_stack as _cwt_stack,
                                aperture_band as _aperture_band,
                                default_apertures as _default_apertures,
                                k_dispersion as _k_dispersion)

__all__ = [
    'mlm', 'imlm', 'emep', 'bdm', 'mlm_kscan',
    'traditional_directional_spectra',
]

# %%

# Steering, regularization and normalization helpers


def _steering(k, pos, theta):
    """Plane-wave steering matrix exp(i k (x cos th + y sin th)), (M, L)."""
    return np.exp(1j * k * (pos[:, 0:1] * np.cos(theta)[None, :]
                            + pos[:, 1:2] * np.sin(theta)[None, :]))


def _regularize(C, eps=1e-3):
    """Diagonal-load a cross-spectral matrix for a stable inverse."""
    tr = np.trace(C).real / C.shape[0]
    return C + eps * tr * np.eye(C.shape[0])


def _normalize(D, theta):
    """Clip negatives and scale so the directional integral is one."""
    D = np.clip(D, 0.0, None)
    area = D.sum(-1) * (theta[1] - theta[0])
    return D / area if area > 0 else np.full_like(D, 1.0 / (2 * np.pi))


def _coherence_pairs(C):
    """Off-diagonal complex coherences and their (m, n) index pairs."""
    M = C.shape[0]
    d = np.sqrt(np.clip(np.real(np.diag(C)), 1e-30, None))
    pairs = [(m, n) for m in range(M) for n in range(m + 1, M)]
    phi = np.array([C[m, n] / (d[m] * d[n]) for m, n in pairs])
    return phi, pairs


# %%

# Fixed-wavenumber directional estimators. Each takes one frequency's Hermitian
# cross-spectral matrix and returns D(theta) with unit integral over radians.


def mlm(C, k, pos, theta, eps=1e-3):
    """Maximum likelihood (Capon) directional distribution at one frequency."""
    H = _steering(k, pos, theta)
    Cinv = np.linalg.inv(_regularize(C, eps))
    denom = np.real(np.einsum('ml,mn,nl->l', np.conj(H), Cinv, H))
    return _normalize(1.0 / np.maximum(denom, 1e-12), theta)


def _forward_csm(D, k, pos, theta):
    """Cross-spectral matrix implied by a directional distribution,
    C[m,n] = int H_m H_n* D(theta) dtheta. The forward half of the IMLM loop."""
    H = _steering(k, pos, theta)
    dth = theta[1] - theta[0]
    return (H * (D * dth)[None, :]) @ H.conj().T


def imlm(C, k, pos, theta, eps=1e-3, n_iter=3, relax=1.0):
    """Iterative maximum likelihood directional distribution at one frequency.

    MLM returns the true distribution convolved with a steering-dependent
    smearing kernel, so a Capon lobe is always broader than the sea that made
    it. Pushing an estimate back through the forward relation and re-applying
    MLM reproduces that kernel, which makes the residual against the original
    MLM estimate a correction that can be applied repeatedly (Pawka 1983;
    Oltman-Shay & Guza 1984).

    Gains flatten after roughly three iterations and sidelobes grow beyond
    that, so `n_iter` is capped low rather than run to convergence. `eps` must
    match the load used on the observed matrix: the synthetic matrix is
    noise-free, and an unmatched load means the two passes see different
    kernels and the correction stops being one."""
    D_mlm = mlm(C, k, pos, theta, eps)
    D = D_mlm.copy()
    for _ in range(n_iter):
        D_hat = mlm(_forward_csm(D, k, pos, theta), k, pos, theta, eps)
        D = _normalize(D + relax * (D_mlm - D_hat), theta)
    return D


def _emep_model_phi(coef, k, dpos, theta):
    """Modeled coherences for EMEP harmonic coefficients `coef`."""
    N = len(coef) // 2
    a, b = coef[:N], coef[N:]
    n = np.arange(1, N + 1)[:, None]
    g = (a[:, None] * np.cos(n * theta) + b[:, None] * np.sin(n * theta)).sum(0)
    dth = theta[1] - theta[0]
    D = np.exp(g - g.max())
    D /= D.sum() * dth
    ph = k * (dpos[:, 0:1] * np.cos(theta)[None, :]
              + dpos[:, 1:2] * np.sin(theta)[None, :])
    return (np.exp(1j * ph) * (D * dth)[None, :]).sum(1)


def emep(C, k, pos, theta, nmax=3, ridge=0.02, return_misfit=False):
    """Extended maximum entropy distribution, harmonic order chosen by AIC.

    A weak ridge scaled by 1/(k*b_max)^2 damps over-concentration on a poorly
    conditioned (small k*b) array and is negligible once the baseline resolves
    the shape. `nmax` caps the harmonic order; the AIC order flaps between
    adjacent frequencies above 3, chopping spurious lobes into the spectrum.

    `return_misfit` also returns the normalized residual between modeled and
    observed coherences. Steering at the wrong wavenumber cannot reproduce the
    observed phases for any D, so that residual is the wavevector scan's
    likelihood in k."""
    phi, pairs = _coherence_pairs(C)
    dpos = np.array([pos[m] - pos[n] for m, n in pairs])
    obs = np.concatenate([phi.real, phi.imag])
    obs_norm = float(np.sum(obs ** 2)) + 1e-30
    bmax = np.sqrt((dpos ** 2).sum(1)).max()
    kb = max(k * bmax, 0.3)
    neff = int(np.clip(round(kb) + 1, 1, nmax))
    reg = ridge / kb ** 2

    def resid(coef):
        mod = _emep_model_phi(coef, k, dpos, theta)
        return np.concatenate([mod.real - obs[:len(mod)],
                               mod.imag - obs[len(mod):],
                               np.sqrt(reg) * coef])

    def data_rss(coef):
        mod = _emep_model_phi(coef, k, dpos, theta)
        return np.sum((np.concatenate([mod.real, mod.imag]) - obs) ** 2)

    best, best_aic, best_rss, prev = None, np.inf, np.inf, np.zeros(0)
    for N in range(1, neff + 1):
        c0 = np.zeros(2 * N)
        npr = prev.size // 2
        if npr:                              # warm-start from the previous order
            c0[:npr] = prev[:npr]
            c0[N:N + npr] = prev[npr:]
        try:
            sol = least_squares(resid, c0, method='lm', max_nfev=2000)
        except Exception:
            continue
        rss = data_rss(sol.x)
        aic = len(obs) * np.log(rss / len(obs) + 1e-30) + 2 * (2 * N)
        if aic < best_aic:
            best_aic, best, prev, best_rss = aic, sol.x, sol.x, rss
    if best is None:
        D = np.full_like(theta, 1.0 / (2 * np.pi))
        return (D, 1.0) if return_misfit else D
    N = len(best) // 2
    a, b = best[:N], best[N:]
    n = np.arange(1, N + 1)[:, None]
    g = (a[:, None] * np.cos(n * theta) + b[:, None] * np.sin(n * theta)).sum(0)
    D = _normalize(np.exp(g - g.max()), theta)
    return (D, best_rss / obs_norm) if return_misfit else D


def _circ_second_diff(L):
    """Circular second-difference operator (L, L)."""
    L2 = -2.0 * np.eye(L)
    idx = np.arange(L)
    L2[idx, (idx + 1) % L] = 1.0
    L2[idx, (idx - 1) % L] = 1.0
    return L2


def bdm(C, k, pos, theta, u_ladder=(0.3, 1.0, 3.0), min_smooth_steps=2,
        return_misfit=False):
    """Bayesian Directional Method (Hashimoto & Kobune 1988; Hashimoto 1997).

    Fits x = ln D(theta) on the full direction grid to the same observed
    coherences as `emep`, under a circular second-difference smoothness prior
    u^2 ||L2 x||^2. The hyperparameter is chosen from `u_ladder` by AIC with
    effective degrees of freedom tr[J (J'J + u^2 L2'L2)^-1 J'], a practical
    stand-in for the full ABIC determinant. Warm-started from MLM.

    `min_smooth_steps` bumps the AIC pick that many rungs toward stronger
    smoothing: AIC systematically under-smooths this prior, and the extra
    smoothing costs nothing in fit while removing row-to-row raggedness."""
    phi, pairs = _coherence_pairs(C)
    dpos = np.array([pos[m] - pos[n] for m, n in pairs])
    obs = np.concatenate([phi.real, phi.imag])
    obs_norm = float(np.sum(obs ** 2)) + 1e-30
    E = np.exp(1j * k * (dpos[:, 0:1] * np.cos(theta)[None, :]
                         + dpos[:, 1:2] * np.sin(theta)[None, :]))
    L2 = _circ_second_diff(len(theta))
    n_obs = len(obs)

    def model_and_jac(x):
        w = np.exp(x - x.max())
        s = w.sum()
        mod = (E * (w / s)[None, :]).sum(1)
        return mod, (E - mod[:, None]) * (w / s)[None, :]

    def resid(x, u):
        mod, _ = model_and_jac(x)
        return np.concatenate([mod.real - obs[:len(mod)],
                               mod.imag - obs[len(mod):], u * (L2 @ x)])

    def jac(x, u):
        _, Jc = model_and_jac(x)
        return np.vstack([Jc.real, Jc.imag, u * L2])

    x0 = np.log(np.clip(mlm(C, k, pos, theta), 1e-6, None))
    x0 -= x0.mean()

    rungs, best_idx, best_aic = [], None, np.inf
    for ui, u in enumerate(u_ladder):
        try:
            sol = least_squares(resid, x0, jac=jac, args=(u,), method='trf',
                                max_nfev=200)
        except Exception:
            continue
        mod, Jc = model_and_jac(sol.x)
        rss = float(np.sum((np.concatenate([mod.real, mod.imag]) - obs) ** 2))
        J = np.vstack([Jc.real, Jc.imag])
        A = J.T @ J + u ** 2 * (L2.T @ L2)
        try:
            edof = float(np.trace(J @ np.linalg.solve(A, J.T)))
        except np.linalg.LinAlgError:
            continue
        aic = n_obs * np.log(rss / n_obs + 1e-30) + 2.0 * edof
        rungs.append((ui, aic, sol.x, rss))
        if aic < best_aic:
            best_aic, best_idx = aic, ui
        x0 = sol.x                           # warm-start the next rung
    if not rungs:
        D = np.full_like(theta, 1.0 / (2 * np.pi))
        return (D, 1.0) if return_misfit else D
    floor_idx = min(best_idx + int(min_smooth_steps), len(u_ladder) - 1)
    avail = [t for t in rungs if t[0] >= floor_idx]
    pick = min(avail, key=lambda t: t[1]) if avail else \
        max(rungs, key=lambda t: t[0])
    D = _normalize(np.exp(pick[2] - pick[2].max()), theta)
    return (D, pick[3] / obs_norm) if return_misfit else D


# %%

# Dispersion-free wavevector scan. The fixed-wavenumber estimators above pin k
# to the linear-dispersion shell; these steer over an independent (k, theta)
# grid, so a wave riding a current or a bound harmonic shows up off-shell.


def _kscan_steering(pos, k_scan, theta):
    """Steering tensor e(k, theta) = exp(i k (x cos th + y sin th)), position-
    only, so it is precomputed once per aperture and reused across frequency."""
    proj = (np.outer(np.cos(theta), pos[:, 0])
            + np.outer(np.sin(theta), pos[:, 1]))
    return np.exp(1j * np.asarray(k_scan)[:, None, None] * proj[None, :, :])


def mlm_kscan(C, steer, load=1e-3):
    """Capon wavevector-scan response P(k, theta) = 1 / Re(e^H C^-1 e)."""
    nk, nth, M = steer.shape
    E = steer.reshape(nk * nth, M)
    quad = np.real(np.einsum('pm,mn,pn->p', np.conj(E),
                             np.linalg.inv(_regularize(C, load)), E,
                             optimize=True))
    return (1.0 / np.maximum(quad, 1e-12)).reshape(nk, nth)


def _fit_kscan(C, k_sub, pos, theta, fit_fn, n_ridge=2):
    """Misfit-localized wavevector response for a fitting estimator.

    Fits D(theta) at candidate wavenumbers on a short ridge around the Capon
    peak and weights each candidate by exp(-misfit/min_misfit): steering at the
    wrong k cannot reproduce the observed phases for any D, so the fit residual
    localizes k without assuming dispersion. A full scan over every candidate
    would cost one nonlinear fit per wavenumber per frequency."""
    P = np.zeros((len(k_sub), len(theta)))
    steer = _kscan_steering(pos, k_sub, theta)
    j0 = int(np.argmax(mlm_kscan(C, steer).sum(1)))
    idx = np.unique(np.clip(np.arange(j0 - n_ridge, j0 + n_ridge + 1),
                            0, len(k_sub) - 1))
    Ds, mis = [], []
    for j in idx:
        D, r = fit_fn(C, float(k_sub[j]), pos, theta, return_misfit=True)
        Ds.append(D)
        mis.append(r)
    mis = np.asarray(mis)
    w = np.exp(-(mis - mis.min()) / max(mis.min(), 1e-6))
    for j, D, wj in zip(idx, Ds, w):
        P[j] = D * wj
    return P


# %%

# Driver: virtual-gauge arrays seeded into the camera elevation field, one
# cross-spectral matrix per aperture per frequency, stitched coarse-to-tight.
#
# N. Laxague 2026


def _wavelet_csm(W, fi):
    """Cross-spectral matrix at one frequency, conjugated into the steering
    convention (the Morlet cross-phase is opposite to the FFT's)."""
    Wf = W[:, fi, :]
    return np.conj((Wf @ Wf.conj().T) / Wf.shape[1])


def _fold_to_reference(D, theta_deg, ref_deg):
    """Roll a directional distribution 180 deg when its circular mean opposes
    `ref_deg`. An elevation-only array measures phase differences, which are
    unchanged by reversing the propagation sense, so every one of these
    estimators is ambiguous by 180 deg at each frequency. The reference is the
    EWDM mean direction for the same record, whose sign comes from the
    long-wave curvature phase and the per-sample wavevector solve.

    One decision per row against a reference is the standard fold; note that it
    can flap where the estimate runs near-orthogonal to the reference, in which
    case neither hemisphere is favored by the data."""
    if not np.isfinite(ref_deg):
        return D
    a = np.radians(theta_deg)
    m = np.degrees(np.arctan2((D * np.sin(a)).sum(), (D * np.cos(a)).sum()))
    if abs((m - ref_deg + 180.0) % 360.0 - 180.0) > 90.0:
        return np.roll(D, len(D) // 2)
    return D


def traditional_directional_spectra(eta, dx, depth, fs, freqs, k_grid,
                                    theta_deg, Sf=None, apertures=None,
                                    n_staff=16, seed=20, omega0=12.0,
                                    methods=('EMEP', 'BDM', 'IMLM'),
                                    kscan=True, sign_reference=None):
    """CSM directional spectra from a camera elevation field.

    Seeds a virtual-gauge array into `eta` for each rung of the aperture ladder,
    forms the wavelet cross-spectral matrix per frequency, and inverts it with
    each requested estimator. Apertures are visited coarse to tight and the
    first whose anti-alias band admits a frequency wins it: that rung has the
    largest baseline there, hence the best angular resolution. Averaging D
    across rungs of different resolution would blur exactly the bimodality
    these estimators exist to resolve.

    Args:
        eta: (ny, nx, T) elevation field [m].
        dx: pixel size [m].
        depth: water depth [m].
        fs: sampling rate [Hz].
        freqs: frequency grid [Hz].
        k_grid: wavenumber grid for the wavevector scan [rad/m].
        theta_deg: direction grid, degrees CW from true North.
        Sf: (nf,) omnidirectional elevation spectrum [m^2/Hz]; when given, the
            returned E(f, theta) is Sf * D(theta|f).
        sign_reference: (nf,) direction per frequency [deg CW from N] used to
            break the estimators' 180-deg ambiguity; the EWDM mean direction
            for the same record. Without it the output is axis-only.
        methods: any of 'EMEP', 'BDM', 'MLM', 'IMLM'.
        kscan: also run the dispersion-free wavevector scan. IMLM shares the
            Capon scan: the deconvolution sharpens D(theta) at fixed k and has
            no counterpart in the steered (k, theta) response.

    Returns:
        dict keyed by method name, each a dict with 'D_f' (nf, ntheta) and,
        when `kscan`, 'D_k' and 'Psi_k' (nk, ntheta). 'E_f' is included when
        `Sf` is given. Rows with no aperture coverage are NaN.
    """
    ny, nx, T = eta.shape
    theta_deg = np.asarray(theta_deg, float)
    # compass (CW from N) -> math (CCW from E), wrapped onto [-180, 180); the
    # estimators steer in math
    theta = np.radians(((90.0 - theta_deg + 180.0) % 360.0) - 180.0)
    order = np.argsort(theta)
    theta_sorted = theta[order]
    unsort = np.argsort(order)

    freqs = np.asarray(freqs, float)
    k_grid = np.asarray(k_grid, float)
    kdisp = _k_dispersion(freqs, depth)
    if apertures is None:
        apertures = _default_apertures()

    out = {m: {'D_f': np.full((len(freqs), len(theta_deg)), np.nan)}
           for m in methods}
    if kscan:
        for m in methods:
            out[m]['Psi_k'] = np.zeros((len(k_grid), len(theta_deg)))
            out[m]['_ck'] = np.zeros(len(k_grid))
    claimed = np.zeros(len(freqs), bool)
    fit_fns = {'EMEP': emep, 'BDM': bdm}       # nonlinear, carry a misfit
    csm_fns = {'MLM': mlm, 'IMLM': imlm}       # closed-form Capon and its deconvolution

    for ai, (_name, ext) in enumerate(apertures):
        ii, jj, px, py, bmax = _seed_aperture(ny, nx, dx, ext, n_staff,
                                              seed + ai)
        # seed_aperture returns positions as (center - index)*dx, the negative of
        # the frame the elevation field is built on; flip back so the steering
        # vector and the field share one coordinate origin
        pos = -np.c_[np.asarray(px, float), np.asarray(py, float)]
        klo, khi = _aperture_band(bmax, lo_frac=(0.05 if ai == 0 else 1.0))
        in_band = (kdisp <= khi) & ~claimed
        if not in_band.any():
            continue
        es = _bilinear_stack(eta, ii, jj)
        es = es - es.mean(1, keepdims=True)
        W = _cwt_stack(es, freqs, fs, omega0)
        ksub = k_grid[(k_grid >= klo) & (k_grid <= khi)]

        for fi in np.flatnonzero(in_band):
            C = _wavelet_csm(W, fi)
            k_f = float(kdisp[fi])
            ref = (np.nan if sign_reference is None
                   else float(np.asarray(sign_reference, float)[fi]))
            for m in methods:
                if m in csm_fns:
                    D = csm_fns[m](C, k_f, pos, theta_sorted)
                else:
                    D = fit_fns[m](C, k_f, pos, theta_sorted)
                out[m]['D_f'][fi] = _fold_to_reference(D[unsort], theta_deg, ref)
            if kscan and ksub.size >= 3:
                for m in methods:
                    if m in csm_fns:
                        P = mlm_kscan(C, _kscan_steering(pos, ksub,
                                                         theta_sorted))
                    else:
                        P = _fit_kscan(C, ksub, pos, theta_sorted, fit_fns[m])
                    sel = np.searchsorted(k_grid, ksub)
                    Pk = np.array([_fold_to_reference(P[j][unsort], theta_deg,
                                                      ref)
                                   for j in range(P.shape[0])])
                    out[m]['Psi_k'][sel] += Pk
                    out[m]['_ck'][sel] += 1.0
        claimed |= in_band

    dth = np.radians(abs(theta_deg[1] - theta_deg[0]))
    for m in methods:
        if Sf is not None:
            out[m]['E_f'] = np.asarray(Sf, float)[:, None] * out[m]['D_f']
        if kscan:
            ck = out[m].pop('_ck')
            Psi = np.divide(out[m]['Psi_k'], np.maximum(ck, 1e-30)[:, None],
                            out=np.zeros_like(out[m]['Psi_k']),
                            where=ck[:, None] > 0)
            row = Psi.sum(1) * dth
            out[m]['Psi_k'] = np.where(ck[:, None] > 0, Psi, np.nan)
            out[m]['D_k'] = np.where((ck[:, None] > 0) & (row[:, None] > 0),
                                     Psi / np.maximum(row, 1e-30)[:, None],
                                     np.nan)
    return out

#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

"""
Cross-spectral-matrix directional estimators: MLM (Capon 1969; Isobe et al.
1984 EMLM), IMLM (Pawka 1983), EMEP (Hashimoto, Nagai & Asai 1994), Shannon
MEM, BDM (Hashimoto & Kobune 1988), the Welch CSM source, wavevector-scan
variants, and the transfer-function (`_h`) single-triplet forms. Copied from
the companion wave-direction-estimation repository (`_codes/csm.py`).

CONVENTION: steering is `exp(i k (x cos theta + y sin theta))`, so the CSM
must carry the math/FFT cross-phase sign; a wavelet CSM `<W conj(W)>` (ewdm
Morlet, opposite sign) must be CONJUGATED first or the lobes land 180 deg
off. `theta` is radians, math convention; distributions integrate to one
over theta. Direction comes through linear dispersion k(f).
"""

import numpy as np
from scipy.optimize import least_squares


def steering(k, pos, theta):
    """Plane-wave steering matrix `exp(i k (x cos th + y sin th))`, (M, L)."""
    x = pos[:, 0:1]
    y = pos[:, 1:2]
    return np.exp(1j * k * (x * np.cos(theta)[None, :]
                            + y * np.sin(theta)[None, :]))


def _regularize(C, eps=1e-3):
    """Diagonal-load a cross-spectral matrix for a stable inverse."""
    tr = np.trace(C).real / C.shape[0]
    return C + eps * tr * np.eye(C.shape[0])


def _pint(D, theta):
    """Periodic directional integral on an endpoint-open grid (sum * dtheta)."""
    return D.sum(-1) * (theta[1] - theta[0])


def _normalize(D, theta):
    """Clip negatives and scale so the directional integral is one."""
    D = np.clip(D, 0.0, None)
    area = _pint(D, theta)
    return D / area if area > 0 else np.full_like(D, 1.0 / (2 * np.pi))


def mlm(C, k, pos, theta, eps=1e-3):
    """Maximum likelihood (Capon) directional distribution for one frequency."""
    H = steering(k, pos, theta)
    Cinv = np.linalg.inv(_regularize(C, eps))
    denom = np.real(np.einsum("ml,mn,nl->l", np.conj(H), Cinv, H))
    return _normalize(1.0 / np.maximum(denom, 1e-12), theta)


def _forward_csm(D, k, pos, theta):
    """Cross-spectral matrix implied by a directional distribution,
    C[m,n] = int H_m H_n* D(theta) dtheta. The forward half of the IMLM loop."""
    H = steering(k, pos, theta)
    dth = theta[1] - theta[0]
    return (H * (D * dth)[None, :]) @ H.conj().T


def imlm(C, k, pos, theta, eps=1e-3, n_iter=3, relax=1.0):
    """Iterative MLM (Pawka 1983; Oltman-Shay & Guza 1984): push the estimate
    through the forward relation, re-apply MLM, and correct by the residual
    against the original estimate. Gains flatten after ~3 iterations and
    sidelobes grow beyond, so `n_iter` stays low; `eps` must match the load
    on the observed matrix or the two passes see different smearing kernels."""
    D_mlm = mlm(C, k, pos, theta, eps)
    D = D_mlm.copy()
    for _ in range(n_iter):
        D_hat = mlm(_forward_csm(D, k, pos, theta), k, pos, theta, eps)
        D = _normalize(D + relax * (D_mlm - D_hat), theta)
    return D


def welch_csm(series, fs, freqs, *, nperseg=512, overlap=0.5):
    """Welch cross-spectral matrix of `series` (nstaff, ntime), band-averaged
    onto `freqs`: Hann-windowed detrended segment rffts, per-segment outer
    products averaged per FFT bin, linear bins averaged into each log band
    (nearest bin where a band is empty). Returns (C (nfreq, M, M), dof).
    PHASE: numpy rfft `exp(-2 pi i f t)`, so C[m, n] = <X_m conj(X_n)> needs
    NO conjugation before the steering-convention estimators."""
    series = np.asarray(series, float)
    M, T = series.shape
    nperseg = int(min(nperseg, T))
    step = max(1, int(round(nperseg * (1.0 - overlap))))
    starts = list(range(0, T - nperseg + 1, step)) or [0]
    win = np.hanning(nperseg)
    fbins = np.fft.rfftfreq(nperseg, d=1.0 / fs)

    Cf = np.zeros((len(fbins), M, M), complex)
    for s in starts:
        seg = series[:, s:s + nperseg]
        seg = seg - seg.mean(1, keepdims=True)
        X = np.fft.rfft(seg * win[None, :], axis=1)          # (M, nfft)
        Cf += np.einsum('mf,nf->fmn', X, np.conj(X))
    Cf /= len(starts)

    edges = np.sqrt(freqs[:-1] * freqs[1:])
    lo = np.concatenate([[0.0], edges])
    hi = np.concatenate([edges, [np.inf]])
    C = np.zeros((len(freqs), M, M), complex)
    dof = np.zeros(len(freqs))
    for i in range(len(freqs)):
        m = (fbins >= lo[i]) & (fbins < hi[i])
        if m.any():
            C[i] = Cf[m].mean(0)
            dof[i] = int(m.sum())
        else:
            j = int(np.argmin(np.abs(fbins - freqs[i])))
            C[i] = Cf[j]
            dof[i] = 1
    return C, dof


def kscan_steering(pos, k_scan, theta_scan):
    """Steering tensor e(k, theta) = exp(i k (x cos th + y sin th)),
    (nk, ntheta, M); position-only, so precompute once per aperture."""
    proj = (np.outer(np.cos(theta_scan), pos[:, 0])
            + np.outer(np.sin(theta_scan), pos[:, 1]))     # (ntheta, M)
    return np.exp(1j * np.asarray(k_scan)[:, None, None] * proj[None, :, :])


def mlm_kscan(C, steer, load=1e-3, bartlett=False):
    """Capon (MLM) wavevector-scan response P(k, theta) = 1 / Re(e^H C^-1 e)
    over the `kscan_steering` grid: the wavenumber is scanned rather than
    pinned to the dispersion shell. `C` (M, M) must be in the steering
    convention. `bartlett=True` returns the beamformer e^H C e instead
    (no inverse, lower resolution). Returns (nk, ntheta), non-negative."""
    nk, nth, M = steer.shape
    E = steer.reshape(nk * nth, M)
    A = C if bartlett else np.linalg.inv(_regularize(C, load))
    quad = np.real(np.einsum('pm,mn,pn->p', np.conj(E), A, E, optimize=True))
    if bartlett:
        return np.maximum(quad, 0.0).reshape(nk, nth)
    return (1.0 / np.maximum(quad, 1e-12)).reshape(nk, nth)


def _coherence_pairs(C):
    """Off-diagonal complex coherences and the (m, n) index pairs."""
    M = C.shape[0]
    d = np.sqrt(np.clip(np.real(np.diag(C)), 1e-30, None))
    pairs = [(m, n) for m in range(M) for n in range(m + 1, M)]
    phi = np.array([C[m, n] / (d[m] * d[n]) for m, n in pairs])
    return phi, pairs


# single-point triplet (pseudo-buoy) estimators {{{
#
# The array estimators above steer with spatial phase factors; a collocated
# (elevation, east-slope, north-slope) triplet instead has per-element
# TRANSFER FUNCTIONS h(theta) = [1, i k cos(theta), i k sin(theta)] in the
# same steering convention (heave-pitch-roll processing: Longuet-Higgins et
# al. 1963; Isobe, Kondo & Horikawa 1984 EMLM; Hashimoto 1997). The
# cross-spectral matrix model is C = integral h h^H S(theta) dtheta, so the
# slope autospectra carry directional information (|h| is theta-dependent,
# unlike the unit-modulus array steering).

def triplet_transfer(k, theta):
    """(3, L) transfer matrix for (eta, deta/dEast, deta/dNorth) at one
    point, steering convention `exp(i k x . u(theta))` evaluated at x = 0."""
    one = np.ones_like(theta)
    return np.vstack([one + 0j,
                      1j * k * np.cos(theta),
                      1j * k * np.sin(theta)])


def triplet_coeffs(C, k):
    """First-five directional Fourier coefficients from a triplet CSM.

    With C = integral h h^H S dtheta and h = triplet_transfer:
    a1 = int cos(theta) D, b1 = int sin(theta) D (from the eta-slope
    quadrature), a2 = int cos(2 theta) D, b2 = int sin(2 theta) D (from the
    slope co-spectra). Also returns k_meas = sqrt((Cxx + Cyy)/Czz), the
    buoy check-ratio wavenumber (independent of any common spatial filter
    on the three signals)."""
    E = max(float(C[0, 0].real), 1e-30)
    kk = max(float(k), 1e-30)
    a1 = -float(C[0, 1].imag) / (kk * E)
    b1 = -float(C[0, 2].imag) / (kk * E)
    a2 = float((C[1, 1] - C[2, 2]).real) / (kk ** 2 * E)
    b2 = 2.0 * float(C[1, 2].real) / (kk ** 2 * E)
    k_meas = np.sqrt(max(float((C[1, 1] + C[2, 2]).real), 0.0) / E)
    return a1, b1, a2, b2, k_meas


def mem_shannon(C, k, theta):
    """Shannon-entropy maximum-entropy distribution (MEP: Hashimoto 1997;
    Kim, Lin & Wang 1994) for one frequency of a triplet CSM.

    D(theta) = exp(l . [cos, sin, cos2, sin2](theta)) with the multipliers
    solved so the distribution's first four angular moments match the
    observed (a1, b1, a2, b2); falls back to uniform when the solve fails."""
    a1, b1, a2, b2, _ = triplet_coeffs(C, k)
    G = np.vstack([np.cos(theta), np.sin(theta),
                   np.cos(2 * theta), np.sin(2 * theta)])       # (4, L)
    obs = np.array([a1, b1, a2, b2])

    def moments(lam):
        g = lam @ G
        w = np.exp(g - g.max())
        return (G * w[None, :]).sum(1) / w.sum()

    try:
        sol = least_squares(lambda lam: moments(lam) - obs, np.zeros(4),
                            method="lm", max_nfev=400)
        lam = sol.x
    except Exception:
        return np.full_like(theta, 1.0 / (2 * np.pi))
    g = lam @ G
    return _normalize(np.exp(g - g.max()), theta)


def mlm_h(C, H, theta, eps=1e-3):
    """Capon/MLM with an arbitrary transfer matrix `H` (M, L) -- the
    transfer-function generalization of :func:`mlm` (Isobe et al. 1984)."""
    Cinv = np.linalg.inv(_regularize(C, eps))
    denom = np.real(np.einsum("ml,mn,nl->l", np.conj(H), Cinv, H))
    return _normalize(1.0 / np.maximum(denom, 1e-12), theta)


def _forward_csm_h(D, H, theta):
    """Cross-spectral matrix implied by `D` through an arbitrary transfer
    matrix `H` (M, L). The forward half of the IMLM loop, `imlm` for `mlm_h`."""
    dth = theta[1] - theta[0]
    return (H * (D * dth)[None, :]) @ H.conj().T


def imlm_h(C, H, theta, eps=1e-3, n_iter=3, relax=1.0):
    """Iterative MLM with an arbitrary transfer matrix `H` -- the
    transfer-function generalization of :func:`imlm`. Same Pawka (1983)
    deconvolution: MLM smears by a steering-dependent kernel, and re-applying
    it to the estimate's own forward CSM reproduces that kernel, so the
    residual is a correction. `eps` must match the load on the observed
    matrix."""
    D_mlm = mlm_h(C, H, theta, eps)
    D = D_mlm.copy()
    for _ in range(n_iter):
        D_hat = mlm_h(_forward_csm_h(D, H, theta), H, theta, eps)
        D = _normalize(D + relax * (D_mlm - D_hat), theta)
    return D


def _model_phi_h(p, H, pairs):
    """Modelled normalized cross-spectra for grid masses `p` (sum 1) under
    transfer `H`: mod_mn = (h_m h_n^* . p) / sqrt((|h_m|^2 . p)(|h_n|^2 . p))."""
    num = np.array([(H[m] * np.conj(H[n]) * p).sum() for m, n in pairs])
    B = (H.real ** 2 + H.imag ** 2)
    den = np.sqrt(np.clip(B @ p, 1e-30, None))
    return num / np.array([den[m] * den[n] for m, n in pairs])


def emep_h(C, H, theta, nmax=2, ridge=0.02, aic_mult=1.0,
           return_misfit=False):
    """EMEP with transfer-function steering (single-point triplet).

    Same exponential-harmonic model and AIC order selection as
    :func:`emep`, but the data vector is the normalized triplet
    cross-spectra and the model keeps the theta-dependent |h| in its
    denominators. A triplet measures two harmonics, so `nmax` defaults
    to 2."""
    phi, pairs = _coherence_pairs(C)
    obs = np.concatenate([phi.real, phi.imag])
    obs_norm = float(np.sum(obs ** 2)) + 1e-30
    dth = theta[1] - theta[0]

    def D_of(coef):
        N = len(coef) // 2
        a, b = coef[:N], coef[N:]
        n = np.arange(1, N + 1)[:, None]
        g = (a[:, None] * np.cos(n * theta)
             + b[:, None] * np.sin(n * theta)).sum(0)
        w = np.exp(g - g.max())
        return w / w.sum()

    def resid(coef):
        mod = _model_phi_h(D_of(coef), H, pairs)
        return np.concatenate([mod.real - obs[:len(mod)],
                               mod.imag - obs[len(mod):],
                               np.sqrt(ridge) * coef])

    def data_rss(coef):
        mod = _model_phi_h(D_of(coef), H, pairs)
        return np.sum((np.concatenate([mod.real, mod.imag]) - obs) ** 2)

    best, best_aic, best_rss = None, np.inf, np.inf
    prev = np.zeros(0)
    for N in range(1, int(nmax) + 1):
        c0 = np.zeros(2 * N)
        npr = prev.size // 2
        if npr:
            c0[:npr] = prev[:npr]
            c0[N:N + npr] = prev[npr:]
        try:
            sol = least_squares(resid, c0, method="lm", max_nfev=2000)
        except Exception:
            continue
        rss = data_rss(sol.x)
        aic = len(obs) * np.log(rss / len(obs) + 1e-30) + aic_mult * 2 * (2 * N)
        if aic < best_aic:
            best_aic, best, prev, best_rss = aic, sol.x, sol.x, rss
    if best is None:
        D = np.full_like(theta, 1.0 / (2 * np.pi))
        return (D, 1.0) if return_misfit else D
    D = _normalize(D_of(best) / dth, theta)
    return (D, best_rss / obs_norm) if return_misfit else D


def bdm_h(C, H, theta, u_ladder=(0.3, 1.0, 3.0), min_smooth_steps=0,
          return_misfit=False):
    """BDM with transfer-function steering (single-point triplet).

    Same ln D parametrization, circular second-difference prior, and
    AIC/edof hyperparameter ladder as :func:`bdm`, with the model (and its
    analytic Jacobian) generalized to theta-dependent |h|."""
    phi, pairs = _coherence_pairs(C)
    obs = np.concatenate([phi.real, phi.imag])
    obs_norm = float(np.sum(obs ** 2)) + 1e-30
    Lg = len(theta)
    A_p = np.array([H[m] * np.conj(H[n]) for m, n in pairs])     # (P, L)
    B = H.real ** 2 + H.imag ** 2                                # (M, L)
    L2 = _circ_second_diff(Lg)
    n_obs = len(obs)

    def model_and_jac(x):
        w = np.exp(x - x.max())
        p = w / w.sum()
        den_m = np.sqrt(np.clip(B @ p, 1e-30, None))             # (M,)
        num = A_p @ p                                            # (P,)
        dd_pair = np.array([den_m[m] * den_m[n] for m, n in pairs])
        mod = num / dd_pair
        # d mod / d x_j = p_j [ (A(j) - num) - num ((B_m(j) - den_m^2)
        #     / (2 den_m^2) + (B_n(j) - den_n^2)/(2 den_n^2)) ] / (den_m den_n)
        Jc = np.empty((len(pairs), Lg), complex)
        for pi, (m, n) in enumerate(pairs):
            corr = ((B[m] - den_m[m] ** 2) / (2 * den_m[m] ** 2)
                    + (B[n] - den_m[n] ** 2) / (2 * den_m[n] ** 2))
            Jc[pi] = p * ((A_p[pi] - num[pi]) - num[pi] * corr) / dd_pair[pi]
        return mod, Jc

    def resid(x, u):
        mod, _ = model_and_jac(x)
        return np.concatenate([mod.real - obs[:len(mod)],
                               mod.imag - obs[len(mod):], u * (L2 @ x)])

    def jac(x, u):
        _, Jc = model_and_jac(x)
        return np.vstack([Jc.real, Jc.imag, u * L2])

    x0 = np.log(np.clip(mlm_h(C, H, theta), 1e-6, None))
    x0 -= x0.mean()

    rungs = []
    best_idx, best_aic = None, np.inf
    for ui, u in enumerate(u_ladder):
        try:
            sol = least_squares(resid, x0, jac=jac, args=(u,),
                                method="trf", max_nfev=200)
        except Exception:
            continue
        mod, Jc = model_and_jac(sol.x)
        r = np.concatenate([mod.real, mod.imag]) - obs
        rss = float(np.sum(r ** 2))
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
        x0 = sol.x
    if not rungs:
        D = np.full_like(theta, 1.0 / (2 * np.pi))
        return (D, 1.0) if return_misfit else D
    floor_idx = min(best_idx + int(min_smooth_steps), len(u_ladder) - 1)
    avail = [t for t in rungs if t[0] >= floor_idx]
    pick = min(avail, key=lambda t: t[1]) if avail else \
        max(rungs, key=lambda t: t[0])
    best, best_rss = pick[2], pick[3]
    D = _normalize(np.exp(best - best.max()), theta)
    return (D, best_rss / obs_norm) if return_misfit else D
# }}}


def _emep_model_phi(coef, k, dpos, theta):
    """Modelled coherences for EMEP harmonic coefficients `coef`."""
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


def emep(C, k, pos, theta, nmax=4, ridge=0.02, aic_mult=1.0,
         return_misfit=False):
    """Extended maximum entropy distribution, harmonic order chosen by AIC.
    A weak ridge scaled by 1/(k*b_max)^2 damps over-concentration on a poorly
    conditioned (small k*b) array; the 0.02 default keeps the second harmonic
    that carries bimodality. `return_misfit` adds the normalized residual
    r = data_rss / sum(obs^2), used as a k-likelihood by the wavevector scan
    (steering at the wrong wavenumber cannot reproduce the observed phases)."""
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

    best, best_aic, best_rss = None, np.inf, np.inf
    prev = np.zeros(0)
    for N in range(1, neff + 1):
        c0 = np.zeros(2 * N)
        npr = prev.size // 2
        if npr:                               # warm-start from the previous order
            c0[:npr] = prev[:npr]
            c0[N:N + npr] = prev[npr:]
        try:
            sol = least_squares(resid, c0, method="lm", max_nfev=2000)
        except Exception:
            continue
        rss = data_rss(sol.x)
        aic = len(obs) * np.log(rss / len(obs) + 1e-30) + aic_mult * 2 * (2 * N)
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


def bdm(C, k, pos, theta, u_ladder=(0.3, 1.0, 3.0), min_smooth_steps=0,
        return_misfit=False):
    """Bayesian Directional Method (Hashimoto & Kobune 1988; Hashimoto 1997).

    Fits x = ln D(theta) on the full direction grid to the observed complex
    cross-spectral coherences (the same data vector as `emep`), under a
    circular second-difference smoothness prior u^2 ||L2 x||^2. The
    hyperparameter is selected from `u_ladder` by AIC with effective degrees
    of freedom tr[J (J'J + u^2 L2'L2)^-1 J'] (a practical stand-in for the
    full ABIC determinant machinery). Warm-started from the MLM estimate.
    `return_misfit` returns the same normalized data residual as `emep`."""
    phi, pairs = _coherence_pairs(C)
    dpos = np.array([pos[m] - pos[n] for m, n in pairs])
    obs = np.concatenate([phi.real, phi.imag])
    obs_norm = float(np.sum(obs ** 2)) + 1e-30
    Lg = len(theta)
    dth = theta[1] - theta[0]
    E = np.exp(1j * k * (dpos[:, 0:1] * np.cos(theta)[None, :]
                         + dpos[:, 1:2] * np.sin(theta)[None, :]))
    L2 = _circ_second_diff(Lg)
    n_obs = len(obs)

    def model_and_jac(x):
        w = np.exp(x - x.max())
        s = w.sum()
        mod = (E * (w / s)[None, :]).sum(1)
        Jc = (E - mod[:, None]) * (w / s)[None, :]
        return mod, Jc

    def resid(x, u):
        mod, _ = model_and_jac(x)
        return np.concatenate([mod.real - obs[:len(mod)],
                               mod.imag - obs[len(mod):], u * (L2 @ x)])

    def jac(x, u):
        _, Jc = model_and_jac(x)
        return np.vstack([Jc.real, Jc.imag, u * L2])

    x0 = np.log(np.clip(mlm(C, k, pos, theta), 1e-6, None))
    x0 -= x0.mean()

    # store every rung so a smoothness floor can pick a stronger-than-AIC rung
    rungs = []                                # (u_index, aic, x, rss)
    best_idx, best_aic = None, np.inf
    for ui, u in enumerate(u_ladder):
        try:
            sol = least_squares(resid, x0, jac=jac, args=(u,),
                                method="trf", max_nfev=200)
        except Exception:
            continue
        mod, Jc = model_and_jac(sol.x)
        r = np.concatenate([mod.real, mod.imag]) - obs
        rss = float(np.sum(r ** 2))
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
        x0 = sol.x                            # warm-start the next rung
    if not rungs:
        D = np.full_like(theta, 1.0 / (2 * np.pi))
        return (D, 1.0) if return_misfit else D
    # floor the smoothness: bump the AIC-chosen rung `min_smooth_steps` toward
    # stronger smoothing (higher u), then take the strongest available rung
    # whose index is >= that floor (u_ladder is ascending in smoothness)
    floor_idx = min(best_idx + int(min_smooth_steps), len(u_ladder) - 1)
    avail = [t for t in rungs if t[0] >= floor_idx]
    pick = min(avail, key=lambda t: t[1]) if avail else \
        max(rungs, key=lambda t: t[0])
    best, best_rss = pick[2], pick[3]
    D = _normalize(np.exp(best - best.max()), theta)
    return (D, best_rss / obs_norm) if return_misfit else D

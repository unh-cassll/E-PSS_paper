"""Pseudo-buoy single-triplet directional estimators: one collocated
(eta, sx, sy) series as a 3 m disc average of the E-PSS field, the disc
(jinc) response divided out of S(f), every estimator steering on
h(theta) = [1, i k cos, i k sin] (`subroutines.csm`). Rows with H^2 < H2_MIN
render blank. From the companion wave-direction-estimation repository."""
import logging

import numpy as np
import xarray as xr
from scipy.special import j1

from ewdm.multiaperture import (k_dispersion, circ_stats,
                                _math_angle_to_cw_from_N,
                                _cw_from_N_to_math_angle)
from ewdm.parameters import VARIABLE_NAMES
from subroutines import csm as csmod

logger = logging.getLogger("pseudo_buoy")

BUOY_DIAM_M = 3.0
H2_MIN = 0.1                  # disc-response power floor of the support band
WELCH_NPERSEG = 1024
WELCH_OVERLAP = 0.75


def asit_triplet(eta, se, sn, dx, r_m=BUOY_DIAM_M / 2.0):
    """Disc-averaged (eta, sx, sy) series from the E-PSS field with native
    polarimeter slopes; at dx = 2.915 m the disc holds the center pixel."""
    ny, nx, _ = eta.shape
    ci, cj = ny // 2, nx // 2
    yy, xxg = np.mgrid[0:ny, 0:nx]
    disk = np.hypot((yy - ci) * dx, (xxg - cj) * dx) <= r_m
    ii, jj = np.where(disk)
    if ii.size == 0:
        ii, jj = np.array([ci]), np.array([cj])
    eta_b = np.asarray(eta, float)[ii, jj, :].mean(0)
    sx_b = se[ii, jj, :].mean(0)
    sy_b = sn[ii, jj, :].mean(0)
    return eta_b, sx_b, sy_b, int(ii.size)


def _row_circ_mean_deg(row, dirs_rad):
    """Circular-mean direction [deg] of one distribution row; NaN if empty."""
    w = np.nan_to_num(row)
    if not (w > 0).any():
        return np.nan
    return np.degrees(np.arctan2((w * np.sin(dirs_rad)).sum(),
                                 (w * np.cos(dirs_rad)).sum()))


def _angdiff_deg(a, b):
    """Absolute angular difference [deg], wrapped to [0, 180]."""
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def _hemisphere_flips(m, x, ref, onshore, x_on_max, downwind, x_dw_min,
                      weights=None):
    """Segment-coherent 180-deg flip mask for hemisphere anchoring. Targets:
    onshore for x <= x_on_max, downwind for x >= x_dw_min, `ref` elsewhere.
    Segments split at target-band changes, >90-deg raw jumps, and empty rows;
    each takes one pooled vote sum(w cos(m - target)) and flips when negative,
    so flips change only at raw discontinuities or band edges."""
    m = np.asarray(m, float)
    x = np.asarray(x, float)
    n = len(m)
    w = (np.ones(n) if weights is None
         else np.abs(np.nan_to_num(np.asarray(weights, float)[:n])))

    target = np.full(n, np.nan)
    band = np.zeros(n, int)                  # 0 = LH reference, 1/2 = assertions
    if ref is not None:
        ref = np.asarray(ref, float)
        target[:len(ref[:n])] = ref[:n]
    if onshore is not None and x_on_max is not None:
        sel = x <= float(x_on_max)
        target[sel] = float(onshore)
        band[sel] = 1
    if downwind is not None and x_dw_min is not None:
        sel = (x >= float(x_dw_min)) & (band == 0)
        target[sel] = float(downwind)
        band[sel] = 2

    c = np.cos(np.radians(m - target))       # NaN where either is undefined

    # segment boundaries: band change, raw >90-degree jump, or empty row
    starts = np.zeros(n, bool)
    for i in range(1, n):
        if (band[i] != band[i - 1] or not np.isfinite(m[i])
                or not np.isfinite(m[i - 1])
                or _angdiff_deg(m[i], m[i - 1]) > 90.0):
            starts[i] = True

    seg = np.cumsum(starts)
    sid = int(seg[-1]) if n else 0
    nseg = sid + 1
    segs = [np.flatnonzero(seg == s) for s in range(nseg)]
    vote = np.zeros(nseg)
    wsum = np.zeros(nseg)
    mass = np.zeros(nseg)
    for s, rows in enumerate(segs):
        ok = rows[np.isfinite(c[rows]) & (w[rows] > 0)]
        if ok.size:
            vote[s] = float((w[ok] * c[ok]).sum())
            wsum[s] = float(w[ok].sum())
        fin = rows[np.isfinite(m[rows])]
        mass[s] = float(w[fin].sum()) if fin.size else 0.0

    # |vote| >= MARGIN * wsum pins a segment; near-orthogonal targets stay
    # undecided for the continuity pass
    MARGIN = 0.15
    state = np.zeros(nseg, int)              # +1 keep, -1 flip, 0 undecided
    for s in range(nseg):
        if wsum[s] > 0 and abs(vote[s]) >= MARGIN * wsum[s]:
            state[s] = 1 if vote[s] > 0 else -1
    if not (state != 0).any() and nseg > 1 and mass.max() > 0:
        state[int(np.argmax(mass))] = 1      # nothing passed the margin:
        #                                      keep the most energetic segment

    def _edge(s, side):
        rows = segs[s]
        fin = rows[np.isfinite(m[rows])]
        if fin.size == 0:
            return np.nan
        i = fin[0] if side == "first" else fin[-1]
        return ((m[i] + (180.0 if state[s] == -1 else 0.0) + 180.0)
                % 360.0) - 180.0

    # undecided segments take the nearest decided neighbor's hemisphere
    changed = True
    while changed and (state == 0).any() and (state != 0).any():
        changed = False
        for s in range(nseg):
            if state[s] != 0:
                continue
            rows = segs[s]
            fin = rows[np.isfinite(m[rows])]
            if fin.size == 0:
                continue
            for step, my_edge in ((-1, float(m[fin[0]])),
                                  (1, float(m[fin[-1]]))):
                # nearest decided segment with a finite edge mean
                nb = s + step
                m_nb = np.nan
                while 0 <= nb < nseg:
                    if state[nb] != 0:
                        m_nb = _edge(nb, "last" if step < 0 else "first")
                        if np.isfinite(m_nb):
                            break
                    nb += step
                if not np.isfinite(m_nb):
                    continue
                state[s] = (1 if _angdiff_deg(my_edge, m_nb) <= 90.0
                            else -1)
                changed = True
                break

    flip = np.zeros(n, bool)
    for s, rows in enumerate(segs):
        if state[s] == -1:
            flip[rows] = True
    return flip


def _math_to_cwN_perm(hist_res):
    """Permutation reading a distribution on arange(-180, 180, res) math
    degrees as the same grid clockwise-from-North: `D_math[perm]`."""
    nfine = int(round(360.0 / hist_res))
    cwN = np.arange(-180.0, 180.0, hist_res)
    math_deg = _cw_from_N_to_math_angle(cwN)
    return (np.round((math_deg + 180.0) / hist_res).astype(int)) % nfine


def _jinc_h2(k, r_m):
    """Power response of a filled-disc average, H^2 = (2 J1(kR)/(kR))^2."""
    x = np.asarray(k, float) * float(r_m)
    H = np.where(x > 1e-9, 2.0 * j1(np.maximum(x, 1e-9)) / np.maximum(x, 1e-9),
                 1.0)
    return H ** 2


def _welch_band_psd(x, fs, freqs, nperseg=WELCH_NPERSEG,
                    overlap=WELCH_OVERLAP):
    """One-sided Welch PSD [x^2/Hz] band-averaged onto the log grid
    (same binning as `csm.welch_csm`; nearest bin where a band is empty)."""
    from scipy.signal import welch
    nseg = int(min(nperseg, len(x)))
    fb, P = welch(np.asarray(x, float), fs=fs, nperseg=nseg,
                  noverlap=int(round(nseg * overlap)), detrend="constant")
    edges = np.sqrt(freqs[:-1] * freqs[1:])
    lo = np.concatenate([[0.0], edges])
    hi = np.concatenate([edges, [np.inf]])
    out = np.empty(len(freqs))
    for i in range(len(freqs)):
        m = (fb >= lo[i]) & (fb < hi[i])
        out[i] = P[m].mean() if m.any() else P[np.argmin(np.abs(fb - freqs[i]))]
    return out


def _triplet_ewdm(eta, sx, sy, fs, freqs, dd, kappa):
    """Krogstad wavelet triplet D(theta|f) and S(f) for one point; +180 deg
    corrects the Morlet slope-phase convention (ADCP-calibrated)."""
    from ewdm import Triplets
    from ewdm.density import estimate_directional_distribution
    t = np.arange(len(eta), dtype=float) / fs
    pt = xr.Dataset(
        {"surface_elevation": ("time", eta - eta.mean()),
         "eastward_slope": ("time", np.asarray(sx - sx.mean(), float)),
         "northward_slope": ("time", np.asarray(sy - sy.mean(), float))},
        coords={"time": t})
    trip = Triplets(pt, fs=fs, interpolate=False, normalise=False)
    trip.freqs = np.asarray(freqs, float)
    trip.dd = float(dd)
    trip.kappa = float(kappa)
    trip.use = "slopes"
    power = trip.estimate_wavelet_power(pt)
    th_math_deg = trip.theta_from_slopes(pt)
    th_cwn = _math_angle_to_cw_from_N(np.radians(th_math_deg)) + 180.0
    th_cwn = ((np.asarray(th_cwn, float) + 180.0) % 360.0) - 180.0
    out = estimate_directional_distribution(
        power, xr.DataArray(th_cwn, dims=power.dims, coords=power.coords),
        dd=float(dd), kappa=float(kappa), power_weighted=True)
    return (np.nan_to_num(out["directional_distribution"].values),
            np.nan_to_num(out["frequency_spectrum"].values))


def estimator_dataset(eta, sx, sy, *, fs, depth, freqs, dd, kappa, r_m,
                      onshore=None, f_on=None, downwind=None, f_dw=None):
    """The pseudo-buoy estimator suite on one triplet: returns the estimator
    dataset (f-domain only) with the support band in attrs."""
    freqs = np.asarray(freqs, float)
    nf = len(freqs)
    dirs = np.arange(-180.0, 180.0, dd)
    nth = len(dirs)
    th_math = np.radians(dirs)
    perm = _math_to_cwN_perm(dd)
    dirs_rad = np.radians(dirs)

    k_lin = k_dispersion(freqs, depth)
    H2 = _jinc_h2(k_lin, r_m)
    support = H2 >= H2_MIN

    C, dof = csmod.welch_csm(np.vstack([eta, sx, sy]), fs, freqs,
                             nperseg=WELCH_NPERSEG, overlap=WELCH_OVERLAP)
    S_f = _welch_band_psd(eta - eta.mean(), fs, freqs)
    S_f = np.where(support, S_f / np.maximum(H2, 1e-12), np.nan)

    D = {tag: np.full((nf, nth), np.nan) for tag in
         ("mem", "csm", "bdm", "mlm", "imlm")}
    k_meas = np.full(nf, np.nan)
    for i in np.flatnonzero(support):
        Ci = np.conj(C[i])                 # locked convention: conjugate first
        Hh = csmod.triplet_transfer(float(k_lin[i]), th_math)
        try:
            D["mlm"][i] = csmod.mlm_h(Ci, Hh, th_math)[perm]
            D["imlm"][i] = csmod.imlm_h(Ci, Hh, th_math)[perm]
            D["mem"][i] = csmod.mem_shannon(Ci, float(k_lin[i]), th_math)[perm]
            D["csm"][i] = csmod.emep_h(Ci, Hh, th_math)[perm]
            D["bdm"][i] = csmod.bdm_h(Ci, Hh, th_math)[perm]
            k_meas[i] = csmod.triplet_coeffs(Ci, float(k_lin[i]))[4]
        except Exception:
            logger.exception("row f=%.3f failed", freqs[i])

    Dt, St = _triplet_ewdm(eta, sx, sy, fs, freqs, dd, kappa)
    Dt = np.where(support[:, None], Dt, np.nan)
    D["triplet"] = Dt

    # hemisphere anchoring: assertion segments only (no array LH reference)
    for tag, Dm in D.items():
        m = np.array([_row_circ_mean_deg(np.nan_to_num(Dm[i]), dirs_rad)
                      for i in range(nf)])
        flips = _hemisphere_flips(m, freqs, None, onshore, f_on,
                                  downwind, f_dw, weights=S_f)
        for i in np.flatnonzero(flips):
            Dm[i] = np.roll(Dm[i], nth // 2)

    data_vars = {"frequency_spectrum": ("frequency", S_f),
                 "pseudo_buoy_k_meas": ("frequency", k_meas)}
    for tag, Dm in D.items():
        # renormalize (deg^-1) and scale to E(f, theta) [m^2/Hz/rad]
        tot = np.nansum(Dm, axis=1, keepdims=True) * dd
        Dn = np.divide(Dm, tot, out=np.full_like(Dm, np.nan), where=tot > 0)
        E = np.nan_to_num(Dn) * np.degrees(1.0) * np.nan_to_num(S_f)[:, None]
        thbar, sigma = circ_stats(np.nan_to_num(E), dirs)
        data_vars["directional_distribution_f_" + tag] = (
            ("frequency", "direction"), Dn)
        data_vars["directional_spectrum_f_" + tag] = (
            ("frequency", "direction"), E)
        data_vars["mean_direction_" + tag] = ("frequency", thbar)
        data_vars["directional_spread_" + tag] = ("frequency", sigma)

    ds = xr.Dataset(data_vars,
                    coords={"frequency": freqs, "direction": dirs})
    for tag in D:
        ds["directional_distribution_f_" + tag].attrs = dict(
            VARIABLE_NAMES.get("directional_distribution", {}))
        ds["directional_spectrum_f_" + tag].attrs = dict(
            VARIABLE_NAMES.get("directional_spectrum_f", {}))
        ds["mean_direction_" + tag].attrs = dict(
            VARIABLE_NAMES.get("mean_direction", {}))
        ds["directional_spread_" + tag].attrs = dict(
            VARIABLE_NAMES.get("directional_spread", {}))
    f_sup = freqs[support]
    ds.attrs.update(
        pseudo_buoy=1, buoy_diameter_m=float(2 * r_m),
        buoy_h2_min=float(H2_MIN),
        buoy_support_f_hz=[float(f_sup.min()), float(f_sup.max())]
        if f_sup.size else [np.nan, np.nan],
        welch_nperseg=WELCH_NPERSEG, welch_overlap=WELCH_OVERLAP,
        welch_median_dof=float(np.median(dof)),
        csm_conjugated=1, direction_convention="cw_from_N")
    return ds



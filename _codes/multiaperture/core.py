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
from ewdm.density import (estimate_directional_distribution,
                          estimate_radial_distribution)

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
        k_new = w**2 / (GRAV * np.tanh(np.clip(k * depth, 1e-9, 50.0)))
        converged = np.allclose(k_new, k, rtol=1e-12, atol=0.0)
        k = k_new
        if converged:
            break
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
def erode_valid(valid):
    """1-px erosion of a boolean footprint: keep pixels whose 4-neighbours are
    all inside it, so central-difference gradients stay within the footprint."""
    ev = valid.copy()
    ev[1:-1, 1:-1] &= (valid[:-2, 1:-1] & valid[2:, 1:-1]
                       & valid[1:-1, :-2] & valid[1:-1, 2:])
    ev[[0, -1], :] = False
    ev[:, [0, -1]] = False
    return ev


def seed_aperture(ny, nx, dx, extent_px, n_staff, seed, valid=None):
    """Random virtual staffs in a centered window. extent_px: window side in
    pixels (scalar = square, (rows, cols) = rectangular). Column axis
    West-positive: East=(cxp-j)dx, North=(cyp-i)dx.

    valid (ny,nx bool): if given, draw staffs only from True pixels; the window
    is recentred on the valid centroid and sampled without replacement (with
    replacement only if it holds fewer than n_staff). Returns
    (ii, jj, px, py, b_max)."""
    rng = np.random.default_rng(seed)
    ey, ex = extent_px if isinstance(extent_px, (tuple, list)) else (extent_px, extent_px)
    ey = int(min(ey, ny - 1))
    ex = int(min(ex, nx - 1))
    cxp, cyp = (nx - 1) / 2, (ny - 1) / 2
    if valid is None:
        i0 = (ny - ey) // 2
        j0 = (nx - ex) // 2
        ii = rng.integers(i0, i0 + ey + 1, n_staff)
        jj = rng.integers(j0, j0 + ex + 1, n_staff)
    else:
        vi, vj = np.where(valid)
        ci, cj = int(round(vi.mean())), int(round(vj.mean()))   # valid centroid
        i0, j0 = max(ci - ey // 2, 0), max(cj - ex // 2, 0)      # slice clips top
        wi, wj = np.where(valid[i0:i0 + ey + 1, j0:j0 + ex + 1])
        if wi.size == 0:                       # window missed the footprint
            wi, wj, i0, j0 = vi, vj, 0, 0       # fall back to all valid pixels
        pick = rng.choice(wi.size, size=n_staff, replace=wi.size < n_staff)
        ii, jj = wi[pick] + i0, wj[pick] + j0
    px = (cxp - jj) * dx
    py = (cyp - ii) * dx
    b = np.hypot(px[:, None] - px[None, :], py[:, None] - py[None, :])
    return ii, jj, px, py, float(b.max())


def aperture_band(b_max, lo_frac=0.30, hi_frac=1.0):
    """Trusted wavenumber band [lo_frac/b_max, hi_frac*pi/b_max] for an aperture
    of longest baseline b_max. hi = anti-alias ceiling, lo = minimum resolvable
    phase over the longest baseline."""
    return lo_frac / b_max, hi_frac * np.pi / b_max


def _log_edge_taper(grid, lo, hi, width):
    """Cosine edge weight on a log2 grid: 1 inside [lo, hi], ->0 over `width`
    octaves at each edge, 0 outside. Down-weights each aperture's anti-alias
    tail where bands overlap; cancels in single-aperture regions."""
    lg = np.log2(grid)
    llo, lhi = np.log2(lo), np.log2(hi)
    ramp = np.clip(np.minimum((lg - llo) / width, (lhi - lg) / width), 0.0, 1.0)
    t = 0.5 * (1.0 - np.cos(np.pi * ramp))
    return np.where((lg > llo) & (lg < lhi), t, 0.0)


def default_apertures():
    """Coarse->tight staggered ladder (name, extent_px), ~1.3x spacing so every
    k octave is covered by >=2 apertures (no single-aperture stitch notch)."""
    return [('A0', 28), ('A1', 22), ('A2', 17), ('A3', 13),
            ('A4', 10), ('A5', 8), ('A6', 6), ('A7', 4)]


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
def cwt_stack(staff_series, freqs, fs, omega0=6.0):
    """CWT each staff time series -> (n_staff, n_freq, T) complex array.
    omega0 = Morlet central frequency; larger -> finer frequency resolution
    (df/f ~ 1/omega0) but coarser time resolution (noisier statistics)."""
    mother = Morlet(omega0)
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
                          apertures=None, n_staff=16, seed=20, dd=4.0, kappa=36.0, omega0=12.0,
                          lo_frac=1.0, lo_frac_broad=0.05, hi_frac=1.0, flip_dir=False,
                          rel_bandwidth=0.08, radial_bandwidth_mode='relative',
                          power_weighted=True, nu_lo_broad=0.04,
                          nu_f_lim='auto', nu_k_lim='auto',
                          stitch_taper=0.7, solve_eta=None, valid=None,
                          antialias_gate=False, antialias_mult=3.0,
                          sign_anchor=None, sign_anchor_rmin=0.15, sign_coh_min=0.6):
    """Composite multi-aperture elevation directional spectra on the array-
    measured (WDM) wavenumber, binned with ewdm.density and stitched over each
    aperture's trusted band [lo_frac/b_max, hi_frac*pi/b_max]. Power pre-scaled
    so int S(f) df = var(eta).

    Returns omni S(f), F(k), Q(nu) and polar F(f,theta), F(k,theta), Q(nu,theta);
    theta is deg CW-from-N.

    radial_bandwidth_mode: 'relative' (log-space kernel of fractional width
    rel_bandwidth), 'histogram' (none), or 'absolute' (upstream fixed width).
    power_weighted=True bins variance rather than occurrence into F(k)/Q(nu);
    S(f)/F(f,theta) stay unweighted.

    stitch_taper: cosine edge-taper width (octaves) blending overlapping aperture
    bands at the stitch; 0/None = hard in-band indicator.

    solve_eta: optional field (same shape as eta) used only for the wavevector
    solve; power/variance still come from eta. Pass the de-pistoned field to keep
    the uniform long wave out of the cross-staff phases (else |k| biases low).

    valid: optional (ny,nx) boolean footprint; eta may carry NaN outside it.
    Variance is taken over the footprint and staffs are drawn from its
    1-px-eroded pixels, so the array spans the whole footprint.

    antialias_gate: an aperture whose ceiling khi = pi/b_max is below
    antialias_mult x the spectral-peak k aliases shorter waves to spuriously low
    |k|; when True it deposits onto k/nu only frequencies whose dispersion
    k <= khi. S(f)/F(f,theta) unaffected. Default off (E-PSS small FOV); enable
    when the widest baseline exceeds the dominant wavelength.

    nu_f_lim, nu_k_lim: (lo, hi) gates applied only to the Q(nu)/Q(nu,theta)
    deposit. 'auto': |k| in [2*pi/(50*b_max), k_FOV] with k_FOV = 2*pi/(nx*dx)
    (the EWDM/direct trust boundary for a small FOV), f = that range's dispersion
    band. For a large FOV set nu_k_lim explicitly at the high-k confidence limit.
    Tuple overrides; None disables a gate.
    """
    ny, nx, T = eta.shape
    if apertures is None:
        apertures = default_apertures()
    if valid is None:
        var_eta = eta.var(axis=2).mean()
        seed_valid = None
    else:
        # variance over the footprint only; staffs drawn from a 1-px eroded mask
        # so gradient central differences stay inside it
        var_eta = float(np.nanmean(np.where(valid, eta.var(axis=2), np.nan)))
        seed_valid = erode_valid(valid)
    gy, gx = np.gradient(eta, dx, axis=(0, 1))
    time = np.arange(T) / fs
    coords = {'frequency': freqs, 'time': time}

    ap = []
    for ai, (name, ext) in enumerate(apertures):
        ii, jj, px, py, bmax = seed_aperture(ny, nx, dx, ext, n_staff, seed + ai,
                                             valid=seed_valid)
        es = np.stack([eta[i, j, :] for i, j in zip(ii, jj)])
        es = es - es.mean(1, keepdims=True)
        We = cwt_stack(es, freqs, fs, omega0)
        # wavevector solve on the de-pistoned field if supplied; power P still
        # comes from We(eta)
        if solve_eta is not None:
            ek = np.stack([solve_eta[i, j, :] for i, j in zip(ii, jj)])
            Wk = cwt_stack(ek - ek.mean(1, keepdims=True), freqs, fs, omega0)
        else:
            Wk = We
        kx, ky, resid = solve_wavevectors(Wk, px, py)
        kmag = np.hypot(kx, ky)
        klo, khi = aperture_band(bmax, lo_frac=(lo_frac_broad if ai == 0 else lo_frac), hi_frac=hi_frac)
        ap.append(dict(name=name, bmax=bmax, klo=klo, khi=khi, ii=ii, jj=jj,
                       kmag=kmag, resid=resid, We=We,
                       dir=_math_angle_to_cw_from_N(np.arctan2(ky, kx), flip=flip_dir),
                       P=(np.abs(We)**2).mean(0)))

    # 180-deg sign from the coarsest aperture's LH first moment (reuse its staffs)
    a0 = ap[0]
    sxs = np.stack([gx[i, j, :] for i, j in zip(a0['ii'], a0['jj'])])
    sys = np.stack([gy[i, j, :] for i, j in zip(a0['ii'], a0['jj'])])
    Wsx = cwt_stack(sxs - sxs.mean(1, keepdims=True), freqs, fs, omega0)
    Wsy = cwt_stack(sys - sys.mean(1, keepdims=True), freqs, fs, omega0)
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

    # anti-alias gate threshold (see docstring)
    kdisp = k_dispersion(freqs, depth)
    gate_k = antialias_mult * float(k_dispersion(freqs[pk], depth))

    # Q(nu) (f, k) gate: trusted scale window from the array baselines (longest
    # wave 50x the longest baseline -> shortest wave 2x the shortest baseline);
    # f window = the dispersion band of that k range. Only Q(nu) is gated.
    bmx = [d['bmax'] for d in ap]
    if nu_k_lim == 'auto':
        # high cut = k_FOV, the frame-fundamental wavenumber (EWDM/direct trust
        # boundary)
        nu_k_lim = (2 * np.pi / (50.0 * max(bmx)), 2 * np.pi / (nx * dx))
    if nu_f_lim == 'auto':
        _fk = lambda kk: float(np.sqrt(GRAV * kk * np.tanh(np.clip(kk * depth, 1e-9, 50))) / (2 * np.pi))
        nu_f_lim = (_fk(nu_k_lim[0]), _fk(nu_k_lim[1])) if nu_k_lim is not None else None

    bins_dir = np.arange(-180.0, 180.0, dd)
    Fk = np.zeros(len(k_grid)); ck = np.zeros(len(k_grid))
    Qn = np.zeros(len(nu_grid)); cn = np.zeros(len(nu_grid))
    Fkd = np.zeros((len(k_grid), len(bins_dir)))
    Qnd = np.zeros((len(nu_grid), len(bins_dir)))
    ap_ok_omni = []   # per-aperture omnidirectional F(k) (full grid), for diagnostics
    for d in ap:
        # anti-alias frequency mask: send would-alias samples off the k/nu grids
        # so they never deposit; power kept intact (power-weighting unperturbed)
        fok = (kdisp <= d['khi']) if (antialias_gate and d['khi'] < gate_k) \
            else np.ones(len(freqs), bool)
        power = xr.DataArray(d['P'] * cal, dims=['frequency', 'time'], coords=coords)
        thd = xr.DataArray(d['dirf'], dims=['frequency', 'time'], coords=coords)
        k_vals = np.where(fok[:, None], d['kmag'], 1e12)
        kk = xr.DataArray(k_vals, dims=['frequency', 'time'], coords=coords)
        ok = estimate_radial_distribution(power, thd, kk, 'wavenumber', k_grid,
                                          dd, kappa, bandwidth=rel_bandwidth,
                                          bandwidth_mode=radial_bandwidth_mode,
                                          power_weighted=power_weighted)
        # gate the Q(nu) deposit to the trusted (f, k) window: send out-of-window
        # samples off the nu grid (F(k)/S(f) unaffected, power kept intact)
        nu_vals = d['kmag'] / (2 * np.pi * freqs[:, None])
        if nu_f_lim is not None or nu_k_lim is not None:
            keep = np.ones(nu_vals.shape, bool)
            if nu_f_lim is not None:
                keep &= ((freqs >= nu_f_lim[0]) & (freqs <= nu_f_lim[1]))[:, None]
            if nu_k_lim is not None:
                keep &= (d['kmag'] >= nu_k_lim[0]) & (d['kmag'] <= nu_k_lim[1])
            nu_vals = np.where(keep, nu_vals, 1e6)
        nu_vals = np.where(fok[:, None], nu_vals, 1e12)   # same anti-alias mask
        nu = xr.DataArray(nu_vals, dims=['frequency', 'time'], coords=coords)
        on = estimate_radial_distribution(power, thd, nu, 'nu', nu_grid,
                                          dd, kappa, bandwidth=rel_bandwidth,
                                          bandwidth_mode=radial_bandwidth_mode,
                                          power_weighted=power_weighted)
        ok_omni, ok_dir = ok['wavenumber_spectrum'].values, ok['directional_spectrum'].values
        on_omni, on_dir = on['nu_spectrum'].values, on['directional_spectrum'].values
        ink = (k_grid >= d['klo']) & (k_grid <= d['khi'])
        # nu band from the aperture's k-band along dispersion; broadest aperture
        # extended down to nu_lo_broad to show the measured-nu tail
        nlo, nhi = _nu_of_k(d['khi'], depth), _nu_of_k(d['klo'], depth)
        lo = min(nlo, nhi)
        if d is ap[0]:
            lo = min(lo, nu_lo_broad)
        # stitch weights: cosine edge taper (in log k / log nu) when enabled, else
        # the hard in-band indicator. Weight cancels in single-aperture regions.
        wk = _log_edge_taper(k_grid, d['klo'], d['khi'], stitch_taper) if stitch_taper \
            else ink.astype(float)
        wn = _log_edge_taper(nu_grid, lo, max(nlo, nhi), stitch_taper) if stitch_taper \
            else ((nu_grid >= lo) & (nu_grid <= max(nlo, nhi))).astype(float)
        ap_ok_omni.append((ok_omni.copy(), ink.copy()))
        Fk += ok_omni * wk; ck += wk
        Fkd += ok_dir * wk[:, None]
        Qn += on_omni * wn; cn += wn
        Qnd += on_dir * wn[:, None]
    Fk = np.where(ck > 0, Fk / np.maximum(ck, 1e-30), np.nan)
    Fkd /= np.maximum(ck, 1e-30)[:, None]
    Qn = np.where(cn > 0, Qn / np.maximum(cn, 1e-30), np.nan)
    Qnd /= np.maximum(cn, 1e-30)[:, None]

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
                ap_bands=[(d['klo'], d['khi']) for d in ap],
                ap_ok_omni=ap_ok_omni, ck=ck)

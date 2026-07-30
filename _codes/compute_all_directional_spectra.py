"""
Compute per-run E-PSS multi-aperture elevation directional spectra (f, k, nu)
from the earth-referenced PSS slope fields. Slope fields -> camera elevation
field (Fourier slope-projection long wave + per-frame g2s) -> multi-aperture
virtual-staff EWDM estimator, with the 180-deg sign resolved by the direct
S_f_theta 3-D-FFT anchor and an onshore swell tiebreaker.

@author: nathanlaxague
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
import netCDF4 as nc
import warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor, as_completed

from subroutines.utils import (DX_M, WATER_DEPTH_M, FS_HZ, NUM_RUNS, NUM_SAMPLES,
                               epss_ewdm_grids)

path = '../_data/'
# input/output files overridable via env vars
slope_field_file = path + os.environ.get('EPSS_FLD', 'ASIT2019_slope_fields_reduced.nc')
output_file = path + os.environ.get('EPSS_OUT', 'ASIT2019_EPSS_directional_spectra.nc')
fs, water_depth_m, num_samples, num_runs = FS_HZ, WATER_DEPTH_M, NUM_SAMPLES, NUM_RUNS
# de-piston corner k_n = 2*pi/(n*L); n=2.0 lifts the high-wind FOV-scale F(k) plateau
depiston_n = float(os.environ.get('EPSS_DEPISTON_N', 1.5))
# disc [px] for the long-wave FOV-mean tilt (None = full frame)
_sa = os.environ.get('EPSS_SLOPE_APERTURE', 'none')
slope_aperture = None if str(_sa).strip().lower() in ('none', '') else int(_sa)
# long-wave estimator: default 'fourier' (per-frequency slope projection); or 'wavelet'
longwave_method = os.environ.get('EPSS_LONGWAVE', 'fourier')
# post-hoc scalar-per-frequency recolor of EWDM S(f)/F(f,theta); corrects long+short
# double-count and short-field scale offset. Leaves D(f,theta), F(k), F(k,theta) unchanged.
# EPSS_RECOLOR_FC='none' disables (raw EWDM omni).
_rf = os.environ.get('EPSS_RECOLOR_FC', '0.55')
recolor_fc = None if str(_rf).strip().lower() in ('none', '') else float(_rf)
_rb = os.environ.get('EPSS_RECOLOR_BAND', '0.5,0.6')
recolor_band = tuple(float(x) for x in _rb.split(','))

# fixed grids (dx constant: 2.915 m / 32 px)
dx = DX_M
freqs, k_grid, nu_grid = epss_ewdm_grids(dx)

# Onshore swell tiebreaker: ASIT is 3 km south of Martha's Vineyard, so long
# swell propagates onshore (northward); flip a swell-dominated run that reads
# offshore. The curvature phase in `fourier_slope_projection` fixes the sign
# directly, so this runs only under EPSS_TIEBREAK=1.
onshore_dir = 0.0
swell_cut = 0.16
swell_frac = 0.15
use_tiebreaker = os.environ.get('EPSS_TIEBREAK', '0') not in ('0', '', 'false')

# traditional cross-spectral-matrix estimators run alongside EWDM on the same
# elevation field; '' disables them
_tm = os.environ.get('EPSS_CSM_METHODS', 'EMEP,BDM,IMLM').strip()
csm_methods = tuple(m for m in _tm.split(',') if m)

_DS = {}


def _ds():
    if not _DS:
        _DS['fld'] = nc.Dataset(slope_field_file)
        # slope_field_file renumbered 0-189; ref matched by index for sftheta sign anchor
        _DS['ref'] = nc.Dataset(os.environ.get(
            'EPSS_REF', path + 'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc'))
    return _DS


def _ewmean(deg, w):
    a = np.radians(deg)
    return np.degrees(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum()))


def _dir_moments(F, theta_deg):
    """Energy-weighted mean direction [deg CW from N] and Kuik spread [deg] per
    frequency row of F(f, theta)."""
    w = np.where(np.isfinite(F), F, 0.0)
    tot = w.sum(1)
    a = np.radians(theta_deg)
    c, s_ = (w * np.cos(a)).sum(1), (w * np.sin(a)).sum(1)
    thbar = np.degrees(np.arctan2(s_, c)) % 360.0
    m1 = np.divide(np.hypot(c, s_), tot, out=np.zeros_like(tot), where=tot > 0)
    sigma = np.degrees(np.sqrt(np.clip(2.0 * (1.0 - m1), 0.0, None)))
    return (np.where(tot > 0, thbar, np.nan),
            np.where(tot > 0, sigma, np.nan))


def _triplet_spectrum(eta, dx, fs, freqs, theta_deg, Sf):
    """Single-point EWDM triplet F(f, theta) [m^2/Hz/rad] from the frame-center
    (elevation, east slope, north slope), the heave-pitch-roll configuration of
    a directional wave buoy. Interpolated onto the project direction grid and
    scaled by the E-PSS omnidirectional spectrum.

    Direction mapping theta -> 270 - theta. `Triplets` reports the math
    convention (CCW from East; `theta_from_slopes` returns
    arctan2(Im W_ny, Im W_ex)) and only the `MultiApertureArrays` path carries
    `direction_convention`, so the output is sampled at the math angles of the
    requested compass directions; reading the math axis as a compass axis
    reflects the spectrum about the 45-deg line. The further 180 deg is the
    going-to flip.

    Calibrated over the 76 ADCP records in 0.08-0.30 Hz, the band the
    directional figures show (panels splice the direct spectrum in above
    0.7 Hz): 21.1 deg median, 3/76 wrong hemisphere, vs 158.9 deg and 73/76
    unflipped. Agreement with the EWDM LSQ product is 8.3 deg in that band and
    12.4 deg over 0.30-0.70 Hz. The triplet sign crosses over at ~0.7 Hz
    (k L = 2 pi), so it disagrees above the splice cut; do not quote it there."""
    import xarray as xr
    from ewdm import Triplets
    ny, nx, T = eta.shape
    ci, cj = ny // 2, nx // 2
    gy, gx = np.gradient(eta, dx, axis=(0, 1))
    ds = xr.Dataset(
        {'surface_elevation': ('time', eta[ci, cj, :]),
         'eastward_slope': ('time', gx[ci, cj, :]),
         'northward_slope': ('time', gy[ci, cj, :])},
        coords={'time': np.arange(T) / fs})
    # interpolate=False: the reconstructed field has no gaps, and the gap
    # filler needs a datetime index rather than elapsed seconds
    out = Triplets(ds, fs=fs, interpolate=False).compute(
        use='slopes', dd=float(abs(theta_deg[1] - theta_deg[0])))
    # compass -> math grid, plus the 180 deg going-to flip; the reflection is
    # measure-preserving, so the directional density needs no Jacobian
    theta_math = ((270.0 - np.asarray(theta_deg, float)) % 360.0 + 180.0) % 360.0 - 180.0
    D = out['directional_distribution'].interp(
        frequency=freqs, direction=theta_math, kwargs={'fill_value': np.nan}).values
    D = np.where(np.isfinite(D), D, 0.0)
    dth = np.radians(abs(theta_deg[1] - theta_deg[0]))
    tot = D.sum(1, keepdims=True) * dth
    D = np.divide(D, tot, out=np.full_like(D, np.nan), where=tot > 0)
    return np.asarray(Sf, float)[:, None] * D


def work(run_ind):
    from multiaperture import (build_eta_field, sftheta_sign_anchor,
                               anchored_freq_recolor)
    from ewdm import MultiApertureArrays
    from ewdm.multiaperture import default_apertures
    d = _ds()
    se = np.ma.filled(d['fld']['slope_east'][run_ind][..., :num_samples], np.nan)
    sn = np.ma.filled(d['fld']['slope_north'][run_ind][..., :num_samples], np.nan)
    if not np.isfinite(se).any():                        # NaN-flagged (corrupt) run
        return run_ind, None
    # gated de-piston; returns solve field with long-wave piston removed above depiston_n corner
    eta, _, eta_solve, eta_long, Zsw = build_eta_field(
                                        np.nan_to_num(se).astype(float),
                                        np.nan_to_num(sn).astype(float),
                                        water_depth_m, fs,
                                        slope_aperture=slope_aperture,
                                        depiston_n=depiston_n,
                                        return_components=True,
                                        longwave_method=longwave_method)
    # sign: 3-D-FFT sftheta anchor at matching index; an empty S_f_theta record
    # (run not present in the reference timeseries) -> LH fallback (None)
    Sft_ref = np.nan_to_num(np.ma.filled(d['ref']['S_f_theta'][run_ind], 0.0))
    anchor = sftheta_sign_anchor(d['ref'], run_ind) if Sft_ref.sum() > 0 else None
    # `from_field`/`seed_aperture` place gauges on a West/South-positive index
    # frame (East = (cxp - j) dx) vs `build_eta_field`'s East = +j dx, negating
    # the recovered wavevector. This pure +-k ambiguity is resolved by the sign
    # anchor, so it is not corrected here.
    ds = MultiApertureArrays.from_field(eta, dx, water_depth_m, fs).compute(
        freqs=freqs, k_grid=k_grid, nu_grid=nu_grid,
        apertures=default_apertures(), n_staff=16, seed=20,
        solve_eta=eta_solve, reliability_gate=None, sign_anchor=anchor)
    # map ewdm dataset to M-dict convention; F(f,theta) per-rad->per-deg (main() reverses);
    # F(k,theta) and Q(nu,theta) already per-radian, passed through unchanged
    M = dict(Sf=np.asarray(ds['frequency_spectrum'].values, float),
             Fk=np.asarray(ds['wavenumber_spectrum'].values, float),
             Qn=np.asarray(ds['nu_spectrum'].values, float),
             Fft=np.asarray(ds['directional_spectrum_f'].values, float) * (np.pi / 180.0),
             Fkd=np.asarray(ds['directional_spectrum_k'].values, float),
             Qnd=np.asarray(ds['directional_spectrum_nu'].values, float),
             thbar=np.asarray(ds['mean_direction'].values, float),
             sigma=np.asarray(ds['directional_spread'].values, float),
             sign_ref=np.asarray(ds['sign_reference'].values, float),
             theta=np.asarray(ds['direction'].values, float),
             var_eta=float(ds['var_eta'].values))
    # scalar-per-frequency recolor of S(f) and F(f,theta); leaves D(f,theta), F(k), Q(nu) unchanged
    if recolor_fc is not None:
        ratio, _R = anchored_freq_recolor(eta_long, Zsw, fs, freqs,
                                          recolor_fc, recolor_band)
        M['Sf'] = M['Sf'] * ratio
        M['Fft'] = M['Fft'] * ratio[:, None]
    Fft, Fkd, Qnd = M['Fft'], M['Fkd'], M['Qnd']
    thbar, sigma, ref = M['thbar'], M['sigma'], M['sign_ref']

    # traditional CSM directional spectra (EMEP/BDM/MLM) and the single-point
    # EWDM triplet, on the same elevation field and direction grid
    extra = {}
    if csm_methods:
        from subroutines.traditional_directional_spectra import (
            traditional_directional_spectra)
        trad = traditional_directional_spectra(
            eta, dx, water_depth_m, fs, freqs, k_grid, M['theta'],
            Sf=M['Sf'], kscan=True, methods=csm_methods,
            sign_reference=M['thbar'])
        for name, r in trad.items():
            extra['Fft_' + name] = r['E_f']
            extra['Fkd_' + name] = r['Psi_k']
    extra['Fft_EWDM_triplet'] = _triplet_spectrum(eta, dx, fs, freqs, M['theta'],
                                                  M['Sf'])

    # onshore swell tiebreaker: flip the whole run 180 deg if its swell reads
    # offshore. Off unless EPSS_TIEBREAK=1
    sb = freqs < swell_cut
    if use_tiebreaker and M['Sf'][sb].sum() > swell_frac * M['Sf'].sum():
        th_sw = _ewmean(thbar[sb], M['Sf'][sb])
        if np.cos(np.radians(th_sw - onshore_dir)) < 0:
            roll = len(M['theta']) // 2
            Fft = np.roll(Fft, roll, axis=1)
            Fkd = np.roll(Fkd, roll, axis=1)
            Qnd = np.roll(Qnd, roll, axis=1)
            thbar = (thbar + 180.0) % 360.0
            ref = (ref + 180.0) % 360.0

    return run_ind, dict(Sf=M['Sf'], Fk=M['Fk'], Qn=M['Qn'], Fft=Fft, Fkd=Fkd,
                         Qnd=Qnd, thbar=thbar, sigma=sigma, sign_ref=ref,
                         theta=M['theta'], var_eta=float(M['var_eta']),
                         **extra)


def main():
    from pathlib import Path
    if Path(output_file).exists():
        print(f"File already exists: {output_file}")
        return
    nw = int(os.environ.get('EPSS_NWORKERS', min(8, (os.cpu_count() or 4) - 1)))
    print(f"Computing E-PSS multi-aperture directional spectra ({num_runs} runs, {nw} workers)...")
    results = {}
    with ProcessPoolExecutor(max_workers=nw) as ex:
        futs = {ex.submit(work, r): r for r in range(num_runs)}
        done = 0
        for fu in as_completed(futs):
            r, out = fu.result()
            done += 1
            if out is not None:
                results[r] = out
            if done % 25 == 0:
                print(f"  {done}/{num_runs}")

    th = results[next(iter(results))]['theta']
    nf, nd, nk, nn = len(freqs), len(th), len(k_grid), len(nu_grid)
    def nan_f4(*shape):
        return np.full(shape, np.nan, 'f4')

    Fft = nan_f4(nf, nd, num_runs)
    Fkd = nan_f4(nk, nd, num_runs)
    Qnd = nan_f4(nn, nd, num_runs)
    Sf = nan_f4(nf, num_runs)
    Fk = nan_f4(nk, num_runs)
    Qn = nan_f4(nn, num_runs)
    Tb = nan_f4(nf, num_runs)
    Sg = nan_f4(nf, num_runs)
    Rf = nan_f4(nf, num_runs)
    var = nan_f4(num_runs)
    # traditional CSM estimators and the EWDM triplet, one set per method
    meth_f = [m for m in list(csm_methods) + ['EWDM_triplet']]
    Fft_m = {m: nan_f4(nf, nd, num_runs) for m in meth_f}
    Fkd_m = {m: nan_f4(nk, nd, num_runs) for m in csm_methods}
    Tb_m = {m: nan_f4(nf, num_runs) for m in meth_f}
    Sg_m = {m: nan_f4(nf, num_runs) for m in meth_f}
    for r, o in results.items():
        Fft[:, :, r] = o['Fft']
        Fkd[:, :, r] = o['Fkd']
        Qnd[:, :, r] = o['Qnd']
        Sf[:, r] = o['Sf']
        Fk[:, r] = o['Fk']
        Qn[:, r] = o['Qn']
        Tb[:, r] = o['thbar']
        Sg[:, r] = o['sigma']
        Rf[:, r] = o['sign_ref']
        var[r] = o['var_eta']
        for m in meth_f:
            F = o.get('Fft_' + m)
            if F is not None:
                Fft_m[m][:, :, r] = F
                Tb_m[m][:, r], Sg_m[m][:, r] = _dir_moments(F, th)
        for m in csm_methods:
            F = o.get('Fkd_' + m)
            if F is not None:
                Fkd_m[m][:, :, r] = F

    # direction -> radians CW from true North; F(f,theta) per-deg -> per-rad [m^2/Hz/rad];
    # F_k_d, Q_nu_d already Bjorkqvist (2019) jacobian-removed per-radian form
    th = np.radians(th)
    Fft *= 180.0 / np.pi
    Tb, Sg, Rf = np.radians(Tb), np.radians(Sg), np.radians(Rf)
    # the CSM estimators and the triplet normalize over radians already, so only
    # their direction-valued outputs convert here
    for m in meth_f:
        Tb_m[m], Sg_m[m] = np.radians(Tb_m[m]), np.radians(Sg_m[m])

    out = nc.Dataset(output_file, 'w')
    out.Conventions = 'CF-1.10'
    out.title = 'ASIT 2019 E-PSS multi-aperture EWDM elevation directional wave spectra'
    out.institution = 'University of New Hampshire'
    out.source = ('ewdm.MultiApertureArrays multi-aperture wavelet directional method '
                  'on virtual-staff arrays seeded into camera slope-derived elevation '
                  'fields (build_eta_field: %s slope-projection long wave + g2s short '
                  'wave, depiston_n=%g; apertures=default_apertures, reliability_gate=None)'
                  % (longwave_method, depiston_n))
    out.references = 'Laxague et al., E-PSS (in prep); Bjorkqvist et al. (2019)'
    out.history = ('built by compute_all_directional_spectra.py; longwave_method=%s, '
                   'depiston_n=%g, slope_aperture=%s'
                   % (longwave_method, depiston_n, str(slope_aperture)))
    out.description = ('E-PSS multi-aperture elevation directional spectra; sign '
                       'resolved by the curvature-phase long-wave inversion; '
                       'Direction in radians CW from true North. Polar k/nu directional '
                       'spectra are the Bjorkqvist et al. (2019) jacobian-removed form: '
                       'S_f = int F_f_d dtheta, F_k = int k F_k_d dtheta, '
                       'Q_nu = int nu Q_nu_d dtheta (theta in radians).')
    out.createDimension('frequency', nf)
    out.createDimension('direction', nd)
    out.createDimension('wavenumber', nk)
    out.createDimension('inverse_phase_speed', nn)
    out.createDimension('run', num_runs)

    def V(name, dims, data, **att):
        v = out.createVariable(name, 'f4', dims, zlib=True, complevel=4)
        v[:] = data
        for k, vv in att.items():
            setattr(v, k, vv)
    V('frequency', ('frequency',), freqs, units='Hz')
    V('direction', ('direction',), th, units='radians clockwise from true North')
    V('wavenumber', ('wavenumber',), k_grid, units='rad/m')
    V('inverse_phase_speed', ('inverse_phase_speed',), nu_grid, units='s/m')
    V('F_f_d', ('frequency', 'direction', 'run'), Fft, units='m^2/Hz/rad')
    V('F_k_d', ('wavenumber', 'direction', 'run'), Fkd, units='m^4/rad')
    V('Q_nu_d', ('inverse_phase_speed', 'direction', 'run'), Qnd, units='m^4/(s^2 rad)')
    V('S_f', ('frequency', 'run'), Sf, units='m^2/Hz')
    V('F_k', ('wavenumber', 'run'), Fk, units='m^2/(rad/m)')
    V('Q_nu', ('inverse_phase_speed', 'run'), Qn, units='m^2/(s/m)')
    V('mean_direction', ('frequency', 'run'), Tb, units='radians clockwise from true North')
    V('directional_spread', ('frequency', 'run'), Sg, units='radians')
    V('sign_reference', ('frequency', 'run'), Rf, units='radians clockwise from true North')
    V('variance', ('run',), var, units='m^2')
    # traditional CSM estimators (EMEP/BDM/MLM) and the single-point EWDM
    # triplet: same grids and conventions, method name appended
    for m in meth_f:
        V('F_f_d_' + m, ('frequency', 'direction', 'run'), Fft_m[m],
          units='m^2/Hz/rad', long_name='Directional wave spectrum (%s)' % m)
        V('mean_direction_' + m, ('frequency', 'run'), Tb_m[m],
          units='radians clockwise from true North')
        V('directional_spread_' + m, ('frequency', 'run'), Sg_m[m],
          units='radians')
    for m in csm_methods:
        V('F_k_d_' + m, ('wavenumber', 'direction', 'run'), Fkd_m[m],
          units='m^4/rad',
          long_name='Directional wavenumber spectrum, native scan (%s)' % m)
    # time coordinate from the 0-189 chronological slope-field source
    fld = nc.Dataset(slope_field_file)
    if 'time' in fld.variables:
        tv = out.createVariable('time', 'f8', ('run',))
        tv[:] = fld['time'][:]
        tv.standard_name = 'time'
        tv.units = 'seconds since 1970-01-01 00:00:00'
        tv.calendar = 'standard'
        tv.axis = 'T'
    fld.close()
    out.close()
    print(f"Done. Wrote {output_file} ({len(results)}/{num_runs} runs; renumbered 0-189).")


if __name__ == '__main__':
    main()

"""
Compute per-run E-PSS multi-aperture elevation directional spectra (f, k, nu)
from the earth-referenced PSS slope fields. Slope fields -> camera elevation
field (small-aperture Krogstad long wave + per-frame g2s) -> multi-aperture
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

path = '../_data/'
# input slope fields and output spectra are overridable via env vars (defaults
# unchanged) so a re-normalized slope-field run can be produced alongside the original
slope_field_file = path + os.environ.get('EPSS_FLD', 'ASIT2019_slope_fields_reduced.nc')
output_file = path + os.environ.get('EPSS_OUT', 'ASIT2019_EPSS_directional_spectra.nc')
fs, water_depth_m, num_samples, num_runs = 10.0, 15.0, 6000, 190
# de-piston cut wavelength as a multiple of the frame size (n*L); larger n = more
# aggressive de-piston. Krogstad long-wave slope averaged over a centered disc of
# this diameter [px]. Overridable per sweep member.
depiston_n = float(os.environ.get('EPSS_DEPISTON_N', 2.0))
krog_disc = int(os.environ.get('EPSS_KROG_DISC', 32))

# fixed grids (dx constant: 2.915 m / 32 px)
dx = 2.915 / 32
freqs = np.logspace(np.log10(0.035), np.log10(3.5), 64)
k_grid = 2.0**np.linspace(np.log2(0.01), np.log2(np.pi / dx), 80)
nu_grid = 2.0**np.linspace(np.log2(0.005), np.log2(2.0), 80)

# onshore swell tiebreaker: 3 km south of Martha's Vineyard, so long swell must
# propagate onshore (northward); flip a swell-dominated run reading offshore.
onshore_dir = 0.0
swell_cut = 0.16
swell_frac = 0.15

_DS = {}


def _ds():
    if not _DS:
        _DS['fld'] = nc.Dataset(slope_field_file)
        _DS['ref'] = nc.Dataset(path + 'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc')
    return _DS


def _ewmean(deg, w):
    a = np.radians(deg)
    return np.degrees(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum()))


def work(run_ind):
    from multiaperture import (build_eta_field, default_apertures,
                               multiaperture_spectra, sftheta_sign_anchor)
    d = _ds()
    se = np.ma.filled(d['fld']['slope_east'][run_ind][..., :num_samples], np.nan)
    sn = np.ma.filled(d['fld']['slope_north'][run_ind][..., :num_samples], np.nan)
    if not np.isfinite(se).any():                        # NaN-flagged (corrupt) run
        return run_ind, None
    # gated de-piston: build_eta_field also returns the solve field with the
    # uniform long-wave piston removed above 0.75*f_FOV, so the FOV-scale |k|
    # solve is not biased low. slope_blend off; lo_frac/apertures from defaults.
    eta, _, eta_solve = build_eta_field(np.nan_to_num(se).astype(float),
                                        np.nan_to_num(sn).astype(float),
                                        water_depth_m, fs,
                                        krog_disc=krog_disc, depiston_n=depiston_n)
    M = multiaperture_spectra(eta, dx, freqs, k_grid, nu_grid, water_depth_m, fs,
                              apertures=default_apertures(), n_staff=16,
                              solve_eta=eta_solve,
                              sign_anchor=sftheta_sign_anchor(d['ref'], run_ind))
    Fft, Fkd, Qnd = M['Fft'], M['Fkd'], M['Qnd']
    thbar, sigma, ref = M['thbar'], M['sigma'], M['sign_ref']

    # onshore swell tiebreaker (flip the whole run 180 deg if its swell is offshore)
    sb = freqs < swell_cut
    if M['Sf'][sb].sum() > swell_frac * M['Sf'].sum():
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
                         theta=M['theta'], var_eta=float(M['var_eta']))


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
            r, out = fu.result(); done += 1
            if out is not None:
                results[r] = out
            if done % 25 == 0:
                print(f"  {done}/{num_runs}")

    th = results[next(iter(results))]['theta']
    nf, nd, nk, nn = len(freqs), len(th), len(k_grid), len(nu_grid)
    nan = lambda *s: np.full(s, np.nan, 'f4')
    Fft = nan(nf, nd, num_runs); Fkd = nan(nk, nd, num_runs); Qnd = nan(nn, nd, num_runs)
    Sf = nan(nf, num_runs); Fk = nan(nk, num_runs); Qn = nan(nn, num_runs)
    Tb = nan(nf, num_runs); Sg = nan(nf, num_runs); Rf = nan(nf, num_runs); var = nan(num_runs)
    for r, o in results.items():
        Fft[:, :, r] = o['Fft']; Fkd[:, :, r] = o['Fkd']; Qnd[:, :, r] = o['Qnd']
        Sf[:, r] = o['Sf']; Fk[:, r] = o['Fk']; Qn[:, r] = o['Qn']
        Tb[:, r] = o['thbar']; Sg[:, r] = o['sigma']; Rf[:, r] = o['sign_ref']; var[r] = o['var_eta']

    out = nc.Dataset(output_file, 'w')
    out.description = ('E-PSS multi-aperture elevation directional spectra; sign '
                       'resolved by S_f_theta 3-D-FFT anchor + onshore swell tiebreaker')
    out.createDimension('frequency', nf); out.createDimension('direction', nd)
    out.createDimension('wavenumber', nk); out.createDimension('inverse_phase_speed', nn)
    out.createDimension('run', num_runs)

    def V(name, dims, data, **att):
        v = out.createVariable(name, 'f4', dims, zlib=True, complevel=4); v[:] = data
        for k, vv in att.items():
            setattr(v, k, vv)
    V('frequency', ('frequency',), freqs, units='Hz')
    V('direction', ('direction',), th, units='degrees clockwise from true North')
    V('wavenumber', ('wavenumber',), k_grid, units='rad/m')
    V('inverse_phase_speed', ('inverse_phase_speed',), nu_grid, units='s/m')
    V('F_f_d', ('frequency', 'direction', 'run'), Fft, units='m^2/Hz/rad')
    V('F_k_d', ('wavenumber', 'direction', 'run'), Fkd, units='m^2/(rad/m)/rad')
    V('Q_nu_d', ('inverse_phase_speed', 'direction', 'run'), Qnd, units='m^2/(s/m)/rad')
    V('S_f', ('frequency', 'run'), Sf, units='m^2/Hz')
    V('F_k', ('wavenumber', 'run'), Fk, units='m^2/(rad/m)')
    V('Q_nu', ('inverse_phase_speed', 'run'), Qn, units='m^2/(s/m)')
    V('mean_direction', ('frequency', 'run'), Tb, units='deg CW-from-N')
    V('directional_spread', ('frequency', 'run'), Sg, units='deg')
    V('sign_reference', ('frequency', 'run'), Rf, units='deg CW-from-N')
    V('variance', ('run',), var, units='m^2')
    out.close()
    print(f"Done. Wrote {output_file} ({len(results)} runs; 137-140 NaN-flagged).")


if __name__ == '__main__':
    main()

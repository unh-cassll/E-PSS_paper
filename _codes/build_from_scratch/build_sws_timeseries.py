"""
Assemble the per-run npz outputs of compute_short_wave_spectra_stats.py into
timeseries .nc files mirroring the published
ASIT2019_wave_spectra_stats_timeseries_{no,lab,empirical}_gain.nc layout, with a
filename suffix so the recomputed dataset can be compared against the published
one before either is adopted. Runs without an npz (no Stokes file available)
are left as NaN fill.

Env:
  EPSS_SWS_OUTDIR   npz dir (default ../_data/_sws_intermediate/)
  EPSS_SWS_SUFFIX   output-name appendage (default '_recomputed')

@author: nathanlaxague
"""
import os
import glob
import re
import datetime as dt

import numpy as np
import netCDF4 as nc

DATA = '../_data/'
NPZDIR = os.environ.get('EPSS_SWS_OUTDIR', DATA + '_sws_intermediate/')
SUFFIX = os.environ.get('EPSS_SWS_SUFFIX', '_recomputed')
GAINS = ['no_gain', 'lab_gain', 'empirical_gain']
NRUN = 190


def main():
    npzs = {}
    for f in glob.glob(NPZDIR + 'sws_run*.npz'):
        m = re.match(r'sws_run(\d+)\.npz$', os.path.basename(f))
        if m:
            npzs[int(m.group(1))] = f
    if not npzs:
        raise SystemExit('no sws_run*.npz found in ' + NPZDIR)
    stats_npzs = {}
    for f in glob.glob(NPZDIR + 'sws_stats_run*.npz'):
        m = re.match(r'sws_stats_run(\d+)\.npz$', os.path.basename(f))
        if m:
            stats_npzs[int(m.group(1))] = f
    all_runs = sorted(set(npzs) | set(stats_npzs))
    print('assembling %d recomputed runs (of %d): %d with spectra, %d with stats'
          % (len(all_runs), NRUN, len(npzs), len(stats_npzs)))

    ref = np.load(next(iter(npzs.values())))
    f_Hz = ref['empirical_gain__f']
    k_rad_m = ref['empirical_gain__k']
    nu_s_m = ref['empirical_gain__nu']
    theta_rad = ref['empirical_gain__theta']
    slope_centers = np.linspace(-1.0, 1.0, 200)

    sup = nc.Dataset(DATA + 'ASIT2019_supporting_environmental_observations.nc')
    tvals = np.asarray(sup['time'][:], float)
    sup.close()

    for gain in GAINS:
        out_name = DATA + 'ASIT2019_wave_spectra_stats_timeseries_%s%s.nc' % (gain, SUFFIX)
        out = nc.Dataset(out_name, 'w')
        out.Conventions = 'CF-1.10'
        out.title = ('ASIT 2019 E-PSS wave spectra and slope statistics timeseries '
                     '(%s), recomputed from the raw Stokes archive' % gain)
        out.institution = 'University of New Hampshire'
        out.source = ('compute_short_wave_spectra_stats.py: pss present-path inversion '
                      '+ slopespectra 3-D FFT directional sub-spectra (512^2, 30 Hz, '
                      '600-frame segments); %d/%d runs recomputed, others fill'
                      % (len(npzs), NRUN))
        out.history = 'built %s by build_sws_timeseries.py' % dt.date.today().isoformat()

        out.createDimension('run', NRUN)
        out.createDimension('YMDHMS', 6)
        out.createDimension('wavenumbers', len(k_rad_m))
        out.createDimension('frequencies', len(f_Hz))
        out.createDimension('inverse phase speeds', len(nu_s_m))
        out.createDimension('directions', len(theta_rad))
        out.createDimension('slope centers', len(slope_centers))
        out.createDimension('samples', 6000)

        def V(name, dims, data=None, **att):
            v = out.createVariable(name, 'f8', dims, zlib=True, complevel=4,
                                   fill_value=np.nan)
            if data is not None:
                v[:] = data
            for kk, vv in att.items():
                setattr(v, kk, vv)
            return v

        V('k_rad_m', ('wavenumbers',), k_rad_m, units='rad/m')
        V('f_Hz', ('frequencies',), f_Hz, units='Hz')
        V('nu_s_m', ('inverse phase speeds',), nu_s_m, units='s/m')
        V('theta_rad', ('directions',), theta_rad,
          units='radians clockwise from true North')
        V('slope_centers', ('slope centers',), slope_centers)
        dv = V('DateVector', ('YMDHMS', 'run'))
        se = V('slope_east', ('run', 'samples'))
        sn = V('slope_north', ('run', 'samples'))
        hh = V('slope_histogram_crosswind_upwind', ('run', 'slope centers', 'slope centers'))
        skt = V('S_k_theta', ('run', 'directions', 'wavenumbers'))
        sft = V('S_f_theta', ('run', 'directions', 'frequencies'))
        qnt = V('Qs_nu_theta', ('run', 'directions', 'inverse phase speeds'))
        mc = V('mss_crosswind', ('run',))
        mu = V('mss_upwind', ('run',))
        tv = V('time', ('run',), tvals, standard_name='time',
               units='seconds since 1970-01-01 00:00:00', calendar='standard', axis='T')
        tv.axis = 'T'

        for run in all_runs:
            p = gain + '__'
            z = np.load(npzs[run]) if run in npzs else None
            # gain-subset npzs (EPSS_SWS_GAINS passes) may lack this gain: fill
            if z is not None and p + 'S_f_theta' in z.files:
                sft[run] = z[p + 'S_f_theta']
                skt[run] = z[p + 'S_k_theta']
                qnt[run] = z[p + 'Qs_nu_theta']
                se[run] = z[p + 'slope_east']
                sn[run] = z[p + 'slope_north']
            # stats: prefer the every-frame stats-only recompute; stats-only runs
            # (no spectra npz yet) still get their mss/histogram records
            zs = np.load(stats_npzs[run]) if run in stats_npzs else z
            if zs is not None and p + 'mss_upwind' in zs.files:
                mu[run] = zs[p + 'mss_upwind']
                mc[run] = zs[p + 'mss_crosswind']
                hh[run] = zs[p + 'slope_histogram_crosswind_upwind']
            t = dt.datetime.fromtimestamp(float(tvals[run]), dt.timezone.utc)
            dv[:, run] = [t.year, t.month, t.day, t.hour, t.minute, t.second]
        out.close()
        print('wrote', out_name)


if __name__ == '__main__':
    main()

"""
End-to-end build of the ASIT2019 canonical products that are not produced by
any other script in this repository and are normally obtained from the
published Zenodo archive (see grab_observational_data.py):

  - ASIT2019_wave_spectra_stats_timeseries_{no,lab,empirical}_gain.nc
  - ASIT2019_slope_fields_reduced.nc

Assumes every raw Stokes .nc file (S0/S1/S2, one per acquisition run) lives
in a single local directory. Reduces each run's frames with
polarimetric-slope-sensing (present-geometry inversion) and folds the
result into slopespectra directional sub-spectra, per-run mean square
slope and slope-histogram statistics, a 10 Hz FOV-mean slope_east/north
series (all three gains), and a 10 Hz 32x32 invert-then-average
earth-referenced slope field (empirical gain).

Per-run results are cached to disk (EPSS_SWS_OUTDIR) so a multi-hour run
can be interrupted and resumed; delete a run's cache file to force a
recompute.

Env:
  EPSS_STOKES_DIR   directory of source Stokes .nc files
  EPSS_SWS_RUNS     comma list of run indices to process (default: all
                    matched runs)
  EPSS_SWS_GAINS    comma list of gain keys to compute (default all three)
  EPSS_SWS_OUTDIR   per-run cache dir (default ../_data/_sws_intermediate/)
  SLOPESPECTRA_FFT_WORKERS  FFT thread count (see slopespectra.spectrum)

@author: nathanlaxague
"""
import os
import datetime as dt

import numpy as np
import netCDF4 as nc

# runnable from _codes/build_from_scratch/: add _codes/ for subroutines + siblings
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compute_short_wave_spectra_stats as sws
from subroutines.utils import N_PX, DX_M, NUM_SAMPLES, NUM_RUNS

REDUCED_GAIN = 'empirical_gain'
BLOCK = sws.FRAMESIZE // N_PX
# EPSS_CANON_SUFFIX: appended to every output .nc name (default '' writes the
# canonical production names); set it to write a separate set for comparison
# without overwriting the production dataset.
SUFFIX = os.environ.get('EPSS_CANON_SUFFIX', '')
# EPSS_CANON_EVERY_FRAME_STATS: default '1' runs the every-frame slope-statistics
# pass (exact mss / slope histograms from all frames) so the timeseries matches
# the canonical; '0' falls back to the subsampled statistics from the spectra
# pass (faster, noisier).
EVERY_FRAME_STATS = os.environ.get('EPSS_CANON_EVERY_FRAME_STATS', '1') == '1'
# EPSS_CANON_NO_WRITE='1': process runs into the per-run npz cache but skip the
# final timeseries / reduced-field .nc write. Use it for parallel workers over
# disjoint run subsets, then a single default invocation to assemble.
NO_WRITE = os.environ.get('EPSS_CANON_NO_WRITE', '0') == '1'


def block_mean(a):
    """(FRAMESIZE, FRAMESIZE) -> (N_PX, N_PX) boxcar downsample."""
    ny, nx = a.shape
    f = BLOCK
    return a[:ny // f * f, :nx // f * f].reshape(
        ny // f, f, nx // f, f).mean((1, 3))


def process_run_full(path, Wdir_deg, max_frames=0, gains=None):
    """One Stokes file's spectra, stats, and (empirical-gain) reduced slope
    field, in a single pass over the frames.

    Returns (spectra_by_gain, reduced_slope_east, reduced_slope_north);
    spectra_by_gain matches sws.process_run's return value (S_f_theta,
    S_k_theta, Qs_nu_theta, mss_upwind, mss_crosswind,
    slope_histogram_crosswind_upwind, slope_east, slope_north, estimator).
    """
    lut = sws.build_lookup_table(sws.N_WATER)
    peak = float(lut[1].max())
    ds = nc.Dataset(path)
    ds.set_auto_maskandscale(False)
    transpose, scale = sws._stokes_read_convention(ds)
    n = ds.dimensions['frames'].size
    if max_frames:
        n = min(n, max_frames)

    sub = list(range(0, n, max(1, n // 12)))
    dref = float(np.nanmedian([np.sqrt(
        (ds['S1'][i].astype(np.float64) * scale) ** 2 +
        (ds['S2'][i].astype(np.float64) * scale) ** 2) for i in sub]))

    gkeys = list(gains) if gains else list(sws.GAINS)
    hif_gain = 'empirical_gain' if 'empirical_gain' in gkeys else gkeys[0]
    seg_buf = {g: (np.empty((sws.FRAMESIZE, sws.FRAMESIZE, sws.TSEG), np.float32),
                   np.empty((sws.FRAMESIZE, sws.FRAMESIZE, sws.TSEG), np.float32))
              for g in gkeys}
    acc = {g: None for g in gkeys}
    slope_en = {g: (np.full(sws.NT_10HZ, np.nan), np.full(sws.NT_10HZ, np.nan))
               for g in gkeys}
    su_acc = {g: [] for g in gkeys}
    sc_acc = {g: [] for g in gkeys}
    hifr = []
    reduced_cross = np.full((N_PX, N_PX, NUM_SAMPLES), np.nan, np.float32)
    reduced_along = np.full((N_PX, N_PX, NUM_SAMPLES), np.nan, np.float32)
    seg = 0
    j10 = 0

    for c0 in range(0, n, sws.CHUNK_RAW):
        b1 = ds['S1'][c0:c0 + sws.CHUNK_RAW].astype(np.float64) * scale
        b2 = ds['S2'][c0:c0 + sws.CHUNK_RAW].astype(np.float64) * scale
        for off in range(b1.shape[0]):
            i = c0 + off
            s1 = b1[off].T if transpose else b1[off]
            s2 = b2[off].T if transpose else b2[off]
            for g in gkeys:
                cross, along, hf = sws._slopes_present(s1, s2, sws.GAINS[g],
                                                        lut, peak, dref)
                seg_buf[g][0][:, :, seg] = np.nan_to_num(cross)
                seg_buf[g][1][:, :, seg] = np.nan_to_num(along)
                if i % sws.STR_10HZ == 0 and j10 < sws.NT_10HZ:
                    sE, sN = sws._earth_components(np.nanmean(cross), np.nanmean(along))
                    slope_en[g][0][j10] = sE
                    slope_en[g][1][j10] = sN
                    if g == REDUCED_GAIN:
                        reduced_cross[:, :, j10] = block_mean(np.nan_to_num(cross))
                        reduced_along[:, :, j10] = block_mean(np.nan_to_num(along))
                if i % (sws.STR_10HZ * 25) == 0:
                    sE, sN = sws._earth_components(cross, along)
                    sup, scw = sws._wind_components(sE, sN, Wdir_deg)
                    su_acc[g].append(sup.astype(np.float32))
                    sc_acc[g].append(scw.astype(np.float32))
                    if g == hif_gain:
                        hifr.append(hf)
            if i % sws.STR_10HZ == 0:
                j10 += 1
            seg += 1
            if seg == sws.TSEG:
                for g in gkeys:
                    power, framesize, s3 = sws._segment_power(*seg_buf[g])
                    if acc[g] is None:
                        acc[g] = dict(n=0, cube=np.zeros(power.shape),
                                     framesize=framesize, s3=s3)
                    acc[g]['cube'] += power
                    acc[g]['n'] += 1
                seg = 0
    ds.close()

    nu_nat, dnu_nat = sws._inverse_phase_speed_grid()[:2]
    hf = np.asarray(hifr)
    thr = np.median(hf) + 4.0 * 1.4826 * np.median(np.abs(hf - np.median(hf)))
    keep = np.flatnonzero(hf <= thr)
    out = {}
    for g in gkeys:
        a = acc[g]
        spec = sws.finish_spectrum(a['cube'] / max(a['n'], 1), a['framesize'],
                                   sws.DX_FULL, a['s3'], sws.FS_RAW)
        ds_g = sws.compute_sub_spectra(spec.Skf, spec.dk, spec.df,
                                       heading_deg=sws.CAMERA_AZ, dtheta=sws.DTHETA)
        su = np.concatenate([su_acc[g][j].ravel() for j in keep])
        sc = np.concatenate([sc_acc[g][j].ravel() for j in keep])
        ok = np.isfinite(su) & np.isfinite(sc)
        su, sc = su[ok] - su[ok].mean(), sc[ok] - sc[ok].mean()
        H, _, _ = np.histogram2d(sc, su, bins=[sws._EDGES, sws._EDGES])
        db = sws.SLOPE_CENTERS[1] - sws.SLOPE_CENTERS[0]
        Hn = H / (H.sum() * db * db)
        Sft = sws._theta_remap(np.nan_to_num(ds_g['S_f_theta'].values), axis=0)
        Skt = sws._theta_remap(np.nan_to_num(ds_g['S_k_theta'].values), axis=1).T
        Qnt = sws._theta_remap(sws._rebin_nu(np.nan_to_num(ds_g['Qs_nu_theta'].values),
                                             nu_nat, dnu_nat), axis=1).T
        out[g] = dict(
            S_f_theta=Sft, S_k_theta=Skt, Qs_nu_theta=Qnt,
            k=ds_g['k'].values, f=ds_g['f'].values, nu=sws.NU_PUB,
            theta=ds_g['theta'].values, estimator=sws.ESTIMATOR_TAG,
            mss_upwind=float(np.var(su)), mss_crosswind=float(np.var(sc)),
            slope_histogram_crosswind_upwind=Hn,
            slope_east=slope_en[g][0], slope_north=slope_en[g][1])

    red_e, red_n = sws._earth_components(reduced_cross, reduced_along)
    return out, red_e, red_n


def write_timeseries(npzs, stats_npzs, tvals, out_dir):
    """ASIT2019_wave_spectra_stats_timeseries_{gain}.nc, one per gain, all
    NUM_RUNS runs (fill where a run's npz is missing)."""
    ref = np.load(next(iter(npzs.values())))
    f_Hz = ref['empirical_gain__f']
    k_rad_m = ref['empirical_gain__k']
    nu_s_m = ref['empirical_gain__nu']
    theta_rad = ref['empirical_gain__theta']
    slope_centers = np.linspace(-1.0, 1.0, 200)
    all_runs = sorted(set(npzs) | set(stats_npzs))

    for gain in sws.GAINS:
        out_name = out_dir + 'ASIT2019_wave_spectra_stats_timeseries_%s%s.nc' % (gain, SUFFIX)
        out = nc.Dataset(out_name, 'w')
        out.Conventions = 'CF-1.10'
        out.title = ('ASIT 2019 E-PSS wave spectra and slope statistics '
                     'timeseries (%s)' % gain)
        out.institution = 'University of New Hampshire'
        out.source = ('build_canonical_from_stokes.py: pss present-path '
                      'inversion + slopespectra directional sub-spectra '
                      '(512^2, 30 Hz, 600-frame segments)')
        out.history = 'built %s by build_canonical_from_stokes.py' % dt.date.today().isoformat()

        out.createDimension('run', NUM_RUNS)
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
            if z is not None and p + 'S_f_theta' in z.files:
                sft[run] = z[p + 'S_f_theta']
                skt[run] = z[p + 'S_k_theta']
                qnt[run] = z[p + 'Qs_nu_theta']
                se[run] = z[p + 'slope_east']
                sn[run] = z[p + 'slope_north']
            zs = np.load(stats_npzs[run]) if run in stats_npzs else z
            if zs is not None and p + 'mss_upwind' in zs.files:
                mu[run] = zs[p + 'mss_upwind']
                mc[run] = zs[p + 'mss_crosswind']
                hh[run] = zs[p + 'slope_histogram_crosswind_upwind']
            t = dt.datetime.fromtimestamp(float(tvals[run]), dt.timezone.utc)
            dv[:, run] = [t.year, t.month, t.day, t.hour, t.minute, t.second]
        out.close()
        print('wrote', out_name)


def write_slope_fields_reduced(reduced_npzs, tvals, out_dir):
    """ASIT2019_slope_fields_reduced.nc: 10 Hz, N_PX x N_PX earth-referenced
    invert-then-average slope field, empirical gain, all NUM_RUNS runs."""
    out_name = out_dir + 'ASIT2019_slope_fields_reduced%s.nc' % SUFFIX
    out = nc.Dataset(out_name, 'w')
    out.Conventions = 'CF-1.10'
    out.title = 'ASIT 2019 E-PSS reduced (%dx%d) earth-referenced slope fields' % (N_PX, N_PX)
    out.institution = 'University of New Hampshire'
    out.source = ('build_canonical_from_stokes.py: pss present-path inversion, '
                  '%d-gain, invert-then-average to %dx%d, 10 Hz' % (1, N_PX, N_PX))
    out.history = 'built %s by build_canonical_from_stokes.py' % dt.date.today().isoformat()

    out.createDimension('run', NUM_RUNS)
    out.createDimension('YMDHMS', 6)
    out.createDimension('y', N_PX)
    out.createDimension('x', N_PX)
    out.createDimension('samples', NUM_SAMPLES)

    def V(name, dims, data=None, **att):
        v = out.createVariable(name, 'f8', dims, zlib=True, complevel=4,
                               fill_value=np.nan)
        if data is not None:
            v[:] = data
        for kk, vv in att.items():
            setattr(v, kk, vv)
        return v

    V('dx_m', (), DX_M, units='m', long_name='reduced-grid pixel size')
    dv = V('DateVector', ('YMDHMS', 'run'))
    se = V('slope_east', ('run', 'y', 'x', 'samples'))
    sn = V('slope_north', ('run', 'y', 'x', 'samples'))
    tv = V('time', ('run',), tvals, standard_name='time',
           units='seconds since 1970-01-01 00:00:00', calendar='standard', axis='T')
    tv.axis = 'T'

    for run in sorted(reduced_npzs):
        z = np.load(reduced_npzs[run])
        se[run] = z['slope_east']
        sn[run] = z['slope_north']
        t = dt.datetime.fromtimestamp(float(tvals[run]), dt.timezone.utc)
        dv[:, run] = [t.year, t.month, t.day, t.hour, t.minute, t.second]
    out.close()
    print('wrote', out_name)


def main():
    tbl = sws.build_run_table()
    maxf = int(os.environ.get('EPSS_SWS_MAXFRAMES', '0'))
    runs = os.environ.get('EPSS_SWS_RUNS')
    runs = [int(x) for x in runs.split(',')] if runs else sorted(tbl)
    gsel = os.environ.get('EPSS_SWS_GAINS')
    gsel = [g.strip() for g in gsel.split(',')] if gsel else list(sws.GAINS)

    os.makedirs(sws.OUTDIR, exist_ok=True)
    npzs, stats_npzs, reduced_npzs = {}, {}, {}
    for run in runs:
        if run not in tbl:
            print('run %d: no Stokes file matched, skip' % run); continue
        path, wdir = tbl[run]
        dest = sws.OUTDIR + 'sws_run%d.npz' % run
        reduced_dest = sws.OUTDIR + 'reduced_field_run%d.npz' % run
        have_spectra = os.path.exists(dest) and all(
            f'{g}__estimator' in np.load(dest).files for g in gsel)
        have_reduced = os.path.exists(reduced_dest)
        if have_spectra and have_reduced:
            print('run %d: cached, skip' % run)
        else:
            print('run %d: %s' % (run, os.path.basename(path)), flush=True)
            got, red_e, red_n = process_run_full(path, wdir, max_frames=maxf, gains=gsel)
            out = {f'{g}__{k}': v for g, d in got.items() for k, v in d.items()}
            if os.path.exists(dest):
                prev = dict(np.load(dest)); prev.update(out); out = prev
            tmp = dest + '.tmp.npz'
            np.savez(tmp, **out); os.replace(tmp, dest)
            np.savez(reduced_dest, slope_east=red_e, slope_north=red_n)
            print('  done -> %s, %s' % (dest, reduced_dest), flush=True)
        npzs[run] = dest
        # every-frame slope statistics (exact mss / histograms) so the
        # timeseries mss/hist match the canonical rather than the spectra
        # pass's subsampled estimate
        stats_npzs_path = sws.OUTDIR + 'sws_stats_run%d.npz' % run
        if EVERY_FRAME_STATS and not os.path.exists(stats_npzs_path):
            st = sws.process_run_stats(path, wdir, max_frames=maxf)
            tmp = stats_npzs_path + '.tmp.npz'
            np.savez(tmp, **{f'{g}__{k}': v for g, d in st.items() for k, v in d.items()})
            os.replace(tmp, stats_npzs_path)
            print('  every-frame stats -> %s' % stats_npzs_path, flush=True)
        if os.path.exists(stats_npzs_path):
            stats_npzs[run] = stats_npzs_path
        reduced_npzs[run] = reduced_dest

    if NO_WRITE:
        print('EPSS_CANON_NO_WRITE=1: per-run npzs written, skipping .nc assembly')
        return
    sup = nc.Dataset(sws.DATA + 'ASIT2019_supporting_environmental_observations.nc')
    tvals = np.asarray(sup['time'][:], float)
    sup.close()
    write_timeseries(npzs, stats_npzs, tvals, sws.DATA)
    write_slope_fields_reduced(reduced_npzs, tvals, sws.DATA)


if __name__ == '__main__':
    main()

"""
Builds the ASIT2019 wave-spectra-stats timeseries files
(ASIT2019_wave_spectra_stats_timeseries_{no,lab,empirical}_gain.nc) from the
raw 512^2 Stokes archive, using polarimetric-slope-sensing (pss) for the
present-geometry slope inversion and slopespectra for the directional
sub-spectra. Present-geometry inversion only (no seapol / sky-aware path).

Per run and per gain treatment {none, lab, empirical}, matched to the
published file conventions:
  - directional slope spectra S_f_theta (theta,f), S_k_theta (theta,k),
    Qs_nu_theta (theta,nu), from 512^2, 30 Hz, 600-frame (df=0.05 Hz)
    segments averaged over the record (framesize 512 -> k[256], 5-deg bins
    -> theta[72], dnu_base 0.05 -> nu[80], fs 30 -> f[300] to 15 Hz).
  - mean square slope mss_upwind/mss_crosswind and the 200x200 wind-rotated
    slope histogram (slope_centers = linspace(-1,1,200)).
  - 10 Hz FOV-mean slope_east/slope_north series (6000 samples; every 3rd
    30-Hz frame).

Every segment's directional power is accumulated in the unshifted, unmasked
FFT layout and averaged before the Nyquist-circle mask, physical
normalization, and polar projection are applied once per run
(slopespectra.compute_segment_power / finish_spectrum), in float32. This
cuts per-run cost roughly in half relative to running the full per-segment
pipeline (slopespectra.compute_wave_spectra) once per 600-frame segment.

Assumes every source Stokes file lives under EPSS_STOKES_DIR. The filename
UTC stamp is matched to each run's timestamp in the supporting-environment
file to assign the 0-189 run index.

Env:
  EPSS_STOKES_DIR    colon-separated list of directories holding the raw
                     Stokes .nc archive (default /mnt/DATA/get_Stoked/)
  EPSS_SWS_RUNS      comma list of run indices to process (default: all
                     matched)
  EPSS_SWS_GAINS     comma list of gain keys to compute for the spectra pass
                     (default all three); per-run npz caches merge across
                     gain-subset invocations
  EPSS_SWS_STATS_ONLY  1 -> recompute mean square slope / slope PDFs from
                     every frame instead of the spectra pass
  EPSS_SWS_OUTDIR    per-run npz cache dir (default ../_data/_sws_intermediate/)
  EPSS_SWS_MAXFRAMES  cap frames read per run (0 = no cap)
  SLOPESPECTRA_FFT_WORKERS  FFT thread count (see slopespectra.spectrum)

@author: nathanlaxague
"""
import os
import re
import glob
import datetime as dt

import numpy as np
import netCDF4 as nc

from slopespectra import circular_tukey, compute_segment_power, finish_spectrum
from slopespectra.subspectra import compute_sub_spectra, _inverse_phase_speed_grid
from pss.fresnel import build_lookup_table, dolp_to_aoi
from pss.gain import apply_gain, DEFAULT_LAB_GAIN

# runnable from _codes/build_from_scratch/: add _codes/ for subroutines
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from subroutines.utils import L_FOV_M, WATER_DEPTH_M

import warnings
warnings.filterwarnings("ignore")

DATA = '../_data/'
STOKES_DIR = os.environ.get('EPSS_STOKES_DIR', '/mnt/DATA/get_Stoked/')
OUTDIR = os.environ.get('EPSS_SWS_OUTDIR', DATA + '_sws_intermediate/')

# ---- geometry / grid config (matched to the published timeseries files) ----
THETA_V, N_WATER, CAMERA_AZ = 30.0, 1.34, 190.0
FRAMESIZE = 512
DX_FULL = L_FOV_M / FRAMESIZE
FS_RAW = 30.0                     # raw Stokes frame rate; spectra computed at 30 Hz
TSEG = 600                        # frames per spectral segment -> df = 30/600 = 0.05 Hz
DTHETA = np.deg2rad(5.0)          # -> 72 directions
STR_10HZ = 3                      # every 3rd 30-Hz frame -> 10 Hz slope_east/north series
NT_10HZ = 6000
CHUNK_RAW = 600                   # contiguous raw frames per bulk read
SLOPE_CENTERS = np.linspace(-1.0, 1.0, 200)
_EDGES = np.concatenate(([-np.inf],
                         0.5 * (SLOPE_CENTERS[1:] + SLOPE_CENTERS[:-1]), [np.inf]))
GAINS = {'no_gain': 'none', 'lab_gain': 'lab', 'empirical_gain': 'empirical'}

# Earth-axis convention: the raw on-disk Stokes frame is stored (columns,
# rows); transposed to (rows, cols) = (y, x) before slope inversion. The
# polarimetric 'cross' component then lies along the image column axis
# (camera look azimuth + 90 deg) and 'along' lies along look azimuth + 180
# deg (positive slope tilts toward the camera).
_ROW_AX = np.deg2rad(CAMERA_AZ % 360.0)
_COL_AX = _ROW_AX + np.pi / 2.0
# Directional-spectrum theta label remap: phi_pub[j] = raw bin at angle
# (THETA_C - 5*j) mod 360, matching the published compass CW-from-N
# convention.
THETA_C = 200.0

# published uniform inverse-phase-speed grid (0.05:0.05:4, 80 bins); the
# native piecewise grid is variance-conserving-rebinned onto it
NU_PUB = np.arange(0.05, 4.0 + 1e-9, 0.05)
DNU_PUB = 0.05

# stats-only mode: recompute mss / slope PDFs from EVERY frame (streaming,
# two passes); spectra are untouched
STATS_ONLY = os.environ.get('EPSS_SWS_STATS_ONLY', '0') == '1'
# fine histogram grid for streaming accumulation: 4x subdivision of the
# production bins over [-1.2, 1.2]; mean-shifted then aggregated to SLOPE_CENTERS
FINE_SUB = 4
_DB = SLOPE_CENTERS[1] - SLOPE_CENTERS[0]
_DFINE = _DB / FINE_SUB
FINE_EDGES = np.arange(-1.2, 1.2 + _DFINE / 2, _DFINE)
FINE_CEN = 0.5 * (FINE_EDGES[:-1] + FINE_EDGES[1:])

# spectra-npz cache-validity tag: bump when the spectra pass changes so
# stale per-run caches are recomputed rather than silently reused
ESTIMATOR_TAG = 'mean-power-project-once-f32-v1'


def _theta_remap(A, axis):
    """Reindex the 5-deg theta axis to the published compass convention."""
    j = np.arange(72)
    src = ((int(round(THETA_C / 5.0)) - j) % 72).astype(int)
    return np.take(A, src, axis=axis)


def _rebin_nu(Q, nu, dnu):
    """Variance-conserving rebin of Qs(nu, theta) from the native grid
    (centers nu, widths dnu) onto the published uniform NU_PUB grid.
    E = Q * nu * dnu per source bin; Q_new = sum(E in bin)/(nu_new * DNU_PUB)."""
    E = Q * (nu * dnu)[:, None]                       # (nu, theta)
    edges = np.concatenate((NU_PUB - DNU_PUB / 2, [NU_PUB[-1] + DNU_PUB / 2]))
    idx = np.clip(np.digitize(nu, edges) - 1, 0, len(NU_PUB) - 1)
    out = np.zeros((len(NU_PUB), Q.shape[1]))
    np.add.at(out, idx, E)
    return out / (NU_PUB[:, None] * DNU_PUB)


def _stokes_read_convention(ds):
    """(transpose, scale) for reading S1/S2 frames. The raw archive stores
    (frames, columns, rows) int16 with an implicit 1e-4 scaling and needs a
    per-frame transpose to reach (y, x); the CF-rewritten archive stores
    (frames, y, x) with an explicit scale_factor and needs no transpose."""
    last2 = tuple(d.lower() for d in ds['S1'].dimensions[-2:])
    transpose = last2 == ('columns', 'rows')
    sf = getattr(ds['S1'], 'scale_factor', None)
    scale = float(sf) if sf is not None else 1e-4
    return transpose, scale


def _slopes_present(s1, s2, gain_mode, lut, peak, dref):
    """pss present-geometry slopes (cross, along) at one gain; saturated
    facets NaN. Also returns hifrac, the frame's fraction of near-saturation
    pixels (aoi > 42 deg, unsaturated), used by the stats frame-quality
    filter."""
    g = apply_gain(s1, s2, mode=gain_mode, lab_gain=DEFAULT_LAB_GAIN,
                   theta_i_mean_deg=THETA_V, n_water=N_WATER, dolp_obs_median=dref)
    dolp = np.clip(np.sqrt(g.s1 ** 2 + g.s2 ** 2), 0.0, 1.0)
    phi = 0.5 * np.arctan2(g.s2, g.s1)
    aoi = dolp_to_aoi(dolp, lut[0], lut[1])
    ta = np.tan(np.deg2rad(aoi))
    cross, along = np.sin(phi) * ta, np.cos(phi) * ta
    sat = aoi >= peak - 1e-6
    cross = cross.copy(); along = along.copy()
    cross[sat] = np.nan; along[sat] = np.nan
    hifrac = float(np.mean((aoi > 42.0) & ~sat))
    return cross, along, hifrac


def _earth_components(cross, along):
    """Camera (cross, along) -> earth (S_E, S_N)."""
    S_E = cross * np.sin(_COL_AX) - along * np.sin(_ROW_AX)
    S_N = cross * np.cos(_COL_AX) - along * np.cos(_ROW_AX)
    return S_E, S_N


def _wind_components(S_E, S_N, wfrom_deg):
    """Earth (S_E, S_N) -> (upwind, crosswind) for wind FROM compass
    wfrom_deg; upwind is positive toward the wind source."""
    w = np.deg2rad(wfrom_deg)
    sup = S_E * np.sin(w) + S_N * np.cos(w)
    scw = S_E * np.cos(w) - S_N * np.sin(w)
    return sup, scw


FFTW = int(os.environ.get('SLOPESPECTRA_FFT_WORKERS', '2'))


def _segment_power(sx, sy):
    """Unshifted directional power of one float32 slope segment (demeaned,
    circular-Tukey tapered)."""
    sx = sx - sx.mean(axis=2, keepdims=True, dtype=np.float64).astype(np.float32)
    sy = sy - sy.mean(axis=2, keepdims=True, dtype=np.float64).astype(np.float32)
    sx = circular_tukey(sx, taper_width=0.2, normalization='power',
                        temporal_alpha=0.1, dtype=np.float32)
    sy = circular_tukey(sy, taper_width=0.2, normalization='power',
                        temporal_alpha=0.1, dtype=np.float32)
    power, framesize, s3 = compute_segment_power(
        sx, sy, DX_FULL, FS_RAW, framesize=FRAMESIZE, dtype=np.float32)
    return power, framesize, s3


def process_run(path, Wdir_deg, max_frames=0, gains=None):
    """Full present-path pss + slopespectra reduction of one Stokes file.
    gains: subset of GAINS keys to compute (default all three).
    Returns dict keyed by gain with the published-format products."""
    lut = build_lookup_table(N_WATER)
    peak = float(lut[1].max())
    ds = nc.Dataset(path)
    ds.set_auto_maskandscale(False)
    transpose, scale = _stokes_read_convention(ds)
    n = ds.dimensions['frames'].size
    if max_frames:
        n = min(n, max_frames)

    # empirical-gain reference DoLP (median over 12 sampled frames)
    sub = list(range(0, n, max(1, n // 12)))
    dref = float(np.nanmedian([np.sqrt(
        (ds['S1'][i].astype(np.float64) * scale) ** 2 +
        (ds['S2'][i].astype(np.float64) * scale) ** 2) for i in sub]))

    gkeys = list(gains) if gains else list(GAINS)
    hif_gain = 'empirical_gain' if 'empirical_gain' in gkeys else gkeys[0]
    seg_buf = {g: (np.empty((FRAMESIZE, FRAMESIZE, TSEG), np.float32),
                   np.empty((FRAMESIZE, FRAMESIZE, TSEG), np.float32)) for g in gkeys}
    acc = {g: None for g in gkeys}
    slope_en = {g: (np.full(NT_10HZ, np.nan), np.full(NT_10HZ, np.nan)) for g in gkeys}
    su_acc = {g: [] for g in gkeys}                  # wind-rotated slope samples (stats)
    sc_acc = {g: [] for g in gkeys}
    hifr = []                                        # per-stats-frame near-saturation fraction
    seg = 0
    j10 = 0

    for c0 in range(0, n, CHUNK_RAW):
        b1 = ds['S1'][c0:c0 + CHUNK_RAW].astype(np.float64) * scale
        b2 = ds['S2'][c0:c0 + CHUNK_RAW].astype(np.float64) * scale
        for off in range(b1.shape[0]):
            i = c0 + off
            s1 = b1[off].T if transpose else b1[off]
            s2 = b2[off].T if transpose else b2[off]
            for g in gkeys:
                cross, along, hf = _slopes_present(s1, s2, GAINS[g], lut, peak, dref)
                seg_buf[g][0][:, :, seg] = np.nan_to_num(cross)
                seg_buf[g][1][:, :, seg] = np.nan_to_num(along)
                if i % STR_10HZ == 0 and j10 < NT_10HZ:
                    sE, sN = _earth_components(np.nanmean(cross), np.nanmean(along))
                    slope_en[g][0][j10] = sE
                    slope_en[g][1][j10] = sN
                if i % (STR_10HZ * 25) == 0:
                    sE, sN = _earth_components(cross, along)
                    sup, scw = _wind_components(sE, sN, Wdir_deg)
                    su_acc[g].append(sup.astype(np.float32))
                    sc_acc[g].append(scw.astype(np.float32))
                    if g == hif_gain:
                        hifr.append(hf)
            if i % STR_10HZ == 0:
                j10 += 1
            seg += 1
            if seg == TSEG:
                for g in gkeys:
                    power, framesize, s3 = _segment_power(*seg_buf[g])
                    if acc[g] is None:
                        acc[g] = dict(n=0, cube=np.zeros(power.shape),
                                     framesize=framesize, s3=s3)
                    acc[g]['cube'] += power
                    acc[g]['n'] += 1
                seg = 0
    ds.close()

    nu_nat, dnu_nat = _inverse_phase_speed_grid()[:2]
    # drop frames whose near-saturation fraction exceeds median + 4*MAD
    hf = np.asarray(hifr)
    thr = np.median(hf) + 4.0 * 1.4826 * np.median(np.abs(hf - np.median(hf)))
    keep = np.flatnonzero(hf <= thr)
    out = {}
    for g in gkeys:
        a = acc[g]
        spec = finish_spectrum(a['cube'] / max(a['n'], 1), a['framesize'],
                               DX_FULL, a['s3'], FS_RAW)
        ds_g = compute_sub_spectra(spec.Skf, spec.dk, spec.df,
                                   heading_deg=CAMERA_AZ, dtheta=DTHETA)
        su = np.concatenate([su_acc[g][j].ravel() for j in keep])
        sc = np.concatenate([sc_acc[g][j].ravel() for j in keep])
        ok = np.isfinite(su) & np.isfinite(sc)
        su, sc = su[ok] - su[ok].mean(), sc[ok] - sc[ok].mean()
        H, _, _ = np.histogram2d(sc, su, bins=[_EDGES, _EDGES])
        db = SLOPE_CENTERS[1] - SLOPE_CENTERS[0]
        Hn = H / (H.sum() * db * db)
        # published layout: S_f_theta (theta,f); S_k_theta (theta,k);
        # Qs_nu_theta (theta,nu) on the uniform NU_PUB grid; theta remapped
        # to the compass CW-from-N convention
        Sft = _theta_remap(np.nan_to_num(ds_g['S_f_theta'].values), axis=0)
        Skt = _theta_remap(np.nan_to_num(ds_g['S_k_theta'].values), axis=1).T
        Qnt = _theta_remap(_rebin_nu(np.nan_to_num(ds_g['Qs_nu_theta'].values),
                                     nu_nat, dnu_nat), axis=1).T
        out[g] = dict(
            S_f_theta=Sft, S_k_theta=Skt, Qs_nu_theta=Qnt,
            k=ds_g['k'].values, f=ds_g['f'].values, nu=NU_PUB,
            theta=ds_g['theta'].values, estimator=ESTIMATOR_TAG,
            mss_upwind=float(np.var(su)), mss_crosswind=float(np.var(sc)),
            slope_histogram_crosswind_upwind=Hn,
            slope_east=slope_en[g][0], slope_north=slope_en[g][1])
    return out


def process_run_stats(path, Wdir_deg, max_frames=0):
    """EVERY-frame slope statistics (streaming, two passes over the Stokes).
    Pass 1: per-frame near-saturation fraction (empirical gain) -> exact
    median+4*MAD frame-quality threshold. Pass 2 (kept frames, all gains):
    running sums for exact mss + a fine 2-D (crosswind, upwind) histogram,
    mean-shifted (pooled de-meaning) and aggregated to the production grid.
    Returns {gain: {mss_upwind, mss_crosswind, slope_histogram_crosswind_upwind}}."""
    lut = build_lookup_table(N_WATER)
    peak = float(lut[1].max())
    ds = nc.Dataset(path)
    ds.set_auto_maskandscale(False)
    transpose, scale = _stokes_read_convention(ds)
    n = ds.dimensions['frames'].size
    if max_frames:
        n = min(n, max_frames)
    sub = list(range(0, n, max(1, n // 12)))
    dref = float(np.nanmedian([np.sqrt(
        (ds['S1'][i].astype(np.float64) * scale) ** 2 +
        (ds['S2'][i].astype(np.float64) * scale) ** 2) for i in sub]))

    def _frame(b, off):
        return b[off].T if transpose else b[off]

    # ---- pass 1: hifrac for every frame (empirical gain) ----
    hif = np.empty(n)
    for c0 in range(0, n, CHUNK_RAW):
        c1 = min(c0 + CHUNK_RAW, n)
        b1 = ds['S1'][c0:c1].astype(np.float64) * scale
        b2 = ds['S2'][c0:c1].astype(np.float64) * scale
        for off in range(b1.shape[0]):
            _, _, hif[c0 + off] = _slopes_present(_frame(b1, off), _frame(b2, off),
                                                  'empirical', lut, peak, dref)
    thr = np.median(hif) + 4.0 * 1.4826 * np.median(np.abs(hif - np.median(hif)))
    keep = hif <= thr

    # ---- pass 2: accumulate over kept frames, all gains ----
    gkeys = list(GAINS)
    nf = len(FINE_CEN)
    acc = {g: dict(n=0, su=0.0, su2=0.0, sc=0.0, sc2=0.0,
                   H=np.zeros(nf * nf, np.float64)) for g in gkeys}
    for c0 in range(0, n, CHUNK_RAW):
        c1 = min(c0 + CHUNK_RAW, n)
        b1 = ds['S1'][c0:c1].astype(np.float64) * scale
        b2 = ds['S2'][c0:c1].astype(np.float64) * scale
        for off in range(b1.shape[0]):
            if not keep[c0 + off]:
                continue
            s1 = _frame(b1, off); s2 = _frame(b2, off)
            for g in gkeys:
                cross, along, _ = _slopes_present(s1, s2, GAINS[g], lut, peak, dref)
                sE, sN = _earth_components(cross, along)
                sup, scw = _wind_components(sE, sN, Wdir_deg)
                m = np.isfinite(sup) & np.isfinite(scw)
                su = sup[m]; sc = scw[m]
                a = acc[g]
                a['n'] += su.size
                a['su'] += su.sum(); a['su2'] += (su * su).sum()
                a['sc'] += sc.sum(); a['sc2'] += (sc * sc).sum()
                ic = np.clip(((sc - FINE_EDGES[0]) / _DFINE).astype(np.int64), 0, nf - 1)
                iu = np.clip(((su - FINE_EDGES[0]) / _DFINE).astype(np.int64), 0, nf - 1)
                a['H'] += np.bincount(ic * nf + iu, minlength=nf * nf)
    ds.close()

    out = {}
    nc_bins = len(SLOPE_CENTERS)
    coarse_idx = np.clip(np.digitize(FINE_CEN, _EDGES) - 1, 0, nc_bins - 1)
    A = np.zeros((nc_bins, nf))
    A[coarse_idx, np.arange(nf)] = 1.0                # fine -> coarse aggregation
    for g in gkeys:
        a = acc[g]
        mu_u = a['su'] / a['n']; mu_c = a['sc'] / a['n']
        var_u = a['su2'] / a['n'] - mu_u ** 2
        var_c = a['sc2'] / a['n'] - mu_c ** 2
        H = a['H'].reshape(nf, nf)
        # pooled de-meaning: shift the fine histogram by the mean (nearest fine bin)
        H = np.roll(H, (-int(round(mu_c / _DFINE)), -int(round(mu_u / _DFINE))), axis=(0, 1))
        Hc = A @ H @ A.T                              # aggregate (cw, up) to 200x200
        Hn = Hc / (Hc.sum() * _DB * _DB)
        out[g] = dict(mss_upwind=float(var_u), mss_crosswind=float(var_c),
                      slope_histogram_crosswind_upwind=Hn)
    return out


# ------------------------------------------------------------------ run table
def build_run_table():
    """Map each Stokes file to a 0-189 run index by matching filename UTC to
    the acquisition time `t_seconds_since_January_1_1970` (the `time`
    coordinate is a synthetic nominal HH:05 stamp, not the true camera
    time). EPSS_STOKES_DIR may be a colon-separated list of directories; the
    closest-in-time file wins per run. Wind direction is COARE with BUZM3
    fallback (matches the production upwind/crosswind rotation)."""
    sup = nc.Dataset(DATA + 'ASIT2019_supporting_environmental_observations.nc')
    t = np.asarray(sup['t_seconds_since_January_1_1970'][:], float)
    cw = np.ma.filled(sup['COARE_Wdir'][:], np.nan)
    bw = np.ma.filled(sup['buzm3_WDIR'][:], np.nan)
    wdir = np.where(np.isfinite(cw), cw, bw)         # COARE else BUZM3
    sup.close()
    best = {}                                        # run -> (offset_s, file, wdir)
    for d_ in STOKES_DIR.split(':'):
        d_ = d_.rstrip('/') + '/'
        for f in sorted(glob.glob(d_ + '*.nc')):
            m = re.search(r'(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})', os.path.basename(f))
            if not m:
                continue
            y, mo, d, h, mi, s = map(int, m.groups())
            ft = dt.datetime(y, mo, d, h, mi, s, tzinfo=dt.timezone.utc).timestamp()
            j = int(np.argmin(np.abs(t - ft)))
            off = abs(t[j] - ft)
            if off <= 15 * 60 and (j not in best or off < best[j][0]):
                best[j] = (off, f, float(wdir[j]))
    return {j: (v[1], v[2]) for j, v in best.items()}


def main():
    tbl = build_run_table()
    maxf = int(os.environ.get('EPSS_SWS_MAXFRAMES', '0'))

    runs = os.environ.get('EPSS_SWS_RUNS')
    runs = [int(x) for x in runs.split(',')] if runs else sorted(tbl)
    # EPSS_SWS_GAINS: comma list of gain keys for the spectra pass (default
    # all three); per-run npz caches merge across gain-subset invocations
    gsel = os.environ.get('EPSS_SWS_GAINS')
    gsel = [g.strip() for g in gsel.split(',')] if gsel else list(GAINS)
    assert all(g in GAINS for g in gsel), 'unknown gain in EPSS_SWS_GAINS'
    # spectra pass: presence of the per-gain estimator tag marks a cached
    # run as current; caches from an earlier estimator version lack it
    probe = 'mss_upwind' if STATS_ONLY else 'estimator'
    os.makedirs(OUTDIR, exist_ok=True)
    for run in runs:
        if run not in tbl:
            print('run %d: no Stokes file matched, skip' % run); continue
        path, wdir = tbl[run]
        dest = OUTDIR + ('sws_stats_run%d.npz' if STATS_ONLY else 'sws_run%d.npz') % run
        if os.path.exists(dest):
            have = np.load(dest).files
            if all(f'{g}__{probe}' in have for g in gsel):
                print('run %d: cached, skip' % run); continue
        print('run %d: %s (gains: %s)' % (run, os.path.basename(path), ','.join(gsel)), flush=True)
        if STATS_ONLY:
            got = process_run_stats(path, wdir, max_frames=maxf)
        else:
            got = process_run(path, wdir, max_frames=maxf, gains=gsel)
        out = {f'{g}__{k}': v for g, d in got.items() for k, v in d.items()}
        if os.path.exists(dest):
            prev = dict(np.load(dest))
            prev.update(out)
            out = prev
        tmp = dest + '.tmp.npz'
        np.savez(tmp, **out)
        os.replace(tmp, dest)
        print('  done -> %s' % dest, flush=True)


if __name__ == '__main__':
    main()

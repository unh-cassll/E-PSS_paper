# Compile the single-triplet ("pseudo-buoy") directional estimator suite into one
# ASIT2019 product on the E-PSS frequency/direction grid.
#
# The suite treats each run as a heave-pitch-roll buoy: one collocated
# (elevation, east slope, north slope) time series taken as a 3 m disc average of
# the E-PSS field, with the disc (jinc) response divided out of S(f). Every
# estimator then steers on the transfer function h(theta) = [1, i k cos, i k sin]
# rather than on gauge positions, so the array baseline never enters and the
# low-frequency smearing that the virtual-gauge arrays carry is absent. The disc
# response sets the support band (H^2 >= 0.1, i.e. f <= ~0.65 Hz); rows outside
# it are masked, so the spectra integrate to S_f wherever they are defined and
# render blank elsewhere.
#
# The per-run estimator files are produced by the companion wave-direction-
# estimation repository; point EPSS_PSEUDOBUOY_DIR at its output tree to rebuild.
# Absent that tree this script leaves any existing product in place and exits,
# so a checkout without the companion repository still builds the rest of _data.
# @author: nathanlaxague

import os
import sys

import numpy as np
import xarray as xr

from subroutines.utils import NUM_RUNS

path = '../_data/'

srcdir = os.environ.get(
    'EPSS_PSEUDOBUOY_DIR',
    os.path.join('..', '..', 'wave-direction-estimation', '_dungeon', 'data',
                 'ASIT_2019', 'pseudo_buoy'))
outfile = path+'ASIT2019_pseudo_buoy_directional_spectra.nc'

# source estimator tag -> name used in this project's variable suffixes
tags = [('triplet','EWDM_triplet'),
        ('csm','EMEP'),
        ('bdm','BDM'),
        ('imlm','IMLM'),
        ('mem','MEM'),
        ('mlm','MLM')]

# %%

# Read every run onto the shared grid, leaving missing runs empty


def run_file(run_ind):
    return os.path.join(srcdir,'run_%03d'%run_ind,'estimators.nc')


first = next((r for r in range(NUM_RUNS) if os.path.exists(run_file(r))),None)
if first is None:
    print('no pseudo-buoy estimator files under '+srcdir+'; leaving '
          +('the existing product' if os.path.exists(outfile) else 'nothing')
          +' in place')
    sys.exit(0)

with xr.open_dataset(run_file(first)) as d0:
    freqs = d0['frequency'].values.astype(np.float32)
    dirs_deg = d0['direction'].values.astype(float)
    attrs0 = dict(d0.attrs)

nf, nd = len(freqs), len(dirs_deg)


def nan_f4(*shape):
    return np.full(shape,np.nan,np.float32)


F = {name: nan_f4(nf,nd,NUM_RUNS) for _,name in tags}
mwd = {name: nan_f4(nf,NUM_RUNS) for _,name in tags}
spr = {name: nan_f4(nf,NUM_RUNS) for _,name in tags}
S_f = nan_f4(nf,NUM_RUNS)
found = []

for run_ind in range(NUM_RUNS):
    fname = run_file(run_ind)
    if not os.path.exists(fname):
        continue
    with xr.open_dataset(fname) as d:
        if not np.allclose(d['frequency'].values,freqs,rtol=1e-5):
            raise ValueError('run %03d: frequency grid mismatch'%run_ind)
        if not np.allclose(d['direction'].values,dirs_deg,rtol=1e-5):
            raise ValueError('run %03d: direction grid mismatch'%run_ind)
        Sf = d['frequency_spectrum'].values
        # the source zero-fills unsupported rows in the spectra but leaves them
        # NaN in S_f; carry the S_f mask across so the two agree
        band = np.isfinite(Sf)
        S_f[:,run_ind] = Sf
        for tag,name in tags:
            F[name][:,:,run_ind] = np.where(
                band[:,None],d['directional_spectrum_f_'+tag].values,np.nan)
            mwd[name][:,run_ind] = np.where(
                band,np.radians(d['mean_direction_'+tag].values),np.nan)
            spr[name][:,run_ind] = np.where(
                band,np.radians(d['directional_spread_'+tag].values),np.nan)
    found.append(run_ind)

print('compiled %d/%d runs from %s'%(len(found),NUM_RUNS,srcdir))

# %%

# Write on the units and direction convention used by the other E-PSS
# directional products. The source labels the spectra per-degree, but they
# integrate to S_f over radians, so only the label changes here.

present = np.zeros(NUM_RUNS,np.int8)
present[found] = 1

data_vars = {'S_f': (('frequency','run'),S_f),
             'run_present': (('run',),present)}
for _,name in tags:
    data_vars['F_f_d_'+name] = (('frequency','direction','run'),F[name])
    data_vars['mean_direction_'+name] = (('frequency','run'),mwd[name])
    data_vars['directional_spread_'+name] = (('frequency','run'),spr[name])

ds = xr.Dataset(data_vars,
                coords={'frequency': freqs,
                        'direction': np.radians(dirs_deg).astype(np.float32),
                        'run': np.arange(NUM_RUNS,dtype=np.int32)})

ds['frequency'].attrs = {'units':'Hz'}
ds['direction'].attrs = {'units':'radians clockwise from true North'}
ds['S_f'].attrs = {'units':'m^2/Hz',
                   'long_name':'disc-response-corrected elevation spectrum'}
ds['run_present'].attrs = {'units':'1',
                           'long_name':'1 where a source estimator file was found'}
for _,name in tags:
    ds['F_f_d_'+name].attrs = {'units':'m^2/Hz/rad',
                               'long_name':'frequency-directional spectrum, '
                                           +name+' on the 3 m triplet'}
    ds['mean_direction_'+name].attrs = {'units':'radians clockwise from true North'}
    ds['directional_spread_'+name].attrs = {'units':'radians'}

ds.attrs.update(
    title='ASIT2019 single-triplet (pseudo-buoy) directional wave spectra',
    buoy_support_note='rows outside the disc-response support band are empty',
    direction_convention='cw_from_N',
    source=os.path.abspath(srcdir),
    runs_present=len(found))
# provenance carried through from the source files, where it is recorded
for key in ('buoy_diameter_m','buoy_h2_min','buoy_support_f_hz',
            'welch_nperseg','welch_overlap','csm_conjugated'):
    if key in attrs0:
        ds.attrs[key] = attrs0[key]

enc = {v: {'zlib':True,'complevel':4} for v in ds.data_vars}
ds.to_netcdf(outfile,encoding=enc)
ds.close()
print('wrote '+outfile)

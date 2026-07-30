# Plot the directional spreading function D(theta|f) for a single run as a
# 2 x 3 panel comparison: the ADCP reference and the multi-aperture wavelet
# estimate on the top row, then the four estimators driven by ONE collocated
# (elevation, east slope, north slope) triplet -- the heave-pitch-roll buoy
# observable, taken here as a 3 m disc average of the E-PSS field.
# Inverting these estimators on a virtual-gauge array seeded across the field
# instead smears their low-frequency lobes, the array baseline being short
# against a swell wavelength. Steering the triplet on the transfer function
# h(theta) = [1, i k cos, i k sin] removes the baseline from the problem and the
# lobes tighten accordingly, at the cost of a band limit: the disc response
# confines the triplet to f <~ 0.65 Hz.
# Panel order puts IMLM directly beneath the ADCP, which inverts its own
# cross-spectral matrix the same way.
# Every panel carries the direct 3-D-FFT spectrum above the splice cut, so each
# estimator is read against the same reference over the band it does not cover.
# @author: nathanlaxague

import numpy as np
import xarray as xr

import netCDF4 as nc
from matplotlib import pyplot as plt

import warnings

from subroutines.utils import *
color_list,fullwidth,fullheight,fsize = figure_style()

warnings.filterwarnings("ignore")

path = '../_data/'
figpath = '../_figures/'

run_ind = 162                       # representative bimodal case
flim = (8e-2,2e1)                   # frequency axis limits [Hz]
splice_cut = 0.7                    # direct reference above, estimator below [Hz]

panel_labels = ['(a)','(b)','(c)','(d)','(e)','(f)']

# panels (c-f): netCDF variable suffix in the pseudo-buoy product -> title.
# IMLM leads the bottom row so it sits under the ADCP panel above it.
triplet_panels = [('EWDM_triplet','EWDM Triplet'),
                  ('IMLM','IMLM'),
                  ('EMEP','EMEP'),
                  ('BDM','BDM')]

# %%

# Load the multi-aperture E-PSS spectrum, the single-triplet suite, the ADCP
# reference and the direct spectrum

ds = xr.open_dataset(path+'ASIT2019_EPSS_directional_spectra.nc')
theta = np.degrees(ds['direction'].values)
freqs = ds['frequency'].values

pb = xr.open_dataset(path+'ASIT2019_pseudo_buoy_directional_spectra.nc')
theta_pb = np.degrees(pb['direction'].values)
freqs_pb = pb['frequency'].values

env = nc.Dataset(path+'ASIT2019_supporting_environmental_observations.nc')
F_adcp = np.ma.filled(env['F_f_theta_m2_Hz_rad_ADCP'][:,:,run_ind],np.nan).T
f_adcp = np.asarray(env['f_Hz_ADCP'][:])
th_adcp = np.degrees(np.asarray(env['theta_rad'][:]))
wind_to = (float(np.ma.filled(env['COARE_Wdir'][run_ind],np.nan)) + 180.0) % 360.0
env.close()

ref = nc.Dataset(path+'ASIT2019_wave_spectra_stats_timeseries_empirical_gain.nc')
S_ref = np.ma.filled(ref['S_f_theta'][run_ind],np.nan).T      # (f, theta)
f_ref = np.asarray(ref['f_Hz'][:])
th_ref = np.degrees(np.asarray(ref['theta_rad'][:]))
ref.close()


dth = float(abs(theta[1]-theta[0]))
dth_pb = float(abs(theta_pb[1]-theta_pb[0]))

D_arrays = spreading(ds['F_f_d'].values[:,:,run_ind],dth)
D_pb = {label: spreading(pb['F_f_d_'+suffix].values[:,:,run_ind],dth_pb)
        for suffix,label in triplet_panels}

D_adcp = spreading(F_adcp,float(abs(th_adcp[1]-th_adcp[0])))
# D is scale-free, so the direct SLOPE spectrum needs no k^-2 conversion here.
# Its frequency grid is ~4x denser than the estimators', hence the wider window.
D_ref = spreading(S_ref,float(abs(th_ref[1]-th_ref[0])),
                  smooth=SPREAD_SMOOTHNUM_DIRECT)

# %%

# 2 x 3 panels: reference, the array solve, then the four triplet estimators,
# each spliced to the direct spectrum

fig,axes = plt.subplots(2,3,figsize=(fullwidth,0.57*fullwidth),
                        sharex=True,sharey=True,layout="constrained")

order = np.argsort(wrap_deg(theta))
th_plot = wrap_deg(theta)[order]
order_pb = np.argsort(wrap_deg(theta_pb))
th_pb_plot = wrap_deg(theta_pb)[order_pb]
order_a = np.argsort(wrap_deg(th_adcp))
th_adcp_plot = wrap_deg(th_adcp)[order_a]
order_r = np.argsort(wrap_deg(th_ref))
th_ref_plot = wrap_deg(th_ref)[order_r]

items = [('IMLM (ADCP)',D_adcp,f_adcp,th_adcp_plot,order_a),
         ('EWDM Arrays',D_arrays,freqs,th_plot,order)] + \
        [(label,D_pb[label],freqs_pb,th_pb_plot,order_pb)
         for _,label in triplet_panels]

lo_ref = (f_ref >= splice_cut) & (f_ref <= flim[1])
vals = [D_ref[np.ix_(lo_ref,order_r)]]
for _,Dm,fax,_,_ in items:
    m = (fax >= flim[0]) & (fax <= splice_cut)
    vals.append(Dm[m])
vmax = panel_vmax(vals)

for ax,(label,Dm,fax,thax,idx) in zip(axes.ravel(),items):
    m = fax <= splice_cut
    im = ax.pcolormesh(thax,fax[m],Dm[np.ix_(m,idx)],cmap='magma',
                       shading='auto',vmin=0.0,vmax=vmax,rasterized=True)
    # the same direct spectrum above the cut in every panel; the gap between
    # the two blocks is left blank so the handoff stays visible. On the triplet
    # panels that gap also carries the disc-response band limit.
    ax.pcolormesh(th_ref_plot,f_ref[lo_ref],D_ref[np.ix_(lo_ref,order_r)],
                  cmap='magma',shading='auto',vmin=0.0,vmax=vmax,
                  rasterized=True)
    ax.axhline(splice_cut,color='w',ls='-',lw=0.8,alpha=0.9)
    # going-to wind direction; cyan reads against magma, which has no green-blue
    ax.plot(wrap_deg(wind_to)*np.float64([1.0,1.0]),np.float64([1e-3,1e3]),
            color='#00E5FF',ls=':',lw=1.5,label='wind direction')
    ax.set_title(label,fontsize=fsize)
    ax.set_yscale('log')
    ax.set_ylim(flim)
    # 45-deg convention kept on the minor ticks; three panels across the page
    # cannot label every 45 deg legibly. Limits come last: set_xticks autoscales.
    ax.set_xticks(np.arange(-180,181,90))
    ax.set_xticks(np.arange(-180,181,45),minor=True)
    ax.set_xlim(-180,180)
    ax.grid(False)

for ax,lab in zip(axes.ravel(),panel_labels):
    ax.text(0.93,0.88,lab,color='white',fontsize=fsize,ha='center',va='center',
            transform=ax.transAxes)

for ax in axes[-1]:
    ax.set_xlabel(r'$\theta$ [$\circ$]',fontsize=fsize)
for ax in axes[:,0]:
    ax.set_ylabel('f [Hz]',fontsize=fsize)

cb = fig.colorbar(im,ax=list(axes.ravel()),location='right',shrink=0.85,
                  aspect=24,pad=0.02)
cb.set_label(r'$D(f,\theta)\;\mathrm{[deg^{-1}]}$',fontsize=fsize)

fig.savefig(figpath+'directional_spreading_panels.pdf',bbox_inches='tight',
            dpi=300)
plt.close(fig)
pb.close()
ds.close()

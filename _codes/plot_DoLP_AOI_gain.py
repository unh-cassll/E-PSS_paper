"""
Created on Fri Sep 12 12:16:43 2025

@author: nathanlaxague
"""

import numpy as np

import netCDF4 as nc
from matplotlib import pyplot as plt

import seaborn as sns

from subroutines.utils import *

sns.set_theme(style="whitegrid",palette="deep",font="DejaVu Sans Mono")

color_array = ['#4C2882', '#C39953', '#A52A2A', '#367588', '#2A52BE', '#006611']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_array)

fn = '../_data/Run051_example_Stokes_parameters.nc'

ds_Stokes = nc.Dataset(fn)

S0 = ds_Stokes["S0"][:]
S1 = ds_Stokes["S1"][:]
S2 = ds_Stokes["S2"][:]

AOI_camera = ds_Stokes["AOI_camera"][:]
DoLP_camera = ds_Stokes["DoLP_camera"][:]

m_per_px = ds_Stokes["m_per_px"][:]

N = ds_Stokes.dimensions["num_cols"].size

DoLP_gain = 0.4441/np.median(DoLP_camera)

DoLP_array = np.sqrt(S1**2+S2**2)

x = m_per_px*np.arange(0,N)

n = 1.33

Ssky = np.float64([1,0,0,0])
Sup = np.float64([0,0,0,0])

out_theta,out_DOLP = mueller_calc_full(n,Ssky,Sup)

DoLP_max = 1.1
fig = plt.figure(1,figsize=(12, 12))

gs = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(out_theta,out_DOLP,linewidth=3,label="Fresnel curve; ideal conditions")
ax1.plot(AOI_camera,DoLP_camera,linewidth=3,label=r"avg. observed DoLP($\theta_i$)")
ax1.plot(AOI_camera,DoLP_gain*DoLP_camera,linewidth=3,label="$ibid.$, empirical gain")
ax1.set_xticks(np.arange(0,105,15))
ax1.set_xlim(0,90)
ax1.set_ylim(0,DoLP_max)
ax1.legend(loc="upper left")
ax1.set_xlabel(r'$\theta_i$ [$\circ$]')
ax1.set_ylabel('DoLP')

ax1.set_aspect(aspect=0.5*90/DoLP_max)

ax2 = fig.add_subplot(gs[1, 0])

pc_emp = ax2.imshow(DoLP_array,cmap='gray',vmin=0,vmax=1)
ax2.set_xlabel('')
ax2.set_xticks([])
ax2.set_ylabel('')
ax2.set_yticks([])

ax2.text(25, 40, 'raw',
         fontsize=12,
         color='black',
         bbox=dict(facecolor=color_array[1], edgecolor='black', boxstyle='round,pad=0.5', alpha=1))

ax2.set_aspect(aspect=1)

ax3 = fig.add_subplot(gs[1, 1])

pc_emp = ax3.imshow(DoLP_gain*DoLP_array,cmap='gray',vmin=0,vmax=1)
ax3.set_xlabel('')
ax3.set_xticks([])
ax3.set_ylabel('')
ax3.set_yticks([])

ax3.set_aspect(aspect=1)

ax3.text(25, 40, 'emp. gain',
         fontsize=12,
         color='white',
         bbox=dict(facecolor=color_array[2], edgecolor='white', boxstyle='round,pad=0.5', alpha=1))

cbar = fig.colorbar(pc_emp, ax=[ax2, ax3], orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('DoLP')

plt.savefig('../_figures/DoLP_AOI_gain_example.pdf',bbox_inches='tight')

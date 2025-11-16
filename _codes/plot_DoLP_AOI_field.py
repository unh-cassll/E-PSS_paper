"""
Created on Thu Oct  9 15:50:13 2025

@author: nathanlaxague
"""

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import netCDF4 as nc

from subroutines.utils import *

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid",palette="deep",font="DejaVu Sans Mono")

color_list = ['#4C2882', '#367588', '#A52A2A', '#C39953', '#2A52BE', '#006611']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

panel_labels = ['(a)','(b)','(c)','(d)','(e)','(f)']

path = '../_data/'

kappa = 0.4

ds = nc.Dataset(path+'Piermont2025_DoLP_AoI_observations.nc')

AoI_sunny = ds['AoI_sunny'][:]
DoLP_sunny = ds['DoLP_sunny'][:]

AoI_overcast = ds['AoI_overcast'][:]
DoLP_overcast = ds['DoLP_overcast'][:]

AoI_sunny_narrow = ds['AoI_sunny_narrow'][:]
DoLP_sunny_narrow = ds['DoLP_sunny_narrow'][:]

AoI_overcast_narrow = ds['AoI_overcast_narrow'][:]
DoLP_overcast_narrow = ds['DoLP_overcast_narrow'][:]

Ssky_overcast = ds['Ssky_overcast'][:]
Sup_overcast = ds['Sup_overcast'][:]

Ssky_sunny = ds['Ssky_sunny'][:]
Sup_sunny = ds['Sup_sunny'][:]

n_refrac_sunny = ds['n_refrac_sunny'][:]
n_refrac_overcast = ds['n_refrac_overcast'][:]

DoLP_overcast_sky = ds['DoLP_overcast_sky'][:]
DoLP_sunny_sky = ds['DoLP_sunny_sky'][:]


n = 1.33

Ssky = np.float64([1,0,0,0])
Sup = np.float64([0,0,0,0])

ideal_AoI,ideal_DoLP = mueller_calc_full(n,Ssky,Sup)

overcast_AoI,overcast_DoLP = mueller_calc_full(n_refrac_overcast,Ssky_overcast,Sup_overcast)
sunny_AoI,sunny_DoLP = mueller_calc_full(n_refrac_sunny,Ssky_sunny,Sup_sunny)

sky_DoLP = [DoLP_overcast_sky,DoLP_sunny_sky]

titles = ['overcast','cloudless']

fig, axs = plt.subplots(2,1, sharex=True, sharey=False, figsize=(6, 10))

axs[0].plot(AoI_overcast,DoLP_overcast,linewidth=2,label='wide')
axs[0].plot(AoI_overcast_narrow,DoLP_overcast_narrow,'*',label='narrow')
axs[0].plot(overcast_AoI,overcast_DoLP,'--',label='fit')

axs[1].plot(AoI_sunny,DoLP_sunny,linewidth=2,label='wide')
axs[1].plot(AoI_sunny_narrow,DoLP_sunny_narrow,'*',label='narrow')
axs[1].plot(sunny_AoI,sunny_DoLP,'--',label='fit')

for panel_index in np.arange(0,2):
    axs[panel_index].plot(ideal_AoI,ideal_DoLP,'-',color=(0.5,0.5,0.5),label='ideal')
    axs[panel_index].set_xlim(0,90)
    axs[panel_index].set_ylim(0,1.1)
    axs[panel_index].set_xticks(np.arange(0,105,15))
    axs[panel_index].set_ylabel('DoLP')
    axs[panel_index].text(0.05,0.95,panel_labels[panel_index],fontsize=12,ha='center',va='center',transform=axs[panel_index].transAxes)
    sky_DoLP_val = sky_DoLP[panel_index]
    axs[panel_index].set_title(titles[panel_index]+ ' (sky DoLP = ' + f"{sky_DoLP_val:.3f}" ')')

axs[0].legend(loc='upper right')
axs[1].set_xlabel(r'$\theta_i$ [$\circ$]')
    
plt.tight_layout()

plt.savefig('../_figures/DoLP_AOI_field.pdf',bbox_inches='tight')

"""
Created on Thu Aug 28 11:39:01 2025

@author: nathanlaxague
"""

import sys
sys.path.append('subroutines/')

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns
sns.set_theme(style="whitegrid")

from utils import *

sns.set_theme(style="whitegrid",palette="deep",font="DejaVu Sans Mono")

n = 1.33

Ssky = np.float64([1,0,0,0])
Sup = np.float64([0,0,0,0])

sky_dolp = 0.25
upwelling_frac = 0.025

out_theta,out_DOLP = mueller_calc_full(n,Ssky,Sup)

num_DOLP = 6

DOLP_upwelling = np.ones((len(out_theta),num_DOLP))
DOLP_downwelling = np.ones((len(out_theta),num_DOLP))
DOLP_both = np.ones((len(out_theta),num_DOLP,2))

for dolp_ind in np.arange(0,num_DOLP):
    
    sea_up = np.float64([1,0,0,0])*upwelling_frac*(dolp_ind)/(num_DOLP-1)
    out_theta,out_DOLP_upwelling = mueller_calc_full(n,Ssky,sea_up)
    DOLP_upwelling[:,dolp_ind] = out_DOLP_upwelling
    
    sky_down = np.add(Ssky,np.float64([0,1,1,0])*sky_dolp/np.sqrt(2)*(dolp_ind)/((num_DOLP-1)))
    out_theta,out_DOLP_downwelling = mueller_calc_full(n,sky_down,Sup)
    DOLP_downwelling[:,dolp_ind] = out_DOLP_downwelling
    
DOLP_both[:,:,0] = DOLP_upwelling
DOLP_both[:,:,1] = DOLP_downwelling

fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey='all')

colormaps = ['cividis', 'magma']
color_mins = [0,0]
color_maxes = [upwelling_frac,sky_dolp]

cbar_labels = ['upwelling radiance fraction','DoLP of sky-leaving radiance']

x = out_theta

xlims = [0,90]
ylims = [0,1.1]

panel_labels = ['(a)','(b)','(c)','(d)']
text_x = 0.01
text_y = 0.95

for i, ax in enumerate(axs):
    
    cmap = plt.get_cmap(colormaps[i])
    colors = [cmap(j) for j in np.linspace(0, 0.9, num_DOLP)]
    
    y_values = DOLP_both[:,:,i].T
    
    for j, y in enumerate(y_values):
        
        ax.plot(x, y, color=colors[j],linewidth=2.5)
    
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xlabel(r'$\theta_i$ [$\circ$]')
        ax.set_xticks(ticks=[0,15,30,45,60,75,90])
            
    if i == 0:
    
        ax.set_ylabel('DoLP')
        
        
    ax.text(text_x,text_y,panel_labels[i], transform=ax.transAxes)
    
    norm = plt.Normalize(vmin=color_mins[i], vmax=color_maxes[i])
    
    sm = plt.cm.ScalarMappable(cmap=colormaps[i], norm=norm)    
    cbar = fig.colorbar(sm, ax=ax, location='top', orientation='horizontal', fraction=0.05, pad=0.05)
    cbar.set_label(cbar_labels[i])

plt.tight_layout()

plt.savefig('../_figures/mueller_calc_example.pdf',bbox_inches='tight')


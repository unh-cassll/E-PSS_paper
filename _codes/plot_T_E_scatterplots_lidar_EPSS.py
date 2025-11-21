"""
Created on Mon Sep 15 12:05:23 2025

@author: nathanlaxague
"""

import pandas as pd
import numpy as np
import xarray as xr

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from subroutines.utils import *

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid",palette="deep",font="DejaVu Sans Mono")

# Set custom property cycle colors
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#4C2882', '#367588', '#A52A2A', '#C39953', '#2A52BE', '#006611'])

path = '../_data/'

f_lp = 0.3
f_hp = 0.08

ds_omnispect = xr.open_dataset(path+'ASIT2019_omnidirectional_spectra.nc')
    
f_Hz = ds_omnispect['frequency'][:].data
nf = len(f_Hz)
mask = np.ones((nf,1))
mask[(ds_omnispect['frequency'][:]<f_hp)|(ds_omnispect['frequency'][:]>f_lp)] = np.nan
df = np.median(np.diff(f_Hz))

T_s = f_Hz.reshape(nf,1)**-1
T_s[0] = 0

Hm0_no_gain = 4*np.sqrt((np.nansum(mask*ds_omnispect['F_f_m2_Hz_no_gain'][:].data,axis=0)*df))
Hm0_lab_gain = 4*np.sqrt((np.nansum(mask*ds_omnispect['F_f_m2_Hz_lab_gain'][:].data,axis=0)*df))
Hm0_emp_gain = 4*np.sqrt((np.nansum(mask*ds_omnispect['F_f_m2_Hz_empirical_gain'][:].data,axis=0)*df))

T_E_no_gain = (np.nansum(mask*T_s*ds_omnispect['F_f_m2_Hz_no_gain'][:].data,axis=0)/np.nansum(mask*ds_omnispect['F_f_m2_Hz_no_gain'][:].data,axis=0))
T_E_lab_gain = (np.nansum(mask*T_s*ds_omnispect['F_f_m2_Hz_lab_gain'][:].data,axis=0)/np.nansum(mask*ds_omnispect['F_f_m2_Hz_lab_gain'][:].data,axis=0))
T_E_emp_gain = (np.nansum(mask*T_s*ds_omnispect['F_f_m2_Hz_empirical_gain'][:].data,axis=0)/np.nansum(mask*ds_omnispect['F_f_m2_Hz_empirical_gain'][:].data,axis=0))
T_E_lidar = (np.nansum(mask*T_s*ds_omnispect['F_f_m2_Hz_lidar'][:].data,axis=0)/np.nansum(mask*ds_omnispect['F_f_m2_Hz_lidar'][:].data,axis=0))

inds_exclude = (T_E_lidar > 20) | (T_E_lidar < 4.5) | (Hm0_emp_gain < 0.1)
T_E_lidar[inds_exclude] = np.nan

data_size = len(T_E_lidar)

big_reference = np.concatenate((T_E_lidar,T_E_lidar,T_E_lidar))
big_test = np.concatenate((T_E_no_gain,T_E_lab_gain,T_E_emp_gain))
category = np.concatenate((np.full(data_size, 'none'),np.full(data_size, 'lab'),np.full(data_size, 'empirical')))

data = pd.DataFrame({
    'lidar': big_reference,
    'EPSS': big_test,
    'DoLP gain': category
})

metrics = {}
proxy_estimates = ['none', 'lab', 'empirical']
x = T_E_lidar
for T_E_estimate in proxy_estimates:
    Y = data[data["DoLP gain"] == T_E_estimate]
    y = Y.EPSS
    inds_keep = (~np.isnan(x) & ~np.isnan(y))
    rmse = np.sqrt(mean_squared_error(x[inds_keep], y[inds_keep]))
    correlation_matrix = np.corrcoef(x[inds_keep], y[inds_keep])
    correlation_coefficient = correlation_matrix[0,1]
    p = np.polyfit(x[inds_keep], y[inds_keep],1)
    slope = p[0]
    intercept = p[1]
    metrics[T_E_estimate] = (correlation_coefficient, rmse, slope, intercept)

g = sns.lmplot(
    data=data,
    x="lidar", y="EPSS", hue="DoLP gain",
    scatter_kws = {'alpha': 0.5,'s':15},
    line_kws = {'alpha': 0.75},
    height=6,
    aspect=1,
    legend=False
)

plt.plot([0,15],[0,15],'--',color='k')

r_values = [metrics['none'][0], metrics['lab'][0], metrics['empirical'][0]]
r2_values = [r**2 for r in r_values]
rmse_values = [metrics['none'][1], metrics['lab'][1], metrics['empirical'][1]]
slope_values = [metrics['none'][2], metrics['lab'][2], metrics['empirical'][2]]
offset_values = [metrics['none'][3], metrics['lab'][3], metrics['empirical'][3]]
colors = ['#4C2882', '#367588', '#A52A2A']

r2_line = 'R² = '
    
rmse_line = 'RMSE = '

slope_line = 'slope = '

offset_line = 'bias = '

textstr = r2_line + '\n' + rmse_line + '\n' + slope_line + '\n' + offset_line

plt.gca().add_patch(plt.Rectangle((3.1, 8.25), 3.8, 1.7, color='k', alpha=0.95, edgecolor='k',linewidth=2))
plt.gca().add_patch(plt.Rectangle((3.1, 8.25), 3.8, 1.7, color='w', alpha=0.95, edgecolor='k',linewidth=0.5))


x_position = 0.03
y_position = 0.88
delta_x = [0.03,0.03,0.04]
plt.text(x_position, y_position, textstr, transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top')

plt.text(0.32,0.98, 'DoLP gain', transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='top',horizontalalignment='center')

x_position += 0.02
for r2, rmse, slope, offset, color, proxy, delta_x_val in zip(r2_values, rmse_values, slope_values, offset_values, colors, proxy_estimates, delta_x):
    x_position += 0.12
    plt.text(x_position+delta_x_val, y_position+0.05, proxy, color=color, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top',horizontalalignment='center')
    plt.text(x_position, y_position, f'{r2:.2f}', color=color, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top')
    plt.text(x_position, y_position-0.03, f'{rmse:.2f}', color=color, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top')
    plt.text(x_position, y_position-0.06, f'{slope:.2f}', color=color, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top')
    plt.text(x_position, y_position-0.09, f'{offset:.2f}', color=color, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top')

x_position += 0.1
plt.text(x_position, y_position-0.03, 's', color='k', transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top')
plt.text(x_position, y_position-0.09, 's', color='k', transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top')
    
plt.xticks(np.linspace(3,11,9))
plt.yticks(np.linspace(3,11,9))
plt.xlim(3,10)
plt.ylim(3,10)

plt.xlabel(r'$T_{E}$, lidar [s]')
plt.ylabel(r'$T_{E}$, E-PSS [s]')

plt.savefig('../_figures/T_E_comparison_lidar_EPSS.pdf',bbox_inches='tight')
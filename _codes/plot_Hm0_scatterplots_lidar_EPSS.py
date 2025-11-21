"""
Created on Mon Sep 15 12:01:13 2025

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
    
mask = np.zeros((len(ds_omnispect['frequency']),1))
mask[(ds_omnispect['frequency'][:]>f_hp)&(ds_omnispect['frequency'][:]<f_lp)] = 1
df = np.median(np.diff(ds_omnispect['frequency'][:]))

Hm0_lidar = 4*np.sqrt(np.sum(mask*ds_omnispect['F_f_m2_Hz_lidar'][:].data,axis=0)*df)

inds_exclude = Hm0_lidar < 0.1
Hm0_lidar[inds_exclude] = np.nan

Hm0_no_gain = 4*np.sqrt((np.sum(mask*ds_omnispect['F_f_m2_Hz_no_gain'][:].data,axis=0)*df))
Hm0_lab_gain = 4*np.sqrt((np.sum(mask*ds_omnispect['F_f_m2_Hz_lab_gain'][:].data,axis=0)*df))
Hm0_emp_gain = 4*np.sqrt((np.sum(mask*ds_omnispect['F_f_m2_Hz_empirical_gain'][:].data,axis=0)*df))

data_size = len(Hm0_lidar)

big_reference = np.concatenate((Hm0_lidar,Hm0_lidar,Hm0_lidar))
big_test = np.concatenate((Hm0_no_gain,Hm0_lab_gain,Hm0_emp_gain))
category = np.concatenate((np.full(data_size, 'none'),np.full(data_size, 'lab'),np.full(data_size, 'empirical')))

data = pd.DataFrame({
    'lidar': big_reference,
    'EPSS': big_test,
    'DoLP gain': category
})

# Calculate metrics for each pair
metrics = {}
proxy_estimates = ['none', 'lab', 'empirical']
x = Hm0_lidar
for Hm0_estimate in proxy_estimates:
    Y = data[data["DoLP gain"] == Hm0_estimate]
    y = Y.EPSS
    inds_keep = (~np.isnan(x) & ~np.isnan(y))
    rmse = np.sqrt(mean_squared_error(x[inds_keep], y[inds_keep]))
    correlation_matrix = np.corrcoef(x[inds_keep], y[inds_keep])
    correlation_coefficient = correlation_matrix[0,1]
    p = np.polyfit(x[inds_keep], y[inds_keep],1)
    slope = p[0]
    intercept = p[1]
    metrics[Hm0_estimate] = (correlation_coefficient, rmse, slope, intercept)

# Set up the figure

g = sns.lmplot(
    data=data,
    x="lidar", y="EPSS", hue="DoLP gain",
    scatter_kws = {'alpha': 0.5,'s':15},
    line_kws = {'alpha': 0.75},
    height=6,
    aspect=1,
    legend=False
)

plt.plot([0,5],[0,5],'--',color='k')

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

plt.gca().add_patch(plt.Rectangle((0.09, 3.75), 2.65, 1.2, color='k', alpha=0.95, edgecolor='k',linewidth=2))
plt.gca().add_patch(plt.Rectangle((0.09, 3.75), 2.65, 1.2, color='w', alpha=0.95, edgecolor='k',linewidth=0.5))

# Add the textbox to the plot
x_position = 0.03  # Starting x position for the text
y_position = 0.88  # Starting y position for the text
delta_x = [0.03,0.03,0.04]
plt.text(x_position, y_position, textstr, transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top')

plt.text(0.32,0.98, 'DoLP gain', transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='top',horizontalalignment='center')

# Split the text into lines and add each line with the specified color
x_position += 0.02  # Move over
for r2, rmse, slope, offset, color, proxy, delta_x_val in zip(r2_values, rmse_values, slope_values, offset_values, colors, proxy_estimates, delta_x):
    x_position += 0.12  # Move over
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

x_position += 0.1  # Move over
plt.text(x_position, y_position-0.03, 'm', color='k', transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top')
plt.text(x_position, y_position-0.09, 'm', color='k', transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top')
    
plt.xticks(np.linspace(0,5,6))
plt.yticks(np.linspace(0,5,6))
plt.xlim(0,5)
plt.ylim(0,5)
plt.xlabel(r'$H_{m0}$, lidar [m]')
plt.ylabel(r'$H_{m0}$, E-PSS [m]')

plt.savefig('../_figures/Hm0_comparison_lidar_EPSS.pdf',bbox_inches='tight')

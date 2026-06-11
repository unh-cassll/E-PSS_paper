"""
Scatterplot of significant wave height Hm0: E-PSS vs. lidar reference.
Compares no/lab/empirical DoLP gain corrections.
"""

import pandas as pd
import numpy as np
import xarray as xr

import seaborn as sns
import matplotlib.pyplot as plt

from subroutines.utils import figure_style, scatter_metrics, draw_metrics_box, write_tex_macros
color_list,fullwidth,fullheight,fsize = figure_style()

import warnings

warnings.filterwarnings("ignore")

path = '../_data/'

f_lp = 0.5    # Hz; jinc-corrected single disc is valid to ~lambda=FOV (2.915 m -> ~0.7 Hz)
f_hp = 0.08   # Hz; lower admits 1/k^2-amplified low-f noise

ds_omnispect = xr.open_dataset(path+'ASIT2019_omnidirectional_spectra.nc')

mask = np.zeros((len(ds_omnispect['frequency']),1))
mask[(ds_omnispect['frequency'][:]>f_hp)&(ds_omnispect['frequency'][:]<f_lp)] = 1
df = np.median(np.diff(ds_omnispect['frequency'][:]))

Hm0_lidar = 4*np.sqrt(np.sum(mask*ds_omnispect['F_f_m2_Hz_lidar'][:].data,axis=0)*df)

inds_exclude = (Hm0_lidar < 0.2) | (Hm0_lidar > 5)
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

# (R^2, RMSE, slope, bias) per gain category
metrics = [scatter_metrics(Hm0_lidar, y)
           for y in (Hm0_no_gain, Hm0_lab_gain, Hm0_emp_gain)]

g = sns.lmplot(
    data=data,
    x="lidar", y="EPSS", hue="DoLP gain",
    scatter_kws = {'alpha': 0.5,'s':10},
    line_kws = {'alpha': 0.75},
    height=fullwidth/2,
    aspect=1,
    legend=False
)

plt.plot([0,5],[0,5],'--',color='k')

# Export the empirical-gain headline metrics as LaTeX macros for paper.tex
r2_emp, rmse_emp, slope_emp, bias_emp = metrics[2]
write_tex_macros('Hm0_values.tex', {
    'HmRtwoEmp':  f'{r2_emp:.2f}',
    'HmRMSEemp':  f'{rmse_emp:.2f}',
    'HmSlopeEmp': f'{slope_emp:.2f}',
    'HmBiasEmp':  f'{bias_emp:.2f}',
}, source='plot_Hm0_scatterplots_lidar_EPSS.py')

draw_metrics_box(plt.gca(), metrics, ['none', 'lab', 'emp'], color_list[:3],
                 ('m', 'm'), box_xy=(0.013, 0.713), box_w=0.69, box_h=0.277,
                 col_step=0.15, unit_dx=0.12, fsize=fsize)

plt.xticks(np.linspace(0,4,9))
plt.yticks(np.linspace(0,4,9))
plt.xlim(0,4)
plt.ylim(0,4)
plt.xlabel(r'$H_{m0}$, lidar [m]')
plt.ylabel(r'$H_{m0}$, E-PSS [m]')

sns.despine(right=False,top=False)

plt.savefig('../_figures/Hm0_comparison_lidar_epss.pdf',bbox_inches='tight')

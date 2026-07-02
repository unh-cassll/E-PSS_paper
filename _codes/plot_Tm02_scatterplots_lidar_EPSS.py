"""
Scatterplot of Tm02 = sqrt(m0/m2): E-PSS vs. lidar reference.
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

f_lp = 0.7    # Hz; jinc-corrected single disc is valid to ~lambda=FOV (2.915 m -> ~0.7 Hz)
f_hp = 0.10   # Hz; above the 1/k^2-amplified low-f noise bump that inflates the period

ds_omnispect = xr.open_dataset(path+'ASIT2019_omnidirectional_spectra.nc')

f_Hz = ds_omnispect['frequency'][:].data
nf = len(f_Hz)
mask = np.ones((nf,1))
mask[(ds_omnispect['frequency'][:]<f_hp)|(ds_omnispect['frequency'][:]>f_lp)] = np.nan
df = np.median(np.diff(f_Hz))

f2 = f_Hz.reshape(nf,1)**2   # second spectral moment weight (m2 = int f^2 F df)

Hm0_no_gain = 4*np.sqrt((np.nansum(mask*ds_omnispect['F_f_m2_Hz_no_gain'][:].data,axis=0)*df))
Hm0_lab_gain = 4*np.sqrt((np.nansum(mask*ds_omnispect['F_f_m2_Hz_lab_gain'][:].data,axis=0)*df))
Hm0_emp_gain = 4*np.sqrt((np.nansum(mask*ds_omnispect['F_f_m2_Hz_empirical_gain'][:].data,axis=0)*df))
Hm0_lidar = 4*np.sqrt((np.nansum(mask*ds_omnispect['F_f_m2_Hz_lidar'][:].data,axis=0)*df))

# Mean zero-crossing period Tm02 = sqrt(m0/m2); df cancels in the moment ratio
Tm02_no_gain = np.sqrt(np.nansum(mask*ds_omnispect['F_f_m2_Hz_no_gain'][:].data,axis=0)/np.nansum(mask*f2*ds_omnispect['F_f_m2_Hz_no_gain'][:].data,axis=0))
Tm02_lab_gain = np.sqrt(np.nansum(mask*ds_omnispect['F_f_m2_Hz_lab_gain'][:].data,axis=0)/np.nansum(mask*f2*ds_omnispect['F_f_m2_Hz_lab_gain'][:].data,axis=0))
Tm02_emp_gain = np.sqrt(np.nansum(mask*ds_omnispect['F_f_m2_Hz_empirical_gain'][:].data,axis=0)/np.nansum(mask*f2*ds_omnispect['F_f_m2_Hz_empirical_gain'][:].data,axis=0))
Tm02_lidar = np.sqrt(np.nansum(mask*ds_omnispect['F_f_m2_Hz_lidar'][:].data,axis=0)/np.nansum(mask*f2*ds_omnispect['F_f_m2_Hz_lidar'][:].data,axis=0))

inds_exclude = (Tm02_lidar > 10) | (Tm02_lidar < 2) | (Hm0_emp_gain < 0.2) | (Hm0_lidar < 0.2)
Tm02_lidar[inds_exclude] = np.nan

data_size = len(Tm02_lidar)

big_reference = np.concatenate((Tm02_lidar,Tm02_lidar,Tm02_lidar))
big_test = np.concatenate((Tm02_no_gain,Tm02_lab_gain,Tm02_emp_gain))
category = np.concatenate((np.full(data_size, 'none'),np.full(data_size, 'lab'),np.full(data_size, 'empirical')))

data = pd.DataFrame({
    'lidar': big_reference,
    'EPSS': big_test,
    'DoLP gain': category
})

# (R^2, RMSE, slope, bias) per gain category
metrics = [scatter_metrics(Tm02_lidar, y)
           for y in (Tm02_no_gain, Tm02_lab_gain, Tm02_emp_gain)]

g = sns.lmplot(
    data=data,
    x="lidar", y="EPSS", hue="DoLP gain",
    scatter_kws = {'alpha': 0.5,'s':10},
    line_kws = {'alpha': 0.75},
    height=fullwidth/2,
    aspect=1,
    legend=False
)

plt.plot([0,15],[0,15],'--',color='k')

# Export the empirical-gain headline metrics as LaTeX macros for paper.tex
r2_emp, rmse_emp, slope_emp, bias_emp = metrics[2]
write_tex_macros('Tm02_values.tex', {
    'TmRtwoEmp':  f'{r2_emp:.2f}',
    'TmRMSEemp':  f'{rmse_emp:.2f}',
    'TmSlopeEmp': f'{slope_emp:.2f}',
    'TmBiasEmp':  f'{bias_emp:.2f}',
}, source='plot_Tm02_scatterplots_lidar_EPSS.py')

draw_metrics_box(plt.gca(), metrics, ['none', 'lab', 'emp'], color_list[:3],
                 ('s', 's'), box_xy=(0.339, 0.012), box_w=0.651, box_h=0.278,
                 col_step=0.14, unit_dx=0.11, fsize=fsize)

plt.xticks(np.linspace(2,8,7))
plt.yticks(np.linspace(2,8,7))
plt.xlim(2,7)
plt.ylim(2,7)

plt.xlabel(r'$T_{m02}$, lidar [s]')
plt.ylabel(r'$T_{m02}$, E-PSS [s]')

sns.despine(right=False,top=False)

plt.savefig('../_figures/Tm02_comparison_lidar_epss.pdf',bbox_inches='tight')

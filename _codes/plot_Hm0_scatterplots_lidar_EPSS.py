"""
Scatterplot of significant wave height Hm0: E-PSS vs. lidar reference.
Compares no/lab/empirical DoLP gain corrections.
"""

import pandas as pd
import numpy as np
import xarray as xr

import seaborn as sns
import matplotlib.pyplot as plt

from subroutines.utils import (figure_style, scatter_metrics, draw_metrics_box,
                               write_tex_macros, york_fit, york_band,
                               lidar_member_spectra, lidar_consistency_flag,
                               MEDIAN3_SIGMA_FACTOR)
color_list,fullwidth,fullheight,fsize = figure_style()

from scipy import stats as _stats

import warnings

warnings.filterwarnings("ignore")

path = '../_data/'
figpath = '../_figures/'

f_lp = 0.5    # Hz; jinc-corrected single disc is valid to ~lambda=FOV (2.915 m -> ~0.7 Hz)
f_hp = 0.08   # Hz; lower admits 1/k^2-amplified low-f noise
MAX_DEV = 0.10  # fractional Hm0 spread across the three Riegl units that rejects a record

ds_omnispect = xr.open_dataset(path+'ASIT2019_omnidirectional_spectra.nc')

mask = np.zeros((len(ds_omnispect['frequency']),1))
mask[(ds_omnispect['frequency'][:]>f_hp)&(ds_omnispect['frequency'][:]<f_lp)] = 1
df = np.median(np.diff(ds_omnispect['frequency'][:]))

Hm0_lidar = 4*np.sqrt(np.sum(mask*ds_omnispect['F_f_m2_Hz_lidar'][:].data,axis=0)*df)

P_members = lidar_member_spectra(path)                       # (n_f, n_run, 3)

# Reject records where one Riegl unit reports a different sea state: at 1.6 m
# spacing the three units sample the same water. The band is this figure's own
# [f_hp, f_lp], so the Tp figure gates on a different band.
lidar_bad, lidar_dev, _ = lidar_consistency_flag(
    P_members, ds_omnispect['frequency'][:].data, f_lo=f_hp, f_hi=f_lp,
    max_dev=MAX_DEV)

inds_exclude_base = (Hm0_lidar < 0.2) | (Hm0_lidar > 5)
inds_exclude = inds_exclude_base | lidar_bad
# unit-disagreement rejections only; the remainder of lidar_bad has too few
# live units, which carry no reference value in the first place
_n_lidar_rej = int((np.isfinite(lidar_dev) & (lidar_dev > MAX_DEV)
                    & ~inds_exclude_base).sum())
Hm0_lidar[inds_exclude] = np.nan
print(f'  lidar-consistency rejections (otherwise-passing records): {_n_lidar_rej} runs')

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

# Per-record uncertainty on the lidar Hm0 from the spread across the three Riegl
# instruments. sigma_x is the published (median-of-three) value's error; sigma_y
# assumes E-PSS carries a single instrument's sampling precision -- but that is
# only a starting shape: `calibrate=True` rescales it until the reduced
# chi-square is unity, so the excess scatter is charged to E-PSS.
Hm0_members = 4*np.sqrt(np.sum(mask[:,:,None]*P_members, axis=0)*df)   # (n_run, 3)
sigma_single = np.nanstd(Hm0_members, axis=1, ddof=1)
sigma_x = sigma_single*MEDIAN3_SIGMA_FACTOR
sigma_y = sigma_single

# (R^2, RMSE, slope, bias) per gain category; the slope is replaced by the
# errors-in-variables estimate, since OLS on x treats the lidar as exact and so
# attenuates the slope by the reference's own noise
metrics = [scatter_metrics(Hm0_lidar, y)
           for y in (Hm0_no_gain, Hm0_lab_gain, Hm0_emp_gain)]
yorks = [york_fit(Hm0_lidar, y, sigma_x, sigma_y, calibrate=True)
         for y in (Hm0_no_gain, Hm0_lab_gain, Hm0_emp_gain)]
metrics = [(r2, rmse, yk['slope'], bias)
           for (r2, rmse, _slope, bias), yk in zip(metrics, yorks)]
for suffix, yk in zip(('none','lab','emp'), yorks):
    print(f'  Hm0 {suffix:4s}: York slope {yk["slope"]:.3f} +/- {yk["se_slope"]:.3f}, '
          f'intercept {yk["intercept"]:+.3f}, chi2_red {yk["chi2_reduced"]:.2f}, '
          f'sigma_y inflation x{yk["sy_inflation"]:.1f}, n={yk["n"]}')

g = sns.lmplot(
    data=data,
    x="lidar", y="EPSS", hue="DoLP gain",
    scatter_kws = {'alpha': 0.5,'s':10},
    fit_reg=False,
    height=fullwidth/2,
    aspect=1,
    legend=False
)

# York fit line and 95% confidence band per gain; hue order in the frame is
# none, lab, empirical. The band pivots about each fit's weighted centroid.
axlim = (0.0, 4.0)
_xl = np.linspace(axlim[0], axlim[1], 200)
for yk, c in zip(yorks, color_list[:3]):
    lo, hi = york_band(yk, _xl)
    plt.fill_between(_xl, lo, hi, color=c, alpha=0.15, linewidth=0)
    plt.plot(_xl, yk['intercept'] + yk['slope']*_xl, '-', color=c, alpha=0.75,
             linewidth=1.5)

plt.plot([0,5],[0,5],'--',color='k')

# Export the empirical-gain headline metrics as LaTeX macros for paper.tex
_tq = _stats.t.ppf(0.975, max(yorks[2]['n'] - 2, 1))     # 95% CI on the slope
r2_emp, rmse_emp, slope_emp, bias_emp = metrics[2]
write_tex_macros('Hm0_values.tex', {
    'HmRtwoEmp':  f'{r2_emp:.2f}',
    'HmRMSEemp':  f'{rmse_emp:.2f}',
    'HmSlopeEmp': f'{slope_emp:.2f}',
    'HmSlopeSE':  f'{yorks[2]["se_slope"]:.2f}',
    'HmSlopeCIlo': f'{yorks[2]["slope"] - _tq*yorks[2]["se_slope"]:.2f}',
    'HmSlopeCIhi': f'{yorks[2]["slope"] + _tq*yorks[2]["se_slope"]:.2f}',
    'HmSigInfl':  f'{yorks[2]["sy_inflation"]:.0f}',
    'HmBiasEmp':  f'{bias_emp:.2f}',
    'HmNfit':     f'{yorks[2]["n"]:d}',
    'HmNrej':     f'{_n_lidar_rej:d}',
}, source='plot_Hm0_scatterplots_lidar_EPSS.py')

draw_metrics_box(plt.gca(), metrics, ['none', 'lab', 'emp'], color_list[:3],
                 ('m', 'm'), box_xy=(0.013, 0.713), box_w=0.69, box_h=0.277,
                 col_step=0.15, unit_dx=0.12, fsize=fsize)

plt.xticks(np.linspace(0,4,9))
plt.yticks(np.linspace(0,4,9))
plt.xlim(*axlim)
plt.ylim(*axlim)
plt.xlabel(r'$H_{m0}$, lidar [m]')
plt.ylabel(r'$H_{m0}$, E-PSS [m]')

sns.despine(right=False,top=False)

plt.savefig(figpath + 'Hm0_comparison_lidar_epss.pdf',bbox_inches='tight')

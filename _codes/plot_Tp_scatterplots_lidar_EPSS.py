"""
Scatterplot of the peak period Tp: E-PSS vs. lidar reference.
Compares no/lab/empirical DoLP gain corrections.

Tp follows Young (1995): int E^q df / int f E^q df with q = 4. The E^q weighting
concentrates the integral near the spectral maximum, so this behaves as a peak
period while remaining an integral quantity -- unlike 1/argmax(E), it does not
depend on which single bin happens to be highest.
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

f_lp = 0.7    # Hz; jinc-corrected single disc is valid to ~lambda=FOV (2.915 m -> ~0.7 Hz)
f_hp = 0.10   # Hz; above the 1/k^2-amplified low-f noise bump
MAX_DEV = 0.10  # fractional Hm0 spread across the three Riegl units that rejects a record

peak_q = 4    # Young (1995) peak-period weighting exponent

ds_omnispect = xr.open_dataset(path+'ASIT2019_omnidirectional_spectra.nc')

f_Hz = ds_omnispect['frequency'][:].data
nf = len(f_Hz)
mask = np.ones((nf,1))
mask[(ds_omnispect['frequency'][:]<f_hp)|(ds_omnispect['frequency'][:]>f_lp)] = np.nan
df = np.median(np.diff(f_Hz))

f2 = f_Hz.reshape(nf,1)**2   # second spectral moment weight (m2 = int f^2 F df)


def Tp_young(F, q=peak_q):
    """Peak period [s] after Young (1995): int E^q df / int f E^q df."""
    E = np.nan_to_num(mask*F)**q
    num = np.nansum(E, axis=0)
    den = np.nansum(f_Hz.reshape(nf,1)*E, axis=0)
    return np.divide(num, den, out=np.full(F.shape[1], np.nan), where=den > 0)


Hm0_no_gain = 4*np.sqrt((np.nansum(mask*ds_omnispect['F_f_m2_Hz_no_gain'][:].data,axis=0)*df))
Hm0_lab_gain = 4*np.sqrt((np.nansum(mask*ds_omnispect['F_f_m2_Hz_lab_gain'][:].data,axis=0)*df))
Hm0_emp_gain = 4*np.sqrt((np.nansum(mask*ds_omnispect['F_f_m2_Hz_empirical_gain'][:].data,axis=0)*df))
Hm0_lidar = 4*np.sqrt((np.nansum(mask*ds_omnispect['F_f_m2_Hz_lidar'][:].data,axis=0)*df))

Tp_no_gain = Tp_young(ds_omnispect['F_f_m2_Hz_no_gain'][:].data)
Tp_lab_gain = Tp_young(ds_omnispect['F_f_m2_Hz_lab_gain'][:].data)
Tp_emp_gain = Tp_young(ds_omnispect['F_f_m2_Hz_empirical_gain'][:].data)
Tp_lidar = Tp_young(ds_omnispect['F_f_m2_Hz_lidar'][:].data)

# Tm02 = sqrt(m0/m2); used only for the record gate below
Tm02_lidar = np.sqrt(np.nansum(mask*ds_omnispect['F_f_m2_Hz_lidar'][:].data,axis=0)/np.nansum(mask*f2*ds_omnispect['F_f_m2_Hz_lidar'][:].data,axis=0))

P_members = lidar_member_spectra(path)                       # (n_f, n_run, 3)

# Reject records where one Riegl unit reports a different sea state: at 1.6 m
# spacing the three units sample the same water. Band-limited Hm0 is the
# discriminator; the band is this figure's own [f_hp, f_lp], so the Hm0 figure
# gates on a different band and the two need not reject the same records.
lidar_bad, lidar_dev, _ = lidar_consistency_flag(P_members, f_Hz, f_lo=f_hp,
                                                 f_hi=f_lp, max_dev=MAX_DEV)

inds_exclude_base = ((Tm02_lidar > 10) | (Tm02_lidar < 2)
                     | (Hm0_emp_gain < 0.2) | (Hm0_lidar < 0.2))
inds_exclude = inds_exclude_base | lidar_bad
# unit-disagreement rejections only; the remainder of lidar_bad has too few
# live units, which carry no reference value in the first place
_n_lidar_rej = int((np.isfinite(lidar_dev) & (lidar_dev > MAX_DEV)
                    & ~inds_exclude_base).sum())
Tp_lidar[inds_exclude] = np.nan
print(f'  lidar-consistency rejections (otherwise-passing records): {_n_lidar_rej} runs')

data_size = len(Tp_lidar)

big_reference = np.concatenate((Tp_lidar,Tp_lidar,Tp_lidar))
big_test = np.concatenate((Tp_no_gain,Tp_lab_gain,Tp_emp_gain))
category = np.concatenate((np.full(data_size, 'none'),np.full(data_size, 'lab'),np.full(data_size, 'empirical')))

data = pd.DataFrame({
    'lidar': big_reference,
    'EPSS': big_test,
    'DoLP gain': category
})

# Per-record uncertainty on the lidar Tp, taken empirically from the spread
# across the three Riegl instruments. sigma_x is the published (median-of-three)
# value's error; sigma_y assumes E-PSS carries the sampling precision of a single
# instrument -- but that is only a starting shape: `calibrate=True` rescales it
# until the reduced chi-square is unity, so the excess scatter is charged to
# E-PSS rather than to the reference.
Tp_members = np.stack([Tp_young(P_members[:, :, li]) for li in range(P_members.shape[2])])
sigma_single = np.nanstd(Tp_members, axis=0, ddof=1)
sigma_x = sigma_single*MEDIAN3_SIGMA_FACTOR
sigma_y = sigma_single

# (R^2, RMSE, slope, bias) per gain category; the slope is replaced below by the
# errors-in-variables estimate, since OLS on x treats the lidar as exact and so
# attenuates the slope by the reference's own noise
metrics = [scatter_metrics(Tp_lidar, y)
           for y in (Tp_no_gain, Tp_lab_gain, Tp_emp_gain)]
yorks = [york_fit(Tp_lidar, y, sigma_x, sigma_y, calibrate=True)
         for y in (Tp_no_gain, Tp_lab_gain, Tp_emp_gain)]
metrics = [(r2, rmse, yk['slope'], bias)
           for (r2, rmse, _slope, bias), yk in zip(metrics, yorks)]
for (suffix, yk) in zip(('none','lab','emp'), yorks):
    print(f'  Tp {suffix:4s}: York slope {yk["slope"]:.3f} +/- {yk["se_slope"]:.3f}, '
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
axlim = (3.0, 11.0)
_xl = np.linspace(axlim[0], axlim[1], 200)
for yk, c in zip(yorks, color_list[:3]):
    lo, hi = york_band(yk, _xl)
    plt.fill_between(_xl, lo, hi, color=c, alpha=0.15, linewidth=0)
    plt.plot(_xl, yk['intercept'] + yk['slope']*_xl, '-', color=c, alpha=0.75,
             linewidth=1.5)

plt.plot([0,15],[0,15],'--',color='k')

# Export the empirical-gain headline metrics as LaTeX macros for paper.tex
_tq = _stats.t.ppf(0.975, max(yorks[2]['n'] - 2, 1))     # 95% CI on the slope
r2_emp, rmse_emp, slope_emp, bias_emp = metrics[2]
write_tex_macros('Tp_values.tex', {
    'TpRtwoEmp':  f'{r2_emp:.2f}',
    'TpRMSEemp':  f'{rmse_emp:.2f}',
    'TpSlopeEmp': f'{slope_emp:.2f}',
    'TpSlopeSE':  f'{yorks[2]["se_slope"]:.2f}',
    'TpSlopeCIlo': f'{yorks[2]["slope"] - _tq*yorks[2]["se_slope"]:.2f}',
    'TpSlopeCIhi': f'{yorks[2]["slope"] + _tq*yorks[2]["se_slope"]:.2f}',
    'TpSigInfl':  f'{yorks[2]["sy_inflation"]:.0f}',
    'TpBiasEmp':  f'{bias_emp:.2f}',
    'TpNfit':     f'{yorks[2]["n"]:d}',
    'TpNrej':     f'{_n_lidar_rej:d}',
}, source='plot_Tp_scatterplots_lidar_EPSS.py')

draw_metrics_box(plt.gca(), metrics, ['none', 'lab', 'emp'], color_list[:3],
                 ('s', 's'), box_xy=(0.339, 0.012), box_w=0.651, box_h=0.278,
                 col_step=0.14, unit_dx=0.11, fsize=fsize)

plt.xticks(np.linspace(3,11,9))
plt.yticks(np.linspace(3,11,9))
plt.xlim(*axlim)
plt.ylim(*axlim)

plt.xlabel(r'$T_{p}$, lidar [s]')
plt.ylabel(r'$T_{p}$, E-PSS [s]')

sns.despine(right=False,top=False)

plt.savefig(figpath + 'Tp_comparison_lidar_epss.pdf',bbox_inches='tight')

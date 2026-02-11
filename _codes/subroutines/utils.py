"""
Created on Wed Oct 15 13:01:03 2025

@author: nathanlaxague
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

from typing import Union

import scipy.signal as signal
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt
from scipy.signal import detrend
from scipy import interpolate

# %%

# Mueller calculus
# N. J. M. Laxague 2019
# From Zappa et al. 2008 (Z08)
# Following Kattawar & Adams 1989 (KA89)

def mueller_calc_full(n,Ssky,Sup):

    theta_i = np.linspace(0,np.pi/2,10000)
    theta_t = np.asin(np.sin(theta_i)/n)
    
    # Mueller matrix element equations, KA89
    # taken from Eq. 3 of Z08
    alpha = 1/2*(np.tan(theta_i-theta_t)/np.tan(theta_i+theta_t))**2;
    eta = 1/2*(np.sin(theta_i-theta_t)/np.sin(theta_i+theta_t))**2;
    
    alpha_prime = 1/2*(2*np.sin(theta_t)*np.sin(theta_i)/(np.sin(theta_i+theta_t)*np.cos(theta_i-theta_t)))**2;
    eta_prime = 1/2*(2*np.sin(theta_t)*np.sin(theta_i)/np.sin(theta_t+theta_i))**2;
    
    gamma_Re = (np.tan(theta_i-theta_t)*np.sin(theta_i-theta_t))/(np.tan(theta_i+theta_t)*np.sin(theta_i+theta_t));
    gamma_Re_prime = 4*(np.sin(theta_t)**2*np.sin(theta_i)**2)/(np.sin(theta_t+theta_i)**2*np.cos(theta_t-theta_i)**2);
    
    # Stokes parameter components, reflected and transmitted radiance
    # taken from Eq. 4 of Z08
    S0_Re = Ssky[0]*(alpha+eta) + Ssky[1]*(alpha-eta);
    S1_Re = Ssky[0]*(alpha-eta) + Ssky[1]*(alpha+eta);
    S2_Re = Ssky[2]*gamma_Re;
    S3_Re = Ssky[3]*gamma_Re;
    
    S0_Tr = Sup[0]*(alpha_prime+eta_prime) + Sup[1]*(alpha_prime-eta_prime);
    S1_Tr = Sup[0]*(alpha_prime-eta_prime) + Sup[1]*(alpha_prime+eta_prime);
    S2_Tr = Sup[2]*gamma_Re_prime;
    S3_Tr = Sup[3]*gamma_Re_prime;
    
    S0 = S0_Re + S0_Tr;
    S1 = S1_Re + S1_Tr;
    S2 = S2_Re + S2_Tr;
    S3 = S3_Re + S3_Tr;
    
    # DOLP calculation
    DoLP = np.sqrt(S1**2+S2**2+S3**2)/S0;
    DoLP[0] = DoLP[1] - (DoLP[2]-DoLP[1]);
    
    # Assign output
    out_theta = 180/np.pi*theta_i;
    out_DOLP = DoLP;
    
    return(out_theta,out_DOLP)

# %%

# Given ten-meter wind speed in m/s, returns wave slope 
# joint probability density function and upwind/crosswind mean square slope

# Procedure following Cox & Munk [1954]

# N. Laxague 2023-2025

def compute_gram_charlier_slope_pdf(U10_m_s):
    
    slope_centers = np.linspace(-1,1,num=200)
    
    mss_up = 1e-3 + 3.16*1e-3*U10_m_s
    mss_cross = 3*1e-3 + 1.85*1e-3*U10_m_s
    
    c21 = -9.1e-4*U10_m_s**2
    c03 = -0.45*(1+np.exp(7-U10_m_s))**-1
    c40 = 0.3
    c04 = 0.4
    c22 = 0.12
    
    # Create meshgrid for xi and zeta (normalized cross and upwind mss)
    xi, zeta = np.meshgrid(slope_centers / np.sqrt(mss_cross), slope_centers / np.sqrt(mss_up))

    coeff = (2 * np.pi * np.sqrt(mss_up) * np.sqrt(mss_cross)) ** -1
    PDF_cross_along = coeff * np.exp(-(xi**2 + zeta**2) / 2) * (
            1 +
            -0.5 * c21 * (xi**2 - 1) * zeta +
            -1/6 * c03 * (zeta**3 - 3 * zeta) +
            1/24 * c40 * (xi**4 - 6 * xi**2 + 3) +
            1/24 * c04 * (zeta**4 - 6 * zeta**2 + 3) +
            1/4 * c22 * (xi**2 - 1) * (zeta**2 - 1)
        )
    
    wave_slope_PDF = xr.DataArray(
        PDF_cross_along,
        coords = {
            'slope_cross': slope_centers,
            'slope_up': slope_centers
            },
        dims = {'slope_cross','slope_up'}
        )
    
    wave_slope_PDF['PDF_cross_along'] = wave_slope_PDF/wave_slope_PDF.integrate('slope_cross').integrate('slope_up')
    
    return wave_slope_PDF, mss_cross, mss_up

# %%

# Given water surface slope joint probability density function, compute
# least-squares Gram-Charlier fit and skewness/kurtosis coefficients

# Procedure following Cox & Munk [1954]

# N. Laxague 2023-2025

def fit_gram_charlier_slope_pdf(slope_centers, P_slope_c_u, mss_u, mss_c):
    
    # Create meshgrid for xi and zeta
    xi, zeta = np.meshgrid(slope_centers / np.sqrt(mss_c), slope_centers / np.sqrt(mss_u))

    # Function to fit
    def fit(b, x, y):
        coeff = (2 * np.pi * np.sqrt(mss_u) * np.sqrt(mss_c)) ** -1
        return coeff * np.exp(-(x**2 + y**2) / 2) * (
            1 +
            -0.5 * b[0] * (x**2 - 1) * y +
            -1/6 * b[1] * (y**3 - 3 * y) +
            1/24 * b[2] * (x**4 - 6 * x**2 + 3) +
            1/24 * b[3] * (y**4 - 6 * y**2 + 3) +
            1/4 * b[4] * (x**2 - 1) * (y**2 - 1)
        )

    # Least-Squares cost function
    def cost_function(b):
        return np.sum((fit(b, xi, zeta) - P_slope_c_u) ** 2)

    # Minimize Least-Squares
    initial_guess = np.zeros(5)
    result = minimize(cost_function, initial_guess)
    
    # Calculate fitted values and residuals
    P_fit = fit(result.x, xi, zeta)
    residuals = P_fit - P_slope_c_u

    # Calculate goodness of fit metrics
    ss_res = np.sum(residuals ** 2)  # Residual sum of squares
    ss_tot = np.sum((P_slope_c_u - np.mean(P_slope_c_u)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)  # R-squared
    rmse = np.sqrt(ss_res / np.size(P_slope_c_u))  # Root Mean Square Error

    # Send G-C fit and skewness/kurtosis coefficients to output structure
    out_struc = {
        'P_fit': P_fit,                    # Gram-Charlier expansion fit
        'c21': result.x[0],                # skewness, upwind
        'c03': result.x[1],                # skewness, upwind
        'c40': result.x[2],                # kurtosis
        'c04': result.x[3],                # kurtosis
        'c22': result.x[4],                # kurtosis
        'R_squared': r_squared,            # R-squared value
        'RMSE': rmse                       # Root Mean Square Error
    }

    return out_struc
    
# %%

# Given wave radian frequency, water depth, and current speed, obtains celerity and group speed

# N. Laxague 2026

def lindisp_with_current(omega, h, current_m_s):
    """
    Linear dispersion relation with current
    
    Parameters:
    -----------
    omega : array-like
        Angular frequency
    h : array-like
        Water depth
    current_m_s : array-like
        Current velocity
    
    Returns:
    --------
    c : array
        Phase speed
    cg : array
        Group velocity
    """
    # Ensure inputs are column vectors
    omega[omega == 0] = np.nan
    omega = np.atleast_1d(omega).flatten()
    h = np.atleast_1d(h).flatten()
    current_m_s = np.atleast_1d(current_m_s).flatten()
    
    # Constants
    g = 9.806  # gravitational acceleration
    rho_w = 1020  # water density
    sigma = 0.072  # surface tension
    
    # Wave number vector
    k_vec = np.logspace(-4, 4, 100)
    
    # Dispersion relation with current
    omega_disp = np.sqrt((g*k_vec + sigma/rho_w*k_vec**3) * np.tanh(k_vec*h)) + k_vec*current_m_s
    
    # Create interpolation function
    k_from_omega = interpolate.interp1d(omega_disp, k_vec, kind='cubic')
    
    # Find wave number for given frequencies
    k = k_from_omega(omega)
    
    # Phase speed
    c = omega / k
    
    # Group velocity
    cg = c/2 * (1 + (2*k*h) / np.sinh(2*k*h))
    
    return c, cg

# %%

# Fourier‑based conversion of surface‑wave slope to elevation.

#     Parameters
#     ----------
#     slope_x, slope_y : array_like
#         1‑D arrays of surface‑wave slope (m‑1) along x and y.
#         Must have the same length.
#     water_depth_m : float
#         Water depth (m).  Use np.inf for deep water.
#     dt : float
#         Sampling interval (s).
#     f_lp : float
#         Low‑pass cut‑off frequency (Hz).  Energy above this is
#         attenuated.
#     f_hp : float
#         High‑pass cut‑off frequency (Hz).  Energy below this is
#         attenuated.

#     Returns
#     -------
#     eta_slope : ndarray
#         1‑D array of surface‑wave elevation (m), same length as
#         the input time series.
# 
# N. Laxague 2025

def slope_to_elev(
    slope_x: np.ndarray,
    slope_y: np.ndarray,
    water_depth_m: float,
    dt: float,
    f_lp: float,
    f_hp: float,
) -> np.ndarray:

    slope_x = np.asarray(slope_x, dtype=float).reshape(-1)
    slope_y = np.asarray(slope_y, dtype=float).reshape(-1)

    if slope_x.size != slope_y.size:
        raise ValueError("slope_x and slope_y must have the same length")

    N = slope_x.size

    # If N is odd → drop last sample so that N is even
    if N % 2 == 1:
        slope_x = slope_x[:-1]
        slope_y = slope_y[:-1]
        N = slope_x.size

    # Half‑length for the shifted frequency array
    half_N = int(np.ceil(N / 2))

    # Define frequency vector
    f = np.arange(-half_N, half_N, dtype=float) / (N * dt)

    # Positive‑frequency part (k is defined only for ω ≥ 0)
    f_pos = f[half_N:]            # length = half_N
    C_m_s_disp, Cg_m_s_disp = lindisp_with_current(2*np.pi*f_pos,water_depth_m,0)
    k_rad_m_disp = 2*np.pi*f_pos / C_m_s_disp

    # Full wavenumber array (mirror symmetry)
    k = np.concatenate([-k_rad_m_disp[::-1], k_rad_m_disp])
    
    # Remove linear trend
    s_x = detrend(slope_x)
    s_y = detrend(slope_y)

    # Compute FFT and shift
    S_x = np.fft.fftshift(np.fft.fft(s_x))
    S_y = np.fft.fftshift(np.fft.fft(s_y))

    # Convert slope to elevation in Fourier space
    S_x = 1j * S_x / k
    S_y = 1j * S_y / k

    # Replace infinities / NaNs caused by k=0 with zero
    S_x = np.where(np.isfinite(S_x), S_x, 0.0)
    S_y = np.where(np.isfinite(S_y), S_y, 0.0)

    # Produce bandpass filter
    # Find first index where f >= f_lp (low‑pass cutoff)
    ind_lp = np.argmax(f >= f_lp)
    # Find last index where f <= f_hp (high‑pass cutoff)
    ind_hp = np.argmax(f <= f_hp)
    
    ind_lp = np.searchsorted(f, f_lp, side="left")
    ind_hp = np.searchsorted(f, f_hp, side="right") - 1

    # Build filter in the *half* domain
    inds = np.arange(1, half_N + 1, dtype=float)

    highpass_filt = 1 / (1 + np.exp(-5 * (inds - ind_hp + half_N)))
    lowpass_filt = 1 - 1 / (1 + np.exp(-5 * (inds - ind_lp + half_N)))

    ind_mid = int(np.floor((ind_lp + ind_hp - len(f)) / 2))
    bandpass_filt = np.concatenate(
        (highpass_filt[: ind_mid], lowpass_filt[ind_mid:])
    )
    combined_filt = np.concatenate([bandpass_filt[::-1], bandpass_filt])

    # Apply the filter
    S_x *= combined_filt
    S_y *= combined_filt

    # Combine Fourier amplitudes and produce real water surface elevation timeseries
    elev_combined_Fourier = S_x + S_y
    eta_slope = np.real(np.fft.ifft(np.fft.ifftshift(elev_combined_Fourier)))

    return eta_slope


# %%

# Given water surface elevation and two-component (earth-referenced) slope,
# computes frequency-directional spectrum via Maximum Entropy Method (MEM)
# technique

# A cut-up of code written by N. Laxague 2025, with large chunks taken from the
# E-WDM code base by D. Pelaez-Zapata

def compute_dirspec_EPSS(elev_m,slope_east,slope_north,fs_Hz,lowcut,highcut,nfft,nperseg,smoothnum=3):

    num_samples = len(elev_m)

    t_s = np.linspace(0,num_samples/fs_Hz,num_samples)
    
    slope_east = bandpass_filter(slope_east,lowcut,highcut,fs_Hz)
    slope_north = bandpass_filter(slope_north,lowcut,highcut,fs_Hz)
    elev_m = bandpass_filter(elev_m,lowcut,highcut,fs_Hz)

    # creating dataset (slope timeseries)
    dataset = xr.Dataset(
        data_vars = {
            "eastward_slope": ("time", slope_east),
            "northward_slope": ("time", slope_north),
            "surface_elevation": ("time", elev_m)
            },
        coords = {"time": t_s},
        attrs = {"sampling_rate": fs_Hz}
        )

    csp = ClassicSpectralAnalysis(dataset, fs=dataset.sampling_rate, nfft=nfft, nperseg=nperseg)

    # computing the cross-spectral matrix
    Phi = csp.cross_spectral_matrix()

    # computing the directional moments, aka first-five Fourier coefficients
    moments = csp.directional_moments(Phi)

    D_EPSS = mem_distribution(moments,smoothing=smoothnum)
    D_EPSS.data = np.nan_to_num(D_EPSS.data,0)
    
    D_EPSS = D_EPSS/D_EPSS.integrate("frequency").integrate("direction")
    
    # compute the directional wave spectrum
    F_EPSS = moments["a0"] * D_EPSS
    
    return F_EPSS

    # Function to create a Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply the bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# %%

# Given frequency-directional spectra computed directly from wave slope fields
# and inferred from long wave surface elevation and tilt, attempts to resolve
# +/-180 degree directional ambiguity in E-PSS spectra by using the 'direct'
# spectrum as the tiebreaker

# N. Laxague 2025

def trim_EPSS_dirspec(F_EPSS,Ffd_direct,theta_halfwidth,f_cut_high,smoothnum=3):
    
    Ff_EPSS = F_EPSS.integrate("direction")
    
    D_EPSS = ((F_EPSS.T / F_EPSS.integrate("direction")).rolling(frequency=smoothnum, center=True).median()).T    
    D_direct = ((Ffd_direct.T / Ffd_direct.integrate("direction")).rolling(frequency=9, center=True).median()).T
    
    D_EPSS.data = np.nan_to_num(D_EPSS.data,0)
    D_direct.data = np.nan_to_num(D_direct.data,0)
    
    D_EPSS_reference = D_EPSS.copy()
    
    inds_consider_short = D_direct["frequency"] > f_cut_high    
    Ftheta_direct_short = Ffd_direct[inds_consider_short,:].integrate("frequency")
        
    ind_peak_short = np.argmax(Ftheta_direct_short.data)
        
    wavedir = D_EPSS["direction"].copy()
    wavedir_short = D_direct["direction"].copy()
    freqs = D_EPSS["frequency"].copy()
        
    ind_cut_direct = next(x[0] for x in enumerate(freqs) if x[1] > f_cut_high)-1
    inds_cut_direct = freqs > f_cut_high
    D_EPSS.data[inds_cut_direct,:] = 0

    
    Dslice = D_EPSS.data[ind_cut_direct,:]
    wavedir_diff = np.mod(np.abs(wavedir.data-wavedir_short.data[ind_peak_short])+360,360)
    inds_trim = (wavedir_diff > theta_halfwidth) & (wavedir_diff < 360-theta_halfwidth)
    Dslice[inds_trim] = 0
    ind_peak_first = np.argmax(Dslice)
    wavedir_diff = np.mod(np.abs(wavedir.data-wavedir_short.data[ind_peak_first])+360,360)
    inds_trim = (wavedir_diff > theta_halfwidth) & (wavedir_diff < 360-theta_halfwidth)
    D_EPSS.data[ind_cut_direct,inds_trim] = 0
    
    for index in np.arange(1,ind_cut_direct):
        i = ind_cut_direct - index
        Dslice = D_EPSS.data[i+1,:]
        
        ind_peak_next = np.argmax(Dslice)
        
        wavedir_diff = np.mod(np.abs(wavedir.data-wavedir.data[ind_peak_next])+360,360)
        inds_trim = (wavedir_diff > theta_halfwidth) & (wavedir_diff < 360-theta_halfwidth)
        D_EPSS.data[i,inds_trim] = 0
        
    D_EPSS = D_EPSS*np.sum(D_EPSS_reference)/np.sum(D_EPSS)
    
    # Re-compute the directional wave spectrum
    F_EPSS = Ff_EPSS * D_EPSS

    return F_EPSS

# %%

# Given frequency-directional or wavenumber-directional spectrum, computes the
# mean wave direction and spectral spreading width

# Also takes as an input 'theta_halfwidth', a frequency (or wavenumber) dependent
# mask which is designed to mitigate the effect of +/-180 degree ambiguity

# N. Laxague 2025

def compute_mean_wave_direction_and_spreading(F_dirspec,theta_halfwidth,smoothnum=3):
    
    F_dirspec.data = np.nan_to_num(F_dirspec.data,0)
    spec_energy_density = F_dirspec.data
    
    wavedir = F_dirspec["direction"].copy()
    dtheta = np.median(np.diff(wavedir.data))
    
    if 'frequency' in F_dirspec.coords:
        
        fourier_scale = F_dirspec["frequency"].data
        fourier_scale_name = 'frequency'
        
    if 'wavenumber' in F_dirspec.coords:
        
        fourier_scale = F_dirspec["wavenumber"].data
        fourier_scale_name = 'wavenumber'
        spec_energy_density = spec_energy_density*np.reshape(fourier_scale,(1,len(wavedir)))
            
    D_array = ((F_dirspec.T / F_dirspec.integrate("direction")).rolling(frequency=smoothnum, center=True).median()).T
    D_array.data = np.nan_to_num(D_array.data,0)
    
    Dtheta = D_array.integrate("frequency")
    ind_p = np.argmax(Dtheta.data)
    
    theta_super = np.concatenate((wavedir-360,wavedir,wavedir+360),axis=0)
    theta_rel = theta_super - wavedir.data[ind_p]
    D_array_super = np.concatenate((D_array.data,D_array.data,D_array.data),axis=1)
    F_array_super = np.concatenate((spec_energy_density,spec_energy_density,spec_energy_density),axis=1)
    
    inds_keep = (theta_rel >= -180) & (theta_rel < 180)
    theta_rel = theta_rel[inds_keep]
    D_array_super = D_array_super[:,inds_keep]
    F_array_super = F_array_super[:,inds_keep]
    
    D_array["direction"] = theta_rel
    D_array.data = D_array_super

    MWD = np.sum(np.reshape(D_array.direction.data,(1,len(D_array.direction.data)))*F_array_super)/np.sum(F_array_super) + wavedir.data[ind_p]

    wavedir_diff = np.abs(wavedir.data)
    inds_trim = (wavedir_diff > theta_halfwidth) & (wavedir_diff < 360-theta_halfwidth)

    D_array_downwave = D_array.copy()
    upwave_val = np.sum(D_array.data[:,inds_trim])
    downwave_val = np.sum(D_array.data[:,~inds_trim])
    D_array_downwave.data = D_array_downwave.data*(upwave_val+downwave_val)/downwave_val
    D_array_downwave.data[:,inds_trim] = 1e-10

    theta_nought_array = np.sum(np.reshape(D_array_downwave.direction.data,(1,len(D_array_downwave.direction.data)))*D_array_downwave.data,axis=1)*dtheta
    theta_array_mat = np.tile(np.reshape(D_array_downwave.direction.data,(1,len(D_array_downwave.direction.data))),(len(theta_nought_array),1))
    theta_nought_array_mat = np.tile(np.reshape(theta_nought_array,(len(theta_nought_array),1)),(1,72))
    d_theta2_array = (theta_array_mat-theta_nought_array_mat)**2

    sigma_theta_array = np.sqrt(np.sum(d_theta2_array*D_array_downwave.data,axis=1)*dtheta)

    sigma_theta_array[sigma_theta_array<1.0] = np.nan
    
    spread = xr.DataArray(
        sigma_theta_array,
        name = 'sigma_theta',
        coords = {fourier_scale_name: fourier_scale},
        dims = fourier_scale_name,
        attrs = {"units": 'degrees'},        
        )

    return MWD, spread

# %%

#
# The following code has been taken from the E-WDM code base, written by
# D. Peláez-Zapata
#
# %
# Classic directional spectral analysis
# -------------------------------------
# The following class implements the conventional cross-spectral analysis
# using Fourier transform. This class takes in a dataset containing eastward
# and northward displacements along with the surface elevation data. The
# input parameters are sampling frequency of the time series and number of
# Fourier components for analysis.
#
# Considering the three time series, :math:`x(t)`, :math:`y(t)` and
# :math:`z(t)` (We normally use :math:`\eta(t)` for the vertical but it is
# changed here to :math:`z(t)` for simplicity and convenience), the
# cross-spectral matrix is computed as:
#
# .. math:: \Phi(f) = \begin{bmatrix}
#                        S_{xx} & S_{xy} & S_{xz} \\
#                        S_{yx} & S_{yy} & S_{yz} \\
#                        S_{zx} & S_{zy} & S_{zz}
#                     \end{bmatrix}
#
# where :math:`S_{xy}(f)` is the Fourier cross-spectrum between
# :math:`x(t)` and :math:`y(t)`, respectively.
#
# Each cross-spectrum can be written in terms of a real (co-spectrum)
# and an imaginary (quad-spectrum) component:
#
# .. math:: S_{xy}(f) = C_{xy} + i Q_{xy}
#
# The auto-spectrum is the cross-spectrum of the same signal, and can be
# written as :math:`E_{xx}(f) = S_{xx} S_{xx}^*`, where :math:`*`
# denotes a complex conjugate.
#
# For typical buoy recordings, e.g., three dimensional wave-induced
# displacements, the circular moments can be written in terms of these auto-,
# co-, and quad-spectra, like:
#
# .. math:: a_0 = S_{zz}(f)
# .. math:: a_1 = \frac{Q_{xz}}{\sqrt{E_{zz} (E_{xx} + E_{yy})}}
# .. math:: a_2 = \frac{Q_{yz}}{\sqrt{E_{zz} (E_{xx} + E_{yy})}}
# .. math:: b_1 = \frac{E_{xx} - E_{yy}}{E_{xx} + E_{yy}}
# .. math:: b_2 = \frac{2 C_{xy}}{E_{xx} + E_{yy}}
#
# See `Kuik et al. (1988)`_ and Appendix C in `Peláez-Zapata et al. (2024)`_ for further details.
#
# .. _Kuik et al. (1988): https://doi.org/10.1175/1520-0485(1988)018<1020:AMFTRA>2.0.CO;2
# .. _Peláez-Zapata et al. (2024): https://theses.fr/2024UPASM004


class ClassicSpectralAnalysis(object):
    """This class implements the classic directional spectral analysis"""

    def __init__(self, dataset: xr.Dataset, fs, nfft, nperseg):
        self.dataset = dataset # time series
        self.fs = fs # sampling frequency
        self.nfft = nfft # number of fourier components
        self.nperseg = nperseg # length of each segment

    def cross_spectral_matrix(self) -> xr.Dataset:

        # extract variables from dataset
        x, y, z = (
            self.dataset["northward_slope"],
            self.dataset["eastward_slope"],
            self.dataset["surface_elevation"]
        )

        # constants
        fft_args = {
            "fs": self.fs,
            "detrend": "constant",
            "nfft": min(self.nfft, len(self.dataset["time"])),
            "nperseg": min(self.nperseg, len(self.dataset["time"]))
        }

        # auto-spectra
        frq, Sxx = signal.welch(x, **fft_args)
        frq, Syy = signal.welch(y, **fft_args)
        frq, Szz = signal.welch(z, **fft_args)

        # cross-spectra
        frq, Sxz = signal.csd(x, z, **fft_args)
        frq, Syz = signal.csd(y, z, **fft_args)
        frq, Sxy = signal.csd(x, y, **fft_args)

        return  xr.Dataset(
            coords = {
                "frequency": ("frequency", frq),
            },
            data_vars = {
                "Sxx": ("frequency", Sxx),
                "Syy": ("frequency", Syy),
                "Szz": ("frequency", Szz),
                "Sxz": ("frequency", Sxz),
                "Syz": ("frequency", Syz),
                "Sxy": ("frequency", Sxy),
            }
        )

    def directional_moments(self, Phi: xr.Dataset) -> xr.Dataset:
        Exx, Eyy, Ezz, Cxy, Qxz, Qyz = (
            np.real(Phi["Sxx"]), np.real(Phi["Syy"]), np.real(Phi["Szz"]),
            np.real(Phi["Sxy"]), np.imag(Phi["Sxz"]), np.imag(Phi["Syz"])
        )

        return  xr.Dataset(
            {
                "a0": Ezz,
                "a1": Qxz / np.sqrt(Ezz * (Exx + Eyy)),
                "b1": Qyz / np.sqrt(Ezz * (Exx + Eyy)),
                "a2": (Exx - Eyy) / (Exx + Eyy),
                "b2": 2 * Cxy / (Exx + Eyy)
            }
        )
    
# %%

# The following code has been taken from the E-WDM code base, written by
# D. Peláez-Zapata
# Tweaked to modify some plotting details

# plot directional wave spectrum {{{
def _smooth(E, ws=(5, 2)):
    """Applies a simple circular smoothing to 2D array.

    Args:
        E (ndarray): Input function.
        ws (tuple): Window size. For example, for a directional spectrum
            size E(360,64) and ws=(10,2), the filter acts averaging
            10 directions and 2 frequencies.

    Returns:
        ndarray: Function smoothed.
    """

    # define window
    nd, nf = ws
    if nf == nd:
        frqwin = signal.hamming(nf)
    else:
        frqwin = np.ones(nf)

    dirwin = signal.hamming(nd)
    window = frqwin[None,:] * dirwin[:,None]
    window = window / window.sum()

    # permorm convolution and return output
    return signal.convolve2d(E, window, mode='same', boundary='wrap')

def _get_axes(
        ax=None, rmin=0.1, rmax=0.5, rstep=0.1, angle=-135,
        color="0.8", as_period=False
    ):
    """Draw polar grid on specific axes.

    Args:
        ax (matplotlib.axes, optional): Axes object to draw the grid on.
            If None, a new figure and axes are created.
        rmin (float, optional): Minimum radius for the grid. Defaults to 0.1.
        rmax (float, optional): Maximum radius for the grid. Defaults to 0.5.
        rstep (float, optional): Step size for the radius. Defaults to 0.1.
        angle (int, optional): Angle for the radius labels. Defaults to -135.
        color (str, optional): Color of the grid lines and labels. Defaults to "0.8".
        as_period (bool, optional): If True, labels are formatted as wave periods.
            Defaults to False.

    Returns:
        tuple: A tuple containing the figure and axes objects if a new figure is created. Otherwise, returns None.
    """

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(5,5))

    ax.set_aspect("equal")

    for radii in np.arange(rmin, rmax+rstep, rstep):
        circle = plt.Circle(
            (0,0), radii, color=color, alpha=0.5,
            linestyle="dashed", fill=False, zorder=2
        )
        ax.add_artist(circle)
        if radii <= rmax:
            if as_period:
                radii_label = f"{1/radii:.1f}"
            else:
                radii_label = f"{radii:.2f}"
            ax.text(
                radii*np.cos(np.radians(angle)),
                radii*np.sin(np.radians(angle)),
                radii_label, fontsize="small",
                ha="center", va="center", zorder=3
            )

    ax.axhline(0, color=color, ls="dashed", alpha=0.5)
    ax.axvline(0, color=color, ls="dashed", alpha=0.5)
    ax.plot([0,1], [0,1], "--", c=color, alpha=0.5, transform=ax.transAxes)
    ax.plot([0,1], [1,0], "--", c=color, alpha=0.5, transform=ax.transAxes)

    _label_args = {"fontsize": "small", "ha": "center", "va": "center"}
    ax.text(0.50, 0.95, "N", transform=ax.transAxes, **_label_args)
    ax.text(0.95, 0.50, "E", transform=ax.transAxes, **_label_args)
    ax.text(0.50, 0.05, "S", transform=ax.transAxes, **_label_args)
    ax.text(0.05, 0.50, "W", transform=ax.transAxes, **_label_args)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if as_period:
        ax.set_ylabel("Wave period [s]")
    else:
        ax.set_ylabel("$f$ [Hz]")

    ax.set_xlim([-rmax+rstep, rmax-rstep])
    ax.set_ylim([-rmax+rstep, rmax-rstep])

    try:
        return fig, ax
    except NameError:
        pass


def _get_cmap(colors=None, N=256):
    """Return colormap for a given color list.

    Args:
        colors (list): List of colours.
        N (int): Number of colours

    Returns:
        Colormap object. If not colors passed, it returns viridis with white bottom.

    """

    if colors is None:
        colors = [(1, 1, 1), *plt.cm.viridis(np.linspace(0, 1, 8))]
    return mcolors.LinearSegmentedColormap.from_list('cmap', colors, N=N)

def _add_cbar(
        pc, ax, cax=None, style="outside", ticks=None, orientation="horizontal",
        label="$\\log_{10} E \\; \\mathrm{[m^2 Hz^{-1} deg^{-1}]}$"
    ):
    """Return a colorbar object"""

    if cax is None:
        if style == "inside":
            ticks = mticker.LinearLocator(ticks)
            orientation = "horizontal"
            cax = ax.inset_axes(
                [0.07, 0.92, 0.3, 0.035], transform=ax.transAxes
            )
        if style == "outside":
            ticks = mticker.AutoLocator()
            orientation = "vertical"
            cax = ax.inset_axes(
                [1.02, 0.0, 0.04, 1.0], transform=ax.transAxes
            )
    else:
        pass

    return plt.colorbar(
        pc, cax=cax, orientation=orientation, ticks=ticks, label=label
    )

def _add_wind_info(ax, wspd, wdir, color="k", wind_sea_radius=True):
    """Add some info relative to wind speed"""

    fwind = 9.8 / (2*np.pi * wspd)
    uwnd, vwnd = (
        fwind * np.sin(np.radians(wdir)), fwind * np.cos(np.radians(wdir))
    )

    ax.arrow(
        uwnd, vwnd, -uwnd, -vwnd, color=color, head_width=0.01,
        length_includes_head=True
    )

    if wind_sea_radius:
        circle = plt.Circle(
            (0,0), fwind, color=color, alpha=0.5,
            linestyle="dashed", fill=False
        )
        ax.add_artist(circle)
        
def _add_current_info(ax, curspd, curdir, color="red", wind_sea_radius=True):
    """Add some info relative to wind speed"""

    fcur = 9.8 / (2*np.pi * curspd)
    ucur, vcur = (
        fcur * np.sin(np.radians(curdir)), fcur * np.cos(np.radians(curdir))
    )

    ax.arrow(
        0, 0, ucur, vcur, color=color, head_width=0.01,
        length_includes_head=True
    )

    if wind_sea_radius:
        circle = plt.Circle(
            (0,0), fcur, color=color, alpha=0.5,
            linestyle="dashed", fill=False
        )
        ax.add_artist(circle)


#def plot_directional_spectrum(
#        da: xr.DataArray, frqs="frequency", dirs="direction", ax=None,
#        smooth=None, cmap=None, levels=30, vmin=None, vmax=None, contours=None,
#        colorbar=False, wspd=None, wdir=None, wind_sea_radius=None,
#        curspd=None, curdir=None, cbar_kw={}, axes_kw={}
#    ):
#    """Make a simple plot of a direcitonal wave spectrum"""

def plot_directional_spectrum(
        da: Union[xr.DataArray, np.ndarray],
        frqs: Union[str, np.ndarray] = "frequency",
        dirs: Union[str, np.ndarray] = "direction",
        ax = None,
        smooth = None,
        cmap = None,
        levels: int = 30,
        vmin: Union[float, int] = None,
        vmax: Union[float, int] = None,
        contours: Union[float, int] = None,
        colorbar: bool = False,
        wspd: Union[float, int] =None,
        wdir: Union[float, int] =None,
        wind_sea_radius: Union[float, int] = None,
        curspd: Union[float, int] = None,
        curdir: Union[float, int] = None,
        cbar_kw={},
        axes_kw={}
    ) -> None:
    """
    Make a simple plot of a directional wave spectrum.

    Args:
        da: Directional spectrum data.
        frqs: Frequency label name or numpy array.
        dirs: Direction label name or numpy array.
        ax (optional): Matplotlib axis object.
        smooth (tuple): Smoothing factor for visualisation. Defaults to None.
        cmap (optional): Colormap for the plot. Defaults to None.
        levels (int): Number of contour levels. Defaults to 30. If None, pseudo-color plot is made.
        vmin (int or float): Minimum value for colormap.
        vmax (int or float]): Maximum value for colormap.
        contours (int or float): Specific contour levels to plot on top.
        colorbar (bool): Whether to display colorbar. Defaults to False.
        wspd (int or float): Wind speed value to draw arrow. Defaults to None.
        wdir (int or float): Wind direction value in cartesian convention.
        wind_sea_radius (int or float):Whether to display wind sea separation radius.
        curspd (int or float): Current speed value.
        curdir (int or float): Current direction value in cartesian convention.
        cbar_kw (dict): Additional arguments for colorbar.
        axes_kw (dict): Additional arguments for plot axes.

    Returns:
        fig, ax
    """
    # get axis if not given
    if ax is None:
        fig, ax = _get_axes(**axes_kw)
    else:
        _get_axes(ax=ax, **axes_kw)

    # wrap spectra around the circle
    if da[dirs][0] != da[dirs][-1]:
        padded = da.pad({dirs: (1,0)}, mode="wrap")
    else:
        padded = da.copy()

    # calculate cartesian fx,fy to mimic polar coordinates
    _deg2rad = np.pi/180
    frqx = padded[frqs].data[:,None]*np.sin(padded[dirs].data[None,:]*_deg2rad)
    frqy = padded[frqs].data[:,None]*np.cos(padded[dirs].data[None,:]*_deg2rad)

    # smooth spectra
    if smooth:
        smoothed = xr.apply_ufunc(lambda x: _smooth(x, ws=smooth), padded)
    else:
        smoothed = padded.copy()

    # get colormap
    if cmap is None:
        cmap = _get_cmap()
    else:
        if isinstance(cmap, list):
            cmap = _get_cmap(cmap)

    # plot pcolormesh if levels is None, otherwise go for contourf
    if levels is None:
        pc = ax.pcolormesh(
            frqx, frqy, smoothed[1:,1:], cmap=cmap,
            shading="flat",rasterized='true', vmin=vmin, vmax=vmax
        )
    else:
        pc = ax.contourf(
            frqx, frqy, smoothed, levels=levels, cmap=cmap,
            vmin=vmin, vmax=vmax
        )

    # plot contour lines if contours is not None
    if contours is not None:
        cts = ax.contour(
            frqx, frqy, smoothed, levels=contours, colors="k"
        )

    # add colobar if True
    if colorbar:
        cbar = _add_cbar(pc, ax, **cbar_kw)

    # add current data
    if (curspd is None) or (curspd <= 0):
        curspd = 100
    if curdir is not None:
        fmax = axes_kw["rmax"]
        curspd = 9.8/(2*np.pi*fmax)*1.25
        _add_current_info(
            ax, curspd=curspd, curdir=curdir+180, color="red",
            wind_sea_radius=wind_sea_radius
        )


    # add wind data
    if (wspd is None) or (wspd <= 0):
        wspd = 10
    if wdir is not None:
        fmax = axes_kw["rmax"]
        wspd = 9.8/(2*np.pi*fmax)*1.25
        _add_wind_info(
            ax, wspd=wspd, wdir=wdir, color="k",
            wind_sea_radius=wind_sea_radius
        )
        
    try:
        return fig, ax, pc
    except NameError:
        pc

# }}}

# %%

# %
# Maximum Entropy Method
# ----------------------
#
# A more sophisticated way of obtaining the directional distribution
# function :math:`D(f,\theta)` is by using a maximum entropy estimator.
#
# Following `Lygre and Krogstad (1983)`_ and `Alves and Melo (1999)`_,
# the form of the directional distribution function can be written as:
#
# .. math:: D(f,\theta) = \frac{1}{2\pi} \left[
#                   \frac{1 - \phi_1 c_1^* - \phi_2 c_2^*}
#                        { |1 - \phi_1 e^{-i\theta} - \phi_2 e^{-i2\theta}|^2 }
#               \right]
#
# where :math:`c_1` and :math:`c_2` are the complex representation of the
# Fourier coefficients, i.e.,
#
# .. math:: c_1(f) = a_1(f) + i b_1(f)
# .. math:: c_2(f) = a_2(f) + i b_2(f)
#
# and
#
# .. math:: \phi_1 = \frac{c_1 - c_2 c_1^*}{1 - |c_1|^2}
# .. math:: \phi_2  = c_2 - c_1^* \phi_1
#
# It is worth noting that this is just one of the possible implementations
# of MEM. There are other variations that might potentially produce better
# results. For more details, see `Christie (2024)`_ and
# `Simanesew et al. (2018)`_.
#
# .. _Lygre and Krogstad (1983): https://doi.org/10.1175/1520-0485(1986)016<2052:MEEOTD>2.0.CO;2
# .. _Alves and Melo (1999): https://doi.org/10.1016/S0141-1187(99)00019-X
# .. _Christie (2024): https://www.sciencedirect.com/science/article/pii/S0141118723003711?via%3Dihub
# .. _Simanesew et al. (2018): https://doi.org/10.1175/JTECH-D-17-0007.1
#

def mem_distribution(moments, smoothing=32):
    """Implementation of the Maximum Entropy Method"""

    dirs =  xr.Variable(dims=("direction"), data=np.arange(-180,180,5))

    c1 = moments["a1"] + 1j*moments["b1"]
    c2 = moments["a2"] + 1j*moments["b2"]

    phi1 = (c1 - c2 * c1.conj()) / (1 - np.abs(c1)**2)
    phi2 = c2 - c1.conj() * phi1

    sigma_e = 1 - phi1 * c1.conj() - phi2 * c2.conj()

    D = (1/(2*np.pi)) * np.real(
        sigma_e.expand_dims({"direction": dirs}) /
        np.abs(
            1 - phi1.expand_dims({"direction": dirs}) * np.exp(-1j*dirs*np.pi/180)
              - phi2.expand_dims({"direction": dirs}) * np.exp(-2j*dirs*np.pi/180)
        )**2
    )
    
    D["direction"]=np.arange(-180,180,5)

    return (
        (D.T / D.integrate("direction"))
        .rolling(frequency=smoothing, center=True)
        .median()
    )

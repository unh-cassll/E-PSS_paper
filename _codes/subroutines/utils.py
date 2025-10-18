"""
Created on Wed Oct 15 13:01:03 2025

@author: nathanlaxague
"""

import numpy as np
import xarray as xr

from scipy.optimize import minimize
from scipy.signal import butter, filtfilt

import ClassicSpectralAnalysis as spec
import mem_distribution as mem

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

    csp = spec.ClassicSpectralAnalysis(dataset, fs=dataset.sampling_rate, nfft=nfft, nperseg=nperseg)

    # computing the cross-spectral matrix
    Phi = csp.cross_spectral_matrix()

    # computing the directional moments, aka first-five Fourier coefficients
    moments = csp.directional_moments(Phi)

    D_EPSS = mem.mem_distribution(moments,smoothing=smoothnum)
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
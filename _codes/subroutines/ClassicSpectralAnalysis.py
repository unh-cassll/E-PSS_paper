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

import numpy as np
import xarray as xr
import scipy.signal as signal

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
    
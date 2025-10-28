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

import xarray as xr
import numpy as np

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
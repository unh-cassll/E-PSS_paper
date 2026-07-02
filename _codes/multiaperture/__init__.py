"""E-PSS PSS slope->elevation front end for the multi-aperture directional method.

The multi-aperture estimator itself now lives in `ewdm.MultiApertureArrays`
(upstreamed via extended-wdm PR #16). This package retains only the project glue
that ewdm has no equivalent for: the slope-field -> elevation reconstruction and
the 3-D-FFT sign anchor.

    pss: build_eta_field, fourier_slope_projection, wavelet_slope_projection,
         sftheta_sign_anchor (needs pyGrad2Surf).

Typical use (validated E-PSS config: gated de-piston |k| solve + staggered apertures):
    from multiaperture import build_eta_field, sftheta_sign_anchor
    from ewdm import MultiApertureArrays
    from ewdm.multiaperture import default_apertures
    eta, dx, eta_solve = build_eta_field(slope_east, slope_north, depth, fs,
                                         depiston_n=2.0)
    out = MultiApertureArrays.from_field(eta, dx, depth, fs).compute(
        apertures=default_apertures(), reliability_gate=None, solve_eta=eta_solve)
"""
from .pss import (build_eta_field, fourier_slope_projection, wavelet_slope_projection,
                  sftheta_sign_anchor, anchored_freq_recolor, L_FOV)

__all__ = ['build_eta_field', 'fourier_slope_projection', 'wavelet_slope_projection',
           'sftheta_sign_anchor', 'anchored_freq_recolor', 'L_FOV']

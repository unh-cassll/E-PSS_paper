"""E-PSS PSS slope->elevation front end for the multi-aperture directional method.

The multi-aperture estimator lives in `ewdm.MultiApertureArrays` (upstreamed
via extended-wdm PR #16). This package holds the project glue that ewdm has no
equivalent for: the slope-field -> elevation reconstruction, the 3-D-FFT sign
anchor, and the small-FOV sign-resolution subclass.

    pss: build_eta_field, fourier_slope_projection, wavelet_slope_projection,
         sftheta_sign_anchor (needs pyGrad2Surf).
    small_fov: SmallFOVArrays (per-frequency anchor fold + sub-anchor onshore
         prior on top of upstream ewdm).

Typical use (validated E-PSS config: gated de-piston |k| solve + staggered apertures):
    from multiaperture import build_eta_field, sftheta_sign_anchor, SmallFOVArrays
    from ewdm.multiaperture import default_apertures
    eta, dx, eta_solve = build_eta_field(slope_east, slope_north, depth, fs,
                                         depiston_n=2.0)
    sp = SmallFOVArrays.from_field(eta, dx, depth, fs)
    sp.sign_prior = (0.0, (0.08, 0.35), 120.0)
    out = sp.compute(apertures=default_apertures(), reliability_gate=None,
                     solve_eta=eta_solve, sign_anchor=anchor)
"""
from .pss import (build_eta_field, fourier_slope_projection, wavelet_slope_projection,
                  sftheta_sign_anchor, anchored_freq_recolor, L_FOV)
from .small_fov import SmallFOVArrays

__all__ = ['build_eta_field', 'fourier_slope_projection', 'wavelet_slope_projection',
           'sftheta_sign_anchor', 'anchored_freq_recolor', 'L_FOV', 'SmallFOVArrays']

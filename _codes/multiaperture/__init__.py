"""Multi-aperture directional wave spectra for small field-of-view virtual-staff
elevation arrays. Resolves each wavenumber octave with a matched aperture and
stitches them, giving F(f,theta), F(k,theta), Q(nu,theta).

core: portable estimator (numpy/xarray/ewdm only).
pss:  E-PSS project glue (build_eta_field, sftheta_sign_anchor; needs pyGrad2Surf).

Typical use (winning E-PSS config: gated de-piston |k| solve + staggered apertures):
    from multiaperture import build_eta_field, multiaperture_spectra, default_grids
    eta, dx, eta_solve = build_eta_field(slope_east, slope_north, depth, fs,
                                         depiston_frac=0.75)
    freqs, k_grid, nu_grid = default_grids(dx)
    out = multiaperture_spectra(eta, dx, freqs, k_grid, nu_grid, depth, fs,
                                solve_eta=eta_solve)
"""
from .core import (
    GRAV, k_dispersion, circ_stats, erode_valid, seed_aperture, aperture_band,
    default_apertures, default_grids, cwt_stack, solve_wavevectors, lh_direction,
    multiaperture_spectra)

# keep core importable where project glue dependencies are absent
try:
    from .pss import build_eta_field, sftheta_sign_anchor, L_FOV
except Exception:   # pragma: no cover
    pass

__all__ = [
    'k_dispersion', 'circ_stats', 'erode_valid', 'seed_aperture', 'aperture_band',
    'default_apertures', 'default_grids', 'cwt_stack', 'solve_wavevectors',
    'lh_direction', 'multiaperture_spectra',
    'build_eta_field', 'sftheta_sign_anchor', 'L_FOV', 'GRAV']

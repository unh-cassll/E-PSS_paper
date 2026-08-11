"""Small-FOV sign resolution for the E-PSS multi-aperture estimator:
`SmallFOVArrays` overrides `MultiApertureArrays._run_multiaperture` with a
per-frequency signed-moment fold reference, per-frequency 3-D-FFT anchor
evidence, coherence-gated branch continuation, and the sub-anchor onshore
prior (`sign_prior` attribute). Deposits, stitching, and calibration are
copied unchanged from ewdm main (dspelaez/extended-wdm); update the copy
when ewdm changes."""
import numpy as np
import xarray as xr

from ewdm import MultiApertureArrays
from ewdm.multiaperture import (GRAV, _log_edge_taper, _nu_of_k, _trapezoid,
                                _wrap180, circ_stats, k_dispersion,
                                estimate_directional_distribution,
                                estimate_radial_distribution)

__all__ = ['SmallFOVArrays']


class SmallFOVArrays(MultiApertureArrays):
    """MultiApertureArrays with the E-PSS small-FOV sign resolution.
    `sign_prior` (attribute): optional (direction_deg, (f_min, f_max),
    min_off_deg); a band reading more than min_off_deg from direction_deg
    rotates 180 deg before the fold, skipping anchor-evidence bins."""

    sign_prior = None

    def _run_multiaperture(self, ap, lh, var_eta, time, freqs, k_grid, nu_grid,
                           dd, kappa, rel_bandwidth, radial_bandwidth_mode,
                           power_weighted, nu_lo_broad, nu_f_lim, nu_k_lim,
                           stitch_taper, antialias_gate, antialias_mult,
                           sign_anchor, sign_anchor_rmin, sign_coh_min,
                           reliability_gate):
        """Resolve the sign per frequency, gate and stitch the per-aperture
        solutions. Sign section is E-PSS-specific; the rest matches ewdm main."""
        depth = self.depth
        coords = {'frequency': freqs, 'time': time}

        # per-frequency signed first moment from the widest alias-valid
        # aperture; +180 corrects the Morlet time convention. coh gates which
        # bins may steer the reference outside the anchored band
        wf = ap[0]['P'].mean(1)
        pk = int(np.argmax(wf))
        kdisp = k_dispersion(freqs, depth)
        valid_af = np.stack([kdisp <= d['khi'] for d in ap])
        sense_valid = valid_af.any(0)
        sel = np.where(sense_valid, valid_af.argmax(0), len(ap) - 1)
        ii = np.arange(len(freqs))
        ca = np.stack([(d['P'] * np.cos(np.radians(d['dir'] + 180.0))).sum(1)
                       for d in ap])[sel, ii]
        sa = np.stack([(d['P'] * np.sin(np.radians(d['dir'] + 180.0))).sum(1)
                       for d in ap])[sel, ii]
        pt = np.stack([d['P'].sum(1) for d in ap])[sel, ii]
        sm = np.array([0.25, 0.5, 0.25])
        cs = np.convolve(ca, sm, mode='same')
        ss = np.convolve(sa, sm, mode='same')
        theta = np.degrees(np.arctan2(ss, cs))
        coh = np.hypot(cs, ss) / np.maximum(
            np.convolve(pt, sm, mode='same'), 1e-30)

        # nearer of the two axis branches to the running value
        def _branch(a, prev):
            c1, c2 = _wrap180(a), _wrap180(a + 180.0)
            return c1 if abs(_wrap180(c1 - prev)) <= \
                abs(_wrap180(c2 - prev)) else c2

        # energy-weighted circular mean of a set of directions
        def _ewmean(deg, w):
            return np.degrees(np.arctan2((w * np.sin(np.radians(deg))).sum(),
                                         (w * np.cos(np.radians(deg))).sum()))

        # fold reference: anchor unit vectors deposited at nearest frequency
        # bins (weight aR * wf), smoothed; anchored bins take the evidence
        # direction, others branch-continue (coherence-gated, steps <= 90 deg);
        # Longuet-Higgins global fallback without an anchor
        has_evid = np.zeros(len(freqs), bool)
        anch_ok = sign_anchor is not None and (
            np.isfinite(np.asarray(sign_anchor[1], float))
            & (np.asarray(sign_anchor[2], float) >= sign_anchor_rmin)).any()
        if anch_ok:
            af, ad, aR = (np.asarray(x, float) for x in sign_anchor)
            ok = np.isfinite(ad) & (aR >= sign_anchor_rmin)
            jc = np.argmin(np.abs(af[ok, None] - freqs[None, :]), axis=1)
            ev = np.zeros(len(freqs), complex)
            np.add.at(ev, jc, aR[ok] * wf[jc]
                      * np.exp(1j * np.radians(ad[ok])))
            ev = np.convolve(ev, sm, mode='same')
            has = np.abs(ev) > 0
            has_evid = has
            ref = np.asarray(_wrap180(theta), float).copy()
            lo = int(np.where(has)[0][0])
            cur = ref[lo]
            for i in range(lo, len(freqs)):
                if has[i]:
                    cur = np.degrees(np.angle(ev[i]))
                elif coh[i] >= sign_coh_min:
                    cur = _branch(theta[i], cur)
                ref[i] = cur
            cur = ref[lo]
            for i in range(lo - 1, -1, -1):
                if coh[i] >= sign_coh_min:
                    cur = _branch(theta[i], cur)
                ref[i] = cur
        elif lh is not None:
            flip = np.cos(np.radians(_ewmean(theta, wf) - _ewmean(lh, wf))) < 0
            ref = _wrap180(theta + 180.0) if flip else theta
        else:
            ref = theta

        # sign prior: rotate the band 180 deg before the fold when its mean
        # reads more than min_off from the prior; anchor bins are exempt
        if self.sign_prior is not None:
            pdir, (pf1, pf2), poff = self.sign_prior
            band = (freqs >= pf1) & (freqs < pf2) & ~has_evid
            if band.any():
                ref = np.asarray(ref, float).copy()
                d = _ewmean(ref[band], wf[band])
                if np.cos(np.radians(d - pdir)) < np.cos(np.radians(poff)):
                    ref[band] = _wrap180(ref[band] + 180.0)

        # --- below here matches ewdm main unchanged ---

        # fold every aperture's direction into the signed hemisphere
        def _fold(dir_cwN):
            mis = np.cos(np.radians(dir_cwN - ref[:, None])) < 0
            return np.where(mis, _wrap180(dir_cwN + 180.0), dir_cwN)
        for d in ap:
            d['dirf'] = _fold(d['dir'])

        # one variance calibration from the full-frame power so that the integral
        # of S(f) over frequency equals the elevation variance
        cal = var_eta / _trapezoid(ap[0]['P'].mean(1), freqs)

        # anti-alias gate threshold (see the `compute` docstring)
        gate_k = antialias_mult * float(k_dispersion(freqs[pk], depth))

        # the Q(nu) deposit is restricted to a trusted scale window from the array
        # baselines, with a high cut at the frame-fundamental wavenumber, and the
        # matching dispersion band in frequency. Only Q(nu) is gated this way
        bmx = [d['bmax'] for d in ap]
        if nu_k_lim == 'auto':
            nu_k_lim = (2 * np.pi / (50.0 * max(bmx)), self._kfov)
        if nu_f_lim == 'auto':
            _fk = lambda kk: float(np.sqrt(
                GRAV * kk * np.tanh(np.clip(kk * depth, 1e-9, 50)))
                / (2 * np.pi))
            nu_f_lim = ((_fk(nu_k_lim[0]), _fk(nu_k_lim[1]))
                        if nu_k_lim is not None else None)

        # accumulate the stitched wavenumber and inverse phase speed spectra
        bins_dir = np.arange(-180.0, 180.0, dd)
        Fk = np.zeros(len(k_grid)); ck = np.zeros(len(k_grid))
        Qn = np.zeros(len(nu_grid)); cn = np.zeros(len(nu_grid))
        Fkd = np.zeros((len(k_grid), len(bins_dir)))
        Qnd = np.zeros((len(nu_grid), len(bins_dir)))
        Ffd = None        # stitched frequency-direction deposit (filled in loop)
        ap_ok_omni = []   # per-aperture omnidirectional F(k), for diagnostics
        for d in ap:
            # anti-alias frequency mask: send the frequencies that would alias off
            # the k and nu grids so they never deposit, while keeping the power
            # intact so the power-weighting is undisturbed
            fok = (kdisp <= d['khi']) if (antialias_gate and d['khi'] < gate_k) \
                else np.ones(len(freqs), bool)
            # dispersion-independent reliability mask: drop (freq, time) samples
            # whose least-squares phase misfit is high (a short wave aliased by
            # this aperture, or noise) so their spurious low wavenumber never
            # deposits. The power is kept intact, so the power-weighting is
            # undisturbed. On by default (reliability_gate=0.6); None disables it.
            relok = (d['misfit'] <= reliability_gate) \
                if reliability_gate is not None else np.ones_like(d['kmag'], bool)
            power = xr.DataArray(d['P'] * cal, dims=['frequency', 'time'],
                                 coords=coords)
            thd = xr.DataArray(d['dirf'], dims=['frequency', 'time'],
                               coords=coords)

            # deposit onto the wavenumber grid
            k_vals = np.where(fok[:, None] & relok, d['kmag'], 1e12)
            kk = xr.DataArray(k_vals, dims=['frequency', 'time'], coords=coords)
            ok = estimate_radial_distribution(power, thd, kk, 'wavenumber',
                                              k_grid, dd, kappa,
                                              bandwidth=rel_bandwidth,
                                              bandwidth_mode=radial_bandwidth_mode,
                                              power_weighted=power_weighted)

            # deposit onto the inverse phase speed grid, gated to the trusted
            # (f, k) window. out-of-window samples are sent off the nu grid, again
            # keeping the power intact so F(k) and S(f) are unaffected
            nu_vals = d['kmag'] / (2 * np.pi * freqs[:, None])
            if nu_f_lim is not None or nu_k_lim is not None:
                keep = np.ones(nu_vals.shape, bool)
                if nu_f_lim is not None:
                    keep &= ((freqs >= nu_f_lim[0]) & (freqs <= nu_f_lim[1]))[:, None]
                if nu_k_lim is not None:
                    keep &= (d['kmag'] >= nu_k_lim[0]) & (d['kmag'] <= nu_k_lim[1])
                nu_vals = np.where(keep, nu_vals, 1e6)
            nu_vals = np.where(fok[:, None] & relok, nu_vals, 1e12)
            nu = xr.DataArray(nu_vals, dims=['frequency', 'time'], coords=coords)
            on = estimate_radial_distribution(power, thd, nu, 'nu', nu_grid,
                                              dd, kappa, bandwidth=rel_bandwidth,
                                              bandwidth_mode=radial_bandwidth_mode,
                                              power_weighted=power_weighted)

            ok_omni = ok['wavenumber_spectrum'].values
            ok_dir = ok['directional_spectrum'].values
            on_omni = on['nu_spectrum'].values
            on_dir = on['directional_spectrum'].values

            # the nu band follows the aperture's k-band along the dispersion
            # relation; the broadest aperture is extended down to `nu_lo_broad` to
            # show the measured inverse phase speed tail
            ink = (k_grid >= d['klo']) & (k_grid <= d['khi'])
            nlo, nhi = _nu_of_k(d['khi'], depth), _nu_of_k(d['klo'], depth)
            lo = min(nlo, nhi)
            if d is ap[0]:
                lo = min(lo, nu_lo_broad)

            # stitch weights: a cosine edge taper in log-k and log-nu when one is
            # requested, otherwise the hard in-band indicator. The weight cancels
            # in the single-aperture regions
            wk = _log_edge_taper(k_grid, d['klo'], d['khi'], stitch_taper) \
                if stitch_taper else ink.astype(float)
            wn = _log_edge_taper(nu_grid, lo, max(nlo, nhi), stitch_taper) \
                if stitch_taper else \
                ((nu_grid >= lo) & (nu_grid <= max(nlo, nhi))).astype(float)

            ap_ok_omni.append((ok_omni.copy(), ink.copy()))
            Fk += ok_omni * wk; ck += wk
            Fkd += ok_dir * wk[:, None]
            Qn += on_omni * wn; cn += wn
            Qnd += on_dir * wn[:, None]

            # stitched frequency-direction deposit (dispersion-free): this
            # aperture contributes E(f, theta) from its reliable, non-aliased
            # samples only (the `relok` phase-misfit gate). summed over apertures,
            # each frequency takes its direction from whichever apertures resolve
            # it, so the coarsest aperture's aliased high-frequency direction does
            # not flip the estimate.
            pgd = xr.DataArray(d['P'] * cal * relok, dims=['frequency', 'time'],
                               coords=coords)
            ofd = estimate_directional_distribution(pgd, thd, dd, kappa)
            Ffd = ofd['directional_spectrum'].values if Ffd is None \
                else Ffd + ofd['directional_spectrum'].values

        # normalise the stitched spectra by the accumulated weights
        Fk = np.where(ck > 0, Fk / np.maximum(ck, 1e-30), np.nan)
        Fkd /= np.maximum(ck, 1e-30)[:, None]
        Qn = np.where(cn > 0, Qn / np.maximum(cn, 1e-30), np.nan)
        Qnd /= np.maximum(cn, 1e-30)[:, None]

        # frequency spectrum S(f) from the coarsest aperture: its power is valid
        # at every frequency (only its direction aliases above the baseline
        # limit). F(f, theta) is the reliability-stitched deposit, renormalised
        # so it integrates over direction to S(f).
        power0 = xr.DataArray(ap[0]['P'] * cal, dims=['frequency', 'time'],
                              coords=coords)
        theta0 = xr.DataArray(ap[0]['dirf'], dims=['frequency', 'time'],
                              coords=coords)
        of = estimate_directional_distribution(power0, theta0, dd, kappa)
        Sf = of['frequency_spectrum'].values
        thg = of['direction'].values
        row = Ffd.sum(1) * np.radians(dd)
        Fft = np.divide(Ffd, np.maximum(row, 1e-30)[:, None]) * Sf[:, None]
        thbar, sigma = circ_stats(Fft, thg)

        return dict(freqs=freqs, theta=thg, k=k_grid, nu=nu_grid,
                    var_eta=var_eta, Sf=Sf, Fft=Fft, Fk=Fk, Qn=Qn, Fkd=Fkd,
                    Qnd=Qnd, thbar=thbar, sigma=sigma,
                    lh=(lh if lh is not None else np.full(len(freqs), np.nan)),
                    sign_ref=ref, ap_names=[d['name'] for d in ap],
                    ap_bands=[(d['klo'], d['khi']) for d in ap],
                    ap_bmax=[d['bmax'] for d in ap],
                    ap_ok_omni=ap_ok_omni, ck=ck)

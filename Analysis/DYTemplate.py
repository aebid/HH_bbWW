"""Smooth DY templates of a DNN-score distribution inside an HME window, for EqualSignalBinning.py.

The DY MC in the most signal-like DNN bins of the same-flavour channels is a few events with large,
partly negative weights; a smooth fit there replaces it. Everything here works on fine-binned arrays
(values y, variances w2, bin centres x) of one window, summed over the eras.

- ``LogPolyFit``: f(x) = exp(a0 + a1 t + ... + a_d t^d), t = (x - x0) / s, fitted with a Poisson
  likelihood in effective counts (one effective weight wbar = sum w^2 / sum w for the window:
  n_i = y_i / wbar). Neighbouring bins are grouped for the likelihood, from the high-score end down,
  until each group is positive with N_eff >= 1, so negative weights stay in without making the
  likelihood unbounded. The leading coefficient is kept negative, so the function falls at high score.
- ``tail_template``: raw MC up to the fitted peak plus `shift` widths, the tail fit (degree `deg`)
  above (resolved categories).
- ``fit_template``: the whole-window exp(poly2) fit below that point and the tail fit above, for
  windows with too few events for a raw core (boosted).
- ``lowmass_boost_shape``: the shape of low-mass DY in boosted, fit(low-mass, reference) x fit(M>50,
  category) / fit(M>50, reference).
- Uncertainty band (``_band``): above the start point, the larger difference to two alternative fits
  (exp(poly2) tail from the peak, whole-window exp(poly4)) combined with the tail's normalisation
  1 / sqrt(N_eff); below it 0 for raw MC, or the window's 1 / sqrt(N_eff) for fit-only templates;
  capped at 100%.
"""

import numpy as np
from iminuit import Minuit

# LogPolyFit's minimisation: the limits of p in the leading coefficient -exp(p) (|a| from 2e-9 to 2e4: below
# is nothing at these scales, above only overflows exp in a stray migrad step); the starts of that
# coefficient (degree >= 2, degree 1); how often a migrad that did not converge is resumed; and by how much
# a later start's NLL must be lower to replace an earlier one.
P_MIN, P_MAX = -20.0, 10.0
LEAD_STARTS = {False: (-0.5, -0.05, -0.0025), True: (-1.0, -0.1, -0.01)}
MIGRAD_RETRIES = 3
FVAL_TIE = 1e-4


class LogPolyFit:
    def __init__(self, x, y, w2, deg=2, mask=None, merge=True):
        """mask: the bins entering the likelihood (default all); the prediction covers every bin."""
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.w2 = np.asarray(w2, float)
        self.deg = deg
        self.mask = (
            np.ones(len(self.x), bool) if mask is None else np.asarray(mask, bool)
        )
        Y, W = self.y[self.mask].sum(), self.w2[self.mask].sum()
        self.ok = Y > 0 and W > 0
        if not self.ok:
            return
        self.wbar = W / Y
        self.A = self._groups(merge)
        self.n = np.clip(self.A @ self.y / self.wbar, 0, None)
        pos = np.clip(self.y, 0, None) * self.mask
        self.x0 = float((pos * self.x).sum() / pos.sum())
        self.s = float(
            np.sqrt(max((pos * (self.x - self.x0) ** 2).sum() / pos.sum(), 0.25))
        )
        t = (self.x - self.x0) / self.s
        self.T = np.vstack([t**k for k in range(deg + 1)]).T
        # start: a Gaussian with the observed mean, width and total (a falling exponential at degree 1)
        start = np.zeros(deg + 1)
        start[0] = np.log(
            max(Y, 1e-9) / (np.sqrt(2 * np.pi) * self.s / (self.x[1] - self.x[0]))
        )
        if deg >= 2:
            start[2] = -0.5
        # The leading coefficient is -exp(p). Started at 0 (p = log 1e-6), its gradient is ~1e-6:
        # migrad stopped on that plateau and called it converged, short of the minimum by up to ~100 in
        # NLL, and whether it drifted off depended on the last bits of the arithmetic -- numpy's AVX2
        # and AVX-512 code paths differ there, so the template changed with the node's CPU. So it
        # starts at a few negative values instead, and the best valid fit is kept. The limit on p
        # makes a minimum where the data prefer the coefficient at 0 a point (|a| <= 2e-9, nothing at
        # these scales), and a migrad that stops short is resumed.
        best = None
        for lead in LEAD_STARTS[deg == 1]:
            start[deg] = lead
            m = Minuit(self._nll, self._to_p(start))
            m.errordef = Minuit.LIKELIHOOD
            m.limits[deg] = (P_MIN, P_MAX)
            m.strategy = 2
            m.migrad()
            for _ in range(MIGRAD_RETRIES):
                if m.valid:
                    break
                m.migrad()
            # strictly lower to replace: equal minima keep the first start, on every machine
            if best is None or (m.valid, -m.fval) > (best.valid, -best.fval + FVAL_TIE):
                best = m
        self.m = best
        self.m.hesse()
        self.ok = self.m.valid

    def _groups(self, merge):
        """(n_groups x n_bins) 0/1 matrix of the likelihood groups over the masked bins."""
        idx = np.flatnonzero(self.mask)
        if not merge:
            groups = [[i] for i in idx]
        else:
            groups, cur = [], []
            for i in idx[::-1]:
                cur.append(i)
                y, w = self.y[cur].sum(), self.w2[cur].sum()
                if (y == 0 and w == 0) or (y > 0 and y * y / w >= 1.0):
                    groups.append(cur)
                    cur = []
            if cur:
                if groups:
                    groups[-1] = groups[-1] + cur
                else:
                    groups.append(cur)
        A = np.zeros((len(groups), len(self.x)))
        for g, b in enumerate(groups):
            A[g, b] = 1.0
        return A

    @staticmethod
    def _to_a(p):
        a = np.array(p, float)
        a[-1] = -np.exp(p[-1])
        return a

    @staticmethod
    def _to_p(a):
        p = np.array(a, float)
        p[-1] = max(np.log(max(-a[-1], np.exp(P_MIN))), P_MIN)
        return p

    def _nll(self, p):
        mu = self.A @ np.exp(np.clip(self.T @ self._to_a(p), -50, 50)) / self.wbar
        mu = np.maximum(mu, 1e-300)
        return float((mu - self.n * np.log(mu)).sum())

    def predict(self):
        return np.exp(np.clip(self.T @ self._to_a(np.array(self.m.values)), -50, 50))


def _neff(y, w2):
    return float(y.sum() ** 2 / w2.sum()) if w2.sum() > 0 and y.sum() > 0 else 1.0


def _band(x, y, w2, f, mask, kp, merge, core_rel=None):
    env = np.zeros(len(x))
    alternatives = (
        LogPolyFit(x, y, w2, 2, mask=np.arange(len(x)) >= kp, merge=merge),
        LogPolyFit(x, y, w2, 4, merge=merge),
    )
    for alt in alternatives:
        if alt.ok:
            env = np.maximum(env, np.abs(alt.predict() - f))
    stat = f / np.sqrt(max(_neff(y[mask], w2[mask]), 1.0))
    err = np.where(
        mask,
        np.sqrt(env**2 + stat**2),
        0.0 if core_rel is None else core_rel * f,
    )
    return np.minimum(err, f)


def _start(x, f2, shift):
    """(peak index, start index of the tail): the fitted peak plus `shift` widths."""
    kp = int(np.argmax(f2))
    w = f2 / f2.sum()
    s = float(np.sqrt((w * (x - x[kp]) ** 2).sum()))
    return kp, min(int(np.searchsorted(x, x[kp] + shift * s)), len(x) - 3)


def _fits(x, y, w2, deg, shift, merge):
    """(whole-window exp(poly2) fit, tail fit, tail mask, peak index), or None if the poly2 fit
    fails; the tail fit falls back to the poly2 fit."""
    g2 = LogPolyFit(x, y, w2, 2, merge=merge)
    if not g2.ok:
        return None
    f2 = g2.predict()
    kp, k = _start(x, f2, shift)
    mask = np.arange(len(x)) >= k
    ft = LogPolyFit(x, y, w2, deg, mask=mask, merge=merge)
    return f2, (ft if ft.ok else g2).predict(), mask, kp


def tail_template(x, y, w2, deg=3, shift=1.0, merge=True):
    """-> template, (up, down), tail mask, ok. Raw MC below the tail start (mask False), the tail
    fit above it."""
    fits = _fits(x, y, w2, deg, shift, merge)
    if fits is None:
        return y.copy(), (y.copy(), y.copy()), np.zeros(len(x), bool), False
    _, f, mask, kp = fits
    tmpl = np.where(mask, f, y)
    err = _band(x, y, w2, f, mask, kp, merge)
    return tmpl, (tmpl + err, np.clip(tmpl - err, 0, None)), mask, True


def fit_template(x, y, w2, deg=3, shift=1.0, merge=True, core_rel=None):
    """-> template, (up, down), ok. Fit-only: exp(poly2) below the tail start, the tail fit above.

    core_rel: the band's relative size below the tail start; default the window's normalisation
    statistics, 1 / sqrt(N_eff). A caller that renormalises the template to another yield (with
    that yield's own statistics) passes 0, so the normalisation statistics are not counted twice.
    """
    fits = _fits(x, y, w2, deg, shift, merge)
    if fits is None:
        return None, (None, None), False
    f2, f, mask, kp = fits
    tmpl = np.where(mask, f, f2)
    if core_rel is None:
        core_rel = 1.0 / np.sqrt(max(_neff(y, w2), 1.0))
    err = _band(x, y, w2, tmpl, mask, kp, merge, core_rel=core_rel)
    return tmpl, (tmpl + err, np.clip(tmpl - err, 0, None)), True


def lowmass_boost_shape(x, low_ref, m50_ref, m50_shape, **fit):
    """Shape (sum 1) of low-mass DY in a category: fit(low-mass, reference) x m50_shape /
    fit(M>50, reference), all in the category's window. low_ref / m50_ref are (y, w2); m50_shape is
    the category's fitted M>50 shape. Falls back to m50_shape if a fit fails."""
    tl, _, ok_low = fit_template(x, *low_ref, **fit)
    tm, _, ok_m50 = fit_template(x, *m50_ref, **fit)
    if not (ok_low and ok_m50):
        return m50_shape
    r = tl / tl.sum() * m50_shape / np.maximum(tm / tm.sum(), 1e-12)
    return r / r.sum() if r.sum() > 0 else m50_shape

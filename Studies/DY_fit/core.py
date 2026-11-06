"""Shared helpers used by more than one script in this directory.

load_hist_2d  - load TH2 (native bins; no rebin)
ContinuousDensity2D family (N × pdf, Gauss–Legendre cell integrals):
  ExpPoly2D, ExpPolyCheb2D, ExpPolyLogY2D, LogBern2D, MixtureExpPoly2D
eval / integrate for templates and plots
eigen_shape_shifts
"""

import math
import os

import numpy as np
import ROOT
from numpy.polynomial.legendre import leggauss

ROOT.gROOT.SetBatch(True)


# --------------------------------------------------------------------------- data


class AxisRange(object):
    def __init__(self, xmin, xmax, bin_width):
        self.xmin = xmin
        self.xmax = xmax
        self.bin_width = bin_width


class Hist2DData(object):
    """Numpy snapshot of a TH2 + fit windows (native binning)."""

    def __init__(
        self,
        mx,
        category,
        process,
        x_centers,
        y_centers,
        x_edges,
        y_edges,
        content,
        error,
        dnn_range,
        hme_range,
        integral,
        n_entries,
        source_file,
    ):
        self.mx = mx
        self.category = category
        self.process = process
        self.x_centers = x_centers
        self.y_centers = y_centers
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.content = content
        self.error = error
        self.dnn_range = dnn_range
        self.hme_range = hme_range
        self.integral = integral
        self.n_entries = n_entries
        self.source_file = source_file

    @property
    def nx(self):
        return len(self.x_centers)

    @property
    def ny(self):
        return len(self.y_centers)

    @property
    def dx(self):
        return float(self.x_edges[1] - self.x_edges[0])

    @property
    def dy(self):
        return float(self.y_edges[1] - self.y_edges[0])


def _hadd_path(mx, category, data_dir):
    return os.path.join(data_dir, "hadd_m%d_%s.root" % (mx, category))


def _load_th2(path, hist_path):
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise FileNotFoundError("Cannot open %s" % path)
    h = f.Get(hist_path)
    if not h:
        f.Close()
        raise KeyError("Histogram '%s' not found in %s" % (hist_path, path))
    h.SetDirectory(0)
    # PyROOT does not own what Get() returns; detached, the histogram would be owned
    # by nobody, never freed
    ROOT.SetOwnership(h, True)
    f.Close()
    return h


def _th2_to_data(h, mx, category, process, dnn_range, hme_range, source_file):
    nx, ny = h.GetNbinsX(), h.GetNbinsY()
    content = np.zeros((nx, ny), dtype=float)
    error = np.zeros((nx, ny), dtype=float)
    x_c = np.zeros(nx)
    y_c = np.zeros(ny)
    x_edges = np.zeros(nx + 1)
    y_edges = np.zeros(ny + 1)
    for ix in range(1, nx + 1):
        x_c[ix - 1] = h.GetXaxis().GetBinCenter(ix)
        x_edges[ix - 1] = h.GetXaxis().GetBinLowEdge(ix)
    x_edges[nx] = h.GetXaxis().GetBinUpEdge(nx)
    for iy in range(1, ny + 1):
        y_c[iy - 1] = h.GetYaxis().GetBinCenter(iy)
        y_edges[iy - 1] = h.GetYaxis().GetBinLowEdge(iy)
    y_edges[ny] = h.GetYaxis().GetBinUpEdge(ny)
    for ix in range(1, nx + 1):
        for iy in range(1, ny + 1):
            content[ix - 1, iy - 1] = h.GetBinContent(ix, iy)
            error[ix - 1, iy - 1] = h.GetBinError(ix, iy)
    return Hist2DData(
        mx=mx,
        category=category,
        process=process,
        x_centers=x_c,
        y_centers=y_c,
        x_edges=x_edges,
        y_edges=y_edges,
        content=content,
        error=error,
        dnn_range=dnn_range,
        hme_range=hme_range,
        integral=float(content.sum()),
        n_entries=float(h.GetEntries()),
        source_file=source_file,
    )


def load_hist_2d(
    mx,
    data_dir,
    category="res2b",
    process="DY",
    dnn_min=-7.5,
    dnn_max=6.5,
    hme_min=240.0,
    hme_max=1200.0,
    hist_name=None,
):
    """Load plots_2d/{process}_2d (or hist_name) with native binning. No rebin."""
    path = _hadd_path(mx, category, data_dir)
    hname = hist_name or ("plots_2d/%s_2d" % process)
    h = _load_th2(path, hname)
    return _th2_to_data(
        h,
        mx=mx,
        category=category,
        process=process,
        dnn_range=AxisRange(dnn_min, dnn_max, h.GetXaxis().GetBinWidth(1)),
        hme_range=AxisRange(hme_min, hme_max, h.GetYaxis().GetBinWidth(1)),
        source_file=path,
    )


# --------------------------------------------------------------------------- hist quality (negatives / empties)


def _window_mask(data, dnn_min=None, dnn_max=None, hme_min=None, hme_max=None):
    """Boolean (nx, ny) mask for bin centres inside the fit window."""
    xr = data.dnn_range
    yr = data.hme_range
    xlo = xr.xmin if dnn_min is None else float(dnn_min)
    xhi = xr.xmax if dnn_max is None else float(dnn_max)
    ylo = yr.xmin if hme_min is None else float(hme_min)
    yhi = yr.xmax if hme_max is None else float(hme_max)
    mx = (data.x_centers >= xlo) & (data.x_centers <= xhi)
    my = (data.y_centers >= ylo) & (data.y_centers <= yhi)
    return mx[:, None] & my[None, :]


def count_negative_bins(data, window_only=True):
    """Number of bins with content < 0 (optionally restricted to fit window)."""
    c = data.content
    if window_only:
        m = _window_mask(data)
        return int(np.sum((c < 0) & m))
    return int(np.sum(c < 0))


def count_empty_bins(data, window_only=True):
    """Number of bins with content == 0 (optionally restricted to fit window)."""
    c = data.content
    if window_only:
        m = _window_mask(data)
        return int(np.sum((c == 0.0) & m))
    return int(np.sum(c == 0.0))


def hist_quality_issues(data, window_only=True, allow_empty=True):
    """Return list of human-readable problems (empty list if clean).

    By default empty bins are not reported (they are acceptable for fits;
    Garwood σ is used).  Negatives always are.
    """
    issues = []
    n_neg = count_negative_bins(data, window_only=window_only)
    scope = "fit-window" if window_only else "full histogram"
    if n_neg:
        issues.append(
            "%d bins with negative yields in %s "
            "(run grid_merge.py to build a non-negative cell grid)" % (n_neg, scope)
        )
    if not allow_empty:
        n_empty = count_empty_bins(data, window_only=window_only)
        if n_empty:
            issues.append("%d empty (zero-yield) bins in %s" % (n_empty, scope))
    return issues


def assert_hist_quality(data, window_only=True, allow_empty=True):
    """Hard-fail if the histogram has negative yields.

    Empty bins are allowed by default (``allow_empty=True``).  Negatives
    must be cleared (via ``grid_merge.py``) before fitting.
    """
    issues = hist_quality_issues(data, window_only=window_only, allow_empty=allow_empty)
    if issues:
        raise ValueError("; ".join(issues))


def th2_bin_stats(h, xmin=None, xmax=None, ymin=None, ymax=None):
    """Count negative / empty / positive bins of a TH2 (optional centre window)."""
    nx, ny = h.GetNbinsX(), h.GetNbinsY()
    n_neg = n_empty = n_pos = n_tot = 0
    for ix in range(1, nx + 1):
        cx = h.GetXaxis().GetBinCenter(ix)
        if xmin is not None and cx < xmin:
            continue
        if xmax is not None and cx > xmax:
            continue
        for iy in range(1, ny + 1):
            cy = h.GetYaxis().GetBinCenter(iy)
            if ymin is not None and cy < ymin:
                continue
            if ymax is not None and cy > ymax:
                continue
            c = float(h.GetBinContent(ix, iy))
            n_tot += 1
            if c < 0:
                n_neg += 1
            elif c == 0.0:
                n_empty += 1
            else:
                n_pos += 1
    return {
        "n_tot": n_tot,
        "n_neg": n_neg,
        "n_empty": n_empty,
        "n_pos": n_pos,
        "nx": nx,
        "ny": ny,
    }


# --------------------------------------------------------------------------- Continuous 2D densities (N × pdf)


class ContinuousDensity2D(object):
    """Base: extended yield N times unit pdf on the fit rectangle.

        s(x,y)  = positive unnormalised shape  (subclass)
        pdf     = s / ∬_window s
        rho     = N * pdf
        mu_cell = N · ∬_cell s / ∬_window s   (Gauss–Legendre)

    Subclasses implement ``_shape_at_arr(x, y, shape_par)`` where
    ``shape_par = par[1:]`` (everything except N).
    """

    def __init__(self, xmin, xmax, ymin, ymax, n_quad=4, key=None):
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.n_quad = max(int(n_quad), 2)
        self._nodes, self._weights = leggauss(self.n_quad)
        self._Is_cache_key = None
        self._Is_cache_val = None
        self._x0 = 0.5 * (self.xmin + self.xmax)
        self._y0 = 0.5 * (self.ymin + self.ymax)
        self._xhalf = max(0.5 * (self.xmax - self.xmin), 1e-9)
        self._yhalf = max(0.5 * (self.ymax - self.ymin), 1e-9)
        self.key = key or "ContinuousDensity2D"
        self.label = self.key
        # subclasses set param_names / initial_params / npar
        self.param_names = ["N"]
        self.initial_params = [1.0]
        self.npar = 1
        # Opt-in fine-bin tiling normalisation (``set_tiling_norm``).  When
        # set, ``_integral_s_window`` returns the tiling window integral
        # instead of the single n_quad×n_quad GL rule over the whole window.
        self._tiling_norm = None

    def _shape_at_arr(self, x, y, shape_par):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Linear-exponent basis hooks (exp-poly type models override these so
    # the tiling path can use one cached basis matrix per pack).
    # ------------------------------------------------------------------
    def _basis_signature(self):
        """Hashable id of the exponent basis (None → no linear basis)."""
        return None

    def _basis_matrix(self, x, y):
        """(n_points × n_terms) matrix B with exponent arg = B @ shape_par."""
        raise NotImplementedError

    def exp_arg_at_arr(self, x, y, shape_par):
        """UNclipped exponent argument at points (None if no linear basis)."""
        if self._basis_signature() is None:
            return None
        xf = np.asarray(x, dtype=float).ravel()
        yf = np.asarray(y, dtype=float).ravel()
        B = self._basis_matrix(xf, yf)
        return B @ np.asarray([float(v) for v in shape_par], dtype=float)

    # ------------------------------------------------------------------
    # Fine-bin TILING normalisation (opt-in; the window integral is the sum
    # of per-fine-bin GL integrals, so ∑_cells μ == N exactly).
    # ------------------------------------------------------------------
    def prepare_fine_tiling(
        self,
        x_edges_fine,
        y_edges_fine,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        n_quad_fine=2,
    ):
        """Tensor GL nodes on every fine bin whose centre lies in the window.

        Returns a dict ``pack`` with flat arrays ``node_x``, ``node_y``,
        ``node_w`` (weights include the bin half-widths: ∑w over a bin = its
        area), ``fine_index`` (node → in-window fine-bin id), ``n_fine_in``,
        the fine-bin ids ``ix``/``iy``, a ``fine_pos`` (nx_fine × ny_fine)
        lookup (fine bin → pack id or -1) and a per-model ``basis_cache``.
        """
        xe = np.asarray(x_edges_fine, dtype=float)
        ye = np.asarray(y_edges_fine, dtype=float)
        xmin = self.xmin if xmin is None else float(xmin)
        xmax = self.xmax if xmax is None else float(xmax)
        ymin = self.ymin if ymin is None else float(ymin)
        ymax = self.ymax if ymax is None else float(ymax)
        xc = 0.5 * (xe[1:] + xe[:-1])
        yc = 0.5 * (ye[1:] + ye[:-1])
        ix_in = np.nonzero((xc >= xmin) & (xc <= xmax))[0]
        iy_in = np.nonzero((yc >= ymin) & (yc <= ymax))[0]
        IX, IY = np.meshgrid(ix_in, iy_in, indexing="ij")
        IX = IX.ravel().astype(np.intp)
        IY = IY.ravel().astype(np.intp)
        n_fine_in = int(IX.shape[0])
        nq = max(int(n_quad_fine), 1)
        nodes, weights = leggauss(nq)
        xlo, xhi = xe[IX], xe[IX + 1]
        ylo, yhi = ye[IY], ye[IY + 1]
        xmid = 0.5 * (xlo + xhi)
        xhalf = 0.5 * (xhi - xlo)
        ymid = 0.5 * (ylo + yhi)
        yhalf = 0.5 * (yhi - ylo)
        node_x = xmid[:, None, None] + xhalf[:, None, None] * nodes[None, :, None]
        node_y = ymid[:, None, None] + yhalf[:, None, None] * nodes[None, None, :]
        node_x = np.broadcast_to(node_x, (n_fine_in, nq, nq)).ravel().copy()
        node_y = np.broadcast_to(node_y, (n_fine_in, nq, nq)).ravel().copy()
        node_w = (
            (weights[:, None] * weights[None, :])[None, :, :]
            * (xhalf * yhalf)[:, None, None]
        ).ravel()
        fine_index = np.repeat(np.arange(n_fine_in, dtype=np.intp), nq * nq)
        fine_pos = -np.ones((len(xe) - 1, len(ye) - 1), dtype=np.intp)
        fine_pos[IX, IY] = np.arange(n_fine_in, dtype=np.intp)
        return {
            "node_x": node_x,
            "node_y": node_y,
            "node_w": node_w,
            "fine_index": fine_index,
            "n_fine_in": n_fine_in,
            "n_nodes": int(node_x.shape[0]),
            "ix": IX,
            "iy": IY,
            "fine_pos": fine_pos,
            "x_edges": xe,
            "y_edges": ye,
            "n_quad_fine": nq,
            "window": (xmin, xmax, ymin, ymax),
            "basis_cache": {},
        }

    def _tiling_basis(self, pack):
        """Cached (nodes × terms) basis matrix for this model on ``pack``."""
        sig = self._basis_signature()
        if sig is None:
            return None
        cache = pack.setdefault("basis_cache", {})
        B = cache.get(sig)
        if B is None:
            B = self._basis_matrix(pack["node_x"], pack["node_y"])
            cache[sig] = B
        return B

    def exp_arg_on_nodes(self, shape_par, pack):
        """UNclipped exponent argument on all tiling nodes (None if no basis)."""
        B = self._tiling_basis(pack)
        if B is None:
            return None
        return B @ np.asarray([float(v) for v in shape_par], dtype=float)

    def shape_on_nodes(self, shape_par, pack):
        """Shape s on all tiling nodes — evaluated ONCE per call."""
        arg = self.exp_arg_on_nodes(shape_par, pack)
        if arg is None:
            return np.asarray(
                self._shape_at_arr(pack["node_x"], pack["node_y"], shape_par),
                dtype=float,
            ).ravel()
        np.clip(arg, -50.0, 50.0, out=arg)
        return np.exp(arg)

    def integrals_fine(self, shape_par, pack):
        """∬_fine-bin s for every in-window fine bin (length n_fine_in)."""
        s = self.shape_on_nodes(shape_par, pack)
        return np.bincount(
            pack["fine_index"], weights=s * pack["node_w"], minlength=pack["n_fine_in"]
        )

    def max_exp_arg(self, shape_par, pack):
        """Max UNclipped exponent argument over the tiling nodes (diagnostic)."""
        arg = self.exp_arg_on_nodes(shape_par, pack)
        if arg is None:
            return None
        return float(np.max(arg))

    def expected_cells_batch_tiling(self, par, pack, fine_to_obs, n_obs=None):
        """μ_obs = N · ∑_{fine∈obs} I_fine / ∑_{fine mapped} I_fine.

        ``fine_to_obs`` maps every in-window fine bin to an observation index
        (-1 = unmapped, excluded from the window integral).  When every fine
        bin is mapped, ∑_obs μ_obs == N exactly.
        """
        N = float(par[0])
        sp = [float(par[i]) for i in range(1, self.npar)]
        f2o = np.asarray(fine_to_obs, dtype=np.intp)
        mask = f2o >= 0
        if n_obs is None:
            n_obs = int(f2o[mask].max()) + 1 if np.any(mask) else 0
        I_fine = self.integrals_fine(sp, pack)
        I_W = float(np.sum(I_fine[mask]))
        if not (I_W > 0.0 and math.isfinite(I_W)):
            return np.zeros(int(n_obs), dtype=float)
        mu = np.bincount(f2o[mask], weights=I_fine[mask], minlength=int(n_obs))
        return N * mu / I_W

    def set_tiling_norm(self, pack, include=None):
        """Normalise the density by the fine-bin tiling window integral.

        ``include`` is an optional boolean mask over the pack's in-window fine
        bins (False = excluded/unmapped, mirroring the fit's I_W).
        """
        self._tiling_norm = None
        self._Is_cache_key = None
        self._Is_cache_val = None
        if pack is None:
            return
        inc = (
            np.ones(pack["n_fine_in"], dtype=bool)
            if include is None
            else np.asarray(include, dtype=bool)
        )
        self._tiling_norm = {"pack": pack, "include": inc}

    def _window_integral_tiling(self, shape_par):
        tn = self._tiling_norm
        I_fine = self.integrals_fine(shape_par, tn["pack"])
        return float(np.sum(I_fine[tn["include"]]))

    def _shape_at(self, x, y, shape_par):
        return float(
            np.asarray(
                self._shape_at_arr(np.array([x]), np.array([y]), shape_par)
            ).ravel()[0]
        )

    def _integral_s_rect(self, shape_par, xlo, xhi, ylo, yhi):
        xmid = 0.5 * (xlo + xhi)
        xhalf = 0.5 * (xhi - xlo)
        ymid = 0.5 * (ylo + yhi)
        yhalf = 0.5 * (yhi - ylo)
        if xhalf <= 0 or yhalf <= 0:
            return 0.0
        acc = 0.0
        for i, wi in enumerate(self._weights):
            x = xmid + xhalf * float(self._nodes[i])
            for j, wj in enumerate(self._weights):
                y = ymid + yhalf * float(self._nodes[j])
                acc += wi * wj * self._shape_at(x, y, shape_par)
        return acc * xhalf * yhalf

    def _integral_s_window(self, shape_par):
        key = tuple(float(c) for c in shape_par)
        if key == self._Is_cache_key and self._Is_cache_val is not None:
            return self._Is_cache_val
        if self._tiling_norm is not None:
            val = self._window_integral_tiling(shape_par)
        else:
            val = self._integral_s_rect(
                shape_par, self.xmin, self.xmax, self.ymin, self.ymax
            )
        self._Is_cache_key = key
        self._Is_cache_val = val
        return val

    def continuous_density(self, x, y, par):
        N = float(par[0])
        sp = [float(par[i]) for i in range(1, self.npar)]
        Is = self._integral_s_window(sp)
        if Is <= 0.0:
            return 0.0
        return N * self._shape_at(x, y, sp) / Is

    def expected_bin(self, par, xlo, xhi, ylo, yhi):
        N = float(par[0])
        sp = [float(par[i]) for i in range(1, self.npar)]
        Is = self._integral_s_window(sp)
        if Is <= 0.0:
            return 0.0
        return N * self._integral_s_rect(sp, xlo, xhi, ylo, yhi) / Is

    @staticmethod
    def _cell_subrects(c):
        rects = getattr(c, "rects", None)
        if rects:
            for r in rects:
                yield (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
            return
        if hasattr(c, "xlo"):
            yield (float(c.xlo), float(c.xhi), float(c.ylo), float(c.yhi))
            return
        yield (
            float(c.get("xlo", c.get("xmin"))),
            float(c.get("xhi", c.get("xmax"))),
            float(c.get("ylo", c.get("ymin"))),
            float(c.get("yhi", c.get("ymax"))),
        )

    def prepare_cell_gl(self, cells):
        """Precompute GL stencils; multi-rect meta-clusters expand to subrects."""
        nq = self.n_quad
        nodes = self._nodes
        weights = self._weights
        n_cells = len(cells)
        sub_xlo, sub_xhi, sub_ylo, sub_yhi = [], [], [], []
        cell_of_sub = []
        for k, c in enumerate(cells):
            for xlo, xhi, ylo, yhi in self._cell_subrects(c):
                if not (xhi > xlo and yhi > ylo):
                    continue
                sub_xlo.append(xlo)
                sub_xhi.append(xhi)
                sub_ylo.append(ylo)
                sub_yhi.append(yhi)
                cell_of_sub.append(k)
        n_sub = len(sub_xlo)
        xs = np.zeros((n_sub, nq, nq), dtype=float)
        ys = np.zeros((n_sub, nq, nq), dtype=float)
        fac = np.zeros((n_sub, nq, nq), dtype=float)
        for s in range(n_sub):
            xlo, xhi = sub_xlo[s], sub_xhi[s]
            ylo, yhi = sub_ylo[s], sub_yhi[s]
            xmid = 0.5 * (xlo + xhi)
            xhalf = 0.5 * (xhi - xlo)
            ymid = 0.5 * (ylo + yhi)
            yhalf = 0.5 * (yhi - ylo)
            for i in range(nq):
                x = xmid + xhalf * float(nodes[i])
                for j in range(nq):
                    y = ymid + yhalf * float(nodes[j])
                    xs[s, i, j] = x
                    ys[s, i, j] = y
                    fac[s, i, j] = float(weights[i]) * float(weights[j]) * xhalf * yhalf
        xmid_w = 0.5 * (self.xmin + self.xmax)
        xhalf_w = 0.5 * (self.xmax - self.xmin)
        ymid_w = 0.5 * (self.ymin + self.ymax)
        yhalf_w = 0.5 * (self.ymax - self.ymin)
        wxs = np.zeros((nq, nq), dtype=float)
        wys = np.zeros((nq, nq), dtype=float)
        wfac = np.zeros((nq, nq), dtype=float)
        for i in range(nq):
            x = xmid_w + xhalf_w * float(nodes[i])
            for j in range(nq):
                y = ymid_w + yhalf_w * float(nodes[j])
                wxs[i, j] = x
                wys[i, j] = y
                wfac[i, j] = float(weights[i]) * float(weights[j]) * xhalf_w * yhalf_w
        return {
            "xs": xs,
            "ys": ys,
            "fac": fac,
            "wxs": wxs,
            "wys": wys,
            "wfac": wfac,
            "n_cells": n_cells,
            "n_sub": n_sub,
            "cell_of_sub": np.asarray(cell_of_sub, dtype=np.intp),
        }

    def expected_cells_batch(self, par, gl_pack):
        N = float(par[0])
        sp = [float(par[i]) for i in range(1, self.npar)]
        n_cells = gl_pack["n_cells"]
        Is = float(
            np.sum(
                gl_pack["wfac"] * self._shape_at_arr(gl_pack["wxs"], gl_pack["wys"], sp)
            )
        )
        if not (Is > 0.0 and math.isfinite(Is)):
            return np.zeros(n_cells, dtype=float)
        s_pts = self._shape_at_arr(gl_pack["xs"], gl_pack["ys"], sp)
        I_sub = np.sum(gl_pack["fac"] * s_pts, axis=(1, 2))
        cell_of_sub = gl_pack.get("cell_of_sub")
        if cell_of_sub is None and I_sub.shape[0] == n_cells:
            return N * I_sub / Is
        I_cell = np.zeros(n_cells, dtype=float)
        np.add.at(I_cell, cell_of_sub, I_sub)
        return N * I_cell / Is

    def __call__(self, x, y, par):
        return self.continuous_density(x, y, par)

    def integrate_bins(self, par, x_edges, y_edges, n_quad=None):
        old_nq = self.n_quad
        if n_quad is not None and int(n_quad) != self.n_quad:
            self.n_quad = max(int(n_quad), 2)
            self._nodes, self._weights = leggauss(self.n_quad)
            self._Is_cache_key = None
        try:
            N = float(par[0])
            sp = [float(par[i]) for i in range(1, self.npar)]
            Is = self._integral_s_window(sp)
            nx = len(x_edges) - 1
            ny = len(y_edges) - 1
            Z = np.zeros((nx, ny), dtype=float)
            if Is <= 0.0:
                return Z
            for ix in range(nx):
                xlo = float(x_edges[ix])
                xhi = float(x_edges[ix + 1])
                for iy in range(ny):
                    ylo = float(y_edges[iy])
                    yhi = float(y_edges[iy + 1])
                    Z[ix, iy] = N * self._integral_s_rect(sp, xlo, xhi, ylo, yhi) / Is
            return Z
        finally:
            if n_quad is not None and int(n_quad) != old_nq:
                self.n_quad = old_nq
                self._nodes, self._weights = leggauss(self.n_quad)
                self._Is_cache_key = None

    def eval_on_grid(self, par, x_centers, y_centers):
        N = float(par[0])
        sp = [float(par[i]) for i in range(1, self.npar)]
        Is = self._integral_s_window(sp)
        Z = np.zeros((len(x_centers), len(y_centers)), dtype=float)
        if Is <= 0.0:
            return Z
        for ix, x in enumerate(x_centers):
            for iy, y in enumerate(y_centers):
                Z[ix, iy] = N * self._shape_at(x, y, sp) / Is
        return Z

    def to_spec(self):
        return {
            "model": "continuous",
            "norm": "continuous",
            "param_names": list(self.param_names),
            "npar": self.npar,
            "n_quad": self.n_quad,
            "key": self.key,
        }


def _total_degree_terms(degree):
    terms = []
    for s in range(1, int(degree) + 1):
        for i in range(s + 1):
            terms.append((i, s - i))
    return terms


def _chebyshev_T_powers(z, max_n):
    """Return list T_0..T_max_n evaluated on array z ∈ [-1,1]."""
    z = np.asarray(z, dtype=float)
    out = [np.ones_like(z), z.copy()]
    for n in range(2, max_n + 1):
        out.append(2.0 * z * out[-1] - out[-2])
    return out[: max_n + 1]


def _bernstein_basis(t, n):
    """Bernstein basis B_0^n .. B_n^n on t ∈ [0,1]. Returns list of arrays."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    om = 1.0 - t
    return [float(math.comb(n, i)) * (t**i) * (om ** (n - i)) for i in range(n + 1)]


class ExpPoly2D(ContinuousDensity2D):
    """exp(polynomial) on normalised (xn,yn) ∈ [-1,1] (monomial basis).

    s = exp( sum_{1≤i+j≤deg} a_ij · xn^i · yn^j )
    """

    def __init__(self, xmin, xmax, ymin, ymax, degree=4, key=None, n_quad=4):
        super(ExpPoly2D, self).__init__(xmin, xmax, ymin, ymax, n_quad=n_quad, key=key)
        self.degree = int(degree)
        self._terms = _total_degree_terms(self.degree)
        self.param_names = ["N"] + ["a_%d_%d" % (i, j) for i, j in self._terms]
        self.npar = len(self.param_names)
        self.key = key or ("ExpPoly2D-%d" % self.degree)
        self.label = self.key
        inits = [1.0]
        for i, j in self._terms:
            inits.append(-1.0 if (i == 0 and j == 1) else 0.0)
        self.initial_params = inits

    def _shape_at_arr(self, x, y, coeffs):
        xn = (np.asarray(x, dtype=float) - self._x0) / self._xhalf
        yn = (np.asarray(y, dtype=float) - self._y0) / self._yhalf
        arg = np.zeros(np.broadcast(xn, yn).shape, dtype=float)
        for (i, j), a in zip(self._terms, coeffs):
            arg = arg + float(a) * (xn**i) * (yn**j)
        np.clip(arg, -50.0, 50.0, out=arg)
        return np.exp(arg)

    def _basis_signature(self):
        return (
            "ExpPoly2D",
            tuple(self._terms),
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
        )

    def _basis_matrix(self, x, y):
        xn = (np.asarray(x, dtype=float) - self._x0) / self._xhalf
        yn = (np.asarray(y, dtype=float) - self._y0) / self._yhalf
        return np.stack([(xn**i) * (yn**j) for i, j in self._terms], axis=1)

    def to_spec(self):
        return {
            "model": "exppoly2d",
            "norm": "continuous",
            "degree": self.degree,
            "param_names": list(self.param_names),
            "npar": self.npar,
            "terms": [list(t) for t in self._terms],
            "n_quad": self.n_quad,
        }


class ExpPolyCheb2D(ContinuousDensity2D):
    """Same total-degree exp-poly, Chebyshev T_i(xn) T_j(yn) basis.

    Better-conditioned at high degree than monomials (same npar as ExpPoly2D).
    """

    def __init__(self, xmin, xmax, ymin, ymax, degree=4, key=None, n_quad=4):
        super(ExpPolyCheb2D, self).__init__(
            xmin, xmax, ymin, ymax, n_quad=n_quad, key=key
        )
        self.degree = int(degree)
        self._terms = _total_degree_terms(self.degree)
        self.param_names = ["N"] + ["c_%d_%d" % (i, j) for i, j in self._terms]
        self.npar = len(self.param_names)
        self.key = key or ("ExpPolyCheb2D-%d" % self.degree)
        self.label = self.key
        inits = [1.0]
        for i, j in self._terms:
            # mild falling HME tilt via T_1(yn)=yn
            inits.append(-0.5 if (i == 0 and j == 1) else 0.0)
        self.initial_params = inits

    def _shape_at_arr(self, x, y, coeffs):
        xn = np.clip((np.asarray(x, dtype=float) - self._x0) / self._xhalf, -1.0, 1.0)
        yn = np.clip((np.asarray(y, dtype=float) - self._y0) / self._yhalf, -1.0, 1.0)
        max_n = self.degree
        Tx = _chebyshev_T_powers(xn, max_n)
        Ty = _chebyshev_T_powers(yn, max_n)
        arg = np.zeros(np.broadcast(xn, yn).shape, dtype=float)
        for (i, j), a in zip(self._terms, coeffs):
            arg = arg + float(a) * Tx[i] * Ty[j]
        np.clip(arg, -50.0, 50.0, out=arg)
        return np.exp(arg)

    def _basis_signature(self):
        return (
            "ExpPolyCheb2D",
            tuple(self._terms),
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
        )

    def _basis_matrix(self, x, y):
        xn = np.clip((np.asarray(x, dtype=float) - self._x0) / self._xhalf, -1.0, 1.0)
        yn = np.clip((np.asarray(y, dtype=float) - self._y0) / self._yhalf, -1.0, 1.0)
        Tx = _chebyshev_T_powers(xn, self.degree)
        Ty = _chebyshev_T_powers(yn, self.degree)
        return np.stack([Tx[i] * Ty[j] for i, j in self._terms], axis=1)

    def to_spec(self):
        return {
            "model": "exppoly_cheb2d",
            "norm": "continuous",
            "degree": self.degree,
            "param_names": list(self.param_names),
            "npar": self.npar,
            "terms": [list(t) for t in self._terms],
            "n_quad": self.n_quad,
        }


class ExpPolyLogY2D(ContinuousDensity2D):
    """ExpPoly on (xn, log-mapped yn): better for steeply falling HME.

    yn_log maps HME ∈ [ymin,ymax] → [-1,1] via
      t = log( (y - ymin + eps) / (ymax - ymin + eps) ) renormalised.
    """

    def __init__(
        self,
        xmin,
        xmax,
        ymin,
        ymax,
        degree=4,
        key=None,
        n_quad=4,
        y_eps=1e-3,
    ):
        super(ExpPolyLogY2D, self).__init__(
            xmin, xmax, ymin, ymax, n_quad=n_quad, key=key
        )
        self.degree = int(degree)
        self._terms = _total_degree_terms(self.degree)
        self.param_names = ["N"] + ["a_%d_%d" % (i, j) for i, j in self._terms]
        self.npar = len(self.param_names)
        self.key = key or ("ExpPolyLogY2D-%d" % self.degree)
        self.label = self.key
        inits = [1.0]
        for i, j in self._terms:
            inits.append(-1.0 if (i == 0 and j == 1) else 0.0)
        self.initial_params = inits
        # log map: u = log1p((y-ymin+eps)/eps) / log1p((ymax-ymin+eps)/eps)
        # → [0,1] then → [-1,1].  ``y_eps`` (GeV) sets how much of the yn
        # range the first HME bins occupy (legacy 1e-3).
        y_eps = float(y_eps)
        if not (y_eps > 0.0 and math.isfinite(y_eps)):
            raise ValueError("ExpPolyLogY2D: y_eps must be > 0 (got %r)" % y_eps)
        self._y_eps = y_eps
        self._log_span = math.log1p((self.ymax - self.ymin + self._y_eps) / self._y_eps)

    def _yn_log(self, y):
        y = np.asarray(y, dtype=float)
        u = np.log1p((y - self.ymin + self._y_eps) / self._y_eps) / max(
            self._log_span, 1e-12
        )
        return np.clip(2.0 * u - 1.0, -1.0, 1.0)

    def _shape_at_arr(self, x, y, coeffs):
        xn = (np.asarray(x, dtype=float) - self._x0) / self._xhalf
        yn = self._yn_log(y)
        arg = np.zeros(np.broadcast(xn, yn).shape, dtype=float)
        for (i, j), a in zip(self._terms, coeffs):
            arg = arg + float(a) * (xn**i) * (yn**j)
        np.clip(arg, -50.0, 50.0, out=arg)
        return np.exp(arg)

    def _basis_signature(self):
        return (
            "ExpPolyLogY2D",
            tuple(self._terms),
            self._y_eps,
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
        )

    def _basis_matrix(self, x, y):
        xn = (np.asarray(x, dtype=float) - self._x0) / self._xhalf
        yn = self._yn_log(y)
        return np.stack([(xn**i) * (yn**j) for i, j in self._terms], axis=1)

    def to_spec(self):
        return {
            "model": "exppoly_logy2d",
            "norm": "continuous",
            "degree": self.degree,
            "param_names": list(self.param_names),
            "npar": self.npar,
            "terms": [list(t) for t in self._terms],
            "n_quad": self.n_quad,
            "y_eps": self._y_eps,
        }


class LogBern2D(ContinuousDensity2D):
    """log-Bernstein density: s = exp( sum c_ij B_i^{nx}(u) B_j^{ny}(v) ).

    u,v map the fit window to [0,1].  Constant term c_00 dropped (absorbed in N).
    More stable than high-degree monomials on a compact domain.
    """

    def __init__(self, xmin, xmax, ymin, ymax, nx=4, ny=4, key=None, n_quad=4):
        super(LogBern2D, self).__init__(xmin, xmax, ymin, ymax, n_quad=n_quad, key=key)
        self.nx = int(nx)
        self.ny = int(ny)
        self._terms = []
        for i in range(self.nx + 1):
            for j in range(self.ny + 1):
                if i == 0 and j == 0:
                    continue
                self._terms.append((i, j))
        self.param_names = ["N"] + ["b_%d_%d" % (i, j) for i, j in self._terms]
        self.npar = len(self.param_names)
        self.key = key or ("LogBern2D-%d-%d" % (self.nx, self.ny))
        self.label = self.key
        inits = [1.0]
        for i, j in self._terms:
            # mild fall in HME: higher j → smaller v → prefer negative b_0_1
            inits.append(-0.5 if (i == 0 and j == 1) else 0.0)
        self.initial_params = inits

    def _uv(self, x, y):
        u = (np.asarray(x, dtype=float) - self.xmin) / max(self.xmax - self.xmin, 1e-12)
        v = (np.asarray(y, dtype=float) - self.ymin) / max(self.ymax - self.ymin, 1e-12)
        return np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)

    def _shape_at_arr(self, x, y, coeffs):
        u, v = self._uv(x, y)
        Bu = _bernstein_basis(u, self.nx)
        Bv = _bernstein_basis(v, self.ny)
        arg = np.zeros(np.broadcast(u, v).shape, dtype=float)
        for (i, j), a in zip(self._terms, coeffs):
            arg = arg + float(a) * Bu[i] * Bv[j]
        np.clip(arg, -50.0, 50.0, out=arg)
        return np.exp(arg)

    def to_spec(self):
        return {
            "model": "logbern2d",
            "norm": "continuous",
            "nx": self.nx,
            "ny": self.ny,
            "param_names": list(self.param_names),
            "npar": self.npar,
            "terms": [list(t) for t in self._terms],
            "n_quad": self.n_quad,
        }


class MixtureExpPoly2D(ContinuousDensity2D):
    """Mixture of K ExpPoly2D components (softmax weights).

        s = sum_k w_k · exp(poly_k),   w = softmax(logits)

    Captures multi-region structure (e.g. low-HME peak vs high-HME tail)
    with moderate per-component degree.
    """

    def __init__(
        self,
        xmin,
        xmax,
        ymin,
        ymax,
        n_comp=2,
        degree=3,
        key=None,
        n_quad=4,
    ):
        super(MixtureExpPoly2D, self).__init__(
            xmin, xmax, ymin, ymax, n_quad=n_quad, key=key
        )
        self.n_comp = max(int(n_comp), 2)
        self.degree = int(degree)
        self._terms = _total_degree_terms(self.degree)
        n_shape = len(self._terms)
        # params: N, logit_1..logit_{K-1}, then K blocks of shape coeffs
        names = ["N"]
        for k in range(1, self.n_comp):
            names.append("logit_%d" % k)
        for k in range(self.n_comp):
            for i, j in self._terms:
                names.append("a%d_%d_%d" % (k, i, j))
        self.param_names = names
        self.npar = len(names)
        self.key = key or ("MixExpPoly2D-k%d-d%d" % (self.n_comp, self.degree))
        self.label = self.key
        inits = [1.0]
        for _k in range(1, self.n_comp):
            inits.append(0.0)  # equal weights
        for k in range(self.n_comp):
            for i, j in self._terms:
                # diversify components slightly
                tilt = -1.0 - 0.3 * k
                inits.append(tilt if (i == 0 and j == 1) else 0.0)
        self.initial_params = inits
        self._n_shape = n_shape

    def _weights_and_blocks(self, shape_par):
        K = self.n_comp
        logits = [0.0]  # component 0 reference
        for k in range(1, K):
            logits.append(float(shape_par[k - 1]))
        m = max(logits)
        ex = [math.exp(min(l - m, 50.0)) for l in logits]
        s = sum(ex)
        w = [e / s for e in ex]
        blocks = []
        off = K - 1
        for k in range(K):
            blocks.append(
                [
                    float(shape_par[off + k * self._n_shape + t])
                    for t in range(self._n_shape)
                ]
            )
        return w, blocks

    def _shape_at_arr(self, x, y, shape_par):
        xn = (np.asarray(x, dtype=float) - self._x0) / self._xhalf
        yn = (np.asarray(y, dtype=float) - self._y0) / self._yhalf
        w, blocks = self._weights_and_blocks(shape_par)
        acc = np.zeros(np.broadcast(xn, yn).shape, dtype=float)
        for wk, coeffs in zip(w, blocks):
            arg = np.zeros_like(acc)
            for (i, j), a in zip(self._terms, coeffs):
                arg = arg + float(a) * (xn**i) * (yn**j)
            np.clip(arg, -50.0, 50.0, out=arg)
            acc = acc + wk * np.exp(arg)
        return acc

    def to_spec(self):
        return {
            "model": "mixture_exppoly2d",
            "norm": "continuous",
            "n_comp": self.n_comp,
            "degree": self.degree,
            "param_names": list(self.param_names),
            "npar": self.npar,
            "terms": [list(t) for t in self._terms],
            "n_quad": self.n_quad,
        }


class ExpPolySep2D(ContinuousDensity2D):
    """Separable exp-poly: s = exp(P_dx(xn) + Q_dy(yn)) (no cross terms).

    Cheap baseline; underfits when DNN–HME correlation is strong.
    """

    def __init__(self, xmin, xmax, ymin, ymax, dx=4, dy=4, key=None, n_quad=4):
        super(ExpPolySep2D, self).__init__(
            xmin, xmax, ymin, ymax, n_quad=n_quad, key=key
        )
        self.dx = int(dx)
        self.dy = int(dy)
        names = ["N"]
        for i in range(1, self.dx + 1):
            names.append("px_%d" % i)
        for j in range(1, self.dy + 1):
            names.append("qy_%d" % j)
        self.param_names = names
        self.npar = len(names)
        self.key = key or ("ExpPolySep2D-%d-%d" % (self.dx, self.dy))
        self.label = self.key
        inits = [1.0] + [0.0] * self.dx + [-1.0] + [0.0] * (self.dy - 1)
        self.initial_params = inits

    def _shape_at_arr(self, x, y, coeffs):
        xn = (np.asarray(x, dtype=float) - self._x0) / self._xhalf
        yn = (np.asarray(y, dtype=float) - self._y0) / self._yhalf
        arg = np.zeros(np.broadcast(xn, yn).shape, dtype=float)
        off = 0
        for i in range(1, self.dx + 1):
            arg = arg + float(coeffs[off]) * (xn**i)
            off += 1
        for j in range(1, self.dy + 1):
            arg = arg + float(coeffs[off]) * (yn**j)
            off += 1
        np.clip(arg, -50.0, 50.0, out=arg)
        return np.exp(arg)

    def to_spec(self):
        return {
            "model": "exppoly_sep2d",
            "norm": "continuous",
            "dx": self.dx,
            "dy": self.dy,
            "param_names": list(self.param_names),
            "npar": self.npar,
            "n_quad": self.n_quad,
        }


def make_density_2d(
    name,
    xmin,
    xmax,
    ymin,
    ymax,
    degree=4,
    n_quad=4,
    nx=None,
    ny=None,
    n_comp=2,
    dx=None,
    dy=None,
    key=None,
    y_eps=1e-3,
):
    """Factory for continuous 2D density models.

    Names (case-insensitive):
      exppoly | expploy2d | poly
      cheb | exppoly_cheb
      logy | exppoly_logy      (``y_eps``: log-map offset in GeV)
      logbern | bern
      mix | mixture
      sep | separable
    """
    n = (name or "exppoly").strip().lower().replace("-", "_")
    if n in ("exppoly", "exppoly2d", "poly", "exp_poly"):
        return ExpPoly2D(xmin, xmax, ymin, ymax, degree=degree, n_quad=n_quad, key=key)
    if n in ("cheb", "exppoly_cheb", "exppolycheb2d", "chebyshev"):
        return ExpPolyCheb2D(
            xmin, xmax, ymin, ymax, degree=degree, n_quad=n_quad, key=key
        )
    if n in ("logy", "exppoly_logy", "exppolylogy2d", "loghme"):
        return ExpPolyLogY2D(
            xmin,
            xmax,
            ymin,
            ymax,
            degree=degree,
            n_quad=n_quad,
            key=key,
            y_eps=y_eps,
        )
    if n in ("logbern", "bern", "logbern2d", "bernstein"):
        return LogBern2D(
            xmin,
            xmax,
            ymin,
            ymax,
            nx=int(nx if nx is not None else degree),
            ny=int(ny if ny is not None else degree),
            n_quad=n_quad,
            key=key,
        )
    if n in ("mix", "mixture", "mixexppoly", "mixture_exppoly2d"):
        return MixtureExpPoly2D(
            xmin,
            xmax,
            ymin,
            ymax,
            n_comp=n_comp,
            degree=degree,
            n_quad=n_quad,
            key=key,
        )
    if n in ("sep", "separable", "exppoly_sep", "exppolysep2d"):
        return ExpPolySep2D(
            xmin,
            xmax,
            ymin,
            ymax,
            dx=int(dx if dx is not None else degree),
            dy=int(dy if dy is not None else degree),
            n_quad=n_quad,
            key=key,
        )
    raise ValueError("unknown density model %r" % name)


def rebuild_exppoly2d(result):
    """Rebuild continuous density from fit JSON (any ContinuousDensity2D)."""
    return rebuild_density_2d(result)


def rebuild_density_2d(result):
    """Rebuild continuous density from fit JSON.

    For ``function.norm == "tiling"`` JSONs the fine-bin tiling pack is
    rebuilt from ``fine_binning`` and attached (``set_tiling_norm``), so every
    normalised evaluation (``integrate_bins``, ``eval_on_grid``, …) uses the
    same window integral as the fit.  Legacy JSONs (no ``y_eps``/``norm``
    tiling) rebuild identically to before.
    """
    model = _rebuild_density_2d_bare(result)
    spec = result.get("function") or {}
    if spec.get("norm") == "tiling":
        pack, include = _tiling_pack_for_result(result, model)
        model.set_tiling_norm(pack, include)
    return model


def _rebuild_density_2d_bare(result):
    spec = result.get("function") or {}
    model_name = spec.get("model") or result.get("model") or "exppoly2d"
    n_quad = int(spec.get("n_quad") or result.get("n_quad") or 4)
    deg = int(spec.get("degree") or result.get("degree") or 4)
    edges = result.get("fit_range_edges") or {}
    fr = result.get("fit_range") or {}
    dr = edges.get("dnn") or fr.get("dnn") or [-7.5, 6.5]
    hr = edges.get("hme") or fr.get("hme") or [240.0, 1200.0]
    xmin, xmax = float(dr[0]), float(dr[1])
    ymin, ymax = float(hr[0]), float(hr[1])
    key = result.get("key")
    m = str(model_name).lower()
    if m in ("exppoly2d", "exppoly", "poly"):
        return ExpPoly2D(xmin, xmax, ymin, ymax, degree=deg, key=key, n_quad=n_quad)
    if m in ("exppoly_cheb2d", "cheb", "exppolycheb2d"):
        return ExpPolyCheb2D(xmin, xmax, ymin, ymax, degree=deg, key=key, n_quad=n_quad)
    if m in ("exppoly_logy2d", "logy", "exppolylogy2d"):
        y_eps = spec.get("y_eps")
        if y_eps is None:
            y_eps = result.get("y_eps")
        if y_eps is None:
            y_eps = 1e-3  # legacy JSONs: the historical hard-coded value
        return ExpPolyLogY2D(
            xmin,
            xmax,
            ymin,
            ymax,
            degree=deg,
            key=key,
            n_quad=n_quad,
            y_eps=float(y_eps),
        )
    if m in ("logbern2d", "logbern", "bernstein"):
        return LogBern2D(
            xmin,
            xmax,
            ymin,
            ymax,
            nx=int(spec.get("nx") or deg),
            ny=int(spec.get("ny") or deg),
            key=key,
            n_quad=n_quad,
        )
    if m in ("mixture_exppoly2d", "mix", "mixture"):
        return MixtureExpPoly2D(
            xmin,
            xmax,
            ymin,
            ymax,
            n_comp=int(spec.get("n_comp") or 2),
            degree=deg,
            key=key,
            n_quad=n_quad,
        )
    if m in ("exppoly_sep2d", "sep", "separable"):
        return ExpPolySep2D(
            xmin,
            xmax,
            ymin,
            ymax,
            dx=int(spec.get("dx") or deg),
            dy=int(spec.get("dy") or deg),
            key=key,
            n_quad=n_quad,
        )
    # default fallback
    return ExpPoly2D(xmin, xmax, ymin, ymax, degree=deg, key=key, n_quad=n_quad)


def eval_model_grid(result, x_c, y_c, params=None):
    """Pointwise density N*pdf (not bin counts). Prefer integrate_model_bins."""
    model = rebuild_density_2d(result)
    par = list(params if params is not None else result["parameters"])
    return model.eval_on_grid(par, list(x_c), list(y_c))


def integrate_model_bins(result, x_edges, y_edges, params=None, n_quad=8):
    """Integrate continuous N*pdf over each (x,y) bin rectangle.

    ``function.norm == "tiling"`` fits (fit_one ``--norm tiling``) are
    normalised by the fine-bin tiling window integral I_W (same as the fit):
    if the template edges coincide with the fine edges stored in the JSON the
    per-fine-bin integrals are reused directly (template sums to N exactly
    inside the window); otherwise each template bin is integrated with
    GL(n_quad) over its intersection with the fit window and divided by I_W.
    The density is zero outside the fit window in tiling mode (the pdf is
    defined on the window only; the legacy path extrapolates and yields NaN
    below the HME window, which callers ``nan_to_num`` → 0).
    """
    model = rebuild_density_2d(result)
    spec = result.get("function") or {}
    if spec.get("norm") == "tiling":
        par = list(params if params is not None else result["parameters"])
        return _integrate_model_bins_tiling(model, par, x_edges, y_edges, n_quad)
    if n_quad is not None:
        model.n_quad = max(int(n_quad), 2)
        model._nodes, model._weights = leggauss(model.n_quad)
        model._Is_cache_key = None
    par = list(params if params is not None else result["parameters"])
    return model.integrate_bins(par, x_edges, y_edges)


_TILING_PACK_CACHE = {}


def _tiling_pack_for_result(result, model):
    """(pack, include-mask) for a norm=tiling fit JSON (cached per binning)."""
    fb = result.get("fine_binning") or {}
    xe = fb.get("x_edges")
    ye = fb.get("y_edges")
    if xe is None or ye is None:
        raise ValueError("norm=tiling fit JSON lacks fine_binning.x_edges/y_edges")
    spec = result.get("function") or {}
    nqf = int(spec.get("n_quad_fine") or 2)
    key = (
        tuple(float(v) for v in xe),
        tuple(float(v) for v in ye),
        model.xmin,
        model.xmax,
        model.ymin,
        model.ymax,
        nqf,
    )
    pack = _TILING_PACK_CACHE.get(key)
    if pack is None:
        if len(_TILING_PACK_CACHE) >= 8:
            _TILING_PACK_CACHE.clear()
        pack = model.prepare_fine_tiling(
            xe, ye, model.xmin, model.xmax, model.ymin, model.ymax, n_quad_fine=nqf
        )
        _TILING_PACK_CACHE[key] = pack
    include = np.ones(pack["n_fine_in"], dtype=bool)
    for ix, iy in fb.get("unmapped") or []:
        p = int(pack["fine_pos"][int(ix), int(iy)])
        if p >= 0:
            include[p] = False
    return pack, include


def _integrate_model_bins_tiling(model, par, x_edges, y_edges, n_quad):
    tn = model._tiling_norm
    if tn is None:
        raise ValueError("model has no tiling normalisation attached")
    pack, include = tn["pack"], tn["include"]
    N = float(par[0])
    sp = [float(par[i]) for i in range(1, model.npar)]
    I_fine = model.integrals_fine(sp, pack)
    I_W = float(np.sum(I_fine[include]))
    xe = np.asarray(x_edges, dtype=float)
    ye = np.asarray(y_edges, dtype=float)
    nx, ny = len(xe) - 1, len(ye) - 1
    Z = np.zeros((nx, ny), dtype=float)
    if not (I_W > 0.0 and math.isfinite(I_W)):
        return Z
    same = (
        len(xe) == len(pack["x_edges"])
        and len(ye) == len(pack["y_edges"])
        and np.allclose(xe, pack["x_edges"])
        and np.allclose(ye, pack["y_edges"])
    )
    if same:
        vals = np.where(include, N * I_fine / I_W, 0.0)
        Z[pack["ix"], pack["iy"]] = vals
        return Z
    # Generic binning: GL(n_quad) over (bin ∩ window) / I_W
    wxmin, wxmax, wymin, wymax = pack["window"]
    old = (model.n_quad, model._nodes, model._weights)
    try:
        if n_quad is not None and int(n_quad) != model.n_quad:
            model.n_quad = max(int(n_quad), 2)
            model._nodes, model._weights = leggauss(model.n_quad)
        for ix in range(nx):
            xlo = max(float(xe[ix]), wxmin)
            xhi = min(float(xe[ix + 1]), wxmax)
            if not xhi > xlo:
                continue
            for iy in range(ny):
                ylo = max(float(ye[iy]), wymin)
                yhi = min(float(ye[iy + 1]), wymax)
                if not yhi > ylo:
                    continue
                Z[ix, iy] = N * model._integral_s_rect(sp, xlo, xhi, ylo, yhi) / I_W
    finally:
        model.n_quad, model._nodes, model._weights = old
    return Z


# --------------------------------------------------------------------------- cov eigenmodes


def eigen_shape_shifts(
    covariance,
    skip_indices=None,
    max_modes=0,
    min_eigenvalue=1e-12,
):
    """Return [(rank, sqrt(lambda), unit eigenvector in full param space), ...]."""
    V = np.array(covariance, dtype=float)
    n = V.shape[0]
    skip = set(skip_indices or [])
    keep = [i for i in range(n) if i not in skip]
    if not keep:
        return []
    Vsub = 0.5 * (V[np.ix_(keep, keep)] + V[np.ix_(keep, keep)].T)
    try:
        evals, evecs = np.linalg.eigh(Vsub)
    except np.linalg.LinAlgError:
        return []
    modes = []
    for rank, idx in enumerate(np.argsort(evals)[::-1]):
        lam = float(evals[idx])
        if lam <= min_eigenvalue:
            continue
        if max_modes and len(modes) >= max_modes:
            break
        u_full = np.zeros(n)
        for j, gi in enumerate(keep):
            u_full[gi] = float(evecs[j, idx])
        norm = float(np.linalg.norm(u_full))
        if norm > 0:
            u_full /= norm
        modes.append((rank, math.sqrt(lam), u_full))
    return modes

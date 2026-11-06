#!/usr/bin/env python3
"""Overlay DY MC vs fit as 1D HME projections in DNN slices -> multi-page PDF.

Plots only (no summary page).  One slice per page by default.  Full-width axes;
legend + annotations live in the right margin (no parameter list).

For each DNN slice:
  * DY HME projection (points) - never labelled "data"
  * fit model HME projection (line)
  * DNN range, model key, global chi2/ndf & p-value
  * Legend labels carry slice integrals with errors, e.g. DY [1.08 +/- 0.12],
    Fit [0.213 +/- 0.04].  DY error = sqrt(sum e_bin^2) (stat).  Fit error =
    sqrt(sum_k delta_I_k^2) with delta_I_k = max(|I0 - I_up|, |I0 - I_dn|)
    over covariance eigenmodes (same as make_hist_from_fit --shape-variations;
    N floated by default).

Two mutually exclusive ways to define slices:

1. Explicit edges (must match histogram bin edges after the fit rebin)::

       --dnn-slices=-7.5,-5,-3,-1,0,1,2,3,6.5

2. Auto equal-signal slices (approx. constant signal yield per slice)::

       --auto-dnn-slice-n-const-signal-bins 5

Usage::

  python3 -u plot_fit_slices.py \\
      --fit m500_fit.json --data-dir /path/to/hadd_files \\
      --auto-dnn-slice-n-const-signal-bins 5 \\
      --output m500_slices.pdf
"""

import argparse
from typing import List, Optional, Tuple
import json
import math
import os
import sys

import numpy as np
import ROOT

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import core

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)


def _load_result(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _load_dy_th2(result: dict, args):
    """Return (TH2 DY, x_centers, y_centers, Hist2DData).

    Content outside the fit DNN/HME window is zeroed so HME projections match
    the fit region (same selection used in the chi2).
    """
    mx = int(args.mx if args.mx is not None else result.get("mx") or 500)
    cat = args.category or result.get("category") or "res2b"
    proc = args.process or result.get("process") or "DY"
    if not args.data_dir:
        sys.exit("[ERROR] --data-dir is required")
    data_dir = args.data_dir
    fr = result.get("fit_range") or {}
    dr = fr.get("dnn") or [-7.5, 6.5]
    hr = fr.get("hme") or [240.0, 1200.0]
    dnn_min = float(args.dnn_min if args.dnn_min is not None else dr[0])
    dnn_max = float(args.dnn_max if args.dnn_max is not None else dr[1])
    hme_min = float(args.hme_min if args.hme_min is not None else hr[0])
    hme_max = float(args.hme_max if args.hme_max is not None else hr[1])

    hist = core.load_hist_2d(
        mx,
        data_dir=data_dir,
        category=cat,
        process=proc,
        dnn_min=dnn_min,
        dnn_max=dnn_max,
        hme_min=hme_min,
        hme_max=hme_max,
    )
    nx, ny = hist.nx, hist.ny
    xr, yr = hist.dnn_range, hist.hme_range
    h = ROOT.TH2D(
        f"dy_m{mx}",
        f"DY mX={mx};DNN;HME [GeV]",
        nx,
        hist.x_edges.astype(float),
        ny,
        hist.y_edges.astype(float),
    )
    h.SetDirectory(0)
    for ix in range(nx):
        x = float(hist.x_centers[ix])
        in_x = xr.xmin <= x <= xr.xmax
        for iy in range(ny):
            y = float(hist.y_centers[iy])
            if not (in_x and yr.xmin <= y <= yr.xmax):
                continue
            h.SetBinContent(ix + 1, iy + 1, float(hist.content[ix, iy]))
            h.SetBinError(ix + 1, iy + 1, float(hist.error[ix, iy]))
    return h, hist.x_centers.copy(), hist.y_centers.copy(), hist


def _fit_th2(result: dict, data: core.Hist2DData) -> ROOT.TH2D:
    """Integrate continuous N*pdf over each hist bin -> TH2 (zero errors)."""
    Z = core.integrate_model_bins(
        result, data.x_edges, data.y_edges, n_quad=int(result.get("n_quad") or 4)
    )
    h = ROOT.TH2D(
        "fit_model",
        "fit;DNN;HME [GeV]",
        data.nx,
        data.x_edges.astype(float),
        data.ny,
        data.y_edges.astype(float),
    )
    h.SetDirectory(0)
    xc, yc = data.x_centers, data.y_centers
    for ix in range(data.nx):
        for iy in range(data.ny):
            x, y = float(xc[ix]), float(yc[iy])
            if not (data.dnn_range.xmin <= x <= data.dnn_range.xmax):
                continue
            if not (data.hme_range.xmin <= y <= data.hme_range.xmax):
                continue
            h.SetBinContent(ix + 1, iy + 1, float(Z[ix, iy]))
            h.SetBinError(ix + 1, iy + 1, 0.0)
    return h


def _fit_eigen_integral_modes(
    result: dict,
    data: core.Hist2DData,
    *,
    vary_yield: bool = True,
    max_modes: int = 0,
) -> list:
    """Cov eigen +/-1sigma grids (same as make_hist_from_fit shape variations).

    Returns a list of ``(Z_up, Z_dn)`` arrays on ``data`` centres.  Used to
    propagate fit-parameter uncertainty into per-slice integrals via
    sqrt(sum_k delta_I_k^2) with
    delta_I_k = max(|I0 - I_up|, |I0 - I_dn|).  By default N is floated
    (full integral uncertainty); pass vary_yield=False for pure shape
    (N frozen).
    """
    cov = result.get("covariance")
    if not cov:
        sys.exit(
            "[ERROR] fit JSON has no covariance - cannot compute Fit integral "
            "errors (refit with strategy 2 / check hesse_ok in the JSON)"
        )
    params = list(result["parameters"])
    n = len(params)
    if len(cov) != n or any(len(row) != n for row in cov):
        sys.exit("[ERROR] covariance shape mismatch with parameters")
    try:
        diags = [float(cov[i][i]) for i in range(n)]
        if max(diags) <= 0.0:
            sys.exit(
                "[ERROR] covariance diagonals are all zero (null Hesse matrix); "
                "refit with --strategy 2"
            )
    except (TypeError, ValueError, IndexError):
        sys.exit("[ERROR] unreadable covariance matrix in fit JSON")

    skip = set()  # type: set
    if not vary_yield:
        skip.add(0)
        for i, name in enumerate(result.get("param_names") or []):
            if name == "N":
                skip.add(i)

    modes = core.eigen_shape_shifts(cov, skip_indices=skip, max_modes=max_modes)
    if not modes:
        sys.exit("[ERROR] no positive cov eigenmodes - cannot compute Fit errors")

    n_quad = int(result.get("n_quad") or 4)
    out = []
    for rank, sigma, u in modes:
        delta = sigma * u
        p_up = [float(params[i] + delta[i]) for i in range(n)]
        p_dn = [float(params[i] - delta[i]) for i in range(n)]
        try:
            Zu = core.integrate_model_bins(
                result, data.x_edges, data.y_edges, p_up, n_quad=n_quad
            )
            Zd = core.integrate_model_bins(
                result, data.x_edges, data.y_edges, p_dn, n_quad=n_quad
            )
        except Exception as exc:
            print("[WARN] eigen mode %s: eval failed (%s)" % (rank, exc))
            continue
        out.append((Zu, Zd))
    return out


def _slice_integral_from_grid(
    Z: np.ndarray,
    data: core.Hist2DData,
    ix_lo: int,
    ix_hi: int,
    hme_min: float,
    hme_max: float,
) -> float:
    """Sum model grid over DNN bins [ix_lo, ix_hi] and HME centres in window."""
    ymask = (data.y_centers >= hme_min) & (data.y_centers <= hme_max)
    return float(Z[ix_lo : ix_hi + 1, :][:, ymask].sum())


def _fit_slice_integral_err(
    modes: list,
    data: core.Hist2DData,
    ix_lo: int,
    ix_hi: int,
    hme_min: float,
    hme_max: float,
    int_central: float,
) -> float:
    """sqrtsum_k (delta_I_k)2 with delta_I_k = max(|I0 - I_up|, |I0 - I_dn|) per eigenmode."""
    if not modes:
        return 0.0
    i0 = float(int_central)
    acc = 0.0
    for Zu, Zd in modes:
        i_up = _slice_integral_from_grid(Zu, data, ix_lo, ix_hi, hme_min, hme_max)
        i_dn = _slice_integral_from_grid(Zd, data, ix_lo, ix_hi, hme_min, hme_max)
        di = max(abs(i0 - i_up), abs(i0 - i_dn))
        acc += di * di
    return math.sqrt(acc)


def _parse_dnn_slices(s: str) -> List[float]:
    """Parse comma-separated floats into a sorted edge list."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) < 2:
        sys.exit(
            "[ERROR] --dnn-slices needs at least two edges " "(e.g. -7.5,-3,0,3,6.5)"
        )
    try:
        edges = [float(p) for p in parts]
    except ValueError as exc:
        sys.exit(f"[ERROR] --dnn-slices: invalid float in {s!r}: {exc}")
    # keep user order for reporting, but require strictly increasing
    for i in range(len(edges) - 1):
        if not (edges[i + 1] > edges[i]):
            sys.exit(
                f"[ERROR] --dnn-slices must be strictly increasing; "
                f"got {edges[i]} >= {edges[i + 1]} at position {i}"
            )
    return edges


def _match_edge(value: float, bin_edges: np.ndarray, *, tol: float = 1e-9) -> int:
    """Return index i such that bin_edges[i] == value (within tol), or -1."""
    for i, e in enumerate(bin_edges):
        if abs(float(e) - float(value)) <= tol:
            return i
    return -1


def _slice_ranges_from_edges(
    x_edges: np.ndarray,
    slice_edges: List[float],
    *,
    tol: float = 1e-9,
) -> list:
    """Map DNN slice edges -> (ix_lo, ix_hi, x_lo, x_hi).

    ``slice_edges`` must each coincide with a histogram bin edge.
    Slice k covers bins with low edge in [slice_edges[k], slice_edges[k+1]).
    Returns 0-based inclusive bin indices for ProjectionY.
    """
    # Validate every requested edge
    bad = []
    edge_idx = []
    for v in slice_edges:
        i = _match_edge(v, x_edges, tol=tol)
        if i < 0:
            bad.append(v)
        else:
            edge_idx.append(i)
    if bad:
        # helpful message: nearest edges + full list of available edges in range
        avail = ", ".join(f"{float(e):g}" for e in x_edges)
        nearest = []
        for v in bad:
            j = int(np.argmin(np.abs(x_edges - v)))
            nearest.append(f"{v:g} (nearest bin edge {float(x_edges[j]):g})")
        sys.exit(
            "[ERROR] --dnn-slices values are not aligned with histogram bin edges:\n"
            + "\n".join(f"  - {n}" for n in nearest)
            + f"\nAvailable DNN bin edges ({len(x_edges)}):\n  {avail}"
        )

    out = []
    for k in range(len(slice_edges) - 1):
        i_lo = edge_idx[k]  # index of low edge = first bin of slice
        i_hi_edge = edge_idx[k + 1]  # index of high edge = first bin AFTER slice
        # bins are between edges: bin j has edges[j] .. edges[j+1]
        # so last included bin index is i_hi_edge - 1
        i_hi = i_hi_edge - 1
        if i_hi < i_lo:
            sys.exit(
                f"[ERROR] empty DNN slice [{slice_edges[k]}, {slice_edges[k + 1]}]: "
                f"no bins between matched edges"
            )
        out.append((i_lo, i_hi, float(slice_edges[k]), float(slice_edges[k + 1])))
    return out


def _load_signal_dnn_projection(
    result: dict, args, data: core.Hist2DData
) -> np.ndarray:
    """Per-DNN-bin signal yield inside the fit HME window (native bins).

    Uses plots_2d/signal_2d (override with --signal-hist).
    """
    mx = int(args.mx if args.mx is not None else result.get("mx") or 500)
    cat = args.category or result.get("category") or "res2b"
    if not args.data_dir:
        raise RuntimeError("--data-dir is required")
    data_dir = args.data_dir
    fr = result.get("fit_range") or {}
    dr = fr.get("dnn") or [-7.5, 6.5]
    hr = fr.get("hme") or [240.0, 1200.0]
    dnn_min = float(args.dnn_min if args.dnn_min is not None else dr[0])
    dnn_max = float(args.dnn_max if args.dnn_max is not None else dr[1])
    hme_min = float(args.hme_min if args.hme_min is not None else hr[0])
    hme_max = float(args.hme_max if args.hme_max is not None else hr[1])
    hist_name = args.signal_hist or "plots_2d/signal_2d"

    try:
        sig = core.load_hist_2d(
            mx,
            data_dir=data_dir,
            category=cat,
            process="signal",
            dnn_min=dnn_min,
            dnn_max=dnn_max,
            hme_min=hme_min,
            hme_max=hme_max,
            hist_name=hist_name,
        )
    except (FileNotFoundError, KeyError) as exc:
        raise RuntimeError(
            "cannot load signal histogram (%s in mX=%s %s): %s"
            % (hist_name, mx, cat, exc)
        )

    if sig.nx != data.nx or not np.allclose(
        sig.x_edges, data.x_edges, atol=1e-9, rtol=0
    ):
        raise RuntimeError(
            "signal DNN binning does not match DY (native bins must agree)"
        )

    ymask = (sig.y_centers >= hme_min) & (sig.y_centers <= hme_max)
    # clamp negatives (rare MC underflow) so cumulative stays monotonic
    cont = np.maximum(sig.content[:, ymask], 0.0)
    return cont.sum(axis=1)  # shape (nx,)


def _auto_const_signal_slice_edges(
    x_edges: np.ndarray,
    x_centers: np.ndarray,
    sig_yield_dnn: np.ndarray,
    n_slices: int,
    dnn_min: float,
    dnn_max: float,
) -> Tuple[List[float], List[float]]:
    """Choose N DNN slices with approximately equal signal yield.

    Returns ``(slice_edges, per_slice_signal)`` where edges are exact histogram
    bin edges.  The partition is performed on the **signal support** inside the
    fit DNN window (first...last positive-yield bin), so masses where the DNN
    score is peaked do not force empty high/low-score slices.  Boundaries snap
    to bin edges via closest cumulative-yield quantiles (>=1 bin per slice).
    """
    if n_slices < 1:
        sys.exit("[ERROR] --auto-dnn-slice-n-const-signal-bins must be >= 1")

    window_idxs = [i for i, x in enumerate(x_centers) if dnn_min <= float(x) <= dnn_max]
    if not window_idxs:
        sys.exit(f"[ERROR] no DNN bins in fit window [{dnn_min:g}, {dnn_max:g}]")

    window_yields = np.array(
        [float(max(sig_yield_dnn[i], 0.0)) for i in window_idxs], dtype=float
    )
    total = float(window_yields.sum())

    if total <= 0.0:
        print(
            "[WARN] signal yield in fit window is zero; "
            "falling back to equal-bin DNN slices over the full window"
        )
        idxs = window_idxs
        yields = window_yields
        if len(idxs) < n_slices:
            sys.exit(
                f"[ERROR] only {len(idxs)} DNN bins but requested {n_slices} slices"
            )
        edges = [float(x_edges[idxs[0]])]
        for k in range(1, n_slices):
            j = int(round(k * len(idxs) / n_slices)) - 1
            j = max(j, 0)
            j = min(j, len(idxs) - 1 - (n_slices - k))
            edges.append(float(x_edges[idxs[j] + 1]))
        edges.append(float(x_edges[idxs[-1] + 1]))
        edges = _dedupe_strict_edges(edges)
        per = _slice_signal_yields(idxs, yields, edges, x_edges)
        return edges, per

    # Restrict to the central signal mass (drop empty / negligible tails).
    # Cumulants [0.5%, 99.5%] keep ~99% of S while avoiding eps-yield edge bins
    # that would otherwise steal an entire equal-signal slice.
    cum_w = np.cumsum(window_yields)
    j0 = int(np.searchsorted(cum_w, 0.005 * total, side="left"))
    j1 = int(np.searchsorted(cum_w, 0.995 * total, side="left"))
    j0 = max(0, min(j0, len(window_idxs) - 1))
    j1 = max(j0, min(j1, len(window_idxs) - 1))
    idxs = window_idxs[j0 : j1 + 1]
    yields = window_yields[j0 : j1 + 1]
    n_support = len(idxs)
    support_tot = float(yields.sum())
    print(
        f"[auto-slices] signal support DNN in "
        f"[{float(x_edges[idxs[0]]):g}, {float(x_edges[idxs[-1] + 1]):g}]  "
        f"({n_support} bins, S={support_tot:.4g} / {total:.4g} total "
        f"= {100.0 * support_tot / total:.1f}%)"
    )
    if n_support < n_slices:
        sys.exit(
            f"[ERROR] only {n_support} positive-signal DNN bins in the fit window "
            f"but requested {n_slices} equal-signal slices - reduce N or rebin finer"
        )

    cum = np.cumsum(yields)
    # work with the support total (may equal `total` if no internal zeros only)
    support_total = float(cum[-1])
    edges = [float(x_edges[idxs[0]])]
    used_end = -1
    for k in range(1, n_slices):
        target = k * support_total / float(n_slices)
        remaining = n_slices - k
        max_j = n_support - 1 - remaining
        min_j = used_end + 1
        if min_j > max_j:
            sys.exit(
                f"[ERROR] cannot place equal-signal boundary {k}/{n_slices}: "
                f"not enough DNN bins in signal support"
            )
        best_j = min_j
        best_err = abs(float(cum[min_j]) - target)
        for j in range(min_j, max_j + 1):
            err = abs(float(cum[j]) - target)
            if err < best_err - 1e-15:
                best_err = err
                best_j = j
        edges.append(float(x_edges[idxs[best_j] + 1]))
        used_end = best_j
    edges.append(float(x_edges[idxs[-1] + 1]))
    edges = _dedupe_strict_edges(edges)
    if len(edges) - 1 != n_slices:
        print(
            f"[WARN] equal-signal slicing produced {len(edges) - 1} slices "
            f"(requested {n_slices}) after edge dedupe - signal may be "
            f"concentrated in few DNN bins"
        )
    per = _slice_signal_yields(idxs, yields, edges, x_edges)
    return edges, per


def _dedupe_strict_edges(edges: List[float], *, tol: float = 1e-12) -> List[float]:
    out = [edges[0]]
    for e in edges[1:]:
        if abs(e - out[-1]) > tol:
            out.append(e)
    if len(out) < 2:
        sys.exit("[ERROR] auto DNN slice edges collapsed to a single point")
    return out


def _slice_signal_yields(
    idxs,
    yields: np.ndarray,
    edges: List[float],
    x_edges: np.ndarray,
) -> List[float]:
    """Sum signal yield per slice from local (idxs, yields) arrays."""
    # map edge value -> x_edges index
    edge_ix = []
    for e in edges:
        i = _match_edge(e, x_edges, tol=1e-6)
        if i < 0:
            i = int(np.argmin(np.abs(x_edges - e)))
        edge_ix.append(i)
    per = []
    for k in range(len(edges) - 1):
        lo, hi = edge_ix[k], edge_ix[k + 1] - 1  # inclusive bin indices
        s = 0.0
        for local, gix in enumerate(idxs):
            if lo <= gix <= hi:
                s += float(yields[local])
        per.append(s)
    return per


def _project_hme(
    h2: ROOT.TH2,
    ix_lo: int,
    ix_hi: int,
    name: str,
) -> ROOT.TH1D:
    """Project Y (HME) for DNN bins [ix_lo, ix_hi] inclusive (0-based)."""
    # ROOT bins are 1-based
    proj = h2.ProjectionY(name, ix_lo + 1, ix_hi + 1, "e")
    proj.SetDirectory(0)
    # PyROOT does not own what ProjectionY() returns; detached, the projection would be
    # owned by nobody, never freed
    ROOT.SetOwnership(proj, True)
    return proj


def _integral_in_range(h: ROOT.TH1D, lo: float, hi: float) -> Tuple[float, float]:
    """Sum bin contents (+ stat err sqrt(sum e^2)) whose centres lie in [lo, hi]."""
    s = 0.0
    e2 = 0.0
    for ib in range(1, h.GetNbinsX() + 1):
        c = h.GetXaxis().GetBinCenter(ib)
        if lo <= c <= hi:
            s += float(h.GetBinContent(ib))
            e = float(h.GetBinError(ib))
            e2 += e * e
    return s, math.sqrt(e2)


def _fmt_int_pm(val: float, err: Optional[float]) -> str:
    """Format ``v`` or ``v +/- e`` compactly for legend labels."""

    def _one(v: float) -> str:
        av = abs(v)
        if av >= 1e4 or (av > 0 and av < 1e-2):
            return f"{v:.3g}"
        if av >= 100:
            return f"{v:.1f}"
        if av >= 10:
            return f"{v:.2f}"
        return f"{v:.3g}"

    if err is None or not math.isfinite(err) or err <= 0:
        return _one(val)
    return f"{_one(val)} #pm {_one(err)}"


def _draw_slice_pad(
    pad: ROOT.TPad,
    h_dy: ROOT.TH1D,
    h_fit: ROOT.TH1D,
    *,
    title: str,
    result: dict,
    hme_min: float,
    hme_max: float,
    process_label: str = "DY",
    slice_label: Optional[str] = None,
    signal_yield: Optional[float] = None,
    signal_frac: Optional[float] = None,
    int_dy: Optional[float] = None,
    int_dy_err: Optional[float] = None,
    int_fit: Optional[float] = None,
    int_fit_err: Optional[float] = None,
):
    """Full plot frame; legend + summary text inside the axes, top-right."""
    pad.cd()
    pad.SetLeftMargin(0.11)
    pad.SetRightMargin(0.05)
    pad.SetBottomMargin(0.12)
    pad.SetTopMargin(0.07)

    h_dy.SetMarkerStyle(20)
    h_dy.SetMarkerSize(0.9)
    h_dy.SetLineColor(ROOT.kBlack)
    h_dy.SetMarkerColor(ROOT.kBlack)
    h_dy.GetXaxis().SetTitle("HME [GeV]")
    h_dy.GetYaxis().SetTitle("Events")
    h_dy.GetXaxis().SetRangeUser(hme_min, hme_max)
    h_dy.SetTitle(title)

    h_fit.SetLineColor(ROOT.kRed + 1)
    h_fit.SetLineWidth(2)
    h_fit.SetMarkerSize(0)
    h_fit.GetXaxis().SetRangeUser(hme_min, hme_max)

    # y-range from visible HME window only (headroom so text does not sit on peaks)
    ymax = 1e-6
    for h in (h_dy, h_fit):
        for ib in range(1, h.GetNbinsX() + 1):
            c = h.GetXaxis().GetBinCenter(ib)
            if hme_min <= c <= hme_max:
                ymax = max(ymax, h.GetBinContent(ib))
    h_dy.SetMinimum(0.0)
    h_dy.SetMaximum(ymax * 1.45)

    h_dy.Draw("E1")
    h_fit.Draw("HIST SAME")

    if int_dy is None or int_dy_err is None:
        int_dy, int_dy_err = _integral_in_range(h_dy, hme_min, hme_max)
    if int_fit is None:
        int_fit, _ = _integral_in_range(h_fit, hme_min, hme_max)

    # Inside the plot frame, top-right: integral +/- err in legend labels
    leg = ROOT.TLegend(0.55, 0.78, 0.93, 0.91)
    leg.SetBorderSize(0)
    leg.SetFillStyle(1001)
    leg.SetFillColor(ROOT.kWhite)
    leg.SetTextSize(0.032)
    leg.AddEntry(
        h_dy,
        f"{process_label} [{_fmt_int_pm(float(int_dy), int_dy_err)}]",
        "lep",
    )
    leg.AddEntry(
        h_fit,
        f"Fit [{_fmt_int_pm(float(int_fit), int_fit_err)}]",
        "l",
    )
    leg.Draw()

    chi2ndf = result.get("chi2ndf")
    pval = result.get("p_value")
    chi2 = result.get("chi2")
    ndf = result.get("ndf")
    key = result.get("key") or result.get("function_name") or "?"

    pave = ROOT.TPaveText(0.62, 0.52, 0.93, 0.77, "NDC")
    pave.SetFillColor(ROOT.kWhite)
    pave.SetFillStyle(1001)
    pave.SetBorderSize(0)
    pave.SetTextAlign(12)
    pave.SetTextSize(0.028)
    if slice_label:
        pave.AddText(slice_label)
    pave.AddText(str(key))
    if chi2ndf is not None and math.isfinite(float(chi2ndf)):
        if chi2 is not None and ndf is not None:
            pave.AddText(
                f"chi^{{2}}/ndf = {float(chi2ndf):.3f}  ({float(chi2):.1f}/{int(ndf)})"
            )
        else:
            pave.AddText(f"chi^{{2}}/ndf = {float(chi2ndf):.3f}")
    if pval is not None and math.isfinite(float(pval)):
        pave.AddText(f"p = {float(pval):.3g}")
    if signal_yield is not None and math.isfinite(float(signal_yield)):
        if signal_frac is not None and math.isfinite(float(signal_frac)):
            pave.AddText(
                f"S = {float(signal_yield):.3g} ({100.0 * float(signal_frac):.1f}%)"
            )
        else:
            pave.AddText(f"S = {float(signal_yield):.3g}")
    pave.Draw()

    pad._leg = leg
    pad._gof = pave
    pad._hdy = h_dy
    pad._hfit = h_fit
    pad._int_dy = int_dy
    pad._int_dy_err = int_dy_err
    pad._int_fit = int_fit
    pad._int_fit_err = int_fit_err


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--fit", required=True, help="Fit-result JSON")
    p.add_argument("--output", "-o", required=True, help="Output PDF path")
    p.add_argument("--mx", type=int, default=None, help="Override mX from JSON")
    p.add_argument("--category", default=None)
    p.add_argument("--process", default=None)
    p.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing hadd_m{MX}_{category}.root",
    )
    p.add_argument("--dnn-min", type=float, default=None)
    p.add_argument("--dnn-max", type=float, default=None)
    p.add_argument("--hme-min", type=float, default=None)
    p.add_argument("--hme-max", type=float, default=None)
    p.add_argument(
        "--fit-int-freeze-n",
        action="store_true",
        help="Freeze N when propagating cov -> fit integral error "
        "(default: float all params including N)",
    )
    p.add_argument(
        "--max-modes",
        type=int,
        default=0,
        help="Max cov eigenmodes for fit integral error (0 = all positive)",
    )
    slice_grp = p.add_mutually_exclusive_group(required=True)
    slice_grp.add_argument(
        "--dnn-slices",
        default=None,
        metavar="EDGES",
        help="Comma-separated DNN slice edges (must match histogram bin edges). "
        "Use equals form so negatives are not parsed as flags: "
        "--dnn-slices=-7.5,-5.1,-2.7,0.5,3.7,6.9",
    )
    slice_grp.add_argument(
        "--auto-dnn-slice-n-const-signal-bins",
        type=int,
        default=None,
        metavar="N",
        dest="auto_n_const_signal",
        help="Auto-choose N DNN slices with approximately equal signal yield "
        "(from plots_2d/signal_2d, same HME window as the fit).",
    )
    p.add_argument(
        "--signal-hist",
        default=None,
        help="ROOT path of the signal TH2 inside the hadd file "
        "(default: plots_2d/signal_2d). Only used with "
        "--auto-dnn-slice-n-const-signal-bins.",
    )
    p.add_argument(
        "--edge-tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance when matching --dnn-slices to bin edges (default 1e-6)",
    )
    p.add_argument(
        "--pads-per-page",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 6, 9],
        help="Number of slice pads per PDF page (default 1 = one canvas per slice)",
    )
    return p.parse_args()


def _pad_layout(n: int) -> Tuple[int, int]:
    return {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        6: (2, 3),
        9: (3, 3),
    }[n]


def main():
    args = parse_args()
    result = _load_result(args.fit)
    h_dy2, x_c, y_c, data = _load_dy_th2(result, args)
    h_fit2 = _fit_th2(result, data)
    process_label = str(args.process or result.get("process") or "DY")

    dnn_min, dnn_max = data.dnn_range.xmin, data.dnn_range.xmax
    hme_min, hme_max = data.hme_range.xmin, data.hme_range.xmax

    # Require a valid fit JSON: covariance + p >= 0.05
    pval = result.get("p_value")
    if pval is None or not math.isfinite(float(pval)):
        sys.exit("[ERROR] fit JSON has no p-value")
    if float(pval) < 0.05:
        sys.exit(
            "[ERROR] fit JSON p-value %.4g < 0.05 - refit before plotting" % float(pval)
        )

    # Cov eigen +/-1sigma grids for fit-integral uncertainty
    fit_modes = _fit_eigen_integral_modes(
        result,
        data,
        vary_yield=not bool(args.fit_int_freeze_n),
        max_modes=int(args.max_modes),
    )
    print(
        "[fit-int] %d cov eigenmodes for Fit integral error (N %s)"
        % (
            len(fit_modes),
            "frozen" if args.fit_int_freeze_n else "floated",
        )
    )
    slice_mode = "manual"
    per_slice_signal: Optional[List[float]] = None
    if args.auto_n_const_signal is not None:
        slice_mode = "auto_const_signal"
        n_auto = int(args.auto_n_const_signal)
        if n_auto < 1:
            sys.exit("[ERROR] --auto-dnn-slice-n-const-signal-bins must be >= 1")
        try:
            sig_dnn = _load_signal_dnn_projection(result, args, data)
        except RuntimeError as exc:
            sys.exit(f"[ERROR] auto equal-signal slices: {exc}")
        slice_edges, per_slice_signal = _auto_const_signal_slice_edges(
            data.x_edges,
            data.x_centers,
            sig_dnn,
            n_auto,
            dnn_min,
            dnn_max,
        )
        total_s = sum(per_slice_signal) if per_slice_signal else 0.0
        print(
            f"[auto-slices] N={n_auto}  mode=const-signal  "
            f"edges=[{', '.join(f'{e:g}' for e in slice_edges)}]"
        )
        for k, (lo, hi) in enumerate(zip(slice_edges[:-1], slice_edges[1:])):
            s = per_slice_signal[k] if k < len(per_slice_signal) else float("nan")
            frac = (100.0 * s / total_s) if total_s > 0 else float("nan")
            print(
                f"  slice {k}: DNN in [{lo:g}, {hi:g}]  "
                f"S={s:.4g} ({frac:.1f}% of total)"
            )
    else:
        slice_edges = _parse_dnn_slices(args.dnn_slices)

    slices = _slice_ranges_from_edges(
        data.x_edges, slice_edges, tol=float(args.edge_tol)
    )
    if not slices:
        sys.exit("[ERROR] No DNN slices produced")

    # If manual mode, still annotate signal yields when signal_2d is available
    if per_slice_signal is None:
        try:
            sig_dnn = _load_signal_dnn_projection(result, args, data)
            idxs = [
                i
                for i, x in enumerate(data.x_centers)
                if dnn_min <= float(x) <= dnn_max
            ]
            yields = np.array([float(max(sig_dnn[i], 0.0)) for i in idxs], dtype=float)
            per_slice_signal = _slice_signal_yields(
                idxs, yields, slice_edges, data.x_edges
            )
        except Exception as exc:
            print(f"[WARN] could not annotate signal per slice: {exc}")
            per_slice_signal = None
    total_sig = float(sum(per_slice_signal)) if per_slice_signal else 0.0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    # multi-page PDF of plots only (no summary page)
    canvas = ROOT.TCanvas("c_slices", "HME slices", 1200, 900)
    pdf = args.output
    if not pdf.lower().endswith(".pdf"):
        pdf = pdf + ".pdf"
    canvas.Print(pdf + "[")

    n_pp = int(args.pads_per_page)
    nrows, ncols = _pad_layout(n_pp)
    keep = []  # prevent GC of histos/pads

    for page_start in range(0, len(slices), n_pp):
        canvas.Clear()
        if n_pp == 1:
            canvas.cd()
            pad = canvas
        else:
            canvas.Divide(ncols, nrows)
        page_slices = slices[page_start : page_start + n_pp]
        for ipad, (i0, i1, xlo, xhi) in enumerate(page_slices, start=1):
            global_k = page_start + ipad - 1
            if n_pp != 1:
                pad = canvas.cd(ipad)
            hd = _project_hme(h_dy2, i0, i1, f"dy_s{page_start}_{ipad}")
            hf = _project_hme(h_fit2, i0, i1, f"fit_s{page_start}_{ipad}")
            for h in (hd, hf):
                h.GetXaxis().SetRangeUser(hme_min, hme_max)
            title = (
                f"slice {global_k}: DNN #in [{xlo:g}, {xhi:g}]  "
                f"(bins {i0 + 1}-{i1 + 1})"
            )
            slice_label = f"DNN #in [{xlo:g}, {xhi:g}]"
            s_y = (
                per_slice_signal[global_k]
                if per_slice_signal is not None and global_k < len(per_slice_signal)
                else None
            )
            s_f = (s_y / total_sig) if (s_y is not None and total_sig > 0) else None
            int_dy, int_dy_err = _integral_in_range(hd, hme_min, hme_max)
            int_fit, _ = _integral_in_range(hf, hme_min, hme_max)
            int_fit_err = _fit_slice_integral_err(
                fit_modes, data, i0, i1, hme_min, hme_max, int_fit
            )
            _draw_slice_pad(
                pad,
                hd,
                hf,
                title=title,
                result=result,
                hme_min=hme_min,
                hme_max=hme_max,
                process_label=process_label,
                slice_label=slice_label,
                signal_yield=s_y,
                signal_frac=s_f,
                int_dy=int_dy,
                int_dy_err=int_dy_err,
                int_fit=int_fit,
                int_fit_err=int_fit_err if int_fit_err > 0 else None,
            )
            print(
                f"  [plot] slice {global_k} DNN=[{xlo:g},{xhi:g}]  "
                f"I_DY={int_dy:.4g}+/-{int_dy_err:.4g}  "
                f"I_Fit={int_fit:.4g}+/-{int_fit_err:.4g}"
                + (f"  ratio={int_fit / int_dy:.3f}" if int_dy > 0 else "")
            )
            keep.extend([hd, hf, pad])
        canvas.Print(pdf)

    canvas.Print(pdf + "]")
    print(f"[plot] {len(slices)} slices ({slice_mode}, {n_pp}/page) -> {pdf}")


if __name__ == "__main__":
    main()

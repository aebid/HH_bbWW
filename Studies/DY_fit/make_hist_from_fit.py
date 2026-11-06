#!/usr/bin/env python3
"""Build TH2 template(s) from an ExpPolyLogY2D (or ExpPoly2D) fit JSON.

Bin content = numerical integral of the continuous model density over each
bin rectangle (2D Gauss-Legendre).

**Accepted uncertainty thr** (from fit JSON ``thr_min`` / ``acceptable_unc_thr``)::

    δ_i = thr · μ_i
    nominal bin error   = δ_i
    acceptedUnc         = δ_i          (absolute, same binning)
    acceptedUncUp/Down  = μ_i ± δ_i    (Down floored at 0)

This is the same thr used in the χ² fit
(σ² = σ_stat² + (thr·|content|)²).  Future steps should use these hists
for the thr systematic (not re-derive thr).

Binning: ``--from-hist FILE[:TH2]`` or explicit ``--x-range/--x-bins`` etc.

``--shape-variations``: covariance eigenmodes → ``eig{k}Up/Down`` (N frozen
unless ``--vary-yield``).

Examples::

  python3 -u make_hist_from_fit.py \\
      --fit m500_fit.json \\
      --from-hist /data/hadd_m500_res2b.root:plots_2d/DY_2d \\
      -o templates.root --shape-variations
"""

import argparse
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

# Default hist path inside hadd files when --from-hist is "file.root" only
_DEFAULT_FROM_HIST = "plots_2d/DY_2d"


def safe_filename(key):
    s = key.replace(" ", "")
    for a, b in (("/", "_"), ("[", "_"), ("]", ""), ("(", ""), (")", ""), (",", "_")):
        s = s.replace(a, b)
    return "".join(ch if (ch.isalnum() or ch in "._-+") else "_" for ch in s)


def bin_edges_uniform(lo, hi, nbins):
    return np.linspace(lo, hi, nbins + 1)


def th2_axis_edges(axis):
    """Full edge array for a ROOT TAxis (variable or fixed binning)."""
    n = axis.GetNbins()
    edges = np.zeros(n + 1, dtype=float)
    for i in range(1, n + 1):
        edges[i - 1] = axis.GetBinLowEdge(i)
    edges[n] = axis.GetBinUpEdge(n)
    return edges


def parse_from_hist_spec(spec):
    """Parse 'file.root' or 'file.root:hist/path' -> (file, hist_path)."""
    spec = spec.strip()
    if not spec:
        sys.exit("[ERROR] empty --from-hist")
    # Split on last ':' that looks like hist path (not Windows drive)
    if ":" in spec:
        # Prefer rightmost colon after .root
        low = spec.lower()
        idx = low.rfind(".root:")
        if idx >= 0:
            root_file = spec[: idx + 5]
            hist_path = spec[idx + 6 :]
            if not hist_path:
                sys.exit("[ERROR] --from-hist: empty hist path after ':'")
            return root_file, hist_path
        # generic last colon
        root_file, hist_path = spec.rsplit(":", 1)
        if (
            hist_path
            and not hist_path.startswith("/")
            and os.path.sep not in hist_path[:2]
        ):
            return root_file, hist_path
    return spec, _DEFAULT_FROM_HIST


def _load_ref_th2(root_file, hist_path):
    f = ROOT.TFile.Open(root_file)
    if not f or f.IsZombie():
        sys.exit("[ERROR] --from-hist: cannot open %s" % root_file)
    h = f.Get(hist_path)
    if not h:
        f.Close()
        sys.exit(
            "[ERROR] --from-hist: histogram '%s' not found in %s"
            % (hist_path, root_file)
        )
    h.SetDirectory(0)
    # PyROOT does not own what Get() returns; detached, the histogram would be owned
    # by nobody, never freed
    ROOT.SetOwnership(h, True)
    f.Close()
    return h


def load_binning_from_hist(spec):
    """Return (x_edges, y_edges, meta_dict) from a reference TH2."""
    root_file, hist_path = parse_from_hist_spec(spec)
    if not os.path.isfile(root_file):
        sys.exit("[ERROR] --from-hist: file not found: %s" % root_file)
    h = _load_ref_th2(root_file, hist_path)
    if not isinstance(h, ROOT.TH2):
        sys.exit(
            "[ERROR] --from-hist: '%s' in %s is not a TH2 (got %s)"
            % (hist_path, root_file, h.ClassName() if h else None)
        )
    x_e = th2_axis_edges(h.GetXaxis())
    y_e = th2_axis_edges(h.GetYaxis())
    meta = {
        "from_hist_file": root_file,
        "from_hist_name": hist_path,
        "x_bins": int(len(x_e) - 1),
        "y_bins": int(len(y_e) - 1),
        "x_range": [float(x_e[0]), float(x_e[-1])],
        "y_range": [float(y_e[0]), float(y_e[-1])],
    }
    print(
        "[hist] binning from %s:%s  nx=%d ny=%d  x=[%g,%g]  y=[%g,%g]"
        % (
            root_file,
            hist_path,
            meta["x_bins"],
            meta["y_bins"],
            meta["x_range"][0],
            meta["x_range"][1],
            meta["y_range"][0],
            meta["y_range"][1],
        )
    )
    return x_e, y_e, meta


def thr_from_fit(result):
    """Accepted-unc thr used in the fit (prefer thr_min, else acceptable_unc_thr)."""
    for key in ("thr_min", "acceptable_unc_thr", "thr"):
        if key in result and result[key] is not None:
            try:
                t = float(result[key])
                if math.isfinite(t) and t >= 0.0:
                    return t
            except (TypeError, ValueError):
                pass
    return 0.0


def fill_th2(name, title, x_edges, y_edges, Z, Zerr=None):
    """Fill TH2 content from Z[ix,iy]; optional Zerr for bin errors."""
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    h = ROOT.TH2D(name, title, nx, x_edges.astype(float), ny, y_edges.astype(float))
    h.SetDirectory(0)
    # main() runs this with a directory of the output file current, which then owns the
    # new histogram, not PyROOT; detached, it would be owned by nobody, never freed
    ROOT.SetOwnership(h, True)
    for ix in range(nx):
        for iy in range(ny):
            h.SetBinContent(ix + 1, iy + 1, float(Z[ix, iy]))
            if Zerr is not None:
                h.SetBinError(ix + 1, iy + 1, float(Zerr[ix, iy]))
            else:
                h.SetBinError(ix + 1, iy + 1, 0.0)
    return h


def process_one_fit(
    result,
    x_edges,
    y_edges,
    shape_variations,
    vary_yield,
    max_modes,
    hist_name="nominal",
    n_quad=8,
    accepted_unc=True,
):
    key = result.get("key") or result.get("function_name") or "fit"
    params = list(result["parameters"])
    Z0 = core.integrate_model_bins(result, x_edges, y_edges, params, n_quad=n_quad)
    # Bins outside the fit window can yield non-finite values (e.g. log-HME map)
    Z0 = np.nan_to_num(Z0, nan=0.0, posinf=0.0, neginf=0.0)
    Z0 = np.maximum(Z0, 0.0)
    thr = thr_from_fit(result)
    # Bin-by-bin accepted unc (same thr as in χ²): δ = thr · μ
    Z_acc = thr * Z0

    hists = {
        hist_name: fill_th2(
            hist_name,
            "%s;DNN;HME [GeV]" % key,
            x_edges,
            y_edges,
            Z0,
            Zerr=Z_acc if accepted_unc else None,
        )
    }

    if accepted_unc:
        # Absolute map δ_i (for Combine/log-normal rate systematics, etc.)
        h_acc = fill_th2(
            "%s_acceptedUnc" % hist_name,
            "%s accepted unc (thr=%.4g)·mu;DNN;HME [GeV]" % (key, thr),
            x_edges,
            y_edges,
            Z_acc,
        )
        hists["%s_acceptedUnc" % hist_name] = h_acc
        # Up / Down yield templates: μ ± thr·μ
        Z_up = Z0 + Z_acc
        Z_dn = np.maximum(Z0 - Z_acc, 0.0)
        hists["%s_acceptedUncUp" % hist_name] = fill_th2(
            "%s_acceptedUncUp" % hist_name,
            "%s accepted unc Up;DNN;HME [GeV]" % key,
            x_edges,
            y_edges,
            Z_up,
        )
        hists["%s_acceptedUncDown" % hist_name] = fill_th2(
            "%s_acceptedUncDown" % hist_name,
            "%s accepted unc Down;DNN;HME [GeV]" % key,
            x_edges,
            y_edges,
            Z_dn,
        )
        print(
            "[hist] %s  thr=%.4g  sum(mu)=%.6g  sum(acceptedUnc)=%.6g"
            % (key, thr, float(Z0.sum()), float(Z_acc.sum()))
        )

    if not shape_variations:
        return hists, thr
    cov = result.get("covariance")
    if not cov:
        print("[warn] %s: no covariance - skip shape variations" % key)
        return hists, thr

    skip = set()
    if not vary_yield:
        skip.add(0)
        for i, n in enumerate(result.get("param_names") or []):
            if n == "N":
                skip.add(i)

    modes = core.eigen_shape_shifts(cov, skip_indices=skip, max_modes=max_modes)
    if not modes:
        print("[warn] %s: no positive eigenmodes" % key)
        return hists, thr

    for rank, sigma, u in modes:
        delta = sigma * u
        p_up = [float(params[i] + delta[i]) for i in range(len(params))]
        p_dn = [float(params[i] - delta[i]) for i in range(len(params))]
        try:
            Zu = core.integrate_model_bins(
                result, x_edges, y_edges, p_up, n_quad=n_quad
            )
            Zd = core.integrate_model_bins(
                result, x_edges, y_edges, p_dn, n_quad=n_quad
            )
        except Exception as exc:
            print("[warn] %s mode%d: %s" % (key, rank, exc))
            continue
        hists["%s_eig%dUp" % (hist_name, rank)] = fill_th2(
            "%s_eig%dUp" % (hist_name, rank),
            "%s eig%d Up;DNN;HME [GeV]" % (key, rank),
            x_edges,
            y_edges,
            Zu,
        )
        hists["%s_eig%dDown" % (hist_name, rank)] = fill_th2(
            "%s_eig%dDown" % (hist_name, rank),
            "%s eig%d Down;DNN;HME [GeV]" % (key, rank),
            x_edges,
            y_edges,
            Zd,
        )
    return hists, thr


def load_fit_jsons(fit, fit_dir):
    paths = []
    if fit:
        paths.append(fit)
    if fit_dir:
        for name in sorted(os.listdir(fit_dir)):
            if name.endswith(".json") and name != "index.json":
                paths.append(os.path.join(fit_dir, name))
    out = []
    for p in paths:
        with open(p) as fh:
            r = json.load(fh)
        if "parameters" not in r:
            print("[skip] %s: no parameters" % p)
            continue
        if not r.get("converged", True):
            print("[skip] %s: not converged" % p)
            continue
        if "stat" not in r:
            # A JSON written by the August-2026 package: the only configuration
            # it could produce is the Neyman chi2 with the window-GL normalisation.
            r["caveats"] = list(r.get("caveats") or []) + [
                "legacy fit JSON (no stat/norm fields): Neyman chi2 with sigma = "
                "observed cell error, blind to the yield on single-MC-event cells, "
                "and window_gl normalisation (template yield depends on --n-quad)"
            ]
        pv = r.get("p_value")
        if pv is not None and pv < 0.05:
            print("[warn] %s: p_value = %.3g < 0.05 (fit did not pass)" % (p, pv))
        if r.get("clip_hit"):
            print(
                "[warn] %s: clip_hit (the +-50 exponent clip was reached at a "
                "quadrature node; the density is not reliable off the fine grid)" % p
            )
        for c in r.get("caveats") or []:
            print("[caveat] %s: %s" % (p, c))
        out.append((p, r))
    return out


def resolve_binning(args):
    """Return (x_edges, y_edges, binning_meta)."""
    has_from = bool(args.from_hist)
    has_explicit = any(
        v is not None for v in (args.x_range, args.x_bins, args.y_range, args.y_bins)
    )
    if has_from and has_explicit:
        sys.exit(
            "[ERROR] use either --from-hist or "
            "--x-range/--x-bins/--y-range/--y-bins, not both"
        )
    if has_from:
        return load_binning_from_hist(args.from_hist)
    if (
        args.x_range is None
        or args.x_bins is None
        or args.y_range is None
        or args.y_bins is None
    ):
        sys.exit(
            "[ERROR] provide --from-hist FILE[:HIST] or all of "
            "--x-range --x-bins --y-range --y-bins"
        )
    x_lo, x_hi = args.x_range
    y_lo, y_hi = args.y_range
    x_e = bin_edges_uniform(x_lo, x_hi, args.x_bins)
    y_e = bin_edges_uniform(y_lo, y_hi, args.y_bins)
    meta = {
        "from_hist_file": None,
        "from_hist_name": None,
        "x_bins": int(args.x_bins),
        "y_bins": int(args.y_bins),
        "x_range": [float(x_lo), float(x_hi)],
        "y_range": [float(y_lo), float(y_hi)],
    }
    return x_e, y_e, meta


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--fit", default=None, help="Single fit-result JSON")
    g.add_argument("--fit-dir", default=None, help="Directory of fit JSONs")
    p.add_argument(
        "--from-hist",
        default=None,
        metavar="FILE[:HIST]",
        help="Copy x/y bin edges from this TH2. "
        "Format: path/to/file.root or path/to/file.root:plots_2d/DY_2d "
        "(default hist path: %s)" % _DEFAULT_FROM_HIST,
    )
    p.add_argument(
        "--x-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="DNN range (with --x-bins; not with --from-hist)",
    )
    p.add_argument(
        "--x-bins",
        type=int,
        default=None,
        help="Number of DNN bins (with --x-range; not with --from-hist)",
    )
    p.add_argument(
        "--y-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="HME range (with --y-bins; not with --from-hist)",
    )
    p.add_argument(
        "--y-bins",
        type=int,
        default=None,
        help="Number of HME bins (with --y-range; not with --from-hist)",
    )
    p.add_argument("-o", "--output", required=True, help="Output ROOT file")
    p.add_argument("--shape-variations", action="store_true")
    p.add_argument("--vary-yield", action="store_true", help="Include N in eigenmodes")
    p.add_argument("--max-modes", type=int, default=0)
    p.add_argument("--hist-name", default="nominal")
    p.add_argument("--flat", action="store_true", help="No per-fit TDirectory")
    p.add_argument(
        "--n-quad",
        type=int,
        default=8,
        help="Gauss-Legendre order per axis for bin integration (default 8)",
    )
    p.add_argument(
        "--no-accepted-unc",
        action="store_true",
        help="Do not write acceptedUnc / Up / Down histograms from fit thr_min",
    )
    p.add_argument(
        "--thr",
        type=float,
        default=None,
        help=(
            "Override thr for acceptedUnc (default: thr_min from fit JSON). "
            "Absolute bin unc = thr · mu."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    fits = load_fit_jsons(args.fit, args.fit_dir)
    if not fits:
        sys.exit("[ERROR] no usable fit JSONs")

    x_e, y_e, bin_meta = resolve_binning(args)
    parent = os.path.dirname(os.path.abspath(args.output))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fout = ROOT.TFile.Open(args.output, "RECREATE")
    if not fout or fout.IsZombie():
        sys.exit("[ERROR] cannot create %s" % args.output)

    for path, result in fits:
        key = result.get("key") or result.get("function_name") or os.path.basename(path)
        safe = safe_filename(key)
        mx = result.get("mx")
        dir_name = ("m%d_%s" % (mx, safe)) if mx is not None else safe
        print("[hist] %s -> %s" % (key, dir_name))
        # Optional thr override for template building only
        result_use = dict(result)
        if args.thr is not None:
            result_use["thr_min"] = float(args.thr)
            result_use["acceptable_unc_thr"] = float(args.thr)
        try:
            hists, thr = process_one_fit(
                result_use,
                x_edges=x_e,
                y_edges=y_e,
                shape_variations=args.shape_variations,
                vary_yield=args.vary_yield,
                max_modes=args.max_modes,
                hist_name=args.hist_name,
                n_quad=args.n_quad,
                accepted_unc=not args.no_accepted_unc,
            )
        except Exception as exc:
            print("[ERROR] %s: %s" % (path, exc))
            continue

        if args.flat:
            for name, h in hists.items():
                h.SetName("%s_%s" % (dir_name, name))
                fout.cd()
                h.Write()
        else:
            d = (
                fout.mkdir(dir_name)
                if not fout.GetDirectory(dir_name)
                else fout.GetDirectory(dir_name)
            )
            d.cd()
            for h in hists.values():
                h.Write()
            func_spec = result.get("function") or {}
            meta = ROOT.TObjString(
                json.dumps(
                    {
                        "key": key,
                        "source_json": path,
                        "x_range": bin_meta["x_range"],
                        "x_bins": bin_meta["x_bins"],
                        "y_range": bin_meta["y_range"],
                        "y_bins": bin_meta["y_bins"],
                        "from_hist_file": bin_meta.get("from_hist_file"),
                        "from_hist_name": bin_meta.get("from_hist_name"),
                        "shape_variations": args.shape_variations,
                        "n_quad": args.n_quad,
                        # Fit provenance (v2 fits: norm "tiling", stat, y_eps)
                        "norm": result.get("norm") or func_spec.get("norm"),
                        "n_quad_fine": func_spec.get("n_quad_fine"),
                        "y_eps": func_spec.get("y_eps", result.get("y_eps")),
                        "stat": result.get("stat"),
                        "fit_mode": result.get("fit_mode"),
                        "preset": result.get("preset"),
                        "gof_stat": (result.get("gof") or {}).get("stat"),
                        "p_value": result.get("p_value"),
                        "objective": result.get("objective"),
                        "clip_hit": result.get("clip_hit"),
                        "caveats": result.get("caveats"),
                        "n_hists": len(hists),
                        "thr_min": thr,
                        "acceptable_unc_thr": thr,
                        "accepted_unc_hists": (
                            None
                            if args.no_accepted_unc
                            else [
                                "%s_acceptedUnc" % args.hist_name,
                                "%s_acceptedUncUp" % args.hist_name,
                                "%s_acceptedUncDown" % args.hist_name,
                            ]
                        ),
                        "accepted_unc_definition": "delta_i = thr * mu_i",
                    }
                )
            )
            meta.Write("meta_json")
        print("         wrote %d hist(s)  thr=%.4g" % (len(hists), thr))

    fout.Write()
    fout.Close()
    print("[hist] done -> %s" % args.output)


if __name__ == "__main__":
    main()

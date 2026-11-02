"""Equal-signal binning of the 2D DNN x HME fit inputs inside fixed HME windows.

The `preprocess:` step of a datacard configuration (StatInference's PreprocessShapesTask calls
it with --input, --output, --era and --config). For every mass point and datacard category it
cuts the category's HME window out of the 2D merged histograms, merges the merged-histogram
processes into the card's backgrounds, and bins the DNN score inside the window into bins of
equal signal, from the top of the score down. A bin is only closed once every background in it
passes the gates of the binning configuration, for the nominal and for every variation.

Input:  <input>/<era>/<variable>/<variable>.root for every era of the card (the merged
        histograms), with <channel>/<input category>/<process>[_<uncertainty>_Up|Down] 2D
        histograms, DNN score on x and HME mass on y.
Output: the same layout under <output>, 1D in the DNN score, one file per era of the card, and
        <output>/binning.json with the bins of every window.

What is merged, dropped and estimated, the windows, the gates and the target number of bins per
window are all in the binning configuration (--binning-config); see
config/Datacards/binning_eqsig_DL_v2610.yaml for what each entry means.
"""

import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from string import Template

import numpy as np
import uproot
import yaml
import ROOT

from DYTemplate import fit_template, lowmass_boost_shape, tail_template

ROOT.gROOT.SetBatch(True)
# config/... paths are relative to the analysis area this script belongs to
ANALYSIS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the datacard configuration is read as StatInference reads it (extends, variables)
sys.path.insert(0, ANALYSIS_PATH)
from StatInference.common.card_config import load_card_config  # noqa: E402


def analysis_file(path):
    return path if os.path.isabs(path) else os.path.join(ANALYSIS_PATH, path)


def signed(value):
    """[down, up] relative shifts of a member uncertainty; a single number v is [-v, v]."""
    if isinstance(value, (list, tuple)):
        return float(value[0]), float(value[1])
    return -float(value), float(value)


def cumulative(arr):
    return np.concatenate([[0.0], np.cumsum(arr)])


def in_range(c, i, j):
    """Sum of fine bins i..j from a cumulative array."""
    return c[j + 1] - c[i]


def window_mask(y_edges, low, high):
    """Flow-inclusive mask of the y (HME) bins inside [low, high]; a cut on the first (last)
    edge takes the underflow (overflow) too."""
    mask = np.zeros(len(y_edges) + 1, bool)
    mask[1:-1] = (y_edges[:-1] >= low - 1e-6) & (y_edges[1:] <= high + 1e-6)
    mask[0] = low <= y_edges[0] + 1e-6
    mask[-1] = high >= y_edges[-1] - 1e-6
    return mask


def project(hist, mask):
    """(values, variances) of a flow-inclusive 2D histogram summed over the masked y bins, the
    x flows folded into the first and last bin."""
    out = []
    for arr in (hist.values(flow=True), hist.variances(flow=True)):
        p = arr[:, mask].sum(axis=1)
        q = p[1:-1].copy()
        q[0] += p[0]
        q[-1] += p[-1]
        out.append(q)
    return tuple(out)


def negligible(y):
    """Variance of a templated bin: a relative MC-stat. error of 1e-3, since the template's own
    uncertainty (a shape nuisance) replaces the MC statistics."""
    return (1e-3 * np.asarray(y)) ** 2


def variation_suffixes(names, process):
    """'_<uncertainty>_Up|Down' of every variation of `process` among `names`."""
    return {
        n[len(process) :]
        for n in names
        if n.startswith(process + "_") and n.endswith(("_Up", "_Down"))
    }


class Card:
    """What the binning needs from the datacard configuration."""

    def __init__(self, path, era):
        cfg = load_card_config(path)
        self.eras = cfg.get("era_groups", {}).get(era, [era])
        self.channels = cfg["channels"]
        mapping = cfg.get("input_categories") or {}
        # card category -> its directory in the input (and output) files
        self.categories = {c: mapping.get(c, c) for c in cfg["categories"]}
        model = cfg["model"]
        self.param = model["parameters"][0]
        self.pattern = model["input_file_pattern"]
        self.signals, self.backgrounds, self.data, masses = [], [], [], set()
        for p in cfg["processes"]:
            if isinstance(p, str):
                self.backgrounds.append(p)
                continue
            masses.update(p.get("param_values") or [])
            name = p.get("hist_name", p["process"])
            if p.get("is_signal"):
                self.signals.append(name)
            elif p.get("is_data"):
                if not p.get("is_asimov_data"):
                    self.data.append(name)
            else:
                self.backgrounds.append(name)
        self.masses = sorted(masses)

    def input_file(self, era, mass):
        return Template(self.pattern).substitute(ERA=era, **{self.param: mass})

    def resolved(self, names, mass):
        return [Template(n).safe_substitute(**{self.param: mass}) for n in names]


class Binning:
    def __init__(self, card, era, cfg, input_dir, output_dir):
        self.card, self.era, self.input_dir, self.output_dir = (
            card,
            era,
            input_dir,
            output_dir,
        )
        self.cfg = cfg
        self.windows = yaml.safe_load(open(analysis_file(cfg["windows"])))["windows"]
        self.n_bins = cfg["n_bins"]
        self.gates = cfg["gates"]
        self.members = {
            b: cfg.get("backgrounds", {}).get(b, [b]) for b in card.backgrounds
        }
        self.drop = cfg.get("drop") or {}
        self.single_bin_regions = set(cfg.get("single_bin_regions") or [])
        self.lowmass = cfg.get("lowmass_dy")
        if self.lowmass is not None:
            # the process that carries the estimate (DY when the low-mass sample is a member of it)
            self.lowmass.setdefault("output", "DY")
            self.lowmass.setdefault("nuisance_factors", [0.0, 2.0])
        self.dy_template = cfg.get("dy_template")
        self.pooled = (
            {}
        )  # (mass, directory, channels) -> fitted DY shape of fit-only windows
        self.member_uncs = {
            name: [(t["member"], signed(t["value"])) for t in terms]
            for name, terms in (cfg.get("member_uncertainties") or {}).items()
        }
        self.record = {}

    @staticmethod
    def window_key(directory):
        """The windows file keys a category by its base name: SR/res2b -> res2b, and so
        does every region of that category (TT_CR/res2b -> res2b)."""
        return directory.rsplit("/", 1)[-1]

    @staticmethod
    def region(directory):
        """SR/res2b -> SR, TT_CR/res2b -> TT_CR."""
        return directory.rsplit("/", 1)[0] if "/" in directory else ""

    def lowmass_applies(self, directory):
        lm = self.lowmass
        return (
            lm is not None
            and self.window_key(directory) in lm["categories"]
            and ("regions" not in lm or self.region(directory) in lm["regions"])
        )

    def template_mode(self, channel, directory):
        """'tail', 'fit_only' or None: how dy_template replaces the DY MC of a category."""
        t = self.dy_template
        if t is None:
            return None
        key, region = self.window_key(directory), self.region(directory)
        for mode in ("tail", "fit_only"):
            m = t.get(mode)
            if (
                m
                and key in m["categories"]
                and region in m.get("regions", [region])
                and channel in m.get("channels", self.card.channels)
            ):
                return mode
        return None

    def pooled_shape(self, mass, channel, directory, files, window, x):
        """(channels fitted, (shape, up, down)) of the fit-only DY template, each normalised to
        the nominal's sum: one fit to the DY MC summed over the eras and over the channels pooled
        with `channel`."""
        t = self.dy_template
        group = next(
            (tuple(g) for g in t["fit_only"].get("pool", []) if channel in g),
            (channel,),
        )
        key = (mass, directory, group)
        if key not in self.pooled:
            proc, lowmass = t["process"], self.lowmass_applies(directory)
            y, w = np.zeros(len(x)), np.zeros(len(x))
            for ch in group:
                raw, _ = self.read(
                    files, ch, directory, window, only=set(self.members[proc])
                )
                for r in raw.values():
                    p = self.build_processes(r, directory, lowmass, mass).get(proc)
                    if p:
                        y, w = y + p[""][0], w + p[""][1]
            tmpl, (up, down), ok = fit_template(
                x, y, np.maximum(w, 0.0), **t.get("fit", {})
            )
            if not ok:
                raise RuntimeError(
                    f"dy_template: the {proc} fit of {'+'.join(group)} {directory} "
                    f"MX={mass} failed"
                )
            norm = tmpl.sum()
            self.pooled[key] = (tmpl / norm, up / norm, down / norm)
        return group, self.pooled[key]

    def apply_template(
        self, mode, per_era, raw, mass, channel, directory, files, window, x
    ):
        """Replace the DY MC of every era by its smooth template (in place); returns the
        fit-only shape (None for 'tail').

        tail: each MC member summed over the eras, raw below its fitted peak and fitted above
          (DYTemplate.tail_template); the members' templates add up to DY. The raw part keeps
          its MC statistical error.
        fit_only: one pooled fit (pooled_shape) times the window's DY MC; the band adds that
          yield's own MC statistical error (1 / sqrt(N_eff) of this channel's DY).
        The total is shared out to the eras by their share_by window yield. Each DY variation of
        an era is the template times that era's (varied / nominal) window yield, except a member
        uncertainty, which moves its own member's template (tail). The template's band is the
        shape nuisance <nuisance>_<channel>_<category>; the fitted part's MC statistical error
        is negligible (the band replaces it)."""
        t = self.dy_template
        proc = t["process"]
        eras = [e for e, p in per_era.items() if proc in p]
        if not eras:
            return None
        zero = np.zeros(len(x))
        fit = t.get("fit", {})
        member_tmpl, shape = {}, None
        raw_var = (
            zero  # MC variance of the bins kept raw (tail mode, below the tail start)
        )
        if mode == "tail":
            members = [
                m
                for m in self.members[proc]
                if m not in self.dropped_in(directory)
                and any(m in raw[e] for e in eras)
                and not (
                    self.lowmass_applies(directory) and m == self.lowmass["process"]
                )
            ]
            total, up, down = zero, zero, zero
            for m in members:
                y = sum((raw[e][m][0] for e in eras if m in raw[e]), zero)
                w = sum((raw[e][m][1] for e in eras if m in raw[e]), zero)
                w = np.maximum(w, 0.0)
                tmpl, (u, d), tail, ok = tail_template(x, y, w, **fit)
                if not ok:
                    raise RuntimeError(
                        f"dy_template: the {m} fit of {channel}/{directory} MX={mass} "
                        "failed"
                    )
                member_tmpl[m] = tmpl
                total, up, down = total + tmpl, up + u, down + d
                raw_var = raw_var + np.where(tail, 0.0, w)
        else:
            _, (shape, s_up, s_down) = self.pooled_shape(
                mass, channel, directory, files, window, x
            )
            n = sum(float(per_era[e][proc][""][0].sum()) for e in eras)
            w_n = sum(float(per_era[e][proc][""][1].sum()) for e in eras)
            # this channel's yield: its own MC statistical error, on top of the pooled band
            norm = 1.0 / np.sqrt(max(n * n / w_n, 1.0)) if n > 0 and w_n > 0 else 0.0
            total = shape * n
            rel_up = np.where(shape > 0, s_up / np.maximum(shape, 1e-300) - 1.0, 0.0)
            rel_dn = np.where(shape > 0, 1.0 - s_down / np.maximum(shape, 1e-300), 0.0)
            up = total * (1.0 + np.minimum(np.hypot(rel_up, norm), 1.0))
            down = total * (1.0 - np.minimum(np.hypot(rel_dn, norm), 1.0))

        by = {
            e: max(float(raw[e].get(t["share_by"], (zero,))[0].sum()), 0.0)
            for e in eras
        }
        by_total = sum(by.values())
        member_shift = {}  # suffix -> template change of a member uncertainty
        for unc, terms in self.member_uncs.items():
            if any(m in member_tmpl for m, _ in terms):
                for k, direction in ((0, "Down"), (1, "Up")):
                    member_shift[f"_{unc}_{direction}"] = sum(
                        (v[k] * member_tmpl[m] for m, v in terms if m in member_tmpl),
                        zero,
                    )
        nuisance = f"_{t['nuisance']}_{channel}_{self.window_key(directory)}"
        for e in eras:
            share = by[e] / by_total if by_total > 0 else 1.0 / len(eras)
            shapes = per_era[e][proc]
            nominal = float(shapes[""][0].sum())
            new = {}
            for s, (y, _) in shapes.items():
                r = 1.0
                if s in member_shift:
                    v = share * (total + member_shift[s])
                else:
                    r = (
                        max(float(y.sum()) / nominal, 0.0)
                        if (s and nominal > 0)
                        else 1.0
                    )
                    v = share * total * r
                new[s] = (v, negligible(v) + share * raw_var * r * r)
            for direction, v in (("Up", up), ("Down", down)):
                new[f"{nuisance}_{direction}"] = (
                    share * v,
                    negligible(share * v) + share * raw_var,
                )
            per_era[e][proc] = new
        return shape

    def read(self, files, channel, directory, window, only=None):
        """{era: {merged-histogram name: (values, variances)}} of one category, projected on
        `window`, plus the DNN-score bin edges; `only`: just these names (nominals)."""
        out, x_edges = {}, None
        for era, f in files.items():
            d = f[f"{channel}/{directory}"]
            names = d.keys(recursive=False, cycle=False)
            if only is not None:
                names = [n for n in names if n in only]
            if not names:
                out[era] = {}
                continue
            first = d[names[0]]
            x_edges = first.axes[0].edges()
            mask = window_mask(first.axes[1].edges(), *window)
            out[era] = {n: project(d[n], mask) for n in names}
        return out, x_edges

    def dropped_in(self, directory):
        """Inputs left out of a category: `drop` lists, per input, jet categories (res2b), regions
        (TT_CR) or region/category directories (TT_CR/res2b)."""
        keys = {self.window_key(directory), self.region(directory), directory}
        return {m for m, cats in self.drop.items() if keys & set(cats)}

    def build_processes(self, raw, directory, lowmass, mass):
        """{output process: {suffix: (values, variances)}} of one era, suffix '' = nominal.

        With `lowmass` the low-mass member's MC is left out of its output process here (which
        keeps that member's variations, so it exists even if the member was its only one); the
        estimate is added per bin range from the transfer (see bin_window)."""
        dropped = self.dropped_in(directory)
        out = {}
        for proc, members in self.members.items():
            members = [m for m in members if m in raw and m not in dropped]
            if not members:
                continue
            mc_members = members
            if lowmass and proc == self.lowmass["output"]:
                mc_members = [m for m in members if m != self.lowmass["process"]]
            suffixes = set().union(*(variation_suffixes(raw, m) for m in members))
            zero = np.zeros_like(raw[members[0]][0])
            shapes = {}
            for s in [""] + sorted(suffixes):
                parts = [raw.get(m + s, raw[m]) for m in mc_members]
                shapes[s] = (
                    sum((p[0] for p in parts), zero),
                    sum((p[1] for p in parts), zero),
                )
            for unc, terms in self.member_uncs.items():
                if any(m in members for m, _ in terms) and not any(
                    m in mc_members for m, _ in terms
                ):
                    # a member uncertainty on the estimate alone: it enters through its factor
                    for direction in ("Down", "Up"):
                        shapes[f"_{unc}_{direction}"] = shapes[""]
                    continue
                terms = [(m, v) for m, v in terms if m in mc_members]
                if not terms:
                    continue
                for k, direction in ((0, "Down"), (1, "Up")):
                    y = shapes[""][0] + sum(v[k] * raw[m][0] for m, v in terms)
                    shapes[f"_{unc}_{direction}"] = (y, shapes[""][1])
            out[proc] = shapes
        for name in self.card.resolved(self.card.signals, mass) + self.card.data:
            if name in raw:
                out[name] = {
                    s: raw[name + s]
                    for s in [""] + sorted(variation_suffixes(raw, name))
                }
        return out

    def lowmass_factors(self, suffixes):
        """How the low-mass estimate enters each DY variation: at its nominal, scaled by a
        member uncertainty on the low-mass process, or 2x / 0x for its own nuisance."""
        factors = {}
        lm = self.lowmass
        for s in suffixes:
            f = 1.0
            for unc, terms in self.member_uncs.items():
                for k, direction in ((0, "Down"), (1, "Up")):
                    if s == f"_{unc}_{direction}":
                        f += sum(v[k] for m, v in terms if m == lm["process"])
            factors[s] = f
        down, up = lm["nuisance_factors"]
        factors[f"_{lm['nuisance']}_Up"] = float(up)
        factors[f"_{lm['nuisance']}_Down"] = float(down)
        return factors

    def bin_window(self, mass, channel, category, directory, files):
        key = self.window_key(directory)
        window = self.windows[mass][key]
        raw, x_edges = self.read(files, channel, directory, window)
        nb = len(x_edges) - 1
        lowmass = self.lowmass_applies(directory)
        per_era = {
            e: self.build_processes(r, directory, lowmass, mass) for e, r in raw.items()
        }
        x = 0.5 * (x_edges[1:] + x_edges[:-1])
        mode = self.template_mode(channel, directory)
        template_shape = None
        if mode:
            template_shape = self.apply_template(
                mode, per_era, raw, mass, channel, directory, files, window, x
            )
        bkgs = [b for b in self.members if any(b in p for p in per_era.values())]
        signals = self.card.resolved(self.card.signals, mass)

        def summed(proc, s):
            parts = [p[proc].get(s, p[proc][""]) for p in per_era.values() if proc in p]
            return sum(p[0] for p in parts), sum(p[1] for p in parts)

        suffixes = {
            b: sorted(
                set().union(*(set(p[b]) for p in per_era.values() if b in p)) - {""}
            )
            for b in bkgs
        }
        L, factors = None, {}
        lm_out = self.lowmass["output"] if self.lowmass else None
        if lowmass and lm_out in bkgs:
            lm = self.lowmass
            # the reference category of the same region, projected on THIS category's window
            ref_dir = f"{self.region(directory)}/{lm['reference']}"
            if ref_dir not in self.card.categories.values():
                raise RuntimeError(
                    f"lowmass_dy: reference {ref_dir} of {directory} is not a datacard "
                    "category input"
                )
            ref, _ = self.read(
                files, channel, ref_dir, window, only={lm["process"], lm["transfer"]}
            )
            zero = np.zeros(nb)
            low = [
                sum(r.get(lm["process"], (zero, zero))[k] for r in ref.values())
                for k in (0, 1)
            ]
            transfer = [
                sum(r.get(lm["transfer"], (zero, zero))[k] for r in ref.values())
                for k in (0, 1)
            ]
            c_low, c_low_w = cumulative(low[0]), cumulative(np.maximum(low[1], 0.0))
            c_tr = cumulative(transfer[0])
            # the transfer sample in this category itself
            m50_here = {
                e: r.get(lm["transfer"], (zero, zero))[0] for e, r in raw.items()
            }
            if mode == "fit_only":
                # its template: the output process's MC here is the transfer sample
                m50_here = {
                    e: p[lm_out][""][0] if lm_out in p else zero
                    for e, p in per_era.items()
                }
            c_m50 = cumulative(sum(m50_here.values()))

            def estimate(i, j):
                """(low-mass DY, its variance) over fine bins i..j"""
                t = in_range(c_tr, i, j)
                if j < i or t <= 0:
                    return 0.0, 0.0
                r = in_range(c_m50, i, j) / t
                return in_range(c_low, i, j) * r, in_range(c_low_w, i, j) * r * r

            L = estimate
            if template_shape is not None:
                # the window's double ratio with a smooth DNN shape: fit(low-mass, reference) x
                # the template here / fit(transfer, reference)
                shape = lowmass_boost_shape(
                    x,
                    (low[0], np.maximum(low[1], 0.0)),
                    (transfer[0], np.maximum(transfer[1], 0.0)),
                    template_shape,
                    **self.dy_template.get("fit", {}),
                )
                c_fine = cumulative(shape * estimate(0, nb - 1)[0])

                def smooth(i, j):
                    v = in_range(c_fine, i, j) if j >= i else 0.0
                    return v, float(negligible(v))

                L = smooth
            factors = self.lowmass_factors(suffixes[lm_out])
            suffixes[lm_out] = sorted(set(suffixes[lm_out]) | set(factors))

        series = {
            b: {
                s: tuple(cumulative(a) for a in summed(b, s))
                for s in [""] + suffixes[b]
            }
            for b in bkgs
        }
        signal = np.zeros(nb)
        for s in signals:
            if any(s in p for p in per_era.values()):
                signal = signal + summed(s, "")[0]
        c_sig = cumulative(signal)

        def get(b, s, i, j):
            c = series[b][s if s in series[b] else ""]
            y, w = in_range(c[0], i, j), in_range(c[1], i, j)
            if L is not None and b == lm_out:
                lv, lw = L(i, j)
                f = factors.get(s, 1.0)
                # the estimate scaled by f: its variance by f^2 (none for f = 0)
                y, w = y + f * lv, w + f * f * lw
            return y, w

        g = self.gates
        totals = {b: get(b, "", 0, nb - 1)[0] for b in bkgs}
        exempt = {
            b for b in bkgs if totals[b] < g["exempt_fraction"] * sum(totals.values())
        }

        def failure(i, j):
            """Why fine bins i..j fail the gates, or None if they pass."""
            nominal = {b: get(b, "", i, j) for b in bkgs}
            bt = sum(v[0] for v in nominal.values())
            bw = sum(v[1] for v in nominal.values())
            if bt < g["min_bkg"] or bw <= 0 or bt * bt / bw < g["min_bkg_neff"]:
                return f"total background {bt:.3g}, N_eff {bt * bt / bw if bw > 0 else 0:.2f}"
            for b in bkgs:
                y, w = nominal[b]
                if y < -1e-12:
                    return f"{b} {y:.3g} < 0"
                strict = b not in exempt
                if strict and (
                    y <= g["min_each"] or w <= 0 or y * y / w < g["min_each_neff"]
                ):
                    return f"{b} {y:.3g}, N_eff {y * y / w if w > 0 else 0:.2f}"
                for s in suffixes[b]:
                    yv, wv = get(b, s, i, j)
                    if yv < -1e-12:
                        return f"{b}{s} {yv:.3g} < 0"
                    if strict and (wv <= 0 or yv * yv / wv < g["min_variation_neff"]):
                        return f"{b}{s} N_eff {yv * yv / wv if wv > 0 else 0:.2f}"
            return None

        def passes(i, j):
            return failure(i, j) is None

        # control regions: one bin over the window (they constrain normalisations, not shapes)
        n = (
            1
            if self.region(directory) in self.single_bin_regions
            else self.n_bins[mass][key]
        )
        bins, hi = [], nb - 1
        # equal signal from the top of the score down, each bin as narrow as the gates allow
        while hi >= 0 and len(bins) < n - 1:
            target = in_range(c_sig, 0, hi) / (n - len(bins))
            found = next(
                (
                    i
                    for i in range(hi, -1, -1)
                    if in_range(c_sig, i, hi) >= target and passes(i, hi)
                ),
                None,
            )
            if found is None or found == 0:
                break
            bins.append((found, hi))
            hi = found - 1
        if hi >= 0:
            # the rest of the window is the lowest bin; merge upwards until it passes
            last = (0, hi)
            while not passes(*last):
                if not bins:
                    why = failure(0, nb - 1)
                    if "< 0" in why:
                        # a negative background cannot be written, at any binning
                        raise RuntimeError(
                            f"{channel}/{directory} MX={mass}: the whole window fails "
                            f"the gates: {why}"
                        )
                    # too few events for any finer binning: the window is one bin
                    print(
                        f"WARNING: {channel}/{directory} MX={mass}: the whole window "
                        f"fails the gates ({why}); it is one bin",
                        flush=True,
                    )
                    break
                last = (0, bins.pop()[1])
            bins.append(last)
        bins.sort()
        edges = [float(x_edges[i]) for i, _ in bins] + [float(x_edges[-1])]

        self.record.setdefault(str(mass), {}).setdefault(channel, {})[category] = {
            "slices": [
                {"y_range": [float(window[0]), float(window[1])], "edges": edges}
            ],
            "n_bins_target": n,
            "n_bins": len(bins),
            "signal": [float(in_range(c_sig, i, j)) for i, j in bins],
            "background": [
                float(sum(get(b, "", i, j)[0] for b in bkgs)) for i, j in bins
            ],
        }

        # the low-mass estimate is shared out to the eras by their M>50 yield in each bin
        # (a negative era sum counts as none, so every share is in [0, 1])
        lowmass_by_era = {}
        if L is not None:
            m50 = m50_here
            for e in m50:
                vals = []
                for i, j in bins:
                    part = {x_: max(float(m50[x_][i : j + 1].sum()), 0.0) for x_ in m50}
                    tot = sum(part.values())
                    share = part[e] / tot if tot > 0 else 1.0 / len(m50)
                    lv, lw = L(i, j)
                    vals.append((lv * share, lw * share))
                lowmass_by_era[e] = vals
        return per_era, bins, edges, factors, lowmass_by_era

    def run_mass(self, mass):
        """Bin and write every window of one mass; returns its binning.json record."""
        card = self.card
        files = {
            e: uproot.open(os.path.join(self.input_dir, card.input_file(e, mass)))
            for e in card.eras
        }
        out = {}
        for e in card.eras:
            path = os.path.join(self.output_dir, card.input_file(e, mass))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            out[e] = ROOT.TFile(path, "RECREATE")
        reached = {}
        for channel in card.channels:
            for category, directory in card.categories.items():
                per_era, bins, edges, factors, lowmass_by_era = self.bin_window(
                    mass, channel, category, directory, files
                )
                reached[f"{channel}/{category}"] = len(bins)
                for e, procs in per_era.items():
                    self.write(
                        out[e].mkdir(f"{channel}/{directory}", "", True),
                        procs,
                        bins,
                        edges,
                        factors,
                        lowmass_by_era.get(e),
                        self.lowmass["output"] if self.lowmass else None,
                    )
        for f in out.values():
            f.Close()
        self.pooled.clear()
        print(f"MX={mass}: bins per window {json.dumps(reached)}", flush=True)
        return self.record.get(str(mass), {})

    def run(self, jobs=1):
        """Every mass of the card, `jobs` at a time (each mass in its own process)."""
        masses = self.card.masses
        if jobs > 1 and len(masses) > 1:
            args = [
                (self.card, self.era, self.cfg, self.input_dir, self.output_dir, m)
                for m in masses
            ]
            # spawned, not forked: ROOT's state does not survive a fork
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(min(jobs, len(masses)), mp_context=ctx) as pool:
                for mass, record in zip(masses, pool.map(_run_mass, args)):
                    self.record[str(mass)] = record
        else:
            for mass in masses:
                self.run_mass(mass)
        with open(os.path.join(self.output_dir, "binning.json"), "w") as f:
            json.dump({"binning": {self.era: self.record}}, f, indent=1)

    @staticmethod
    def write(directory, procs, bins, edges, factors, lowmass, lowmass_process):
        """One era's histograms of a category: the variations that era's input has, and
        for the process carrying the low-mass estimate also its own variations (every era).
        """
        directory.cd()
        edges = np.array(edges, dtype=float)

        def hist(name, y, w):
            h = ROOT.TH1D(name, name, len(edges) - 1, edges)
            h.Sumw2()
            for k, (i, j) in enumerate(bins, start=1):
                h.SetBinContent(k, float(y[i : j + 1].sum()))
                h.SetBinError(k, float(np.sqrt(max(w[i : j + 1].sum(), 0.0))))
            return h

        for proc, shapes in procs.items():
            names = set(shapes) - {""}
            carries_estimate = proc == lowmass_process and lowmass is not None
            if carries_estimate:
                # the variations that move the estimate exist in every era
                names |= {s for s, f in factors.items() if f != 1.0}
            for s in [""] + sorted(names):
                y, w = shapes.get(s, shapes[""])
                h = hist(proc + s, y, w)
                if carries_estimate:
                    f = factors.get(s, 1.0)
                    for k, (lv, lw) in enumerate(lowmass, start=1):
                        h.SetBinContent(k, h.GetBinContent(k) + f * lv)
                        h.SetBinError(
                            k, float(np.sqrt(h.GetBinError(k) ** 2 + f * f * lw))
                        )
                h.Write()
                h.Delete()


def _run_mass(args):
    card, era, cfg, input_dir, output_dir, mass = args
    return Binning(card, era, cfg, input_dir, output_dir).run_mass(mass)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--input", required=True, help="merged histograms, <era>/<var>/<var>.root"
    )
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--era", required=True, help="the card's era (a group era is summed)"
    )
    ap.add_argument("--config", required=True, help="the datacard configuration")
    ap.add_argument("--binning-config", required=True)
    ap.add_argument(
        "--masses",
        default="",
        help="comma-separated subset of the card's masses (tests)",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="masses processed in parallel, one process each (match the job's CPUs)",
    )
    args = ap.parse_args()

    card = Card(args.config, args.era)
    if args.masses:
        card.masses = [int(m) for m in args.masses.split(",")]
    binning = Binning(
        card,
        args.era,
        yaml.safe_load(open(analysis_file(args.binning_config))),
        args.input,
        args.output,
    )
    os.makedirs(args.output, exist_ok=True)
    binning.run(args.jobs)


if __name__ == "__main__":
    main()

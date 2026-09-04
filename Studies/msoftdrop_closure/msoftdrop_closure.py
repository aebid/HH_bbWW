"""Data/MC closure of the boosted msoftdrop cut.

The boosted Hbb candidate requires ``SelectedFatJet_msoftdrop > 30`` (the
``FatBJet_Sel`` definition in ``Analysis/hh_bbww.py``).  This script measures,
per era / lepton channel / process, the efficiency of that requirement and
compares data with simulation.

Input is a merged histogram file from the HistMerger step:

    <hists>/<era>/<variable>/<variable>.root

with the usual ``<channel>/<region>/<category>/<process>`` layout.

Which variable to run on
------------------------
``fatbjet_msoftdrop`` CANNOT measure this.  It is only filled from a fat jet
that already passes the full ``FatBJet_Sel``
(``particleNetWithMass_HbbvsQCD > 0.92 && msoftdrop > 30``); everything else
gets the default 0 from

    fatbjet_isValid ? FatBJet_msoftdrop[0] : value_type()

so the 0-30 range is empty by construction and the bin at 0 is a sentinel that
also absorbs Hbb-tagger failures.  Dropping the msoftdrop cut from the
*category* does not help, because the cut lives in the *variable*.

Use instead one of the probe variables, which keep the same jet ordering but
drop the mass requirement and use -1 (underflow) as the "no candidate"
sentinel:

* ``fatbjetProbe_msoftdrop`` - Hbb-tagged fat jet, no msoftdrop cut.  This is
  the efficiency in the selection sequence actually used.
* ``leadfatjet_msoftdrop`` - leading-Hbb-score fat jet, no cuts at all.  The
  tagger is mass-aware, so this avoids the correlation the tagged denominator
  carries.

Examples
--------
    # the real measurement, once a production with the probes exists
    python3 Studies/msoftdrop_closure/msoftdrop_closure.py \\
        --input-dir <hists> --variable fatbjetProbe_msoftdrop --sentinel-value -1

    # what the v2605a_msoftdrop production can show (see README)
    python3 Studies/msoftdrop_closure/msoftdrop_closure.py --input-dir <hists>
"""

import argparse
import ctypes
import math
import os
import sys

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

DEFAULT_INPUT = "/eos/user/d/daebi/HH_bbWW/v2605a_msoftdrop/Hists_merged"
DEFAULT_ERAS = ["Run3_2022", "Run3_2022EE", "Run3_2023", "Run3_2023BPix"]
DEFAULT_CHANNELS = ["e", "mu", "eE", "eMu", "muMu"]
DEFAULT_BACKGROUNDS = ["TT", "DY", "ST", "VV", "W", "SingleHiggs"]
DATA_PROCESS = "Data_Full"


class Counts:
    """Yields of one process in one (era, channel) cell.

    ``total`` splits into ``sentinel`` (no fat-jet candidate at all) plus
    ``candidates``; ``candidates`` splits into ``below`` (fails the cut) plus
    ``passing``.  The cut efficiency is passing/candidates, which is what the
    sentinel must be kept out of.
    """

    __slots__ = ("total", "total_e2", "sentinel", "sentinel_e2",
                 "below", "below_e2", "straddle", "present")

    def __init__(self):
        self.total = 0.0
        self.total_e2 = 0.0
        self.sentinel = 0.0
        self.sentinel_e2 = 0.0
        self.below = 0.0
        self.below_e2 = 0.0
        self.straddle = 0.0     # bin containing the threshold, if interior
        self.present = False

    def __iadd__(self, o):
        self.total += o.total
        self.total_e2 += o.total_e2
        self.sentinel += o.sentinel
        self.sentinel_e2 += o.sentinel_e2
        self.below += o.below
        self.below_e2 += o.below_e2
        self.straddle += o.straddle
        self.present = self.present or o.present
        return self

    @property
    def candidates(self):
        return self.total - self.sentinel

    @property
    def candidates_e2(self):
        # sentinel and candidates are disjoint bin ranges, so variances subtract
        return max(self.total_e2 - self.sentinel_e2, 0.0)

    @property
    def passing(self):
        return self.candidates - self.below

    def efficiency(self):
        """passing / candidates, with a binomial error on weighted counts."""
        cand = self.candidates
        if cand <= 0.0:
            return float("nan"), float("nan")
        eff = self.passing / cand
        n_eff = (cand ** 2 / self.candidates_e2) if self.candidates_e2 > 0 else 0.0
        if n_eff <= 0.0:
            return eff, float("nan")
        return eff, math.sqrt(max(eff * (1.0 - eff), 0.0) / n_eff)

    def candidate_fraction(self):
        """candidates / total: how often a fat-jet candidate exists at all."""
        if self.total <= 0.0:
            return float("nan")
        return self.candidates / self.total


def _integral(h, first, last):
    if last < first:
        return 0.0, 0.0
    err = ctypes.c_double(0.0)
    val = h.IntegralAndError(first, last, err)
    return float(val), float(err.value)


def read_counts(h, threshold, sentinel_value):
    c = Counts()
    c.present = True
    ax = h.GetXaxis()
    n = ax.GetNbins()

    c.total, e = _integral(h, 0, n + 1)
    c.total_e2 = e * e

    # FindBin maps a below-range sentinel (e.g. -1) onto the underflow bin 0.
    sbin = ax.FindBin(sentinel_value)
    c.sentinel, e = _integral(h, sbin, sbin)
    c.sentinel_e2 = e * e

    tbin = ax.FindBin(threshold)
    straddles = ax.GetBinLowEdge(tbin) < threshold
    c.below, e = _integral(h, sbin + 1, tbin - 1)
    c.below_e2 = e * e
    if straddles:
        c.straddle = h.GetBinContent(tbin)
    return c


def collect(input_dir, eras, channels, region, category, variable, processes,
            threshold, sentinel_value):
    counts, missing, warned = {}, [], [False]
    for era in eras:
        path = os.path.join(input_dir, era, variable, f"{variable}.root")
        if not os.path.exists(path):
            missing.append(path)
            continue
        f = ROOT.TFile.Open(path)
        if not f or f.IsZombie():
            missing.append(path)
            continue
        counts[era] = {}
        for ch in channels:
            counts[era][ch] = {}
            for proc in processes:
                key = f"{ch}/{region}/{category}/{proc}"
                h = f.Get(key)
                if not h:
                    missing.append(f"{path}:{key}")
                    counts[era][ch][proc] = Counts()
                    continue
                if not warned[0]:
                    ax = h.GetXaxis()
                    tbin = ax.FindBin(threshold)
                    if ax.GetBinLowEdge(tbin) < threshold:
                        print(
                            f"[warn] threshold {threshold:g} falls inside bin {tbin} = "
                            f"[{ax.GetBinLowEdge(tbin):g}, {ax.GetBinUpEdge(tbin):g}); "
                            "that bin is counted as passing and reported separately. "
                            "Rebin so an edge lands on the threshold.",
                            file=sys.stderr)
                    warned[0] = True
                counts[era][ch][proc] = read_counts(h, threshold, sentinel_value)
        f.Close()
    return counts, missing


def sum_over(counts, eras, channels, procs):
    out = Counts()
    for era in eras:
        for ch in channels:
            for p in procs:
                c = counts.get(era, {}).get(ch, {}).get(p)
                if c is not None:
                    out += c
    return out


def format_table(rows, headers):
    w = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            w[i] = max(w[i], len(cell))
    out = ["| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(headers)) + " |",
           "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    for r in rows:
        out.append("| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |")
    return "\n".join(out)


def yield_table(counts, eras, channel, processes, threshold):
    headers = ["Process", "nTotalEvents", "no candidate", "candidates",
               f"msd<{threshold:g}", f"msd>{threshold:g}", "cut eff [%]"]
    rows = []
    mc_procs = [p for p in processes if p != DATA_PROCESS]

    def row(label, c):
        eff, err = c.efficiency()
        if eff != eff:
            eff_s = "n/a"
        elif err == err:
            eff_s = f"{100 * eff:.2f} +- {100 * err:.2f}"
        else:
            eff_s = f"{100 * eff:.2f}"
        return [label, f"{c.total:.1f}", f"{c.sentinel:.1f}", f"{c.candidates:.2f}",
                f"{c.below:.3f}", f"{c.passing:.2f}", eff_s]

    for p in processes:
        c = sum_over(counts, eras, [channel], [p])
        if c.present:
            rows.append(row(p, c))
    mc = sum_over(counts, eras, [channel], mc_procs)
    data = sum_over(counts, eras, [channel], [DATA_PROCESS])
    if mc.present:
        rows.append(row("Total MC", mc))
    if mc.present and data.present:
        e_mc, s_mc = mc.efficiency()
        e_da, s_da = data.efficiency()
        if e_mc == e_mc and e_mc > 0 and e_da == e_da:
            ratio = e_da / e_mc
            if e_da > 0 and s_da == s_da and s_mc == s_mc:
                rerr = ratio * math.sqrt((s_da / e_da) ** 2 + (s_mc / e_mc) ** 2)
                cell = f"{ratio:.4f} +- {rerr:.4f}"
            else:
                cell = f"{ratio:.4f}"
            rows.append(["Data/MC eff", "", "", "", "", "", cell])
    return headers, rows


def efficiency_matrix(counts, eras, channels, processes):
    """Cut efficiency [%] per process (rows) x era (columns), one per channel."""
    tables = []
    mc_procs = [p for p in processes if p != DATA_PROCESS]
    for ch in channels:
        headers = ["Process"] + list(eras) + ["All eras"]
        rows = []
        for p in processes + ["Total MC", "Data/MC"]:
            cells, ok = [p], False
            for era_list in [[e] for e in eras] + [list(eras)]:
                if p == "Data/MC":
                    e_mc, _ = sum_over(counts, era_list, [ch], mc_procs).efficiency()
                    e_da, _ = sum_over(counts, era_list, [ch], [DATA_PROCESS]).efficiency()
                    val = e_da / e_mc if (e_mc == e_mc and e_mc > 0 and e_da == e_da) \
                        else float("nan")
                    cells.append("n/a" if val != val else f"{val:.4f}")
                    ok = ok or val == val
                    continue
                procs = mc_procs if p == "Total MC" else [p]
                eff, _ = sum_over(counts, era_list, [ch], procs).efficiency()
                cells.append("n/a" if eff != eff else f"{100 * eff:.2f}")
                ok = ok or eff == eff
            if ok:
                rows.append(cells)
        tables.append((ch, headers, rows))
    return tables


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=DEFAULT_INPUT)
    ap.add_argument("--eras", nargs="+", default=DEFAULT_ERAS)
    ap.add_argument("--channels", nargs="+", default=DEFAULT_CHANNELS)
    ap.add_argument("--region", default="OS_Iso")
    ap.add_argument("--category", default="baseline_boosted")
    ap.add_argument("--variable", default="fatbjet_msoftdrop",
                    help="use fatbjetProbe_msoftdrop or leadfatjet_msoftdrop for the "
                         "real measurement; see the module docstring")
    ap.add_argument("--processes", nargs="+",
                    default=DEFAULT_BACKGROUNDS + [DATA_PROCESS],
                    help="backgrounds and data; signal is deliberately excluded")
    ap.add_argument("--threshold", type=float, default=30.0)
    ap.add_argument("--sentinel-value", type=float, default=0.0,
                    help="the 'no candidate' default value; 0 for fatbjet_msoftdrop, "
                         "-1 for the probe variables (default: %(default)s)")
    ap.add_argument("--per-era", action="store_true")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    counts, missing = collect(args.input_dir, args.eras, args.channels, args.region,
                              args.category, args.variable, args.processes,
                              args.threshold, args.sentinel_value)
    if not counts:
        print(f"No '{args.variable}' input found under {args.input_dir}", file=sys.stderr)
        for m in missing:
            print("  missing: " + m, file=sys.stderr)
        return 1
    eras = [e for e in args.eras if e in counts]

    print(f"# msoftdrop closure: {args.variable} in "
          f"<channel>/{args.region}/{args.category}, threshold {args.threshold:g}, "
          f"sentinel {args.sentinel_value:g}")
    print(f"# input: {args.input_dir}")
    print(f"# eras: {', '.join(eras)}\n")

    print(f"# ===== Cut efficiency [%] = N(msd>{args.threshold:g}) / "
          "N(candidates) per era =====\n")
    for ch, headers, rows in efficiency_matrix(counts, eras, args.channels,
                                               args.processes):
        print(f"## channel {ch}")
        print(format_table(rows, headers))
        print()

    print("# ===== Yields, all eras summed =====\n")
    for ch in args.channels:
        headers, rows = yield_table(counts, eras, ch, args.processes, args.threshold)
        if rows:
            print(f"## channel {ch}")
            print(format_table(rows, headers))
            print()

    if args.per_era:
        print("# ===== Yields per era =====\n")
        for era in eras:
            for ch in args.channels:
                headers, rows = yield_table(counts, [era], ch, args.processes,
                                            args.threshold)
                if rows:
                    print(f"## {era} - channel {ch}")
                    print(format_table(rows, headers))
                    print()

    overall = sum_over(counts, eras, args.channels, args.processes)
    if abs(overall.below) < 1e-9:
        print(f"# NOTE: not one event anywhere has "
              f"{args.sentinel_value:g} < {args.variable} < {args.threshold:g}, so "
              "every efficiency above is a trivial 100%. That is the signature of "
              "the cut being baked into the variable rather than the category: "
              "re-run with --variable fatbjetProbe_msoftdrop --sentinel-value -1 "
              "on a production that defines it.")

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("era,channel,process,total,no_candidate,candidates,"
                     "fail_msd,pass_msd,efficiency,efficiency_err\n")
            for era in eras:
                for ch in args.channels:
                    for p in args.processes:
                        c = counts.get(era, {}).get(ch, {}).get(p)
                        if c is None or not c.present:
                            continue
                        eff, err = c.efficiency()
                        fh.write(f"{era},{ch},{p},{c.total:.6g},{c.sentinel:.6g},"
                                 f"{c.candidates:.6g},{c.below:.6g},{c.passing:.6g},"
                                 f"{eff:.6g},{err:.6g}\n")
        print(f"# wrote {args.csv}")

    if missing:
        print(f"# {len(missing)} missing histogram(s):", file=sys.stderr)
        for m in missing[:20]:
            print("#   " + m, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

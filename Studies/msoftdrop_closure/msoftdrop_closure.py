"""Data/MC closure of the boosted msoftdrop cut.

The boosted Hbb candidate requires ``SelectedFatJet_msoftdrop > 30`` (the
``FatBJet_Sel`` definition in ``Analysis/hh_bbww.py``).  This script measures,
per era / lepton channel / process, the efficiency of that requirement relative
to the ``baseline_boosted`` category (>=1 SelectedFatJet, no msoftdrop cut), so
the efficiency can be compared between data and simulation.

Input is the merged ``fatbjet_msoftdrop`` histogram file from the HistMerger
step:

    <hists>/<era>/fatbjet_msoftdrop/fatbjet_msoftdrop.root

with the usual ``<channel>/<region>/<category>/<process>`` layout.

IMPORTANT caveat on what this can measure
-----------------------------------------
``fatbjet_msoftdrop`` is only filled from a fat jet that already passes the full
``FatBJet_Sel`` (``particleNetWithMass_HbbvsQCD > 0.92 && msoftdrop > 30``).
Events without such a jet get the default value 0 from

    fatbjet_isValid ? FatBJet_msoftdrop[0] : value_type()

so the bin at 0 is a sentinel, not a measurement.  Dropping the msoftdrop cut
from the *category* does not populate the 0-30 range, because the cut lives in
the *variable* definition.  Consequently the failing-events column below is the
combined "no valid FatBJet" rate (fails the Hbb tagger OR fails msoftdrop OR,
in the single-lepton channels, is boosted only through the W-jet leg), and the
``0<msd<thr`` column is identically empty.

To isolate msoftdrop alone, re-produce with the msoftdrop term removed from
``FatBJet_Sel``; this script then reports the real efficiency without changes,
and the ``0<msd<thr`` column becomes the numerator of interest.
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
SENTINEL = 0.0


class Counts:
    """Yields of one process in one (era, channel) cell."""

    __slots__ = ("total", "total_e2", "sentinel", "sentinel_e2",
                 "below", "below_e2", "straddle", "present")

    def __init__(self):
        self.total = 0.0
        self.total_e2 = 0.0
        self.sentinel = 0.0     # msoftdrop == 0 -> no valid FatBJet
        self.sentinel_e2 = 0.0
        self.below = 0.0        # 0 < msoftdrop < threshold (real failures)
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
    def passing(self):
        """Events with a valid FatBJet, i.e. msoftdrop above the threshold."""
        return self.total - self.sentinel - self.below

    @property
    def passing_e2(self):
        return max(self.total_e2 - self.sentinel_e2 - self.below_e2, 0.0)

    def efficiency(self):
        """passing / total, with a binomial error on the weighted counts."""
        if self.total <= 0.0:
            return float("nan"), float("nan")
        eff = self.passing / self.total
        # Effective-entries binomial error: sigma^2 = eff(1-eff)/N_eff
        n_eff = (self.total ** 2 / self.total_e2) if self.total_e2 > 0 else 0.0
        if n_eff <= 0.0:
            return eff, float("nan")
        err = math.sqrt(max(eff * (1.0 - eff), 0.0) / n_eff)
        return eff, err


def _integral(h, first, last):
    """Integral over bins [first, last] inclusive, with its stat error."""
    if last < first:
        return 0.0, 0.0
    err = ctypes.c_double(0.0)
    val = h.IntegralAndError(first, last, err)
    return float(val), float(err.value)


def read_counts(h, threshold):
    c = Counts()
    c.present = True
    ax = h.GetXaxis()
    n = ax.GetNbins()

    c.total, e = _integral(h, 0, n + 1)
    c.total_e2 = e * e

    sbin = ax.FindBin(SENTINEL)
    c.sentinel, e = _integral(h, sbin, sbin)
    c.sentinel_e2 = e * e

    tbin = ax.FindBin(threshold)
    straddles = ax.GetBinLowEdge(tbin) < threshold
    last_below = tbin - 1
    first_below = sbin + 1
    c.below, e = _integral(h, first_below, last_below)
    c.below_e2 = e * e
    if straddles:
        c.straddle = h.GetBinContent(tbin)
    return c


def collect(input_dir, eras, channels, region, category, variable, processes,
            threshold):
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
                            "that bin is counted as passing and reported separately.",
                            file=sys.stderr)
                    warned[0] = True
                counts[era][ch][proc] = read_counts(h, threshold)
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
    """Per-process yields and cut efficiency for one channel, eras summed."""
    headers = ["Process", "nTotalEvents", f"msd<{threshold:g}", f"msd>{threshold:g}",
               "eff [%]", f"0<msd<{threshold:g}"]
    rows = []
    mc_procs = [p for p in processes if p != DATA_PROCESS]

    def row(label, c):
        eff, err = c.efficiency()
        eff_s = "n/a" if eff != eff else (f"{100 * eff:.2f} +- {100 * err:.2f}"
                                          if err == err else f"{100 * eff:.2f}")
        return [label, f"{c.total:.1f}", f"{c.sentinel:.1f}", f"{c.passing:.2f}",
                eff_s, f"{c.below:.4g}"]

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
            rerr = ratio * math.sqrt((s_da / e_da) ** 2 + (s_mc / e_mc) ** 2) \
                if (e_da > 0 and s_da == s_da and s_mc == s_mc) else float("nan")
            rows.append(["Data/MC eff", "", "", "",
                         f"{ratio:.4f} +- {rerr:.4f}" if rerr == rerr else f"{ratio:.4f}",
                         ""])
    return headers, rows


def efficiency_matrix(counts, eras, channels, processes):
    """Efficiency [%] per process (rows) x era (columns), one table per channel."""
    tables = []
    mc_procs = [p for p in processes if p != DATA_PROCESS]
    for ch in channels:
        headers = ["Process"] + list(eras) + ["All eras"]
        rows = []
        for p in processes + ["Total MC", "Data/MC"]:
            cells = [p]
            ok = False
            for era_list in [[e] for e in eras] + [list(eras)]:
                if p == "Total MC":
                    c = sum_over(counts, era_list, [ch], mc_procs)
                    eff, err = c.efficiency()
                elif p == "Data/MC":
                    mc = sum_over(counts, era_list, [ch], mc_procs)
                    da = sum_over(counts, era_list, [ch], [DATA_PROCESS])
                    e_mc, _ = mc.efficiency()
                    e_da, _ = da.efficiency()
                    eff = e_da / e_mc if (e_mc == e_mc and e_mc > 0) else float("nan")
                    cells.append("n/a" if eff != eff else f"{eff:.4f}")
                    ok = ok or eff == eff
                    continue
                else:
                    c = sum_over(counts, era_list, [ch], [p])
                    eff, err = c.efficiency()
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
    ap.add_argument("--variable", default="fatbjet_msoftdrop")
    ap.add_argument("--processes", nargs="+",
                    default=DEFAULT_BACKGROUNDS + [DATA_PROCESS],
                    help="backgrounds and data; signal is deliberately excluded")
    ap.add_argument("--threshold", type=float, default=30.0)
    ap.add_argument("--per-era", action="store_true",
                    help="also print the per-process yield table for each era")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    counts, missing = collect(args.input_dir, args.eras, args.channels, args.region,
                              args.category, args.variable, args.processes,
                              args.threshold)
    if not counts:
        print("No input files found under " + args.input_dir, file=sys.stderr)
        for m in missing:
            print("  missing: " + m, file=sys.stderr)
        return 1
    eras = [e for e in args.eras if e in counts]

    print(f"# msoftdrop closure: {args.variable} in "
          f"<channel>/{args.region}/{args.category}, threshold {args.threshold:g}")
    print(f"# input: {args.input_dir}")
    print(f"# eras: {', '.join(eras)}\n")

    print("# ============ Cut efficiency [%] per era ============")
    print(f"# efficiency = N(msd>{args.threshold:g}) / nTotalEvents in {args.category}\n")
    for ch, headers, rows in efficiency_matrix(counts, eras, args.channels,
                                               args.processes):
        print(f"## channel {ch}")
        print(format_table(rows, headers))
        print()

    print("# ============ Yields, all eras summed ============\n")
    for ch in args.channels:
        headers, rows = yield_table(counts, eras, ch, args.processes, args.threshold)
        if rows:
            print(f"## channel {ch}")
            print(format_table(rows, headers))
            print()

    if args.per_era:
        print("# ============ Yields per era ============\n")
        for era in eras:
            for ch in args.channels:
                headers, rows = yield_table(counts, [era], ch, args.processes,
                                            args.threshold)
                if rows:
                    print(f"## {era} - channel {ch}")
                    print(format_table(rows, headers))
                    print()

    total_below = sum_over(counts, eras, args.channels, args.processes).below
    if abs(total_below) < 1e-9:
        print(f"# NOTE: not one event anywhere has 0 < {args.variable} < "
              f"{args.threshold:g}.  The msoftdrop cut is baked into the variable "
              f"(FatBJet_Sel), not the category, so the '{args.category}' category "
              "cannot isolate it: every failing event collapses onto the "
              f"{args.variable} == 0 sentinel together with Hbb-tagger failures.")

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("era,channel,process,total,fail_msd_lt_thr,pass_msd_gt_thr,"
                     "efficiency,efficiency_err,below_nonsentinel\n")
            for era in eras:
                for ch in args.channels:
                    for p in args.processes:
                        c = counts.get(era, {}).get(ch, {}).get(p)
                        if c is None or not c.present:
                            continue
                        eff, err = c.efficiency()
                        fh.write(f"{era},{ch},{p},{c.total:.6g},{c.sentinel:.6g},"
                                 f"{c.passing:.6g},{eff:.6g},{err:.6g},{c.below:.6g}\n")
        print(f"# wrote {args.csv}")

    if missing:
        print(f"# {len(missing)} missing histogram(s):", file=sys.stderr)
        for m in missing[:20]:
            print("#   " + m, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

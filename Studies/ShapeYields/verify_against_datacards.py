#!/usr/bin/env python3
"""Check a ShapeYields CSV against the datacard shapes CreateDatacardsTask wrote.

Two independent readings of the same preprocessed shapes: this tool sums the source eras
itself, the datacard maker sums them its own way. They have to agree bin for bin, and
where they do not, one of the two is wrong.

The comparison runs in both directions on purpose. Every nominal histogram in the datacard
must be reproducible from the CSV, and every background in the CSV must be consumed by the
datacard. Checking only the first direction is what let a real defect through: the merged
`TotalBkg` categories were skipped as "explained by the merge", and the merge turned out to
include a process the CSV was leaving out -- DY in eMu boosted, up to 13.5% of the
background in that category. A histogram that cannot be matched is reported, never skipped.

What this does and does not establish: it establishes that ShapeYields reads and sums the
shapes the way the datacard chain does. It cannot establish that the shapes themselves are
right -- both sides read the same files, so an error upstream in the rebinning or the
merge would be reproduced identically by both.

Usage:
    python3 Studies/ShapeYields/verify_against_datacards.py \\
        --csv Studies/ShapeYields/output/uncv2/yields.csv \\
        --datacards data/uncv2/Datacards/Run3_Early \\
        --config config/Datacards/x_hh_bbww_DL_run3.yaml \\
        --era Run3_Early
"""

import argparse
import csv
import re
import sys
from collections import defaultdict

import numpy as np
import uproot
import yaml


def merged_templates(cfg):
    """{merged process: {constituent, ...}} from the datacard configuration."""
    return {
        e["process"]: set(e["subprocesses"])
        for e in cfg.get("processes", [])
        if e.get("subprocesses")
    }


def signal_names(cfg):
    return {
        e["process"].split("${")[0]
        for e in cfg.get("processes", [])
        if e.get("is_signal")
    }


def load_rows(path, era):
    """(mass, channel, category) -> process -> {bin: yield}"""
    rows = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["era"] != era:
                continue
            key = (r["mass"], r["channel"], r["category"])
            rows[key][r["process"]][int(r["bin"])] = float(r["yield"])
    return rows


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--csv", required=True)
    p.add_argument("--datacards", required=True, help="Datacards/<era> directory.")
    p.add_argument("--config", required=True)
    p.add_argument("--era", required=True)
    p.add_argument("--mass", action="append", type=int, default=None)
    p.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Largest relative bin difference treated as agreement (default 1e-9, which "
        "is far above the ~1e-16 round-off two correct summations differ by).",
    )
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    merged = merged_templates(cfg)
    signals = signal_names(cfg)
    masses = args.mass or next(
        e["param_values"] for e in cfg["processes"] if e.get("is_signal")
    )

    rows = load_rows(args.csv, args.era)
    if not rows:
        sys.exit(f"{args.csv} holds no rows for era {args.era}.")

    sig_file = next(e for e in cfg["processes"] if e.get("is_signal"))["process"]
    param = cfg["model"]["parameters"][0]
    pattern = re.compile(rf"^{re.escape(args.era)}_hh_bbww_(?P<ch>[^_]+)_(?P<pre>.+)$")

    worst, worst_at, n_cmp = 0.0, None, 0
    problems = []

    for mass in masses:
        name = sig_file.replace("${" + param + "}", str(mass))
        path = f"{args.datacards}/{name}.root"
        try:
            f = uproot.open(path)
        except Exception as exc:
            problems.append(f"m{mass}: cannot open {path} ({exc})")
            continue
        used = defaultdict(set)

        for top_key in f.keys(recursive=False):
            top_name = top_key.split(";")[0]
            m = pattern.match(top_name)
            if not m:
                problems.append(f"m{mass}: unparsed directory {top_name}")
                continue
            channel, prefix = m.group("ch"), m.group("pre")
            top = f[top_name]
            # The card nests <era>_hh_bbww_<ch>_<prefix>/<slice>/<process>; the CSV
            # spells that category "<prefix>/<slice>".
            for sub_key in top.keys(recursive=False):
                slice_name = sub_key.split(";")[0]
                category = f"{prefix}/{slice_name}"
                d = top[slice_name]
                mine = rows.get((str(mass), channel, category), {})

                for h_key in d.keys(recursive=False):
                    hname = h_key.split(";")[0]
                    if hname.endswith("Up") or hname.endswith("Down"):
                        continue
                    if hname == "data_obs":
                        continue
                    card = d[hname].values()

                    if hname in merged:
                        parts = [c for c in merged[hname] if c in mine]
                        if not parts:
                            problems.append(
                                f"m{mass} {channel} {category}: {hname} has no "
                                "constituent in the CSV"
                            )
                            continue
                        n_bins = max(max(mine[c]) for c in parts)
                        ours = np.zeros(n_bins)
                        for c in parts:
                            for b, v in mine[c].items():
                                ours[b - 1] += v
                        used[(channel, category)].update(parts)
                    else:
                        if hname not in mine:
                            problems.append(
                                f"m{mass} {channel} {category}: {hname} is in the "
                                "datacard and absent from the CSV"
                            )
                            continue
                        ours = np.array([mine[hname][b] for b in sorted(mine[hname])])
                        used[(channel, category)].add(hname)

                    if len(card) != len(ours):
                        problems.append(
                            f"m{mass} {channel} {category} {hname}: {len(card)} bins in "
                            f"the datacard, {len(ours)} in the CSV"
                        )
                        continue
                    rel = np.max(np.abs(card - ours) / np.maximum(np.abs(ours), 1e-9))
                    n_cmp += 1
                    if rel > worst:
                        worst, worst_at = rel, f"m{mass} {channel} {category} {hname}"

        # The other direction: a background in the CSV that the datacard never used means
        # this tool is reporting something the fit does not see.
        for (m_, channel, category), procs in rows.items():
            if m_ != str(mass):
                continue
            for proc in procs:
                if any(proc.startswith(s) for s in signals):
                    continue
                if proc not in used[(channel, category)]:
                    problems.append(
                        f"m{mass} {channel} {category}: the CSV has {proc}, the "
                        "datacard never used it"
                    )

    print(f"compared {n_cmp} histograms over masses {masses}")
    print(
        f"worst relative bin difference: {worst:.3e}"
        + (f"  at {worst_at}" if worst_at else "")
    )
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for line in problems[:60]:
            print("  " + line)
        if len(problems) > 60:
            print(f"  ... and {len(problems) - 60} more")
        sys.exit(1)
    if worst > args.tolerance:
        print(f"\nFAIL: worst difference exceeds tolerance {args.tolerance:g}")
        sys.exit(1)
    print("\nno unmatched histograms in either direction; every bin agrees")


if __name__ == "__main__":
    main()

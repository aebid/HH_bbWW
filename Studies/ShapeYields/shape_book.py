#!/usr/bin/env python3
"""A book of the rebinned shapes: one page per mass, channels down, categories across.

Each panel shows the stacked backgrounds with their MC-statistical band and the summed
signal, read the same way shape_yields.py reads them (the era group's sum). The panel is
labelled with the selection its category stands for, e.g. "670 < HME < 1170" for a
window, taken from the binning.json the rebinning wrote beside the shapes.

Usage:
    python3 Studies/ShapeYields/shape_book.py \\
        --input /eos/user/d/daebi/HH_bbWW/<production>/Hists_preprocessed/Run3_Early \\
        --config config/Datacards/x_hh_bbww_DL_run3.yaml \\
        --output Studies/ShapeYields/output/<production>
"""

import argparse
import json
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shape_yields import (  # noqa: E402
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
    gather,
    load_config,
    read_csv,
)

# Backgrounds in the card's order; muted so the signal line reads on top.
BKG_COLORS = [
    "#8fb3d9",
    "#f2b880",
    "#9fcf9a",
    "#c9a7d8",
    "#e8a3a3",
    "#b5b09a",
    "#9ad1d4",
    "#d9c77a",
]
SIGNAL_COLOR = "#c0392b"
PANEL_W, PANEL_H = 3.3, 2.5  # inches


def fmt_edge(v):
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.2f}".rstrip("0").rstrip(".")


def fmt_selection(lo, hi, var):
    if lo is None and hi is None:
        return ""
    if lo is None:
        return f"{var} < {fmt_edge(hi)}"
    if hi is None:
        return f"{var} > {fmt_edge(lo)}"
    return f"{fmt_edge(lo)} < {var} < {fmt_edge(hi)}"


def selection_labels(path, group):
    """{(mass, channel, category): ("lo < var < hi", binned axis)} from a binning.json."""
    with open(path) as f:
        record = json.load(f)
    var = record.get("slice_var", "")
    pattern = record.get("category_pattern") or "{base_category}_dnn{slice_idx}"
    labels = {}
    for mass, by_channel in record.get("binning", {}).get(group, {}).items():
        for channel, by_category in by_channel.items():
            for base, entry in by_category.items():
                for idx, sl in enumerate(entry["slices"]):
                    # the selection is on y for a window, on x for a greedy slice
                    axis = "y" if "y_range" in sl else "x"
                    lo, hi = sl.get(f"{axis}_edges", [None, None])
                    cat = pattern.format(base_category=base, slice_idx=idx)
                    binned = "x" if axis == "y" else "y"
                    labels[(int(mass), channel, cat)] = (
                        fmt_selection(lo, hi, var),
                        binned,
                    )
    return labels


def panel_data(rows):
    """(edges, {process: (values, errors)}, is_signal by process) for one panel."""
    bins = sorted({r["bin"] for r in rows})
    index = {b: i for i, b in enumerate(bins)}
    lo = {r["bin"]: r["bin_lo"] for r in rows}
    hi = {r["bin"]: r["bin_hi"] for r in rows}
    edges = [lo[b] for b in bins] + [hi[bins[-1]]]
    procs, is_signal = {}, {}
    for r in rows:
        values, errors = procs.setdefault(
            r["process"], (np.zeros(len(bins)), np.zeros(len(bins)))
        )
        values[index[r["bin"]]] = r["yield"]
        errors[index[r["bin"]]] = r["error"]
        is_signal[r["process"]] = bool(r["is_signal"])
    return edges, procs, is_signal


def draw_panel(ax, edges, procs, is_signal, bkg_order, x_label):
    n = len(edges) - 1
    x = np.arange(n + 1)
    bottom = np.zeros(n)
    total_var = np.zeros(n)
    for i, proc in enumerate(bkg_order):
        if proc not in procs:
            continue
        values, errors = procs[proc]
        top = bottom + np.clip(values, 0, None)  # negative bins cannot be stacked
        ax.stairs(
            top,
            x,
            baseline=bottom,
            fill=True,
            color=BKG_COLORS[i % len(BKG_COLORS)],
            lw=0,
        )
        bottom = top
        total_var += errors**2
    err = np.sqrt(total_var)
    ax.stairs(bottom, x, color=INK_SECONDARY, lw=0.6)
    ax.stairs(
        bottom + err,
        x,
        baseline=np.clip(bottom - err, 0, None),
        fill=False,
        hatch="////",
        color=INK_MUTED,
        lw=0,
    )
    signal = sum(v for p, (v, _) in procs.items() if is_signal[p])
    if np.ndim(signal):
        ax.stairs(signal, x, color=SIGNAL_COLOR, lw=1.6)

    positive = [v for v in np.concatenate([bottom, np.atleast_1d(signal)]) if v > 0]
    ax.set_yscale("log")
    if positive:
        ax.set_ylim(min(positive) * 0.3, max(positive) * 30)
    ax.set_xlim(0, n)
    step = max(1, n // 8)  # keep the edge labels readable on finely binned panels
    ticks = list(range(0, n + 1, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt_edge(edges[t]) for t in ticks], rotation=90, fontsize=6)
    ax.tick_params(axis="y", labelsize=6, colors=INK_SECONDARY)
    ax.set_xlabel(x_label, fontsize=7, color=INK_SECONDARY)
    ax.grid(axis="y", color=GRIDLINE, lw=0.5)
    ax.set_facecolor(SURFACE)


def draw_mass_page(
    pdf,
    rows,
    mass,
    group,
    channels,
    categories,
    bkg_order,
    labels,
    axis_names,
    param_name,
):
    by_panel = {}
    for r in rows:
        by_panel.setdefault((r["channel"], r["category"]), []).append(r)
    columns = [c for c in categories if any((ch, c) in by_panel for ch in channels)]
    if not columns:
        return False
    fig, axes = plt.subplots(
        len(channels),
        len(columns),
        figsize=(PANEL_W * len(columns), PANEL_H * len(channels) + 1.0),
        squeeze=False,
    )
    for i, channel in enumerate(channels):
        for j, category in enumerate(columns):
            ax = axes[i][j]
            panel = by_panel.get((channel, category))
            if not panel:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    "not binned",
                    ha="center",
                    va="center",
                    color=INK_MUTED,
                    fontsize=8,
                    transform=ax.transAxes,
                )
                continue
            edges, procs, is_signal = panel_data(panel)
            label, binned = labels.get((mass, channel, category), ("", None))
            draw_panel(
                ax, edges, procs, is_signal, bkg_order, axis_names.get(binned, "")
            )
            if i == 0:
                ax.set_title(category, fontsize=9, color=INK_PRIMARY)
            if j == 0:
                ax.set_ylabel(f"{channel}\nevents", fontsize=8, color=INK_PRIMARY)
            if label:
                ax.text(
                    0.03,
                    0.95,
                    label,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color=INK_PRIMARY,
                    bbox=dict(
                        boxstyle="round,pad=0.25", fc="white", ec=GRIDLINE, lw=0.5
                    ),
                )
    handles = [
        Patch(color=BKG_COLORS[i % len(BKG_COLORS)], label=p)
        for i, p in enumerate(bkg_order)
        if any(r["process"] == p for r in rows)
    ]
    handles.append(
        Patch(fill=False, hatch="////", color=INK_MUTED, label="bkg. MC stat.")
    )
    handles.append(plt.Line2D([], [], color=SIGNAL_COLOR, lw=1.6, label="signal (sum)"))
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle(
        f"{group}   {param_name} = {mass} GeV", fontsize=11, color=INK_PRIMARY, y=0.995
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(fig)
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input",
        default=None,
        help="Hists_preprocessed/<era-group> directory, as for shape_yields.py.",
    )
    p.add_argument("--config", required=True, help="Datacard configuration YAML.")
    p.add_argument(
        "--binning-config", default=None, help="Binning YAML, for category_pattern."
    )
    p.add_argument(
        "--output", required=True, help="Directory for shapes_<era-group>.pdf."
    )
    p.add_argument(
        "--mass", action="append", type=int, default=None, help="Repeatable."
    )
    p.add_argument("--channel", action="append", default=None, help="Repeatable.")
    p.add_argument(
        "--era-group", default=None, help="Defaults to the single entry of `eras:`."
    )
    p.add_argument(
        "--from-csv",
        default=None,
        help="Draw from shape_yields.py's yields.csv instead of the shapes.",
    )
    p.add_argument(
        "--binning-json",
        default=None,
        help="Selection labels; defaults to <input>/binning.json.",
    )
    p.add_argument("--x-name", default="DNN score", help="The 2D input's x variable.")
    p.add_argument("--y-name", default="HME [GeV]", help="The 2D input's y variable.")
    args = p.parse_args()

    cfg = load_config(args.config)
    param_name = cfg["model"]["parameters"][0]
    group = args.era_group
    if group is None:
        if len(cfg.get("eras", [])) != 1:
            sys.exit(
                "--era-group not given and `eras:` does not have exactly one entry."
            )
        group = cfg["eras"][0]
    source_eras = cfg.get("era_groups", {}).get(group, [group])
    knobs = {
        "category_pattern": cfg.get(
            "category_pattern", "{base_category}_dnn{slice_idx}"
        ),
        "era_group": group,
    }
    if args.binning_config:
        knobs["category_pattern"] = load_config(args.binning_config).get(
            "category_pattern", knobs["category_pattern"]
        )

    masses = args.mass or next(
        e["param_values"] for e in cfg["processes"] if e.get("is_signal")
    )
    channels = args.channel or cfg["channels"]
    categories = cfg["categories"]
    if args.from_csv:
        rows = read_csv(args.from_csv, [group], masses, channels, categories)
    else:
        rows = gather(
            args.input,
            cfg,
            knobs,
            source_eras,
            masses,
            channels,
            categories,
            param_name,
        )
    rows = [r for r in rows if r["era"] == group]
    if not rows:
        sys.exit("No shapes read -- check --input/--from-csv and the selectors.")

    json_path = args.binning_json or (
        os.path.join(args.input, "binning.json") if args.input else None
    )
    labels = {}
    if json_path and os.path.exists(json_path):
        labels = selection_labels(json_path, group)
    else:
        print("[warn] no binning.json; panels are drawn without their selection")

    bkg_order = [
        e["process"]
        for e in cfg["processes"]
        if isinstance(e, dict)
        and not (e.get("is_signal") or e.get("is_data") or e.get("subprocesses"))
    ]

    os.makedirs(args.output, exist_ok=True)
    out = os.path.join(args.output, f"shapes_{group}.pdf")
    pages = 0
    with PdfPages(out) as pdf:
        for mass in sorted({r["mass"] for r in rows}):
            mass_rows = [r for r in rows if r["mass"] == mass]
            pages += draw_mass_page(
                pdf,
                mass_rows,
                mass,
                group,
                channels,
                categories,
                bkg_order,
                labels,
                {"x": args.x_name, "y": args.y_name},
                param_name,
            )
    print(f"Wrote {out}  ({pages} pages)")


if __name__ == "__main__":
    main()

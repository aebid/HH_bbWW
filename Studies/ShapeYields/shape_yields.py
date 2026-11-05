#!/usr/bin/env python3
"""Report the per-bin yields of the rebinned shapes that combine is fitting.

Reads the output of the `preprocess:` step declared in the datacard configuration --
StatInference/bin_opt_2d/rebin_2d.py for HH->bbWW -- which is the same thing
CreateDatacardsTask reads as its input. That layout is "<era>/<variable>/<variable>.root"
with the histograms at "<channel>/<category>/<process>", one file per sub-era per mass.

Nominal shapes only: every histogram whose name is a bare process name, with the
Up/Down variations left alone.

Two products, from one pass over the files:

  yields.csv   one row per (era, mass, channel, category, bin, process): the bin's
               content, its MC-statistical error, and the bin's HME edges.
  yields.pdf   the same numbers as annotated tables, one page per
               (era, mass, channel, base category), four DNN slices to a page.

This measures the shapes and nothing else. It applies no thresholds and makes no
judgement about which bins are acceptable -- the numbers are the report.

Usage:
    python3 Studies/ShapeYields/shape_yields.py \\
        --input /eos/user/d/daebi/HH_bbWW/uncv2/Hists_preprocessed/Run3_Early \\
        --config config/Datacards/x_hh_bbww_DL_run3.yaml \\
        --output Studies/ShapeYields/output/uncv2
"""

import argparse
import csv
import math
import os
import re
import sys

import numpy as np
import uproot
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- palette -----------------------------------------------------------------------
#
# Magnitude is a sequential encoding, so it is one hue light->dark (blue). The ramp is
# cut off at step 350 rather than running to 700: every cell carries its number, and the
# number has to stay legible in primary ink, which it does not on the dark steps.
# Negative content is not a smaller magnitude, it is a different state, so it gets the
# reserved critical red -- and, since colour never carries meaning alone, bold text and
# a leading marker as well.
SEQ_BLUE = ["#eef5fe", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7"]
NEG_FILL = "#f9dede"
NEG_INK = "#d03b3b"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

# Rows that are sums rather than a process read from the file, drawn on their own wash so
# the eye does not add them into the stack above.
DERIVED_FILL = "#f0efec"


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class CategoryNaming:
    """Take a sliced category name apart with the pattern that put it together.

    The same pattern rebin_2d.py names its output with (config/Datacards/binning_2d.yaml,
    echoed by the datacard configuration's `category_pattern`). Kept here rather than
    imported so this tool reads only the configuration and the files, and cannot be
    broken by a change to the binner's internals.
    """

    def __init__(self, pattern):
        self.pattern = pattern
        regex = re.escape(pattern)
        regex = regex.replace(re.escape("{base_category}"), "(?P<base>.+)")
        regex = regex.replace(re.escape("{slice_idx}"), r"(?P<idx>\d+)")
        self.regex = re.compile("^" + regex + "$")

    def split(self, category):
        """(base_category, slice_index). An unsliced name is its own base, index None."""
        m = self.regex.match(category)
        if not m:
            return category, None
        return m.group("base"), int(m.group("idx"))


def substitute(template, **kwargs):
    """Resolve ${NAME} placeholders, the form the datacard configuration uses."""
    out = template
    for key, value in kwargs.items():
        out = out.replace("${" + key + "}", str(value))
    return out


def collect_processes(cfg, mass, param_name):
    """The processes to report for one mass, in the order they should be drawn.

    Driven entirely by the datacard configuration's `processes:` block, so this reports
    what the analysis declares rather than a list hardcoded here. Three kinds are left
    out, each for its own reason:

      is_data          the observation is Asimov (is_asimov_data), so it carries no
                       information the summed background does not.
      subprocesses     a merged template such as TotalBkg is assembled by the datacard
                       maker and does not exist in these files. Its constituents do, and
                       they are reported individually.
      commented out    absent from the configuration, so absent here.

    Returns a list of (label, hist_name, is_signal, channels, categories), where channels
    and categories are None when the entry places no restriction.
    """
    out = []
    for entry in cfg.get("processes", []):
        if entry.get("is_data"):
            continue
        if entry.get("subprocesses"):
            continue
        name = substitute(entry["process"], **{param_name: mass})
        hist_name = substitute(
            entry.get("hist_name", entry["process"]), **{param_name: mass}
        )
        out.append(
            (
                name,
                hist_name,
                bool(entry.get("is_signal")),
                entry.get("channels"),
                entry.get("categories"),
            )
        )
    # Backgrounds first, signal last: the table reads as a stack with the thing being
    # searched for underneath it.
    out.sort(key=lambda p: p[2])
    return out


def applies(restriction, value):
    return restriction is None or value in restriction


def merged_scopes(cfg):
    """Where a merged template pulls its subprocesses in, ignoring their own restrictions.

    The datacard maker assembles a `subprocesses` template from its constituents'
    histograms by path, and does not consult a constituent's own `channels` list on the
    way. So a process excluded from a channel by its own entry is nonetheless part of the
    merged background there, and reporting it as absent understates the background that
    is actually fitted.

    This is not hypothetical: DY declares `channels: [eE, muMu]`, and applying that
    uniformly left eMu boosted short of the datacard's TotalBkg by up to 13.5% at m500 --
    eE and muMu agreed exactly, eMu did not.

    Returns [(subprocess names, channels, categories)] for each merged entry, where the
    two restrictions are the merged entry's own and None means it places none.
    """
    return [
        (set(entry["subprocesses"]), entry.get("channels"), entry.get("categories"))
        for entry in cfg.get("processes", [])
        if entry.get("subprocesses")
    ]


def process_applies(name, chs, cats, channel, category, base, scopes):
    """Whether a process contributes to the background the datacard fits here.

    Either on its own terms, or as a constituent of a merged template that reaches into
    this channel and category.
    """
    if applies(chs, channel) and (applies(cats, base) or applies(cats, category)):
        return True
    for names, m_chs, m_cats in scopes:
        if name not in names:
            continue
        if applies(m_chs, channel) and (
            applies(m_cats, base) or applies(m_cats, category)
        ):
            return True
    return False


def scale_note(cfg):
    """How the datacard maker will rescale these histograms, if it will.

    `scale:` on a process is applied by StatInference/dc_make/maker.py when it reads the
    histogram, so a configuration that declares one produces datacard rates that are not
    the numbers in these files. Whether it does is a property of the configuration, not
    something to assert on the page: the uncv2 shapes were built by a configuration
    carrying no scale, and there the two agree exactly. So the note is written from what
    the configuration actually says.
    """
    scales = {}
    for entry in cfg.get("processes", []):
        raw = entry.get("scale", 1)
        try:
            value = eval(raw, {"__builtins__": {}}, {}) if isinstance(raw, str) else raw
        except Exception:
            continue
        if value != 1:
            scales[entry["process"]] = value
    if not scales:
        return (
            "This configuration declares no process `scale`, so these are also the "
            "datacard rates."
        )
    parts = ", ".join(f"{k} ×{v:.4g}" for k, v in scales.items())
    return f"The datacard rescales before writing its rate: {parts}."


def read_hist(rootfile, path):
    """(values, errors, edges) for one histogram, or None if it is not there.

    A missing histogram is normal -- rebin_2d.py skips a channel/category whose signal is
    below min_signal, and writes nothing for it -- so this is not an error.
    """
    try:
        h = rootfile[path]
    except (KeyError, uproot.KeyInFileError):
        return None
    return h.values(), h.errors(), h.axis().edges()


def gather(input_dir, cfg, knobs, eras, masses, channels, categories, param_name):
    """One pass over the files. Returns a list of row dicts, the tidy CSV in memory.

    Each source era is read separately and a synthetic row set for the era group is
    accumulated alongside, because that is the shape of the thing: rebin_2d.py derives one
    binning from the group's summed statistics and then writes each member out under it,
    and the datacard step sums the members back up. So the per-era numbers and their sum
    are both real, and both are reported.
    """
    naming = CategoryNaming(knobs["category_pattern"])
    pattern = cfg["model"]["input_file_pattern"]
    group_name = knobs["era_group"]
    scopes = merged_scopes(cfg)

    rows = []
    for era in eras:
        for mass in masses:
            rel = substitute(pattern, ERA=era, **{param_name: mass})
            path = os.path.join(input_dir, rel)
            if not os.path.exists(path):
                print(f"[skip] {era} m{mass}: no {path}")
                continue
            procs = collect_processes(cfg, mass, param_name)
            with uproot.open(path) as f:
                for channel in channels:
                    for category in categories:
                        base, idx = naming.split(category)
                        for label, hist_name, is_sig, chs, cats in procs:
                            if not process_applies(
                                label, chs, cats, channel, category, base, scopes
                            ):
                                continue
                            got = read_hist(f, f"{channel}/{category}/{hist_name}")
                            if got is None:
                                continue
                            values, errors, edges = got
                            for b in range(len(values)):
                                rows.append(
                                    {
                                        "era": era,
                                        "mass": mass,
                                        "channel": channel,
                                        "category": category,
                                        "base_category": base,
                                        "slice_idx": idx,
                                        "bin": b + 1,
                                        "bin_lo": float(edges[b]),
                                        "bin_hi": float(edges[b + 1]),
                                        "process": label,
                                        "is_signal": int(is_sig),
                                        "yield": float(values[b]),
                                        "error": float(errors[b]),
                                    }
                                )
    if group_name and len(eras) > 1:
        rows.extend(sum_eras(rows, group_name))
    return rows


def sum_eras(rows, group_name):
    """The era group's rows: yields added, MC-stat errors added in quadrature.

    Independent samples, so quadrature is right. Keyed on everything but the era, which
    is safe only because the group's members share a binning by construction -- rebin_2d
    derives the edges once for the group and applies them to each member.
    """
    acc = {}
    for r in rows:
        key = (
            r["mass"],
            r["channel"],
            r["category"],
            r["bin"],
            r["process"],
        )
        if key not in acc:
            acc[key] = dict(r, era=group_name)
            acc[key]["error"] = r["error"] ** 2
        else:
            acc[key]["yield"] += r["yield"]
            acc[key]["error"] += r["error"] ** 2
    for v in acc.values():
        v["error"] = math.sqrt(v["error"])
    return list(acc.values())


def write_csv(rows, path):
    fields = [
        "era",
        "mass",
        "channel",
        "base_category",
        "slice_idx",
        "category",
        "bin",
        "bin_lo",
        "bin_hi",
        "process",
        "is_signal",
        "yield",
        "error",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(
            rows,
            key=lambda r: (
                r["era"],
                r["mass"],
                r["channel"],
                r["base_category"],
                r["slice_idx"] if r["slice_idx"] is not None else -1,
                r["bin"],
                r["is_signal"],
                r["process"],
            ),
        ):
            w.writerow({k: r[k] for k in fields})
    print(f"Wrote {path}  ({len(rows)} rows)")


def read_csv(path, eras, masses, channels, categories):
    """Rows back out of a yields.csv, narrowed by the same selectors as a fresh run.

    So that reworking a page costs nothing: the files sit on EOS and reading all of them
    dominates a run, while the numbers, once written, do not change.
    """
    eras, masses = set(eras), set(masses)
    channels, categories = set(channels), set(categories)
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["mass"] = int(r["mass"])
            if r["era"] not in eras or r["mass"] not in masses:
                continue
            if r["channel"] not in channels or r["category"] not in categories:
                continue
            r["bin"] = int(r["bin"])
            r["is_signal"] = int(r["is_signal"])
            r["slice_idx"] = int(r["slice_idx"]) if r["slice_idx"] != "" else None
            for k in ("bin_lo", "bin_hi", "yield", "error"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


# --- drawing ------------------------------------------------------------------------


def fmt_yield(value):
    """Enough significant figures to be discussed, few enough to fit the cell."""
    a = abs(value)
    if a == 0:
        return "0"
    if a >= 1000:
        return f"{value:,.0f}"
    if a >= 100:
        return f"{value:.0f}"
    if a >= 10:
        return f"{value:.1f}"
    if a >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def fmt_cell(value, error):
    return f"{fmt_yield(value)}\n±{fmt_yield(error)}"


def shade(value, vmax):
    """A sequential step for a positive magnitude; the negative state's own fill."""
    if value < 0:
        return NEG_FILL
    if vmax <= 0 or value <= 0:
        return SEQ_BLUE[0]
    # Log scale: these yields run over four or five orders of magnitude within a single
    # slice, and a linear ramp puts every background but the largest in the same box.
    frac = math.log10(1 + value) / math.log10(1 + vmax)
    step = int(round(frac * (len(SEQ_BLUE) - 1)))
    return SEQ_BLUE[max(0, min(len(SEQ_BLUE) - 1, step))]


ROW_H = 0.27  # inches per table row
LABEL_W = 2.05  # inches for the process-name column
COL_W = 1.16  # inches per HME bin column
TITLE_H = 0.52
FOOTER_H = 0.34


def build_panel(per_proc, bkg_procs, signal_procs):
    """One slice's numbers, ready to draw: {process: (values, errors)}, edges, row order.

    "Total bkg" is added as a row of its own -- the summed background is the quantity the
    binning was built against and the one the Asimov observation is, so it belongs on the
    page even though it is not a histogram in the file. Errors add in quadrature.
    """
    n_bins = max((r["bin"] for rs in per_proc.values() for r in rs), default=0)
    rows_by_proc = {}
    edges = None
    for proc, rs in per_proc.items():
        rs = sorted(rs, key=lambda r: r["bin"])
        rows_by_proc[proc] = (
            np.array([r["yield"] for r in rs]),
            np.array([r["error"] for r in rs]),
        )
        if edges is None:
            edges = np.array([r["bin_lo"] for r in rs] + [rs[-1]["bin_hi"]])

    bkg_here = [p for p in bkg_procs if p in per_proc]
    if bkg_here:
        tot = np.zeros(n_bins)
        toterr2 = np.zeros(n_bins)
        for p in bkg_here:
            tot += rows_by_proc[p][0]
            toterr2 += rows_by_proc[p][1] ** 2
        rows_by_proc["Total bkg"] = (tot, np.sqrt(toterr2))

    sig_here = [p for p in per_proc if p in signal_procs]
    order = bkg_here + (["Total bkg"] if bkg_here else []) + sorted(sig_here)
    return rows_by_proc, edges, order, n_bins


def draw_page(fig, panels, n_cols, header, param_name, slice_var, note):
    """Draw every slice of one page into a single axes.

    One coordinate system for the whole page rather than one axes per slice: the rows
    then have the same height everywhere, which is both what makes the four slices
    comparable by eye and what stops a slice with few rows from stretching its cells into
    its neighbours.
    """
    total_rows = sum(2 + len(p["order"]) + 1 for p in panels)
    width = LABEL_W + COL_W * n_cols
    height = TITLE_H + ROW_H * total_rows + FOOTER_H
    fig.set_size_inches(width, height)

    ax = fig.add_axes(
        [
            0,
            FOOTER_H / height,
            1,
            1 - (TITLE_H + FOOTER_H) / height,
        ]
    )
    ax.set_axis_off()
    # x runs in column units, with the label column exactly LABEL_W wide.
    label_cols = LABEL_W / COL_W
    ax.set_xlim(0, label_cols + n_cols)
    ax.set_ylim(0, total_rows)

    y = total_rows
    for panel in panels:
        rows_by_proc = panel["rows_by_proc"]
        edges = panel["edges"]
        order = panel["order"]
        n_bins = panel["n_bins"]

        y -= 1
        ax.text(
            0.06,
            y + 0.28,
            f"{panel['category']}   ({slice_var} slice {panel['slice_idx']})",
            ha="left",
            va="center",
            fontsize=9,
            color=INK_PRIMARY,
            weight="bold",
        )

        if n_bins == 0:
            y -= 1
            ax.text(
                label_cols + n_cols / 2.0,
                y + 0.5,
                "no shapes written for this slice",
                ha="center",
                va="center",
                fontsize=8,
                color=INK_MUTED,
                style="italic",
            )
            y -= 1
            continue

        # Per-slice scale: the comparison the page invites is between the bins of one
        # slice, not between a boosted slice and a res2b one.
        vmax = max(
            (float(np.max(v[0])) if len(v[0]) else 0.0) for v in rows_by_proc.values()
        )

        y -= 1
        ax.text(
            label_cols - 0.12,
            y + 0.5,
            "HME (GeV)",
            ha="right",
            va="center",
            fontsize=7,
            color=INK_MUTED,
            style="italic",
        )
        for b in range(n_bins):
            ax.text(
                label_cols + b + 0.5,
                y + 0.5,
                f"{edges[b]:g}–{edges[b + 1]:g}",
                ha="center",
                va="center",
                fontsize=7,
                color=INK_SECONDARY,
            )
        ax.plot(
            [0.06, label_cols + n_bins],
            [y, y],
            color=GRIDLINE,
            lw=0.9,
            solid_capstyle="butt",
        )

        for proc in order:
            y -= 1
            derived = proc == "Total bkg"
            ax.text(
                label_cols - 0.12,
                y + 0.5,
                proc,
                ha="right",
                va="center",
                fontsize=7.5,
                color=INK_SECONDARY if derived else INK_PRIMARY,
                weight="bold" if derived else "normal",
            )
            if proc not in rows_by_proc:
                ax.text(
                    label_cols + n_bins / 2.0,
                    y + 0.5,
                    "not in this category",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=INK_MUTED,
                    style="italic",
                )
                continue
            values, errors = rows_by_proc[proc]
            for b in range(n_bins):
                v = float(values[b])
                e = float(errors[b])
                negative = v < 0
                fill = DERIVED_FILL if derived and not negative else shade(v, vmax)
                # A gap between fills, so adjacent cells read as separate marks.
                ax.add_patch(
                    plt.Rectangle(
                        (label_cols + b + 0.025, y + 0.07),
                        0.95,
                        0.86,
                        facecolor=fill,
                        edgecolor=NEG_INK if negative else "none",
                        lw=1.0 if negative else 0,
                        zorder=1,
                    )
                )
                ax.text(
                    label_cols + b + 0.5,
                    y + 0.5,
                    fmt_cell(v, e),
                    ha="center",
                    va="center",
                    fontsize=6.6,
                    linespacing=1.25,
                    color=NEG_INK if negative else INK_PRIMARY,
                    weight="bold" if negative else "normal",
                    zorder=2,
                )
        y -= 1  # spacer before the next slice

    fig.text(
        0.5,
        1 - TITLE_H / height * 0.62,
        header,
        ha="center",
        va="center",
        fontsize=12,
        color=INK_PRIMARY,
        weight="bold",
    )
    fig.text(
        0.5,
        FOOTER_H / height * 0.62,
        "Nominal yields ± MC-statistical error per HME bin, as written by the rebinning "
        "step and read by CreateDatacardsTask.",
        ha="center",
        va="center",
        fontsize=7,
        color=INK_MUTED,
    )
    fig.text(
        0.5,
        FOOTER_H / height * 0.24,
        "Negative content is boxed and printed in red.    "
        "Cell shade: log magnitude within the slice.    " + note,
        ha="center",
        va="center",
        fontsize=7,
        color=INK_MUTED,
    )


def draw_pages(rows, pdf, cfg, knobs, param_name, note):
    """One page per (era, mass, channel, base category)."""
    slice_var = knobs.get("slice_var", "DNN")

    index = {}
    for r in rows:
        key = (r["era"], r["mass"], r["channel"], r["base_category"])
        index.setdefault(key, {}).setdefault(
            (r["slice_idx"], r["category"]), {}
        ).setdefault(r["process"], []).append(r)

    signal_procs = {r["process"] for r in rows if r["is_signal"]}
    present = {r["process"] for r in rows if not r["is_signal"]}
    # Configuration order, so the dominant background leads and the pages of a CSV redraw
    # match the pages of a fresh read.
    bkg_procs = [
        e["process"]
        for e in cfg.get("processes", [])
        if not e.get("is_data")
        and not e.get("subprocesses")
        and not e.get("is_signal")
        and e["process"] in present
    ]
    bkg_procs += [p for p in sorted(present) if p not in bkg_procs]

    n_pages = 0
    for key in sorted(index, key=lambda k: (k[0], k[1], k[2], k[3])):
        era, mass, channel, base = key
        slices = index[key]
        slice_keys = sorted(slices, key=lambda s: (s[0] if s[0] is not None else -1))

        panels = []
        for idx, category in slice_keys:
            rows_by_proc, edges, order, n_bins = build_panel(
                slices[(idx, category)], bkg_procs, signal_procs
            )
            panels.append(
                {
                    "slice_idx": idx,
                    "category": category,
                    "rows_by_proc": rows_by_proc,
                    "edges": edges,
                    "order": order,
                    "n_bins": n_bins,
                }
            )
        n_cols = max((p["n_bins"] for p in panels), default=1)

        fig = plt.figure(facecolor=SURFACE)
        draw_page(
            fig,
            panels,
            n_cols,
            f"{era}    {param_name} = {mass} GeV    {channel}    {base}",
            param_name,
            slice_var,
            note,
        )
        pdf.savefig(fig, facecolor=SURFACE)
        plt.close(fig)
        n_pages += 1
    return n_pages


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input",
        default=None,
        help="Hists_preprocessed/<era-group> directory written by the preprocess step, "
        "holding one sub-directory per source era.",
    )
    p.add_argument("--config", required=True, help="Datacard configuration YAML.")
    p.add_argument(
        "--binning-config",
        default=None,
        help="Binning YAML, for category_pattern and slice_var. Defaults to the "
        "datacard configuration's own category_pattern.",
    )
    p.add_argument("--output", required=True, help="Directory for yields.csv/.pdf.")
    p.add_argument("--era", action="append", default=None, help="Repeatable.")
    p.add_argument(
        "--mass", action="append", type=int, default=None, help="Repeatable."
    )
    p.add_argument("--channel", action="append", default=None, help="Repeatable.")
    p.add_argument(
        "--category",
        action="append",
        default=None,
        help="Repeatable; the sliced name, e.g. SR/res2b_dnn0.",
    )
    p.add_argument(
        "--era-group",
        default=None,
        help="Era-group key from the datacard configuration whose members are read and "
        "whose sum is reported alongside them. Defaults to the single entry of `eras:`.",
    )
    p.add_argument("--no-pdf", action="store_true", help="Write only the CSV.")
    p.add_argument(
        "--from-csv",
        default=None,
        help="Redraw from an existing yields.csv instead of reading the shapes. The "
        "selectors still apply, so this is how to cut one page for a slide, or to "
        "change the drawing, without going back to the files.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    param_name = cfg["model"]["parameters"][0]

    group = args.era_group
    if group is None:
        eras_cfg = cfg.get("eras", [])
        if len(eras_cfg) != 1:
            sys.exit(
                f"--era-group not given and `eras:` has {len(eras_cfg)} entries; "
                "name the group explicitly."
            )
        group = eras_cfg[0]
    source_eras = cfg.get("era_groups", {}).get(group, [group])

    knobs = {
        "category_pattern": cfg.get(
            "category_pattern", "{base_category}_dnn{slice_idx}"
        ),
        "slice_var": "DNN",
        "era_group": group,
    }
    if args.binning_config:
        b = load_config(args.binning_config)
        knobs["category_pattern"] = b.get("category_pattern", knobs["category_pattern"])
        knobs["slice_var"] = b.get("slice_var", knobs["slice_var"])

    masses = args.mass or next(
        e["param_values"] for e in cfg["processes"] if e.get("is_signal")
    )
    channels = args.channel or cfg["channels"]
    categories = args.category or cfg["categories"]
    # What "every era" means depends on which end we are reading from. Reading the shapes,
    # it is the source eras: the group has no files of its own, its rows are made by
    # summing theirs. Reading a CSV, those group rows are already in it and are usually
    # the ones wanted, so leaving the group out of the default silently drops a fifth of
    # the file -- and the one era the datacards are actually built from.
    eras = args.era or (
        source_eras + [group]
        if args.from_csv and group not in source_eras
        else source_eras
    )

    os.makedirs(args.output, exist_ok=True)
    if args.from_csv:
        rows = read_csv(args.from_csv, eras, masses, channels, categories)
        if not rows:
            sys.exit(f"Nothing in {args.from_csv} matched the selectors.")
        print(f"Read {len(rows)} rows from {args.from_csv}")
    else:
        rows = gather(
            args.input, cfg, knobs, eras, masses, channels, categories, param_name
        )
        if not rows:
            sys.exit("No histograms read -- check --input and the selectors.")
        write_csv(rows, os.path.join(args.output, "yields.csv"))

    note = scale_note(cfg)
    if not args.no_pdf:
        # One document per era rather than one of several hundred pages. An era is the
        # unit a discussion is held in -- the group is what the limit is set on, a member
        # is where a feature of that group came from -- and a document per era is the
        # thing you open to answer "and what does 2022 look like".
        for era in sorted({r["era"] for r in rows}):
            era_rows = [r for r in rows if r["era"] == era]
            pdf_path = os.path.join(args.output, f"yields_{era}.pdf")
            with PdfPages(pdf_path) as pdf:
                n = draw_pages(era_rows, pdf, cfg, knobs, param_name, note)
            print(f"Wrote {pdf_path}  ({n} pages)")


if __name__ == "__main__":
    main()

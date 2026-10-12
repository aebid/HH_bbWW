#!/usr/bin/env python3
"""
Permutation feature importance of a trained pDNN, per regime and mass hypothesis.

For every parity cycle, the cycle's ONNX models are evaluated on the validation parity (never
used in training) at several mass hypotheses. Each input feature, and each group of strongly
correlated features (|r| > --corr-threshold in background; a correlated partner otherwise
hides the information of a shuffled feature), is shuffled in turn and the change in
  - signal-vs-background ROC AUC (signal at the hypothesis mass vs all backgrounds, physical
    weights), absolute and relative to the AUC deficit 1 - AUC, and
  - class-balanced cross-entropy (each output class with equal total weight)
is recorded. Mass-scaled features (*_scaled) are recomputed from their shuffled base.
Results are averaged over the cycles, with the spread between cycles as the uncertainty.
By default every mass in the config's signal_mass_points is scanned; importance_evolution_*.png
show the top inputs and the correlated groups vs mass.

Usage:
    python3 feature_importance.py --config model_configs/v2610_pDNN_v1.yaml [--folds 0]
    python3 feature_importance.py --config ... --summarize   # combine per-fold JSONs
"""

import os
import json
import argparse
import yaml
import numpy as np
import pandas as pd
import onnxruntime as ort
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_auc_score

from src.model_helper import (
    ParametricDNN,
    eval_parity_expr,
    get_class_setup,
    load_physical_fold,
)


class OnnxAsTorch:
    """
    Runs an exported pDNN on the GPU with torch: rebuilds ParametricDNN from the ONNX
    initializers (input standardization 'mean' / 'scale' and the 'model.*' weights).
    onnxruntime's CUDA provider fails on the BatchNormalization nodes
    (CUDNN_STATUS_NOT_SUPPORTED).
    """

    def __init__(self, path, device):
        import onnx
        from onnx import numpy_helper

        weights = {
            i.name: torch.tensor(numpy_helper.to_array(i).copy())
            for i in onnx.load(path).graph.initializer
        }
        linear = sorted(
            (
                k
                for k in weights
                if k.startswith("model.network.") and weights[k].dim() == 2
            ),
            key=lambda k: int(k.split(".")[2]),
        )
        hidden = [weights[k].shape[0] for k in linear]
        n_in = weights[linear[0]].shape[1]
        n_out = weights["model.output_layer.weight"].shape[0]
        self.model = ParametricDNN(n_in, n_out, hidden_dims=hidden)
        self.model.load_state_dict(
            {
                k[len("model.") :]: v
                for k, v in weights.items()
                if k.startswith("model.")
            },
            strict=False,
        )
        self.model = self.model.to(device).eval()
        self.mean = weights["mean"].to(device)
        self.scale = weights["scale"].to(device)
        self.device = device

    def __call__(self, X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 65536):
                x = torch.tensor(X[i : i + 65536], device=self.device)
                out.append(
                    torch.softmax(self.model((x - self.mean) / self.scale), dim=1)
                )
        return torch.cat(out).double().cpu().numpy()


class OnnxRuntime:
    def __init__(self, path):
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def __call__(self, X):
        name = self.session.get_inputs()[0].name
        return np.concatenate(
            [
                self.session.run(None, {name: X[i : i + 65536]})[0]
                for i in range(0, len(X), 65536)
            ]
        ).astype(np.float64)


def load_model(path):
    if torch.cuda.is_available():
        return OnnxAsTorch(path, torch.device("cuda"))
    return OnnxRuntime(path)


MASS_BANDS = {"low": (250, 400), "mid": (450, 900), "high": (1000, 2000)}


def correlated_groups(X, features, threshold):
    """Connected components of |Pearson r| > threshold among the features (background events)."""
    names = [f for f in features if f != "X_mass"]
    corr = np.abs(np.corrcoef(X[:, [features.index(f) for f in names]], rowvar=False))
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if corr[i, j] > threshold:
                parent[find(i)] = find(j)
    groups = {}
    for i, name in enumerate(names):
        groups.setdefault(find(i), []).append(name)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def build_matrix(df, features, mass):
    """Raw (unscaled) input matrix at a mass hypothesis; *_scaled columns hold base * mass."""
    cols = []
    for f in features:
        if f == "X_mass":
            cols.append(np.full(len(df), mass, dtype=np.float32))
        elif f.endswith("_scaled"):
            cols.append(df[f.replace("_scaled", "")].values.astype(np.float32) * mass)
        else:
            cols.append(df[f].values.astype(np.float32))
    return np.stack(cols, axis=1)


def predict(model, X):
    return model(X)


def metrics(probs, y, w_phys, w_bal):
    p = np.clip(probs[:, 0], 1e-15, 1 - 1e-15)
    score = np.log(p) - np.log1p(-p)
    is_sig = y == 0
    sw = np.where(is_sig, w_phys / w_phys[is_sig].sum(), w_phys / w_phys[~is_sig].sum())
    auc = roc_auc_score(is_sig, score, sample_weight=sw)
    ce = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1))
    loss = np.sum(w_bal * ce) / np.sum(w_bal)
    return auc, loss


def run_fold(cfg, i_fold, args, rng):
    class_setup = get_class_setup(cfg)
    n_classes = len(class_setup["class_names"])
    feature_map = {
        r: cfg["common_features"] + cfg["regime_features"].get(r, [])
        for r in cfg["regimes"]
    }
    model_name_fmt = cfg.get("model_name", "pdnn_model_{regime}_nparity{nParity}.onnx")
    train_parity = eval_parity_expr(cfg["train_parity"]["index"], i_fold)
    val_parity = eval_parity_expr(cfg["val_parity"]["index"], i_fold)
    fold_dir = os.path.join(cfg["output_folder"], f"nParity{train_parity}_validation")
    path = os.path.join(cfg["input_folder"], f"nParity{val_parity}_Merged.root")
    df = load_physical_fold(path, None, feature_map, cfg["extra_variables"], cfg)
    df = df[df["class_targets_multiclass"] >= 0]
    # The boosted background is small: keep more of it (default all) for a stable AUC
    bkg_fraction = np.where(
        df["boosted"].values == 1, args.boosted_bkg_fraction, args.bkg_fraction
    )
    keep = (df["class_targets_multiclass"].values == 0) | (
        rng.random(len(df)) < bkg_fraction
    )
    df = df[keep].reset_index(drop=True)
    print(
        f"cycle {i_fold}: validation parity {val_parity}, {len(df)} events", flush=True
    )

    out = {"cycle": i_fold, "regimes": {}}
    for regime in cfg["regimes"]:
        features = feature_map[regime]
        session = load_model(
            os.path.join(fold_dir, model_name_fmt.format(nParity=i_fold, regime=regime))
        )
        d = df[(df["boosted"] == 1) if regime == "boosted" else (df["boosted"] == 0)]
        y_all = d["class_targets_multiclass"].values
        w_all = np.clip(d["weight_Central"].values.astype(np.float64), 0, None)
        X_bkg = build_matrix(d[y_all > 0], features, 1000)
        groups = correlated_groups(X_bkg, features, args.corr_threshold)
        items = [(f, [f]) for f in features if f != "X_mass"] + [
            ("group:" + "+".join(g), g) for g in groups
        ]
        reg_out = {"groups": groups, "masses": {}}
        for mass in args.masses:
            sel = ((y_all == 0) & (d["X_mass"].values == mass)) | (y_all > 0)
            n_sig = int(((y_all == 0) & (d["X_mass"].values == mass)).sum())
            if n_sig < args.min_signal:
                print(f"  {regime} M{mass}: {n_sig} signal events, skipped", flush=True)
                continue
            dm = d[sel]
            y = y_all[sel]
            w = w_all[sel]
            # Class-balanced weights for the loss, as in training
            w_bal = w.copy()
            for c in range(n_classes):
                if w_bal[y == c].sum() > 0:
                    w_bal[y == c] /= w_bal[y == c].sum()
            X = build_matrix(dm, features, mass)
            auc0, loss0 = metrics(predict(session, X), y, w, w_bal)
            res = {"auc": auc0, "loss": loss0, "n_signal": n_sig, "features": {}}
            for name, members in items:
                cols = [features.index(f) for f in members]
                d_auc, d_loss = [], []
                for _ in range(args.repeats):
                    Xp = X.copy()
                    perm = rng.permutation(len(X))
                    Xp[:, cols] = X[perm][:, cols]
                    auc, loss = metrics(predict(session, Xp), y, w, w_bal)
                    d_auc.append(auc0 - auc)
                    d_loss.append(loss - loss0)
                res["features"][name] = {
                    "d_auc": float(np.mean(d_auc)),
                    "d_auc_rel": float(np.mean(d_auc) / max(1 - auc0, 1e-6)),
                    "d_loss": float(np.mean(d_loss)),
                }
            reg_out["masses"][str(mass)] = res
            top = sorted(res["features"].items(), key=lambda kv: -kv[1]["d_auc_rel"])[
                :3
            ]
            print(
                f"  {regime} M{mass}: AUC {auc0:.4f}, top "
                + ", ".join(f"{k} {v['d_auc_rel']:.2f}" for k, v in top),
                flush=True,
            )
        out["regimes"][regime] = reg_out
    return out


def plot_evolution(d, ranked, regime, out_dir, top_n):
    """
    Line plots of the most important inputs vs mass hypothesis, one per metric: single inputs
    (top) and correlated groups (bottom). Points are the mean over cycles, bands its uncertainty
    (spread between cycles / sqrt(n cycles)).
    """
    masses = sorted(d["mass"].unique())
    singles = [f for f in ranked if not f.startswith("group:")][:top_n]
    groups = [f for f in ranked if f.startswith("group:")]
    g = d.groupby(["feature", "mass"])
    metrics_ = [
        ("d_auc_rel", "relative AUC loss\n(AUC - AUC_shuffled) / (1 - AUC)", ""),
        ("d_loss", "increase of class-balanced\ncross-entropy", "_loss"),
    ]
    for col, label, tag in metrics_:
        mean = g[col].mean()
        err = (g[col].std() / np.sqrt(g[col].count())).fillna(0)
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(13, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1]},
        )
        for ax, names, title in [
            (axes[0], singles, f"top {len(singles)} single inputs"),
            (axes[1], groups, "correlated groups (|r| > threshold), shuffled together"),
        ]:
            cmap = plt.cm.tab20 if len(names) > 10 else plt.cm.tab10
            for i, name in enumerate(names):
                m = mean.loc[name].reindex(masses)
                e = err.loc[name].reindex(masses)
                color = cmap(i % cmap.N)
                ax.plot(
                    masses,
                    m,
                    marker="o",
                    markersize=3.5,
                    linewidth=1.8,
                    color=color,
                    label=name.replace("group:", "").replace("+", " + "),
                )
                ax.fill_between(masses, m - e, m + e, color=color, alpha=0.18, lw=0)
            if col == "d_auc_rel":
                ax.set_yscale("log")
                vals = mean.loc[names].values
                floor = 0.5 * vals[vals > 0].min() if (vals > 0).any() else 1e-2
                ax.set_ylim(bottom=max(1e-2, floor))
            else:
                ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel(label)
            ax.set_title(title, fontsize=11, loc="left")
            ax.grid(True, which="major", linestyle="--", alpha=0.4)
            ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8.5)
        fig.suptitle(
            f"{regime.upper()}: permutation importance vs mass hypothesis "
            "(band: uncertainty of the mean over cycles)",
            fontsize=13,
            fontweight="bold",
        )
        axes[1].set_xlabel("$m_X$ [GeV]", fontsize=12)
        # Every mass as a minor tick, labels on a subset that does not overlap
        labelled = [m for m in masses if m in (250, 300, 350, 400) or m % 100 == 0]
        axes[1].set_xticks(labelled)
        axes[1].set_xticks(masses, minor=True)
        axes[1].tick_params(axis="x", rotation=60, labelsize=9)
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"importance_evolution{tag}_{regime}.png"), dpi=200
        )
        plt.close(fig)


def summarize(cfg, out_dir, top_n=12):
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("importance_cycle"))
    folds = [json.load(open(os.path.join(out_dir, f))) for f in files]
    rows = []
    for fold in folds:
        for regime, reg in fold["regimes"].items():
            for mass, res in reg["masses"].items():
                for name, v in res["features"].items():
                    rows.append(
                        {
                            "cycle": fold["cycle"],
                            "regime": regime,
                            "mass": int(mass),
                            "feature": name,
                            **v,
                        }
                    )
    df = pd.DataFrame(rows)

    def band(m):
        for b, (lo, hi) in MASS_BANDS.items():
            if lo <= m <= hi:
                return b
        return "other"

    df["band"] = df["mass"].map(band)
    for regime in df["regime"].unique():
        d = df[df["regime"] == regime]
        # mean over masses within each cycle, then mean and spread over cycles
        per_cycle = d.groupby(["feature", "cycle"])[
            ["d_auc_rel", "d_auc", "d_loss"]
        ].mean()
        summary = per_cycle.groupby("feature").agg(["mean", "std"])
        summary.columns = [f"{a}_{b}" for a, b in summary.columns]
        for b in MASS_BANDS:
            summary[f"d_auc_rel_{b}"] = (
                d[d["band"] == b].groupby("feature")["d_auc_rel"].mean()
            )
        summary = summary.sort_values("d_auc_rel_mean", ascending=False)
        summary.to_csv(os.path.join(out_dir, f"importance_{regime}.csv"))
        print(
            f"\n=== {regime}: relative AUC loss when shuffled (mean over masses and cycles)"
        )
        print(
            summary[
                [
                    "d_auc_rel_mean",
                    "d_auc_rel_std",
                    "d_auc_rel_low",
                    "d_auc_rel_mid",
                    "d_auc_rel_high",
                    "d_loss_mean",
                ]
            ]
            .round(4)
            .to_string()
        )

        # Bar chart with cycle spread
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(summary) + 1.5))
        s = summary.iloc[::-1]
        colors = [
            "darkorange" if f.startswith("group:") else "royalblue" for f in s.index
        ]
        ax.barh(
            range(len(s)), s["d_auc_rel_mean"], xerr=s["d_auc_rel_std"], color=colors
        )
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(
            [f if len(f) < 60 else f[:57] + "..." for f in s.index], fontsize=7
        )
        ax.set_xlabel(
            "relative AUC loss when shuffled, (AUC - AUC_shuffled) / (1 - AUC)"
        )
        ax.set_title(f"{regime}: permutation importance (orange: correlated groups)")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"importance_{regime}.png"), dpi=150)
        plt.close(fig)

        # Heat map: feature x mass
        hm = (
            d.groupby(["feature", "mass"])["d_auc_rel"]
            .mean()
            .unstack()
            .loc[summary.index]
        )
        fig, ax = plt.subplots(figsize=(1 + 0.6 * hm.shape[1], 0.3 * hm.shape[0] + 1.5))
        # Log colour scale: values span 0.01 to > 100; anything below 0.01 (or negative) at the floor
        im = ax.imshow(
            np.clip(hm.values, 1e-2, None),
            aspect="auto",
            cmap="viridis",
            norm=LogNorm(vmin=1e-2, vmax=max(np.nanmax(hm.values), 1e-1)),
        )
        ax.set_xticks(range(hm.shape[1]))
        ax.set_xticklabels(hm.columns, rotation=45)
        ax.set_yticks(range(hm.shape[0]))
        ax.set_yticklabels(
            [f if len(f) < 60 else f[:57] + "..." for f in hm.index], fontsize=7
        )
        ax.set_xlabel("$m_X$ [GeV]")
        plt.colorbar(im, label="relative AUC loss (log, floor 0.01)")
        ax.set_title(f"{regime}: importance vs mass")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"importance_vs_mass_{regime}.png"), dpi=150)
        plt.close(fig)

        plot_evolution(d, list(summary.index), regime, out_dir, top_n)
    print(f"\n--> Wrote importance_*.csv / .png in {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Permutation feature importance of a pDNN"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--masses",
        type=int,
        nargs="+",
        default=None,
        help="Mass hypotheses (default: the config's signal_mass_points)",
    )
    parser.add_argument(
        "--top-n", type=int, default=12, help="Inputs in the evolution line plot"
    )
    parser.add_argument(
        "--bkg-fraction", type=float, default=0.1, help="Background subsample"
    )
    parser.add_argument(
        "--boosted-bkg-fraction",
        type=float,
        default=1.0,
        help="Background subsample in the boosted regime",
    )
    parser.add_argument("--repeats", type=int, default=2, help="Shuffles per feature")
    parser.add_argument("--corr-threshold", type=float, default=0.8)
    parser.add_argument(
        "--min-signal",
        type=int,
        default=200,
        help="Skip masses with fewer signal events",
    )
    parser.add_argument(
        "--summarize", action="store_true", help="Only combine the per-cycle JSONs"
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = args.output_dir or os.path.join(
        cfg["output_folder"], "feature_importance"
    )
    os.makedirs(out_dir, exist_ok=True)
    if args.masses is None:
        args.masses = cfg["signal_mass_points"]
    if not args.summarize:
        folds = args.folds if args.folds is not None else range(cfg.get("nParity", 4))
        for i_fold in folds:
            rng = np.random.default_rng(1000 + i_fold)
            res = run_fold(cfg, i_fold, args, rng)
            with open(
                os.path.join(out_dir, f"importance_cycle{i_fold}.json"), "w"
            ) as f:
                json.dump(res, f, indent=1)
    summarize(cfg, out_dir, args.top_n)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import gc
import yaml
import numpy as np
import awkward as ak
import onnxruntime as ort
import psutil
import yaml
import os
import ROOT
import FLAF.Common.OnnxSession as OnnxSession
import FLAF.Common.Utilities as Utilities
import Analysis.hh_bbww as analysis
from contextlib import contextmanager


class DNNProducer:
    def __init__(self, cfg: dict, payload_name: str, era: str):
        """
        Inference producer for Parametric DNN models with NaN/Inf diagnostics.

        Parameters
        ----------
        cfg : dict
            Master payload configuration dictionary containing channel definitions and columns.
        payload_name : str
            Prefix used for output branches in the snapshot array.
        era : str
            FLAF era (e.g. "Run3_2022").
        """
        self.cfg = cfg
        self.payload_name = payload_name
        self.era = era

        # Ensure analysis module path is loaded
        analysis_path = os.environ.get("ANALYSIS_PATH", ".")
        if analysis_path not in sys.path:
            sys.path.append(analysis_path)

        self.channels = ["DL_Resolved", "DL_Boosted", "SL_Resolved", "SL_Boosted"]
        self.cfg_dict = {ch: self.cfg.get(ch, None) for ch in self.channels}

        self.target_columns = self.cfg.get("columns", [])
        self.target_masses = set()
        self.classes_to_save = set()

        # Parse requested target masses and class names from the column list
        for col in self.target_columns:
            parts = col.split("_", 1)
            if len(parts) == 2 and parts[0].startswith("M"):
                try:
                    self.target_masses.add(float(parts[0][1:]))
                    self.classes_to_save.add(parts[1])
                except ValueError:
                    pass

        self.target_masses = sorted(list(self.target_masses))
        self.dnnConfig = {}
        load_features = set()

        # Build channel-specific configurations and locate ONNX models
        for channel, ch_cfg in self.cfg_dict.items():
            if ch_cfg is None:
                continue

            version = ch_cfg.get("version", "")
            dnnFolder = os.path.join(analysis_path, "config", "DNN", version)
            config_file = os.path.join(dnnFolder, "dnn_config.yaml")

            if not os.path.exists(config_file):
                print(
                    f"[WARNING] Config not found: '{config_file}'. Skipping channel {channel}."
                )
                continue

            with open(config_file, "r") as f:
                model_config = yaml.safe_load(f)

            nParity = model_config.get("nParity", 4)
            features = model_config.get("features", [])
            regime = "boosted" if "Boosted" in channel else "resolved"
            model_name = model_config.get(
                "model_name", "pdnn_model_{regime}_nparity{fold}.onnx"
            )

            # `{fold}` is the cross-validation fold index (0..nParity-1). It is
            # deliberately not called `{nParity}`: nParity is the *number* of
            # folds, and conflating the two is what allows a fold index to be
            # silently substituted where a fold count is meant.
            model_paths = []
            for fold_idx in range(nParity):
                onnx_path = os.path.join(
                    dnnFolder,
                    model_name.format(fold=fold_idx, nParity=nParity, regime=regime),
                )
                model_paths.append(onnx_path)

            model_config["model_paths"] = model_paths
            self.dnnConfig[channel] = model_config

            load_features.update(features)

        self.vars_to_save = set()
        for f in load_features:
            if f == "X_mass":
                continue
            if f.endswith("_scaled"):
                # Dynamically determine unscaled base branch
                self.vars_to_save.add(f.replace("_scaled", ""))
            else:
                self.vars_to_save.add(f)

        mandatory_branches = {"FullEventId", "event", "SL", "DL", "boosted"}
        self.vars_to_save.update(mandatory_branches)

        print(f"[pDNNProducer] Initialized for payload '{self.payload_name}'.")
        print(f" -> Configured Masses : {self.target_masses}")
        print(f" -> Output Classes    : {self.classes_to_save}")
        print(f" -> Input Variables   : {len(self.vars_to_save)} branches")

    def run(self, array: ak.Array) -> ak.Array:
        """
        Executes inference on an input Awkward Array and returns a pruned payload array.
        """
        print(f"[pDNNProducer] Processing batch of {len(array)} events...")

        # 1. Apply ONNX evaluation across configured channels/models
        array = self.ApplyDNN(array)

        # 2. Select prediction per event based on SL/DL and Boosted/Resolved topology
        array = self.selectDNN(array)

        # 3. Assemble and prefix final output columns
        out_fields = {}
        if "FullEventId" in array.fields:
            out_fields["FullEventId"] = array["FullEventId"]

        for col in self.target_columns:
            target_name = f"{self.payload_name}_{col}"
            if col in array.fields:
                out_fields[target_name] = array[col]
            else:
                print(
                    f"[WARNING] Column '{col}' missing in predictions! Filling with zeros."
                )
                out_fields[target_name] = np.zeros(len(array), dtype=np.float32)

        return ak.Array(out_fields)

    def _audit_and_sanitize_inputs(
        self, X_mat: np.ndarray, features: list[str], channel: str, mass: float
    ) -> np.ndarray:
        """
        Scans input feature matrix for NaN/Inf, reports offending variables, and sanitizes array.
        """
        nan_mask = np.isnan(X_mat)
        inf_mask = np.isinf(X_mat)

        if nan_mask.any() or inf_mask.any():
            print("\n" + "!" * 80)
            print(
                f"[pDNNProducer DIAGNOSTIC ERROR] Invalid values detected in Input Features!"
            )
            print(f" -> Channel: {channel} | Mass Point: {mass} GeV")

            for col_idx, feat_name in enumerate(features):
                col_nans = nan_mask[:, col_idx].sum()
                col_infs = inf_mask[:, col_idx].sum()

                if col_nans > 0 or col_infs > 0:
                    bad_rows = np.where(nan_mask[:, col_idx] | inf_mask[:, col_idx])[0]
                    sample_events = bad_rows[:5]  # Display up to first 5 row indices
                    print(
                        f"   [!] Feature '{feat_name}' (col {col_idx}): {col_nans} NaNs, {col_infs} Infs. (Example row indices: {sample_events})"
                    )

            print(
                " -> Automating sanitization: Replacing NaNs/Infs with 0.0 to prevent model failure."
            )
            print("!" * 80 + "\n")

            X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)

        return X_mat

    def _audit_predictions(
        self, array: np.ndarray, stage_label: str, channel: str, mass: float
    ) -> None:
        """
        Checks intermediate predictions and logits for NaN or Inf values.
        """
        if np.isnan(array).any() or np.isinf(array).any():
            nan_count = np.isnan(array).sum()
            inf_count = np.isinf(array).sum()
            print("\n" + "!" * 80)
            print(
                f"[pDNNProducer DIAGNOSTIC ERROR] Invalid outputs found at stage '{stage_label}'!"
            )
            print(f" -> Channel: {channel} | Mass: {mass} GeV")
            print(f" -> NaNs count: {nan_count} | Infs count: {inf_count}")
            print("!" * 80 + "\n")

    def _compute_fold_masks(
        self,
        dnnConfig: dict,
        channel: str,
        event_number: np.ndarray,
        nParity: int,
    ) -> list[np.ndarray]:
        """
        Builds the per-fold application masks, one boolean array per fold.

        Fold `f` is applied to events satisfying
        ``event_number % nParity == (f + offset) % nParity``, where ``offset``
        comes from the ``app_parity`` section of the model configuration -- the
        same bucket training assigns to model `f` for that split. The masks
        are required to partition the events: every event must be claimed by
        exactly one fold, or the ensemble below would silently drop or
        double-count it.
        """
        app_parity_cfg = dnnConfig.get("app_parity")
        if app_parity_cfg is None:
            raise RuntimeError(
                f"[pDNNProducer] Missing required 'app_parity' section in channel '{channel}' configuration!"
            )

        offset = (
            app_parity_cfg.get("offset")
            if isinstance(app_parity_cfg, dict)
            else app_parity_cfg
        )

        if isinstance(offset, bool) or not isinstance(offset, int):
            # Legacy configurations expressed the fold assignment as a Python
            # snippet under 'func'. Support it, but evaluate it with no builtins
            # and only the fold variables in scope -- a model directory is data,
            # and data must not be able to execute arbitrary code.
            legacy_expr = (
                app_parity_cfg.get("func")
                if isinstance(app_parity_cfg, dict)
                else app_parity_cfg
            )
            if not isinstance(legacy_expr, str):
                raise RuntimeError(
                    f"[pDNNProducer] Channel '{channel}': 'app_parity' must provide an integer "
                    f"'offset' (or a legacy 'func' string); got {app_parity_cfg!r}."
                )

            print(
                f"[WARNING] Channel '{channel}': 'app_parity' uses the deprecated 'func' "
                "expression. Replace it with an integer 'offset' -- see the comment in "
                "dnn_config.yaml."
            )

            masks = []
            for fold_idx in range(nParity):
                try:
                    formatted = legacy_expr.format(fold=fold_idx, nParity=nParity)
                except (KeyError, IndexError) as exc:
                    raise RuntimeError(
                        f"[pDNNProducer] Channel '{channel}': cannot substitute fold index into "
                        f"app_parity func {legacy_expr!r} ({exc!r}). It must contain '{{fold}}'."
                    ) from exc

                value = eval(
                    formatted,
                    {"__builtins__": {}},
                    {"event_number": event_number, "nParity": nParity},
                )
                masks.append(np.asarray(value) == 0)
        else:
            bucket = event_number % nParity
            masks = [
                bucket == (fold_idx + offset) % nParity for fold_idx in range(nParity)
            ]

        # The folds must tile the events exactly once each. This single check
        # catches a wrong modulus, a fold offset that does not vary with the
        # fold index, and an event_number that does not match the training one.
        coverage = np.sum(np.stack(masks, axis=0), axis=0)
        if not np.all(coverage == 1):
            unclaimed = int(np.count_nonzero(coverage == 0))
            multiclaimed = int(np.count_nonzero(coverage > 1))
            raise RuntimeError(
                f"[pDNNProducer] Channel '{channel}': the {nParity} application folds do not "
                f"partition the events ({unclaimed} events claimed by no fold, {multiclaimed} "
                f"claimed by more than one). Check 'nParity' and the 'app_parity' offset in the "
                f"model configuration."
            )

        return masks

    def ApplyDNN(self, branches: ak.Array) -> ak.Array:
        """
        Evaluates ONNX inference sessions across channels and parametric mass hypotheses.
        """
        nEvents = len(branches)
        if nEvents == 0:
            return branches

        # The fold assignment must be computed from the *same* quantity that
        # defined the training folds. There is deliberately no fallback to
        # FullEventId: that is a packed (crc16(dataset), crc16(file), entry)
        # identifier with no relation to the event number, so falling back to it
        # would still produce a valid-looking partition while scoring events
        # with folds that were trained on them.
        if "event" not in branches.fields:
            raise RuntimeError(
                "[pDNNProducer] Required branch 'event' is missing from the input array; "
                "it defines the cross-validation folds and has no safe substitute."
            )

        event_number = np.asarray(branches.event)
        if not np.issubdtype(event_number.dtype, np.integer):
            raise RuntimeError(
                f"[pDNNProducer] Branch 'event' has non-integer dtype '{event_number.dtype}'. "
                "Fold assignment uses modular arithmetic and needs exact integers "
                "(float64 silently loses the low bits of large event numbers)."
            )

        output_fields = {}

        for channel, dnnConfig in self.dnnConfig.items():
            if not dnnConfig:
                continue

            features = dnnConfig.get("features", [])
            class_names = dnnConfig.get("class_names", ["Signal", "TT", "Other"])
            nClasses = len(class_names)
            nParity = dnnConfig.get("nParity", 4)
            model_paths = dnnConfig.get("model_paths", [])

            # Fold assignment depends only on the event number, not on the mass
            # hypothesis, so it is computed once per channel rather than once
            # per (mass, fold) pair.
            fold_masks = self._compute_fold_masks(
                dnnConfig, channel, event_number, nParity
            )

            # Prepare ONNX Inference Sessions. Every fold must be present: the
            # fold masks partition the events, so a single missing model leaves
            # its events with all-zero probabilities, which the logit transform
            # below turns into a finite, plausible-looking score (~-16.1) rather
            # than an obvious failure.
            missing = [path for path in model_paths if not os.path.exists(path)]
            if missing or len(model_paths) != nParity:
                raise RuntimeError(
                    f"[pDNNProducer] Channel '{channel}': expected {nParity} ONNX models, "
                    f"found {len(model_paths) - len(missing)}. Missing: {missing}"
                )

            sessions = []
            for p_idx, path in enumerate(model_paths):
                sess = OnnxSession.make_session(path)
                sessions.append((p_idx, sess))

            for mass in self.target_masses:
                # Build feature inputs per event
                feature_cols = []
                for feat in features:
                    if feat == "X_mass":
                        feature_cols.append(np.full(nEvents, mass, dtype=np.float32))
                    elif feat.endswith("_scaled"):
                        base_feat = feat.replace("_scaled", "")
                        if base_feat in branches.fields:
                            raw_val = np.asarray(branches[base_feat], dtype=np.float32)
                        else:
                            print(
                                f"[WARNING] Missing base feature '{base_feat}' for '{feat}'! Filling with 0.0."
                            )
                            raw_val = np.zeros(nEvents, dtype=np.float32)
                        feature_cols.append(raw_val * mass)
                    elif feat in branches.fields:
                        feature_cols.append(
                            np.asarray(branches[feat], dtype=np.float32)
                        )
                    else:
                        print(
                            f"[WARNING] Missing feature branch '{feat}' in input array! Filling with 0.0."
                        )
                        feature_cols.append(np.zeros(nEvents, dtype=np.float32))

                X_mat = np.column_stack(feature_cols).astype(np.float32)

                # AUDIT STEP 1: Audit and clean input feature matrix
                X_mat = self._audit_and_sanitize_inputs(X_mat, features, channel, mass)

                all_predictions = np.zeros((nEvents, nClasses), dtype=np.float32)

                # Run inference fold by fold. Each fold is evaluated only on the
                # events it owns, rather than on all events followed by masking,
                # which costs nParity times less ONNX work.
                for parity_idx, sess in sessions:
                    app_mask = fold_masks[parity_idx]
                    if not app_mask.any():
                        continue

                    input_name = sess.get_inputs()[0].name
                    preds = sess.run(None, {input_name: X_mat[app_mask]})[0]

                    # AUDIT STEP 2: Audit raw ONNX model prediction outputs
                    self._audit_predictions(
                        preds, f"Raw ONNX fold {parity_idx}", channel, mass
                    )

                    all_predictions[app_mask] = preds

                # AUDIT STEP 3: Audit ensemble probabilities before logit transformation
                self._audit_predictions(
                    all_predictions, "Ensembled Probabilities", channel, mass
                )

                # The models emit a softmax over the classes, so every event must
                # carry exactly one fold's worth of probability. A row summing to
                # 0 means an event was scored by no fold; anything else means the
                # ensemble is not the per-event probability it is treated as below.
                prob_sums = all_predictions.sum(axis=1)
                if not np.allclose(prob_sums, 1.0, atol=1e-4):
                    bad = np.flatnonzero(~np.isclose(prob_sums, 1.0, atol=1e-4))
                    raise RuntimeError(
                        f"[pDNNProducer] Channel '{channel}', mass {mass}: {bad.size} events have "
                        f"class probabilities summing to something other than 1 "
                        f"(e.g. rows {bad[:5].tolist()} -> {prob_sums[bad[:5]].tolist()}). "
                        "The fold ensemble is incomplete or the models are not emitting probabilities."
                    )

                # Transform raw probabilities to logit scores (numerically safe subtraction)
                probs_clipped = np.clip(all_predictions, 1e-7, 1.0 - 1e-7)
                logits = np.log(probs_clipped) - np.log(1.0 - probs_clipped)

                # AUDIT STEP 4: Audit final logit values
                self._audit_predictions(logits, "Logit Calculation", channel, mass)

                mass_str = f"{mass:.0f}" if mass.is_integer() else f"{mass}"
                for class_idx, class_name in enumerate(class_names):
                    field_name = f"{channel}_M{mass_str}_{class_name}"
                    output_fields[field_name] = logits[:, class_idx].copy()

        # Attach intermediate outputs to array
        for field_name, values in output_fields.items():
            branches[field_name] = values

        gc.collect()
        return branches

    def selectDNN(self, branches: ak.Array) -> ak.Array:
        """
        Combines topology and lepton channel predictions into unified branches (e.g., M300_Signal).
        """
        nEvents = len(branches)
        output_fields = {}

        sl_mask = (
            np.asarray(branches.SL, dtype=bool)
            if "SL" in branches.fields
            else np.zeros(nEvents, dtype=bool)
        )
        boosted_mask = (
            np.asarray(branches.boosted, dtype=bool)
            if "boosted" in branches.fields
            else np.zeros(nEvents, dtype=bool)
        )

        for mass in self.target_masses:
            mass_str = f"{mass:.0f}" if mass.is_integer() else f"{mass}"
            for class_name in self.classes_to_save:
                target_field = f"M{mass_str}_{class_name}"

                sl_boosted = (
                    np.asarray(branches[f"SL_Boosted_{target_field}"])
                    if f"SL_Boosted_{target_field}" in branches.fields
                    else np.zeros(nEvents, dtype=np.float32)
                )
                sl_resolved = (
                    np.asarray(branches[f"SL_Resolved_{target_field}"])
                    if f"SL_Resolved_{target_field}" in branches.fields
                    else np.zeros(nEvents, dtype=np.float32)
                )
                dl_boosted = (
                    np.asarray(branches[f"DL_Boosted_{target_field}"])
                    if f"DL_Boosted_{target_field}" in branches.fields
                    else np.zeros(nEvents, dtype=np.float32)
                )
                dl_resolved = (
                    np.asarray(branches[f"DL_Resolved_{target_field}"])
                    if f"DL_Resolved_{target_field}" in branches.fields
                    else np.zeros(nEvents, dtype=np.float32)
                )

                sl_sel = np.where(boosted_mask, sl_boosted, sl_resolved)
                dl_sel = np.where(boosted_mask, dl_boosted, dl_resolved)

                output_fields[target_field] = np.where(sl_mask, sl_sel, dl_sel)

        for field_name, values in output_fields.items():
            branches[field_name] = values

        return branches


def make_session(path):
    """
    Helper function to create onnx sessions with options
    """
    so = OnnxSession.session_options()
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    return ort.InferenceSession(
        path, sess_options=so, providers=["CPUExecutionProvider"]
    )


@contextmanager
def onnx_session(path):
    """
    RAII session helper
    """
    sess = make_session(path)
    try:
        yield sess
    finally:
        del sess
        gc.collect()


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def sigmoid(logits):
    # exp overflows to inf for very negative logits, which correctly gives 0
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-logits))


# what a model returns, what the producer saves, and how logits become probabilities
DNN_OUTPUT_TYPES = ["logit", "probability"]
DNN_SAVE_TYPES = ["logit", "probability", "recomputed_logit"]
DNN_ACTIVATIONS = {"softmax": softmax, "sigmoid": sigmoid}


def parse_output_spec(label, spec, needs_signal_node):
    """
    Validates the output description of one model:
    num_nodes, type (logit | probability), activation (softmax | sigmoid),
    save_as (logit | probability | recomputed_logit) and, for the binary model
    with more than one node, signal_node.
    """
    if spec is None:
        raise RuntimeError(f"{label}: output specification is missing.")
    for key in ["num_nodes", "type", "activation", "save_as"]:
        if key not in spec:
            raise RuntimeError(f"{label}: output specification has no '{key}'.")

    num_nodes = spec["num_nodes"]
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 1:
        raise RuntimeError(f"{label}: num_nodes must be a positive integer.")
    if spec["type"] not in DNN_OUTPUT_TYPES:
        raise RuntimeError(f"{label}: type must be one of {DNN_OUTPUT_TYPES}.")
    if spec["save_as"] not in DNN_SAVE_TYPES:
        raise RuntimeError(f"{label}: save_as must be one of {DNN_SAVE_TYPES}.")
    if spec["activation"] not in DNN_ACTIVATIONS:
        raise RuntimeError(
            f"{label}: activation must be one of {list(DNN_ACTIVATIONS)}."
        )
    if spec["activation"] == "softmax" and num_nodes < 2:
        raise RuntimeError(f"{label}: softmax needs at least 2 output nodes.")
    if spec["type"] == "probability" and spec["save_as"] == "logit":
        raise RuntimeError(
            f"{label}: the model returns probabilities, its raw logits cannot be "
            "saved. Use save_as: recomputed_logit."
        )

    parsed = dict(spec)
    if needs_signal_node:
        if num_nodes == 1:
            parsed["signal_node"] = spec.get("signal_node", 0)
        elif "signal_node" not in spec:
            raise RuntimeError(
                f"{label}: signal_node is required for {num_nodes} output nodes."
            )
        if not 0 <= parsed["signal_node"] < num_nodes:
            raise RuntimeError(f"{label}: signal_node is out of range.")
    return parsed


def convert_dnn_output(scores, spec, clip_eps):
    """
    Converts the model output (spec["type"]) to the saved type (spec["save_as"]).
    recomputed_logit is log(p / (1 - p)) of every node, p clipped to
    [clip_eps, 1 - clip_eps].
    """
    scores = scores.astype(np.float64)
    if spec["save_as"] == spec["type"]:
        return scores

    if spec["type"] == "logit":
        probabilities = DNN_ACTIVATIONS[spec["activation"]](scores)
    else:
        probabilities = scores

    if spec["save_as"] == "probability":
        return probabilities

    clipped = np.clip(probabilities, clip_eps, 1.0 - clip_eps)
    return np.log(clipped / (1.0 - clipped))


def model_output(sess, inputs, spec, label):
    """Runs the model and checks that its output has spec["num_nodes"] columns."""
    sess_in = sess.get_inputs()[0].name
    scores = sess.run(None, {sess_in: inputs})[0]
    scores = scores.reshape(len(scores), -1)
    if scores.shape[1] != spec["num_nodes"]:
        raise RuntimeError(
            f"{label}: model returned {scores.shape[1]} output node(s), "
            f"num_nodes is {spec['num_nodes']}."
        )
    return scores


class TwoStageDNNProducer:
    def __init__(self, cfg, payload_name, era):
        self.cfg = cfg
        self.payload_name = payload_name
        self.era = era
        self.masses = self.cfg.get("masses")

        sys.path.append(os.environ["ANALYSIS_PATH"])

        load_features = set()
        columns_to_save = set()

        # categories with dedicated networks/configs, each in its own subdir
        self.categories = ["boosted", "resolved"]

        # maps channel -> dnn specs
        self.channel_dnn_specs = {
            "DL": self.cfg.get("DL", None),
            "SL": self.cfg.get("SL", None),
        }

        self.dnn_configs = {}
        # channel -> category -> models directory
        self.models_folders = {}
        # channel -> {"multiclass": spec, "binary": spec} and clipping for logits
        self.output_specs = {}
        self.logit_clip_eps = {}

        for channel, specs in self.channel_dnn_specs.items():
            if specs is None:
                print(f"Channel {channel} does not have DNN defined, skip.")
                continue

            # one mass-independent multiclass and one binary parametrized by the
            # resonance mass per parity, so the model names carry no mass
            for pattern_key in ["binary_name_pattern", "multiclass_name_pattern"]:
                if "{mass}" in specs[pattern_key]:
                    raise RuntimeError(
                        f"{channel}: {pattern_key} '{specs[pattern_key]}' contains "
                        "{mass}. Models per mass point are no longer supported."
                    )

            mass_feature = specs["mass_feature"]

            self.output_specs[channel] = {
                "multiclass": parse_output_spec(
                    f"{channel}/multiclass_output",
                    specs.get("multiclass_output"),
                    needs_signal_node=False,
                ),
                "binary": parse_output_spec(
                    f"{channel}/binary_output",
                    specs.get("binary_output"),
                    needs_signal_node=True,
                ),
            }
            num_classes = self.output_specs[channel]["multiclass"]["num_nodes"]
            if num_classes != len(self.cfg["class_names"]):
                raise RuntimeError(
                    f"{channel}: multiclass_output num_nodes ({num_classes}) does not "
                    f"match the {len(self.cfg['class_names'])} class_names."
                )

            clip_eps = float(specs.get("logit_clip_eps", 1e-7))
            if not 0.0 < clip_eps < 0.5:
                raise RuntimeError(f"{channel}: logit_clip_eps must be in (0, 0.5).")
            self.logit_clip_eps[channel] = clip_eps

            base_folder = os.path.join(
                os.environ["ANALYSIS_PATH"], "config", "DNN", specs["version"]
            )

            self.dnn_configs[channel] = {}
            self.models_folders[channel] = {}

            for category in self.categories:
                # boosted/resolved live in their own subdirectories
                category_folder = os.path.join(base_folder, category)
                self.models_folders[channel][category] = category_folder

                self.dnn_configs[channel][category] = {}

                binary_cfg_path = os.path.join(
                    base_folder, f"binary_{category}_dnn_config.yaml"
                )
                multiclass_cfg_path = os.path.join(
                    base_folder, f"multiclass_{category}_dnn_config.yaml"
                )

                with open(binary_cfg_path, "r") as f:
                    self.dnn_configs[channel][category]["binary"] = yaml.safe_load(f)
                with open(multiclass_cfg_path, "r") as f:
                    self.dnn_configs[channel][category]["multiclass"] = yaml.safe_load(
                        f
                    )

                # the mass hypothesis is fed as the last column of the binary inputs
                binary_features = self.dnn_configs[channel][category]["binary"][
                    "features"
                ]
                if not binary_features or binary_features[-1] != mass_feature:
                    raise RuntimeError(
                        f"{channel}/{category}: mass feature '{mass_feature}' must be "
                        f"the last binary feature in {binary_cfg_path}."
                    )

                # the mass column is filled per hypothesis, the X_mass branch is never read
                load_features.update(binary_features[:-1])
                load_features.update(
                    self.dnn_configs[channel][category]["multiclass"]["features"]
                )

        columns_to_save.update(
            [f"{self.payload_name}_{col}" for col in self.cfg["columns"]]
        )

        load_features.update(["FullEventId", "event", "SL", "DL", "boosted"])
        self.vars_to_save = load_features

    def run(self, array):
        print("Running TwoStageDNNProducer producer")

        array = self.ApplyDNN(array)
        array = self.SelectDNN(array)

        # Delete not-needed branches
        for col in array.fields:
            if col not in self.cfg["columns"]:
                if col != "FullEventId":
                    del array[col]

        # Rename the branches
        for col in self.cfg["columns"]:
            if col in array.fields:
                array[f"{self.payload_name}_{col}"] = array[f"{col}"]
                del array[f"{col}"]
            else:
                print(f"Expected column {col} not found in your payload array!")
                print(f"Available columns were {array.fields}")

        return array

    def ApplyDNN(self, branches):
        output_fields = {}
        class_names_list = self.cfg["class_names"]

        # a feature value is flagged if it is non-finite (NaN/inf) OR a large
        # sentinel/padding value. Inputs are still fed to the model as-is; this is
        # warning-only. Real features never legitimately reach this scale.
        SENTINEL_CAP = 1e15

        def bad_values(col):
            return ~np.isfinite(col) | (np.abs(col) > SENTINEL_CAP)

        for channel, specs in self.channel_dnn_specs.items():
            if specs is None:
                print(f"Channel {channel} does not have DNN defined, skip.")
                continue

            num_parities = specs["num_parities"]
            multiclass_spec = self.output_specs[channel]["multiclass"]
            binary_spec = self.output_specs[channel]["binary"]
            num_classes = multiclass_spec["num_nodes"]
            signal_node = binary_spec["signal_node"]
            clip_eps = self.logit_clip_eps[channel]
            num_events = len(branches)
            event_id = np.asarray(branches.event, dtype=np.int64)

            bin_name_pattern = specs["binary_name_pattern"]
            multiclass_name_pattern = specs["multiclass_name_pattern"]

            for category in self.categories:
                models_folder = self.models_folders[channel][category]

                binary_feature_list = self.dnn_configs[channel][category]["binary"][
                    "features"
                ]
                multiclass_feature_list = self.dnn_configs[channel][category][
                    "multiclass"
                ]["features"]

                # the last binary feature is the mass, which is overwritten with each
                # hypothesis below; the stored value is never fed to the model
                binary_event_features = binary_feature_list[:-1]

                reuse_inputs = binary_event_features == multiclass_feature_list

                # union of all event features actually fed to either model
                all_features = list(binary_event_features)
                if not reuse_inputs:
                    all_features += [
                        f
                        for f in multiclass_feature_list
                        if f not in binary_event_features
                    ]

                # detect flagged (non-finite or large sentinel) feature values and
                # warn. inputs are still fed to the model as-is.
                flagged_mask = np.zeros(num_events, dtype=bool)
                flagged_by_feature = {}
                for fn in all_features:
                    col = np.asarray(getattr(branches, fn), dtype=np.float32)
                    bad = bad_values(col)
                    if bad.any():
                        flagged_by_feature[fn] = bad
                        flagged_mask |= bad

                n_flagged = int(flagged_mask.sum())
                if n_flagged:
                    flagged_idx = np.where(flagged_mask)[0]
                    flagged_event_ids = event_id[flagged_idx]
                    print(
                        f"\n=== WARNING {channel}/{category}: "
                        f"{n_flagged} event(s) have flagged feature(s) "
                        f"(non-finite or |value| > {SENTINEL_CAP:g}); "
                        f"inputs fed as-is ==="
                    )
                    print(
                        f"event ids: {flagged_event_ids[:100].tolist()}"
                        f"{' ...' if n_flagged > 100 else ''}"
                    )
                    print(
                        "features with flagged values "
                        "(feature -> #events -> affected event ids):"
                    )
                    for fn, bad in flagged_by_feature.items():
                        ids = event_id[bad]
                        print(
                            f"  {fn}: {int(bad.sum())} -> "
                            f"{ids[:100].tolist()}{' ...' if bad.sum() > 100 else ''}"
                        )

                # NaN until filled: every event must get a score from its parity
                multiclass_predictions = np.full(
                    (num_events, num_classes), np.nan, dtype=np.float32
                )
                # one column per mass hypothesis
                binary_predictions = np.full(
                    (num_events, len(self.masses)), np.nan, dtype=np.float32
                )
                assigned_mask = np.zeros(num_events, dtype=bool)

                for train_parity in range(num_parities):
                    application_parity = (train_parity + 3) % num_parities
                    application_mask = event_id % num_parities == application_parity

                    if not np.any(application_mask):
                        continue

                    assigned_mask |= application_mask
                    selected = branches[application_mask]

                    def build(feature_list):
                        return np.stack(
                            [
                                np.asarray(getattr(selected, fn), dtype=np.float32)
                                for fn in feature_list
                            ],
                            axis=1,
                        )

                    event_inputs = build(binary_event_features)
                    multiclass_inputs = (
                        event_inputs if reuse_inputs else build(multiclass_feature_list)
                    )

                    # event features followed by the mass column, filled per hypothesis
                    binary_inputs = np.empty(
                        (event_inputs.shape[0], len(binary_feature_list)),
                        dtype=np.float32,
                    )
                    binary_inputs[:, :-1] = event_inputs

                    multiclass_model_path = os.path.join(
                        models_folder,
                        multiclass_name_pattern.format(train_parity=train_parity),
                    )
                    binary_model_path = os.path.join(
                        models_folder,
                        bin_name_pattern.format(train_parity=train_parity),
                    )

                    with onnx_session(multiclass_model_path) as ms:
                        multiclass_scores = model_output(
                            ms,
                            multiclass_inputs,
                            multiclass_spec,
                            multiclass_model_path,
                        )
                        multiclass_predictions[application_mask] = convert_dnn_output(
                            multiclass_scores, multiclass_spec, clip_eps
                        )

                    with onnx_session(binary_model_path) as bs:
                        for mass_idx, mp in enumerate(self.masses):
                            binary_inputs[:, -1] = mp
                            binary_scores = model_output(
                                bs, binary_inputs, binary_spec, binary_model_path
                            )
                            binary_predictions[application_mask, mass_idx] = (
                                convert_dnn_output(
                                    binary_scores, binary_spec, clip_eps
                                )[:, signal_node]
                            )

                    del selected, event_inputs, multiclass_inputs, binary_inputs

                # report events whose model OUTPUTS are non-finite.
                # only consider events actually assigned to a parity (i.e. not
                # left at the NaN sentinel).
                def report_nonfinite(predictions, label):
                    out_bad_mask = assigned_mask & ~np.isfinite(predictions).all(axis=1)
                    n_out_bad = int(out_bad_mask.sum())
                    if not n_out_bad:
                        return
                    out_bad_idx = np.where(out_bad_mask)[0]
                    out_bad_ids = event_id[out_bad_idx]
                    also_flagged = flagged_mask[out_bad_idx]
                    print(
                        f"\n=== WARNING {channel}/{category}/{label}: "
                        f"{n_out_bad} event(s) have non-finite MODEL OUTPUTS ==="
                    )
                    print(
                        f"event ids: {out_bad_ids[:100].tolist()}"
                        f"{' ...' if n_out_bad > 100 else ''}"
                    )
                    print(
                        f"  of these, {int(also_flagged.sum())} also had "
                        f"flagged input feature(s); "
                        f"{int((~also_flagged).sum())} had clean inputs "
                        f"(suggests model/overflow issue)."
                    )

                report_nonfinite(multiclass_predictions, "multiclass")
                for mass_idx, mp in enumerate(self.masses):
                    report_nonfinite(
                        binary_predictions[:, mass_idx : mass_idx + 1],
                        f"binary/M{mp}",
                    )

                # check that all events have been processed. NaN model outputs of
                # processed events are only reported above.
                assert (
                    assigned_mask.all()
                ), f"All predictions must be filled for {channel}/{category}"

                for class_idx, class_name in enumerate(class_names_list):
                    mc_field_name = f"multiclass_{channel}_{category}_{class_name}"
                    output_fields[mc_field_name] = multiclass_predictions[
                        :, class_idx
                    ].copy()

                for mass_idx, mp in enumerate(self.masses):
                    bin_field_name = f"binary_{channel}_{category}_M{mp}"
                    output_fields[bin_field_name] = binary_predictions[
                        :, mass_idx
                    ].copy()

                del multiclass_predictions, binary_predictions

        for field_name, values in output_fields.items():
            branches[field_name] = values

        del output_fields
        gc.collect()
        return branches

    def SelectDNN(self, branches):
        nEvents = len(branches)
        output_fields = {}
        class_names_list = self.cfg["class_names"]

        def get_field(name):
            """Fetch a branch as float32, or zeros if it's missing."""
            if name in branches.fields:
                return np.asarray(branches[name], dtype=np.float32)
            return np.zeros(nEvents, dtype=np.float32)

        def get_mask(name):
            if name in branches.fields:
                return np.asarray(branches[name], dtype=bool)
            return np.zeros(nEvents, dtype=bool)

        sl_mask = get_mask("SL")
        boosted_mask = get_mask("boosted")

        def select(field_builder):
            """
            field_builder(channel, category) -> branch field name.
            Picks boosted/resolved per event, then SL/DL per event.
            """
            sl_sel = np.where(
                boosted_mask,
                get_field(field_builder("SL", "boosted")),
                get_field(field_builder("SL", "resolved")),
            )
            dl_sel = np.where(
                boosted_mask,
                get_field(field_builder("DL", "boosted")),
                get_field(field_builder("DL", "resolved")),
            )
            return np.where(sl_mask, sl_sel, dl_sel)

        # multiclass scores, mass-independent
        for class_name in class_names_list:
            output_fields[class_name] = select(
                lambda ch, cat: f"multiclass_{ch}_{cat}_{class_name}"
            )

        # binary score per mass hypothesis
        for mass in self.masses:
            m = int(mass)
            target_field = f"M{m}_binary"
            output_fields[target_field] = select(
                lambda ch, cat: f"binary_{ch}_{cat}_M{m}"
            )

        for field_name, values in output_fields.items():
            branches[field_name] = values

        return branches

import tensorflow as tf
import numpy as np
import uproot
import os
import yaml
import tf2onnx
import onnx
import copy
import psutil
import matplotlib.pyplot as plt
import onnxruntime as ort
import ROOT
import sklearn.metrics


class DataWrapper:
    def __init__(self):
        print("Init data wrapper")

        self.feature_names = None

        self.features_no_param = None
        self.features = None

        self.param_values = None

        self.labels = None

        self.class_weight = None
        self.class_target = None
        self.multiclass_weight = None
        self.class_target_binary = None

        self.res2b = None
        self.recovery = None
        self.boosted = None

        self.param_list = [
            300,
            400,
            500,
            600,
            700,
            800,
            1000,
        ]
        self.use_parametric = False

        self.features_paramSet = None

        self.X_mass = None

    def UseParametric(self, use_parametric):
        self.use_parametric = use_parametric
        print(f"Parametric feature set to {use_parametric}")

    def SetParamList(self, param_list):
        self.param_list = param_list

    def SetPredictParamValue(self, param_value):
        # During predict, we want to use a truly random param value even for signal!
        if param_value not in self.param_list:
            print(f"This param value {param_value} is not an option!")
        new_params = np.array(
            [[param_value for x in self.features]], dtype=np.float32
        ).transpose()

        self.features_paramSet = np.append(self.features_no_param, new_params, axis=1)

    def AddInputFeatures(self, features):
        if self.feature_names == None:
            self.feature_names = features
        else:
            self.feature_names = self.feature_names + features

        print(f"Added features {features}")
        print(f"New feature list {self.feature_names}")

    def ReadFile(self, file_name, entry_start=None, entry_stop=None):
        if self.feature_names == None:
            print("Unknown branches to read! DefineInputFeatures first!")
            return

        print(f"Reading file {file_name}")

        features_to_load = self.feature_names.copy()

        features_to_load.append("X_mass")
        features_to_load.append("weight_Central")
        features_to_load.append("class_value")
        features_to_load.append("res2b")
        features_to_load.append("recovery")
        features_to_load.append("boosted")

        print(f"Only loading these features {features_to_load}")

        print(
            f"Going to open file. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}"
        )

        with uproot.open(file_name) as file:
            tree = file["Events"]
            branches = tree.arrays(
                features_to_load, entry_start=entry_start, entry_stop=entry_stop
            )

            print(
                f"Loaded branches. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}"
            )

            self.features = np.array(
                [
                    getattr(branches, feature_name)
                    for feature_name in self.feature_names
                ],
                dtype="float32",
            ).transpose()

            print(
                f"Set Features. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}"
            )

            # Add parametric variable
            # self.param_values = np.array([[x if (x > 0) else np.random.choice(self.param_list) for x in getattr(branches, 'X_mass') ]]).transpose()
            self.X_mass = getattr(branches, "X_mass")
            self.physics_weight = getattr(branches, "weight_Central")
            self.class_value = getattr(branches, "class_value")
            self.res2b = getattr(branches, "res2b")
            self.recovery = getattr(branches, "recovery")
            self.boosted = getattr(branches, "boosted")
            self.param_values = np.array(
                [getattr(branches, "X_mass")], dtype="float32"
            ).transpose()  # Init wrong parametric, later we will fill with random sample
            print("Got the param values")

        self.features_no_param = self.features
        if self.use_parametric:
            self.features = np.append(self.features, self.param_values, axis=1)

        print(
            f"End read. Memory usage in MB is {psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)}"
        )

    def ReadWeightFile(self, weight_name, entry_start=None, entry_stop=None):
        print(f"Reading weight file {weight_name}")
        with uproot.open(weight_name) as file:
            tree = file["weight_tree"]
            branches = tree.arrays(entry_start=entry_start, entry_stop=entry_stop)
            self.class_weight = np.array(
                getattr(branches, "class_weight"), dtype="float32"
            )
            self.class_target = np.array(
                getattr(branches, "class_target"), dtype="float32"
            )
            self.multiclass_weight = np.array(
                getattr(branches, "multiclass_weight", None), dtype="float32"
            )
            self.class_target_binary = np.array(
                getattr(branches, "class_targets_binary"), dtype="float32"
            )
            file.close()

    def GetHME(self, file_name, entry_start=None, entry_stop=None):
        print(f"Reading HME mass from file {file_name}")
        hme_mass = None
        with uproot.open(file_name) as file:
            tree = file["Events"]
            branches = tree.arrays(
                ["DeepHME_mass"],
                entry_start=entry_start,
                entry_stop=entry_stop,
            )
            hme_mass = np.array(getattr(branches, "DeepHME_mass"), dtype="float32")

            # branches = tree.arrays(
            #     ["m_bbllmet"],
            #     entry_start=entry_start,
            #     entry_stop=entry_stop,
            # )
            # hme_mass = np.array(
            #     getattr(branches, "m_bbllmet"), dtype="float32"
            # )
        return hme_mass

    def GetFatJetBTag(self, file_name, entry_start=None, entry_stop=None):
        print(f"Reading FatJet BTag from file {file_name}")
        fatjet_btag = None
        with uproot.open(file_name) as file:
            tree = file["Events"]
            branches = tree.arrays(
                ["fatbjet_particleNetWithMass_HbbvsQCD"],
                entry_start=entry_start,
                entry_stop=entry_stop,
            )
            fatjet_btag = np.array(
                getattr(branches, "fatbjet_particleNetWithMass_HbbvsQCD"),
                dtype="float32",
            )
        return fatjet_btag


class ModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(
        self,
        filepath,
        monitor="val_loss",
        verbose=0,
        mode="min",
        min_delta=None,
        min_rel_delta=None,
        save_callback=None,
        patience=None,
        predicate=None,
        input_signature=None,
    ):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.epochs_since_last_save = 0
        self.msg = None
        self.save_callback = save_callback
        self.patience = patience
        self.predicate = predicate
        self.input_signature = input_signature

        if os.path.exists(filepath):
            shutil.rmtree(filepath)

        self.best = None
        self.monitor_op = self._make_monitor_op(mode, min_delta, min_rel_delta)

    def _make_monitor_op(self, mode, min_delta, min_rel_delta):
        if mode == "min":
            if min_delta is None and min_rel_delta is None:
                return lambda current, best: best is None or best - current > 0
            if min_delta is None:
                return (
                    lambda current, best: best is None
                    or (best - current) > min_rel_delta * best
                )
            if min_rel_delta is None:
                return lambda current, best: best is None or best - current > min_delta
            return (
                lambda current, best: best is None
                or (best - current) > min_rel_delta * best
                or best - current > min_delta
            )
        elif mode == "max":
            if min_delta is None and min_rel_delta is None:
                return lambda current, best: best is None or current - best > 0
            if min_delta is None:
                return (
                    lambda current, best: best is None
                    or (current - best) > min_rel_delta * best
                )
            if min_rel_delta is None:
                return lambda current, best: best is None or current - best > min_delta
            return (
                lambda current, best: best is None
                or (current - best) > min_rel_delta * best
                or current - best > min_delta
            )
        else:
            raise ValueError(f"Unrecognized mode: {mode}")

    def _print_msg(self):
        if self.msg is not None:
            print(self.msg)
            self.msg = None

    def on_epoch_begin(self, epoch, logs=None):
        self._print_msg()

    def on_train_end(self, logs=None):
        self._print_msg()

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best) and (
            self.predicate is None or self.predicate(self.model, logs)
        ):
            # if self.predicate is None or self.predicate(self.model, logs):
            os.makedirs(self.filepath, exist_ok=True)
            path_best = os.path.join(self.filepath, f"{self.monitor}.onnx")
            if os.path.exists(path_best):
                os.remove(path_best)
            if self.input_signature is not None:
                onnx_model, _ = tf2onnx.convert.from_keras(
                    self.model, self.input_signature, opset=13
                )
                onnx.save(onnx_model, path_best)

            if self.verbose > 0:
                self.msg = f"\nEpoch {epoch+1}: {self.monitor} "
                if self.best is None:
                    self.msg += f"= {current:.5f}."
                else:
                    self.msg += f"improved from {self.best:.5f} to {current:.5f} after {self.epochs_since_last_save} epochs."
                self.msg += f" Saving model to {path_best}\n"
            self.best = current
            self.epochs_since_last_save = 0
        if self.patience is not None and self.epochs_since_last_save >= self.patience:
            self.model.stop_training = True
            if self.verbose > 0:
                if self.msg is None:
                    self.msg = "\n"
                self.msg = f"Epoch {epoch+1}: early stopping after {self.epochs_since_last_save} epochs."


class WeightedBackgroundAtSignalYield(tf.keras.metrics.Metric):
    def __init__(
        self,
        threshold_yield,
        max_events=100000,
        nParity=4,
        name="weighted_bkg_at_sig_yield",
        dtype=tf.float32,
    ):
        super().__init__(name=name, dtype=dtype)
        self.threshold_yield = tf.constant(threshold_yield, dtype=dtype)
        self.max_events = max_events
        self.nParity = nParity

        self.scores = self.add_weight(
            name="scores", shape=(max_events,), dtype=dtype, initializer="zeros"
        )
        self.labels = self.add_weight(
            name="labels", shape=(max_events,), dtype=dtype, initializer="zeros"
        )
        self.weights_var = self.add_weight(
            name="weights", shape=(max_events,), dtype=dtype, initializer="zeros"
        )
        self.count = self.add_weight(
            name="count", shape=(), dtype=tf.int32, initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        if sample_weight is None:
            sample_weight = tf.ones_like(y_true, dtype=tf.float32)
        else:
            sample_weight = tf.cast(tf.reshape(sample_weight, [-1]), tf.float32)

        batch_size = tf.shape(y_true)[0]
        start = self.count
        end = tf.minimum(start + batch_size, self.max_events)
        actual_batch = end - start

        y_true = y_true[:actual_batch]
        y_pred = y_pred[:actual_batch]
        sample_weight = sample_weight[:actual_batch]

        indices = tf.range(start, end)
        indices_2d = tf.expand_dims(indices, 1)

        self.scores.assign(tf.tensor_scatter_nd_update(self.scores, indices_2d, y_pred))
        self.labels.assign(tf.tensor_scatter_nd_update(self.labels, indices_2d, y_true))
        self.weights_var.assign(
            tf.tensor_scatter_nd_update(self.weights_var, indices_2d, sample_weight)
        )

        self.count.assign(end)

    def result(self):
        n = tf.cast(self.count, tf.int32)  # ensure int32 for indexing

        def no_samples():
            return (
                tf.constant(-2.0, dtype=self.dtype),
                tf.constant(-2.0, dtype=self.dtype),
                tf.constant(-2.0, dtype=self.dtype),
            )

        def compute():
            # Safe slicing useing tf.gather instead of tf.slice
            idx = tf.range(n)
            scores = tf.gather(self.scores, idx)
            labels = tf.gather(self.labels, idx)
            weights = tf.gather(self.weights_var, idx)

            sorted_idx = tf.argsort(scores, direction="DESCENDING")
            sorted_labels = tf.gather(labels, sorted_idx)
            sorted_weights = tf.gather(weights, sorted_idx)

            is_signal = tf.cast(tf.equal(sorted_labels, 1.0), tf.float32)
            is_bkg = 1.0 - is_signal

            sig_weights = sorted_weights * is_signal
            bkg_weights = sorted_weights * is_bkg

            cum_sig = tf.cumsum(sig_weights)
            cum_bkg = tf.cumsum(bkg_weights)
            cum_bkg_w2 = tf.cumsum(bkg_weights * bkg_weights)

            # Scale by nParity to 'fake' realistic values
            cum_sig = (
                self.nParity * 0.0264215349425664 * cum_sig
            )  # Scale signal to BR too
            cum_bkg = self.nParity * cum_bkg
            cum_bkg_w2 = self.nParity * self.nParity * cum_bkg_w2

            total_sig = cum_sig[-1]
            threshold = tf.minimum(self.threshold_yield, total_sig)

            def no_signal():
                return (
                    tf.constant(-1.0, dtype=self.dtype),
                    tf.constant(-1.0, dtype=self.dtype),
                    tf.constant(-1.0, dtype=self.dtype),
                )

            def with_signal():
                mask = cum_sig >= threshold
                idx = tf.argmax(tf.cast(mask, tf.int32))
                return (
                    tf.cast(cum_bkg[idx], self.dtype),
                    tf.keras.ops.power(tf.cast(cum_bkg_w2[idx], self.dtype), 0.5),
                    tf.cast(scores[idx], self.dtype),
                )

            return tf.cond(
                tf.logical_or(tf.equal(total_sig, 0.0), tf.equal(threshold, 0.0)),
                no_signal,
                with_signal,
            )

        value, error, score = tf.cond(tf.equal(n, 0), no_samples, compute)

        return value, error, score

    def reset_state(self):
        self.scores.assign(tf.zeros_like(self.scores))
        self.labels.assign(tf.zeros_like(self.labels))
        self.weights_var.assign(tf.zeros_like(self.weights_var))
        self.count.assign(0)

    def _compute(self):
        return self.result()


class WeightedBackgroundAtSignalYieldValue(tf.keras.metrics.Metric):
    def __init__(
        self, parent_metric, name="weighted_bkg_at_sig_yield_value", dtype=tf.float32
    ):
        super().__init__(name=name, dtype=dtype)
        self.parent = parent_metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        # raise RuntimeError("Dummy but second one")
        self.parent.update_state(y_true, y_pred, sample_weight)

    def result(self):
        value, _, _ = self.parent._compute()
        return tf.cast(value, self.dtype)

    def reset_state(self):
        self.parent.reset_state()


class WeightedBackgroundAtSignalYieldError(tf.keras.metrics.Metric):
    def __init__(
        self, parent_metric, name="weighted_bkg_at_sig_yield_error", dtype=tf.float32
    ):
        super().__init__(name=name, dtype=dtype)
        self.parent = parent_metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.parent.update_state(y_true, y_pred, sample_weight)

    def result(self):
        _, error, _ = self.parent._compute()
        return tf.cast(error, self.dtype)

    def reset_state(self):
        self.parent.reset_state()


class WeightedBackgroundAtSignalYieldScore(tf.keras.metrics.Metric):
    def __init__(
        self, parent_metric, name="weighted_bkg_at_sig_yield_score", dtype=tf.float32
    ):
        super().__init__(name=name, dtype=dtype)
        self.parent = parent_metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.parent.update_state(y_true, y_pred, sample_weight)

    def result(self):
        _, _, score = self.parent._compute()
        return tf.cast(score, self.dtype)

    def reset_state(self):
        self.parent.reset_state()


@tf.function
def binary_entropy(target, output):
    epsilon = tf.constant(1e-7, dtype=tf.float32)
    x = tf.clip_by_value(output, epsilon, 1 - epsilon)
    return -target * tf.math.log(x) - (1 - target) * tf.math.log(1 - x)


@tf.function
def categorical_entropy(target, output):
    epsilon = tf.constant(1e-7, dtype=tf.float32)
    x = tf.clip_by_value(output, epsilon, 1 - epsilon)
    return -tf.reduce_sum(target * tf.math.log(x), -1)


@tf.function
def binary_focal_crossentropy(target, output, gamma1=2, gamma2=0.5):
    epsilon = tf.constant(1e-7, dtype=tf.float32)

    # Un-nest the output (currently in shape [ [1], [2], [3], ...] and we want in shape [1, 2, 3])
    y_true = target[:, 0]
    y_pred = output[:, 0]

    bce = binary_entropy(y_true, y_pred)

    # Custom focal
    gamma_signal = 0  # DO NOT TOUCH
    gamma_bkg = gamma1
    # Target y_pred -> 1
    p_t = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    # Split into a 'signal' gamma and a 'bkg' gamma
    gamma = y_true * gamma_signal + (1 - y_true) * gamma_bkg
    # Calc factor of pT^gamma
    focal_factor = tf.keras.ops.power(p_t, gamma)

    # focal_factor = 1.0
    # focal_bce = focal_factor * bce

    gamma2_signal = gamma2
    gamma2_bkg = 1  # DO NOT TOUCH

    gamma2 = y_true * gamma2_signal + (1 - y_true) * gamma2_bkg
    # tf.print("ff")
    # tf.print(focal_factor)
    # tf.print("bce")
    # tf.print(bce)

    focal_bce = focal_factor * tf.keras.ops.power(bce, gamma2)

    # tf.print("Min?")
    # tf.print(tf.reduce_min(focal_bce))

    return focal_bce


class Model(tf.keras.Model):
    def __init__(self, setup, max_events, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup = setup
        self.gamma1 = setup["gamma1"]
        self.gamma2 = setup["gamma2"]
        self.signal_loss_scale = setup["signal_loss_scale"]
        self.multiclass_loss_scale = setup["multiclass_loss_scale"]

        self.nClasses = setup["nClasses"]

        # self.class_loss = tf.keras.losses.CategoricalCrossentropy(reduction=None)
        # self.class_loss = tf.keras.losses.CategoricalFocalCrossentropy(reduction=None)
        self.class_loss = binary_focal_crossentropy
        self.multiclass_loss = categorical_entropy

        self.class_accuracy = tf.keras.metrics.categorical_accuracy

        self.class_loss_tracker = tf.keras.metrics.Mean(name="class_loss")
        self.class_accuracy_tracker = tf.keras.metrics.Mean(name="class_accuracy")

        self.signal_class_loss_tracker = tf.keras.metrics.Mean(name="signal_class_loss")
        self.multiclass_loss_tracker = tf.keras.metrics.Mean(name="multiclass_loss")

        self.l2_loss_tracker = tf.keras.metrics.Mean(name="l2_loss")

        # self.bkgAtSignal = WeightedBackgroundAtSignalYield(
        #     threshold_yield=5.0, max_events=max_events
        # )
        # self.bkgAtSignal_value = WeightedBackgroundAtSignalYieldValue(self.bkgAtSignal)
        # self.bkgAtSignal_error = WeightedBackgroundAtSignalYieldError(self.bkgAtSignal)
        # self.bkgAtSignal_score = WeightedBackgroundAtSignalYieldScore(self.bkgAtSignal)

        self.class_min_tracker = tf.keras.metrics.Mean(name="class_min")
        self.class_max_tracker = tf.keras.metrics.Mean(name="class_max")

        self.lr_tracker = tf.keras.metrics.Mean(name="learning_rate")

        self.other_class_min_tracker = [
            tf.keras.metrics.Mean(name=f"other_class_min{n}")
            for n in range(self.nClasses)
            if n != 0
        ]
        self.other_class_max_tracker = [
            tf.keras.metrics.Mean(name=f"other_class_max{n}")
            for n in range(self.nClasses)
            if n != 0
        ]

        self.blocks = []

        n_units = setup["n_units"]
        activation = setup["activation"]

        for n in range(setup["n_layers"]):
            name = f"layer_{n}"
            block = {}
            block["dense"] = tf.keras.layers.Dense(
                n_units,
                activation=activation,
                name=name,
                kernel_initializer="random_normal",
                bias_initializer="random_normal",
                kernel_regularizer=tf.keras.regularizers.l2(setup["l2_rate"]),
            )

            if setup["use_batch_norm"]:
                block["bn"] = tf.keras.layers.BatchNormalization(
                    name=name + "_batch_norm"
                )

            if setup["dropout"] > 0:
                block["dropout"] = tf.keras.layers.Dropout(
                    setup["dropout"], name=name + "_dropout"
                )

            self.blocks.append(block)

        self.class_output = tf.keras.layers.Dense(
            setup["nClasses"], activation="softmax", name="class_output"
        )

        self.output_names = ["class_output"]

    def call(self, x):
        h = x
        for i, block in enumerate(self.blocks):
            h = block["dense"](h)

            if self.setup["use_batch_norm"]:
                h = block["bn"](h)

            if self.setup["dropout"] > 0:
                h = block["dropout"](h)

        class_output = self.class_output(h)
        return class_output

    def _step(self, data, training):
        x, y = data

        y_class = tf.cast(y[0], dtype=tf.float32)

        class_weight = tf.cast(y[1], dtype=tf.float32)

        physics_weight = tf.cast(y[2], dtype=tf.float32)

        def compute_losses():
            y_pred_class = self(x, training=training)

            signal_class_loss_vec = self.class_loss(
                y_class, y_pred_class, self.gamma1, self.gamma2
            )

            # signal_class_loss_vec = self.class_loss(
            #     y_class, y_pred_class
            # )

            multiclass_loss_vec = self.multiclass_loss(y_class, y_pred_class)

            class_loss_vec = (
                self.signal_loss_scale * signal_class_loss_vec
                + self.multiclass_loss_scale * multiclass_loss_vec
                # signal_class_loss_vec
            )

            class_loss = tf.reduce_mean(class_loss_vec * class_weight)

            l2_loss = tf.add_n(self.losses)

            combined_loss = class_loss + l2_loss

            return (
                y_pred_class,
                class_loss_vec,
                class_loss,
                l2_loss,
                combined_loss,
                signal_class_loss_vec,
                multiclass_loss_vec,
            )

        if training:
            with tf.GradientTape() as class_tape:
                (
                    y_pred_class,
                    class_loss_vec,
                    class_loss,
                    l2_loss,
                    combined_loss,
                    signal_class_loss_vec,
                    multiclass_loss_vec,
                ) = compute_losses()
        else:
            (
                y_pred_class,
                class_loss_vec,
                class_loss,
                l2_loss,
                combined_loss,
                signal_class_loss_vec,
                multiclass_loss_vec,
            ) = compute_losses()

        self.class_min_tracker.update_state(tf.reduce_min(y_pred_class[:, 0]))
        self.class_max_tracker.update_state(tf.reduce_max(y_pred_class[:, 0]))

        # self.bkgAtSignal.update_state(
        #     y_class[:, 0], y_pred_class[:, 0], sample_weight=physics_weight
        # )

        for n in range(self.nClasses):
            if n == 0:
                continue
            self.other_class_min_tracker[n - 1].update_state(
                tf.reduce_min(y_pred_class[:, n])
            )
            self.other_class_max_tracker[n - 1].update_state(
                tf.reduce_max(y_pred_class[:, n])
            )

        class_accuracy_vec = self.class_accuracy(y_class, y_pred_class)

        self.class_loss_tracker.update_state(class_loss_vec, sample_weight=class_weight)
        self.class_accuracy_tracker.update_state(
            class_accuracy_vec, sample_weight=class_weight
        )

        self.signal_class_loss_tracker.update_state(
            signal_class_loss_vec, sample_weight=class_weight
        )
        self.multiclass_loss_tracker.update_state(
            multiclass_loss_vec, sample_weight=class_weight
        )

        self.lr_tracker.update_state(self.optimizer.learning_rate)

        self.l2_loss_tracker.update_state(l2_loss)

        if training:
            grad = class_tape.gradient(combined_loss, self.trainable_variables)
            # grad = class_tape.gradient(class_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        return self._step(data, training=True)

    def test_step(self, data):
        return self._step(data, training=False)

    @property
    def metrics(self):
        metric_list = [
            self.class_loss_tracker,
            self.class_accuracy_tracker,
            self.signal_class_loss_tracker,
            self.multiclass_loss_tracker,
            self.l2_loss_tracker,
            # self.bkgAtSignal_value,
            # self.bkgAtSignal_error,
            # self.bkgAtSignal_score,
            self.class_min_tracker,
            self.class_max_tracker,
            self.lr_tracker,
        ]
        metric_list = (
            metric_list
            + [x for x in self.other_class_min_tracker]
            + [x for x in self.other_class_max_tracker]
        )

        return metric_list


def train_dnn(
    setup,
    training_file,
    weight_file,
    test_training_file,
    test_weight_file,
    output_folder,
):
    output_dnn_name = os.path.join(output_folder, f"best.onnx")

    dw = DataWrapper()
    dw.AddInputFeatures(setup["features"])

    dw.UseParametric(setup["UseParametric"])
    dw.SetParamList(setup["parametric_list"])

    # Prep a test dw
    # Must copy before reading file so we can read the test file instead
    test_dw = copy.deepcopy(dw)

    entry_start = 0
    # entry_stop = batch_size * 500 # Only load 500 batches for debuging now

    # Do you want to make a larger batch? May increase speed
    entry_stop = None

    dw.ReadFile(
        training_file,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )
    dw.ReadWeightFile(weight_file, entry_start=entry_start, entry_stop=entry_stop)

    test_dw.ReadFile(
        test_training_file,
        entry_start=entry_start,
        entry_stop=entry_stop,
    )
    test_dw.ReadWeightFile(
        test_weight_file, entry_start=entry_start, entry_stop=entry_stop
    )

    dw.physics_weight = np.where(
        (dw.class_target == 0) & (dw.X_mass != 600), 0.0, dw.physics_weight
    )
    test_dw.physics_weight = np.where(
        (test_dw.class_target == 0) & (test_dw.X_mass != 600),
        0.0,
        test_dw.physics_weight,
    )

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(42)

    nClasses = setup["nClasses"]
    batch_size = setup["batch_size"]
    train_tf_dataset = tf.data.Dataset.from_tensor_slices(
        (
            dw.features,
            (
                # Multiclass
                # tf.one_hot(dw.class_target, nClasses),
                # dw.multiclass_weight,
                # dw.physics_weight,
                # Binary
                tf.one_hot(dw.class_target_binary, nClasses),
                dw.class_weight,
                dw.physics_weight,
            ),
        )
    )
    train_tf_dataset = train_tf_dataset.cache("train_cache.tfdata")
    train_tf_dataset = train_tf_dataset.shuffle(
        len(train_tf_dataset), reshuffle_each_iteration=True
    )
    batch_size_train = min(batch_size, train_tf_dataset.cardinality().numpy())
    train_tf_dataset = train_tf_dataset.batch(batch_size_train, drop_remainder=True)

    test_tf_dataset = tf.data.Dataset.from_tensor_slices(
        (
            test_dw.features,
            (
                # Multiclass
                # tf.one_hot(test_dw.class_target, nClasses),
                # test_dw.multiclass_weight,
                # test_dw.physics_weight,
                # Binary
                tf.one_hot(test_dw.class_target_binary, nClasses),
                test_dw.class_weight,
                test_dw.physics_weight,
            ),
        )
    )
    test_tf_dataset = test_tf_dataset.cache("test_cache.tfdata")
    test_tf_dataset = test_tf_dataset.shuffle(
        len(test_tf_dataset), reshuffle_each_iteration=True
    )
    batch_size_test = min(batch_size, test_tf_dataset.cardinality().numpy())
    test_tf_dataset = test_tf_dataset.batch(batch_size_test, drop_remainder=True)

    print("Train multiclass_weight stats:")
    print(
        f"  mean={np.mean(dw.multiclass_weight):.4f}, std={np.std(dw.multiclass_weight):.4f}"
    )
    print(
        f"  min={np.min(dw.multiclass_weight):.4f}, max={np.max(dw.multiclass_weight):.4f}"
    )

    print("Val multiclass_weight stats:")
    print(
        f"  mean={np.mean(test_dw.multiclass_weight):.4f}, std={np.std(test_dw.multiclass_weight):.4f}"
    )
    print(
        f"  min={np.min(test_dw.multiclass_weight):.4f}, max={np.max(test_dw.multiclass_weight):.4f}"
    )

    for split, d in [("Train", dw), ("Val", test_dw)]:
        vals, counts = np.unique(d.class_target, return_counts=True)
        print(f"{split}: {dict(zip(vals, counts))}")
        print(
            f"{split} weighted: {[np.sum(d.multiclass_weight[d.class_target==v]) for v in vals]}"
        )

    parametric_mass_probability = np.ones(len(dw.param_list)) * 1.0 / len(dw.param_list)
    log_probs = tf.math.log([parametric_mass_probability])

    mass_values = tf.constant(dw.param_list)
    mass_keys = tf.constant(np.arange(len(dw.param_list)))
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(mass_keys, mass_values),
        default_value=-1,
    )

    def new_param_map(*x):
        dataset = x
        features = dataset[0]

        # Need to randomize the features parametric mass

        random_param_mass = tf.random.categorical(
            log_probs,
            tf.shape(features)[0],
            dtype=tf.int64,
        )

        actual_new_mass = table.lookup(random_param_mass)
        actual_new_mass = tf.cast(actual_new_mass, tf.float32)

        # Lastly we need to keep the signal events the correct mass
        class_targets = dataset[1][0]
        old_mass_mask = tf.cast(class_targets[:, 0], tf.float32)
        new_mass_mask = tf.cast((class_targets[:, 0] == 0), tf.float32)

        actual_mass = old_mass_mask * features[:, -1] + new_mass_mask * actual_new_mass
        actual_mass = tf.transpose(actual_mass)

        features = tf.concat([features[:, :-1], actual_mass], axis=-1)
        new_dataset = (features, dataset[1])
        return new_dataset

    if setup["UseParametric"]:
        train_tf_dataset = train_tf_dataset.map(new_param_map)
        train_tf_dataset = train_tf_dataset.prefetch(tf.data.AUTOTUNE)
        test_tf_dataset = test_tf_dataset.map(new_param_map)
        test_tf_dataset = test_tf_dataset.prefetch(tf.data.AUTOTUNE)

    input_shape = [None, dw.features.shape[1]]
    input_signature = [tf.TensorSpec(input_shape, tf.float32, name="x")]

    nBatches = max(
        train_tf_dataset.cardinality().numpy(), test_tf_dataset.cardinality().numpy()
    )
    max_events = nBatches * max(batch_size_train, batch_size_test)

    if setup["UseParametric"]:
        features_no_mass = dw.features[:, :-1]
    else:
        features_no_mass = dw.features
    # mean = np.mean(features_no_mass, axis=0)
    # std = np.std(features_no_mass, axis=0) + 1e-6
    # setup["feature_mean"] = mean.tolist()
    # setup["feature_std"] = std.tolist()
    model = Model(setup, max_events)
    model.compile(
        loss=None,
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=setup["learning_rate"],
            weight_decay=setup["weight_decay"],
            clipnorm=1.0,
        ),
    )
    model(dw.features)
    model.summary()

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_class_loss",
        factor=setup["lr_decay"],
        patience=setup["lr_patience"],
        min_lr=0.0000000001,
    )

    callbacks = [
        ModelCheckpoint(
            output_folder,
            verbose=1,
            monitor="val_class_loss",
            mode="min",
            min_rel_delta=1e-3,
            patience=setup["patience"],
            save_callback=None,
            input_signature=input_signature,
        ),
        # ModelCheckpoint(
        #     output_folder,
        #     verbose=1,
        #     monitor="val_weighted_bkg_at_sig_yield_value",
        #     mode="min",
        #     min_rel_delta=1e-3,
        #     patience=None,
        #     save_callback=None,
        #     input_signature=input_signature,
        # ),
        reduce_lr,
    ]

    verbose = setup["verbose"] if "verbose" in setup else 0
    # verbose = 1
    print("Fit model")
    history = model.fit(
        train_tf_dataset,
        validation_data=test_tf_dataset,
        verbose=verbose,
        epochs=setup["n_epochs"],
        shuffle=True,
        callbacks=callbacks,
    )

    os.makedirs(output_folder, exist_ok=True)

    def PlotMetric(history, metric, output_folder):
        if metric not in history.history:
            print(f"Metric {metric} not found in history")
            return
        os.makedirs(os.path.join(output_folder, "metrics"), exist_ok=True)
        plt.plot(history.history[metric], label=f"train_{metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
        plt.title(f"{metric}")
        plt.ylabel(metric)
        plt.xlabel("Epoch")
        plt.legend(loc="upper right")
        plt.grid(True)
        # plt.yscale("log")
        # plt.ylim(bottom=0.0001, top=10.0)
        plt.savefig(
            os.path.join(output_folder, "metrics", f"{metric}.pdf"), bbox_inches="tight"
        )
        plt.clf()

    PlotMetric(history, "class_loss", output_folder)
    PlotMetric(history, "learning_rate", output_folder)
    PlotMetric(history, "l2_loss", output_folder)
    PlotMetric(history, "class_min", output_folder)
    PlotMetric(history, "class_max", output_folder)

    for i in range(1, nClasses):
        PlotMetric(history, f"other_class_min{i}", output_folder)
        PlotMetric(history, f"other_class_max{i}", output_folder)

    PlotMetric(history, "weighted_bkg_at_sig_yield_value", output_folder)
    PlotMetric(history, "weighted_bkg_at_sig_yield_error", output_folder)
    PlotMetric(history, "weighted_bkg_at_sig_yield_score", output_folder)

    print(f"Finished training, how is the per-class accuracy?")
    for cls in range(setup["nClasses"]):
        mask = dw.class_target == cls
        preds = model(dw.features[mask])
        predicted_class = np.argmax(preds, axis=1)
        accuracy = np.mean(predicted_class == cls)
        print(
            f"Class {setup["class_names"][cls]} accuracy: {accuracy:.3f}  (n={np.sum(mask)})"
        )

    input_shape = [None, dw.features.shape[1]]
    input_signature = [tf.TensorSpec(input_shape, tf.float32, name="x")]

    # Convert model to ONNX (reuse what you already do later)
    onnx_model_stage1, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

    stage1_model_path = os.path.join(output_folder, "stage1.onnx")
    onnx.save(onnx_model_stage1, stage1_model_path)

    features_config = {
        "features": dw.feature_names,
        "use_parametric": dw.use_parametric,
        "parametric_list": dw.param_list,
        "model_setup": setup,
        "nClasses": setup["nClasses"],
        "nParity": 4,
    }

    with open(os.path.join(output_folder, "dnn_config.yaml"), "w") as file:
        yaml.dump(features_config, file)

    with open(os.path.join(output_folder, "history.yaml"), "w") as file:
        yaml.dump(history, file)

    # Experimental 2-stage DNN
    # Binary stage 2 runs signal vs background using stage 1 score as additional input
    if getattr(setup, "train_stage_2", False):
        print("Running Stage 1 inference to build Stage 2 dataset")

        sess_stage1 = ort.InferenceSession(stage1_model_path)

        # Run inference on FULL training dataset
        features_stage1 = dw.features
        preds_stage1 = sess_stage1.run(None, {"x": features_stage1})[0]

        signal_scores = preds_stage1[:, 0]

        # -------------------------
        # Stage 2 selection
        # -------------------------

        # Filter training data
        preds_stage1_col = preds_stage1

        # Option to use stage 1 scores as input to stage 2
        X2 = np.concatenate([dw.features, preds_stage1_col], axis=1)

        class_weight2 = dw.class_weight
        physics_weight2 = dw.physics_weight

        # Binary labels: signal vs background
        y2 = (dw.class_target != 0).astype(int)

        # Stage 2 test dataset
        features_test = test_dw.features
        preds_stage1_test = sess_stage1.run(None, {"x": features_test})[0]

        preds_stage1_col_test = preds_stage1_test

        # Option to use stage 1 scores as input to stage 2
        X2_test = np.concatenate([test_dw.features, preds_stage1_col_test], axis=1)

        class_weight2_test = test_dw.class_weight
        physics_weight2_test = test_dw.physics_weight

        y2_test = (test_dw.class_target != 0).astype(int)

        nClasses_stage2 = 2
        train_tf_dataset_stage2 = (
            tf.data.Dataset.from_tensor_slices(
                (
                    X2,
                    (
                        tf.one_hot(y2, nClasses_stage2),
                        class_weight2,
                        physics_weight2,
                    ),
                )
            )
            .shuffle(len(X2))
            .batch(batch_size_train, drop_remainder=True)
            .cache("train_cache_stage2.tfdata")
        )

        test_tf_dataset_stage2 = (
            tf.data.Dataset.from_tensor_slices(
                (
                    X2_test,
                    (
                        tf.one_hot(y2_test, nClasses_stage2),
                        class_weight2_test,
                        physics_weight2_test,
                    ),
                )
            )
            .shuffle(len(X2_test))
            .batch(batch_size_test, drop_remainder=True)
            .cache("test_cache_stage2.tfdata")
        )

        print("Training Stage 2 binary model")

        setup_stage2 = copy.deepcopy(setup)
        setup_stage2["nClasses"] = 2
        setup_stage2["loss_scale"] = 0

        model_stage2 = Model(setup_stage2, max_events)

        model_stage2.compile(
            loss=None,
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=setup_stage2["learning_rate"],
                weight_decay=setup_stage2["weight_decay"],
                clipnorm=1.0,
            ),
        )

        model_stage2(X2)
        model_stage2.summary()

        input_shape_stage2 = [None, X2.shape[1]]
        input_signature_stage2 = [
            tf.TensorSpec(input_shape_stage2, tf.float32, name="x")
        ]

        history_stage2 = model_stage2.fit(
            train_tf_dataset_stage2,
            validation_data=test_tf_dataset_stage2,
            verbose=verbose,
            epochs=setup_stage2["n_epochs"],
            shuffle=True,
            callbacks=[
                ModelCheckpoint(
                    os.path.join(output_folder, "stage2"),
                    verbose=1,
                    monitor="val_class_loss",
                    mode="min",
                    min_rel_delta=1e-3,
                    patience=setup_stage2["patience"],
                    input_signature=input_signature_stage2,
                )
            ],
        )

        output_folder_stage_2 = os.path.join(output_folder, "stage2")
        PlotMetric(history_stage2, "class_loss", output_folder_stage_2)
        PlotMetric(history_stage2, "learning_rate", output_folder_stage_2)
        PlotMetric(history_stage2, "l2_loss", output_folder_stage_2)
        PlotMetric(history_stage2, "class_min", output_folder_stage_2)
        PlotMetric(history_stage2, "class_max", output_folder_stage_2)

        onnx_model_stage2, _ = tf2onnx.convert.from_keras(
            model_stage2, input_signature_stage2, opset=13
        )

        stage2_model_path = os.path.join(output_folder, "stage2.onnx")
        onnx.save(onnx_model_stage2, stage2_model_path)

    return


# =========================================================
# 1. LOAD MODEL + CONFIG
# =========================================================
def load_model_and_config(model_name, model_config):
    print(f"Loading model: {model_name}")
    sess = ort.InferenceSession(model_name)

    with open(model_config, "r") as f:
        config = yaml.safe_load(f)

    return sess, config


# =========================================================
# 2. DATA PREPARATION
# =========================================================
def prepare_datawrapper(setup, validation_file, validation_weight_file):
    dw = DataWrapper()
    dw.AddInputFeatures(setup["features"])
    dw.UseParametric(setup["UseParametric"])
    dw.SetParamList(setup["parametric_list"])

    dw.ReadFile(validation_file)
    dw.ReadWeightFile(validation_weight_file)

    return dw


# =========================================================
# 3. INFERENCE
# =========================================================
def run_inference(sess, features):
    return sess.run(None, {"x": features})


def get_scores(sess, X, stage="multi"):
    preds = run_inference(sess, X)[0]

    # multi-class
    if preds.ndim == 2:
        return preds

    # binary
    return preds[:, 0]


# =========================================================
# 4. EVENT MASKING
# =========================================================
def build_event_masks(dw, hme_values, cat, para_masspoint, setup=None, hme_cut=False):

    default_class_names = [
        "Signal",
        "TT",
        "DY",
        "ST",
        "SMHiggs",
        "VV",
        "Other",
    ]
    class_names = (
        setup.get("class_names", default_class_names)
        if setup is not None
        else default_class_names
    )
    if len(class_names) < 2:
        raise ValueError(
            "build_event_masks requires at least two class names: signal plus backgrounds"
        )

    signal_mask = (dw.X_mass == para_masspoint) & (dw.class_target == 0)
    Background_mask = dw.class_target != 0

    physics_weight = np.copy(dw.physics_weight)
    # Starting in production v2605a, we now include the BR inside signal XS
    # physics_weight = np.where(
    #     dw.class_target == 0,
    #     physics_weight * 0.0264215349425664,
    #     physics_weight,
    # )

    cat_name = "recovery" if cat == "res1b" else cat
    cat_mask = getattr(dw, cat_name) == 1
    physics_weight = np.where(cat_mask, physics_weight, 0)

    signal_and_cat_mask = signal_mask & cat_mask

    if hme_cut:
        hme_mean = np.mean(hme_values[signal_and_cat_mask])
        hme_std = np.std(hme_values[signal_and_cat_mask])
        hme_mask = (hme_values > hme_mean - hme_std) & (hme_values < hme_mean + hme_std)

        # Do a quick significance scan for HME bounds
        # Take the mean of signal's HME, then scan an asymmetric window around it to maximize significance s/sqrt(s+b)
        best_significance = 0.0
        best_hme_low = 0
        best_hme_high = 0
        for hme_low_scan in np.linspace(hme_mean - 5 * hme_std, hme_mean, 10):
            for hme_high_scan in np.linspace(hme_mean, hme_mean + 5 * hme_std, 10):
                hme_mask_scan = (hme_values > hme_low_scan) & (
                    hme_values < hme_high_scan
                )
                s = np.sum(physics_weight[signal_and_cat_mask & hme_mask_scan])
                b = np.sum(physics_weight[Background_mask & hme_mask_scan])
                significance = s / np.sqrt(s + b + 1e-6)
                if significance > best_significance:
                    best_significance = significance
                    best_hme_low = hme_low_scan
                    best_hme_high = hme_high_scan

        print(
            f"For mass {para_masspoint} we have HME bounds [{best_hme_low}, {best_hme_high}]"
        )
        print(f"This had a significance of {best_significance}")
        print(
            f"Original mean, low, high was: {hme_mean}, {hme_mean - hme_std}, {hme_mean + hme_std}"
        )
        hme_mask = (hme_values > best_hme_low) & (hme_values < best_hme_high)

    else:
        hme_mask = True  # Skip hme filtering for now

    mask_dict = {
        class_names[0]: signal_mask & hme_mask,
    }
    for idx, class_name in enumerate(class_names[1:], start=1):
        mask_dict[class_name] = (dw.class_value == idx) & hme_mask

    return mask_dict, physics_weight


# =========================================================
# 5. CLASS HISTOGRAMS
# =========================================================
def make_hist(values, mask, weights, bins):
    values_masked = values[mask]
    weights_masked = weights[mask]
    h, _ = np.histogram(values_masked, bins=bins, weights=weights_masked)
    h2, _ = np.histogram(values_masked, bins=bins, weights=weights_masked**2)
    return h, np.sqrt(h2)


def make_hist2d(values_x, values_y, mask, weights, bins):
    h, _, _ = np.histogram2d(
        values_x[mask], values_y[mask], bins=bins, weights=weights[mask]
    )
    h2, _, _ = np.histogram2d(
        values_x[mask], values_y[mask], bins=bins, weights=weights[mask] ** 2
    )
    return h, np.sqrt(h2)


# =========================================================
# 6. ROOT WRITING
# =========================================================
def get_quantile_bins(values, mask, n_bins=50):
    """
    Build quantile bins from signal-only distribution
    """
    v = values[mask]

    if len(v) < 100:
        # fallback to uniform if statistics are too low
        return np.linspace(0, 1, n_bins + 1)

    return np.quantile(v, np.linspace(0, 1, n_bins + 1))


def write_root_outputs(
    ROOTOut,
    pred,
    dw,
    mask_dict,
    physics_weight,
    feature_values,
    setup,
    para_masspoint,
    process_tag,
    class_idx=0,
):

    # =========================================================
    # QUANTILE BINNING FROM SIGNAL (stage-2 target)
    # =========================================================

    default_class_names = [
        "Signal",
        "TT",
        "DY",
        "ST",
        "SMHiggs",
        "VV",
        "Other",
    ]
    class_names = setup.get("class_names", default_class_names)
    if class_idx >= len(class_names):
        raise IndexError(
            f"class_idx {class_idx} is out of range for class_names {class_names}"
        )
    class_idx_mask = mask_dict[class_names[class_idx]]

    # Quantile version
    nBins_quant = 50
    bin_low_quant, bin_high_quant = 0, 1
    pred_plot_quant = pred
    bins_quant = get_quantile_bins(pred_plot_quant, class_idx_mask, n_bins=nBins_quant)
    bins_quant[0] = 0.0
    bins_quant[-1] = 1.0

    # Raw version
    nBins_raw = 50
    bin_low_raw, bin_high_raw = 0, 1
    pred_plot_raw = pred
    bins_raw = np.linspace(0, 1, nBins_raw + 1)

    # Logit version
    nBins_logit = 150
    bin_low_logit, bin_high_logit = -7.5, 7.5
    pred_plot_logit = np.clip(pred, 1e-7, 1 - 1e-7)
    pred_plot_logit = np.log(pred_plot_logit / (1 - pred_plot_logit))
    bins_logit = np.linspace(bin_low_logit, bin_high_logit, nBins_logit + 1)

    print(
        f"Starting root output for class {setup["class_names"][class_idx]} on process {process_tag}"
    )

    ROOTOut.cd()
    plot_dir_2d = f"2D_plots_m{para_masspoint}"
    if not ROOTOut.GetDirectory(plot_dir_2d):
        ROOTOut.mkdir(plot_dir_2d)  # Always try, returns None or pointer

    dir2d = ROOTOut.GetDirectory(plot_dir_2d)
    if not dir2d:  # null-pointer check, not 'is None'
        # Try alternative: sometimes in ROOT, directories are not attached until written
        raise RuntimeError(
            f"Failed to create or access ROOT directory '{plot_dir_2d}'."
        )

    plot_dir_hme = f"HME_plots_m{para_masspoint}"
    if not ROOTOut.GetDirectory(plot_dir_hme):
        ROOTOut.mkdir(plot_dir_hme)  # Always try, returns None or pointer

    dirHME = ROOTOut.GetDirectory(plot_dir_hme)
    if not dirHME:  # null-pointer check, not 'is None'
        # Try alternative: sometimes in ROOT, directories are not attached until written
        raise RuntimeError("Failed to create or access ROOT directory 'HME_plots'.")

    plot_dir_hme2D = f"HME_vs_DNN_plots_m{para_masspoint}"
    if not ROOTOut.GetDirectory(plot_dir_hme2D):
        ROOTOut.mkdir(plot_dir_hme2D)  # Always try, returns None or pointer

    dirHMEvsDNN = ROOTOut.GetDirectory(plot_dir_hme2D)
    if not dirHMEvsDNN:  # null-pointer check, not 'is None'
        # Try alternative: sometimes in ROOT, directories are not attached until written
        raise RuntimeError(
            f"Failed to create or access ROOT directory '{plot_dir_hme2D}'."
        )

    plot_dir_raw = f"DNNRaw_m{para_masspoint}"
    if not ROOTOut.GetDirectory(plot_dir_raw):
        ROOTOut.mkdir(plot_dir_raw)  # Always try, returns None or pointer

    dirDNNRaw = ROOTOut.GetDirectory(plot_dir_raw)
    if not dirDNNRaw:  # null-pointer check, not 'is None'
        # Try alternative: sometimes in ROOT, directories are not attached until written
        raise RuntimeError(
            f"Failed to create or access ROOT directory '{plot_dir_raw}'."
        )

    plot_dir_quant = f"DNNQuantile_m{para_masspoint}"
    if not ROOTOut.GetDirectory(plot_dir_quant):
        ROOTOut.mkdir(plot_dir_quant)  # Always try, returns None or pointer

    dirDNNQuant = ROOTOut.GetDirectory(plot_dir_quant)
    if not dirDNNQuant:  # null-pointer check, not 'is None'
        # Try alternative: sometimes in ROOT, directories are not attached until written
        raise RuntimeError(
            f"Failed to create or access ROOT directory '{plot_dir_quant}'."
        )

    plot_dir_logit = f"DNNLogit_m{para_masspoint}"
    if not ROOTOut.GetDirectory(plot_dir_logit):
        ROOTOut.mkdir(plot_dir_logit)  # Always try, returns None or pointer

    dirDNNLogit = ROOTOut.GetDirectory(plot_dir_logit)
    if not dirDNNLogit:  # null-pointer check, not 'is None'
        # Try alternative: sometimes in ROOT, directories are not attached until written
        raise RuntimeError(
            f"Failed to create or access ROOT directory '{plot_dir_logit}'."
        )

    for pname, mask in mask_dict.items():

        print(f"On pname {pname}")

        # -----------------------
        # 1D DNN histogram
        # -----------------------
        h_raw = ROOT.TH1D(
            f"DNNRaw_{pname}_m{para_masspoint}_c{class_idx}",
            "",
            nBins_raw,
            bin_low_raw,
            bin_high_raw,
        )

        hist, err = make_hist(pred_plot_raw, mask, physics_weight, bins_raw)

        for i in range(nBins_raw):
            h_raw.SetBinContent(i + 1, hist[i])
            h_raw.SetBinError(i + 1, err[i])

        dirDNNRaw.WriteObject(
            h_raw, f"m{para_masspoint}_{pname}_class{setup['class_names'][class_idx]}"
        )

        h_quant = ROOT.TH1D(
            f"DNNQuant_{pname}_m{para_masspoint}_c{class_idx}",
            "",
            nBins_quant,
            bin_low_quant,
            bin_high_quant,
        )

        hist, err = make_hist(pred_plot_quant, mask, physics_weight, bins_quant)

        for i in range(nBins_quant):
            h_quant.SetBinContent(i + 1, hist[i])
            h_quant.SetBinError(i + 1, err[i])

        dirDNNQuant.WriteObject(
            h_quant, f"m{para_masspoint}_{pname}_class{setup['class_names'][class_idx]}"
        )

        h_logit = ROOT.TH1D(
            f"DNNLogit_{pname}_m{para_masspoint}_c{setup['class_names'][class_idx]}",
            "",
            nBins_logit,
            bin_low_logit,
            bin_high_logit,
        )

        hist, err = make_hist(pred_plot_logit, mask, physics_weight, bins_logit)

        for i in range(nBins_logit):
            h_logit.SetBinContent(i + 1, hist[i])
            h_logit.SetBinError(i + 1, err[i])

        dirDNNLogit.WriteObject(
            h_logit, f"m{para_masspoint}_{pname}_class{setup['class_names'][class_idx]}"
        )

        # -----------------------
        # 1D HME histogram
        # -----------------------
        if class_idx == 0:
            # Only do HME 1D on the class idx 0 since it should be repeats
            hme_bins = np.linspace(0.0, 2500.0, 251)
            dirHME.cd()
            h = ROOT.TH1D(f"HME_{pname}_m{para_masspoint}", "", 250, 0.0, 2500.0)

            hist, err = make_hist(feature_values["hme"], mask, physics_weight, hme_bins)

            for i in range(250):
                h.SetBinContent(i + 1, hist[i])
                h.SetBinError(i + 1, err[i])

            dirHME.WriteObject(h, f"HME_m{para_masspoint}_{pname}")

        # -----------------------
        # 2D DNN vs HME
        # -----------------------
        dirHMEvsDNN.cd()

        hme_bins = np.linspace(0.0, 2500.0, 251)

        h2_raw = ROOT.TH2D(
            f"DNNRaw_vs_HME_{pname}_m{para_masspoint}_c{class_idx}",
            "",
            nBins_raw,
            bin_low_raw,
            bin_high_raw,
            250,
            0,
            2500,
        )

        # Quantile binning must be done this way as it requires the binning to be applied (not a math transform on predictions)
        h2_quant = ROOT.TH2D(
            f"DNNQuant_vs_HME_{pname}_m{para_masspoint}_c{class_idx}",
            "",
            nBins_quant,
            bin_low_quant,
            bin_high_quant,
            250,
            0,
            2500,
        )
        hist, err = make_hist2d(
            pred_plot_quant,
            feature_values["hme"],
            mask,
            physics_weight,
            [bins_quant, hme_bins],
        )

        h2_logit = ROOT.TH2D(
            f"DNNLogit_vs_HME_{pname}_m{para_masspoint}_c{class_idx}",
            "",
            nBins_logit,
            bin_low_logit,
            bin_high_logit,
            250,
            0,
            2500,
        )

        for i in range(nBins_quant):
            for j in range(len(hme_bins) - 1):
                h2_quant.SetBinContent(i + 1, j + 1, hist[i][j])
                h2_quant.SetBinError(i + 1, j + 1, err[i][j])

        pred_plot_raw_masked = np.ascontiguousarray(
            pred_plot_raw[mask], dtype=np.float64
        )
        pred_plot_logit_masked = np.ascontiguousarray(
            pred_plot_logit[mask], dtype=np.float64
        )
        hme_masked = np.ascontiguousarray(feature_values["hme"][mask], dtype=np.float64)
        w_masked = np.ascontiguousarray(physics_weight[mask], dtype=np.float64)
        n_events = len(pred_plot_raw_masked)

        h2_raw.Sumw2()
        h2_logit.Sumw2()

        h2_raw.FillN(n_events, pred_plot_raw_masked, hme_masked, w_masked)
        h2_logit.FillN(n_events, pred_plot_logit_masked, hme_masked, w_masked)

        dirHMEvsDNN.WriteObject(
            h2_raw,
            f"DNNRaw_vs_HME_m{para_masspoint}_{pname}_class{setup['class_names'][class_idx]}",
        )

        dirHMEvsDNN.WriteObject(
            h2_quant,
            f"DNNQuant_vs_HME_m{para_masspoint}_{pname}_class{setup['class_names'][class_idx]}",
        )

        dirHMEvsDNN.WriteObject(
            h2_logit,
            f"DNNLogit_vs_HME_m{para_masspoint}_{pname}_class{setup['class_names'][class_idx]}",
        )

        # -----------------------
        # 2D DNN vs features
        # -----------------------
        for i, feat in enumerate(setup["features"]):

            feat_masked = np.ascontiguousarray(
                feature_values["features"][:, i], dtype=np.float64
            )
            x_min, x_max = np.min(feat_masked), np.max(feat_masked)

            h2f = ROOT.TH2D(
                f"DNN_vs_{feat}_{pname}_m{para_masspoint}_c{class_idx}",
                "",
                100,
                0,
                1,
                100,
                x_min,
                x_max,
            )

            h2f.Sumw2()
            h2f.FillN(n_events, pred_plot_raw_masked, feat_masked, w_masked)

            dir2d.WriteObject(
                h2f,
                f"DNN_vs_{feat}_m{para_masspoint}_{pname}_class{setup['class_names'][class_idx]}",
            )

        # -----------------------
        # 2D DNN vs Stage1 Score
        # -----------------------
        if "stage1_score" in feature_values.keys() and np.any(mask):
            stage1_values = feature_values["stage1_score"][mask]
            if stage1_values.size > 0:
                h2 = ROOT.TH2D(
                    f"DNN_vs_Stage1_Score_{pname}_m{para_masspoint}_c{class_idx}",
                    "",
                    100,
                    0,
                    1,
                    100,
                    0,
                    1,
                )

                for dnn, stage1 in zip(pred[mask], stage1_values):
                    # If stage1 is an array, take first entry
                    if isinstance(stage1, np.ndarray):
                        stage1 = stage1[0]
                    h2.Fill(dnn, stage1)

                dir2d.WriteObject(
                    h2,
                    f"DNN_vs_Stage1_Score_{pname}_m{para_masspoint}_class{setup['class_names'][class_idx]}",
                )

        ROOTOut.cd()


# =========================================================
# 7. ROC
# =========================================================
def plot_roc(ax, y_true, pred, weights, label):
    display = sklearn.metrics.RocCurveDisplay.from_predictions(
        y_true,
        pred,
        sample_weight=weights,
        ax=ax,
        name=label,
    )
    return display


def plot_confusion_matrix(
    ax, y_true, y_pred, labels, sample_weight=None, normalize="true", title=None
):
    display = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        sample_weight=sample_weight,
        normalize=normalize,
        ax=ax,
        cmap="Blues",
    )
    if title is not None:
        ax.set_title(title)
    return display


# =========================================================
# 8. OPTIONAL FEATURE IMPORTANCE (NOT USED IN FLOW)
# =========================================================
def permutation_importance_onnx(session, X, y, w, n_repeats=3):
    base = get_scores(session, X)[:, 0]
    base_auc = sklearn.metrics.roc_auc_score(y, base, sample_weight=w)

    imps = []

    for i in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            np.random.shuffle(Xp[:, i])
            p = get_scores(session, Xp)[:, 0]
            auc = sklearn.metrics.roc_auc_score(y, p, sample_weight=w)
            drops.append(base_auc - auc)

        imps.append(np.mean(drops))

    return base_auc, np.array(imps)


def feature_importance_scan(
    sess, X, y_true, w, cat, mass, feature_names, output_folder
):
    weights = np.clip(w, 0, None)

    X_feat = X
    y_feat = y_true
    w_feat = weights

    # subsample for speed (important for ROOT-scale datasets)
    max_events = 200_000_000
    if len(X_feat) > max_events:
        idx = np.random.choice(len(X_feat), max_events, replace=False)
        X_feat = X_feat[idx]
        y_feat = y_feat[idx]
        w_feat = w_feat[idx]

    baseline_auc, importances = permutation_importance_onnx(
        sess, X_feat, y_feat, w_feat, n_repeats=3  # stage-1 model
    )

    print(f"[FEATURE IMPORTANCE] baseline AUC = {baseline_auc:.5f}")

    # sort features
    sorted_idx = np.argsort(importances)[::-1]

    print("\nFeature ranking:")
    for rank, i in enumerate(sorted_idx):
        print(f"{rank+1:2d}. {feature_names[i]:30s} " f"ΔAUC = {importances[i]:.6f}")

    # =========================================================
    # SAVE PLOT
    # =========================================================

    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(importances)), importances[sorted_idx])

    plt.xticks(
        range(len(importances)), [feature_names[i] for i in sorted_idx], rotation=90
    )

    plt.ylabel("ΔAUC (importance)")
    plt.title(f"Feature Importance {cat} m{mass}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"importance_{cat}_m{mass}.pdf"))
    plt.close()


# =========================================================
# 9. MAIN PIPELINE (WITH 2-STAGE SUPPORT)
# =========================================================
def validate_dnn_pipeline(
    setup,
    validation_file,
    validation_weight_file,
    output_folder,
    model_stage1,
    model_stage2=None,
    model_config=None,
):

    os.makedirs(output_folder, exist_ok=True)

    if not setup.get("class_names", False):
        setup["class_names"] = [f"Class_{i}" for i in class_labels]

    sess1, config = load_model_and_config(model_stage1, model_config)

    sess2 = None
    if model_stage2 is not None:
        print("Loading Stage-2 model")
        sess2 = ort.InferenceSession(model_stage2)

    dw = prepare_datawrapper(setup, validation_file, validation_weight_file)

    hme_values = dw.GetHME(file_name=validation_file)
    nClasses = setup["nClasses"]

    for cat in ["res2b", "boosted", "res1b"]:

        for hme_cut in [True, False]:
            hme_cut_string = "_hme_cut" if hme_cut else ""

            ROOTOut = ROOT.TFile(
                os.path.join(output_folder, f"validation_{cat}{hme_cut_string}.root"),
                "RECREATE",
            )

            ROOTOut2 = (
                ROOT.TFile(
                    os.path.join(output_folder, f"validation_{cat}_stage2.root"),
                    "RECREATE",
                )
                if sess2
                else None
            )

            for mass in config["parametric_list"]:

                print(f"{cat} mass {mass} hme cut {hme_cut}")

                if dw.use_parametric:
                    dw.SetPredictParamValue(mass)

                X = dw.features_paramSet if dw.use_parametric else dw.features_no_param

                preds1 = get_scores(sess1, X)
                mask_dict, w = build_event_masks(
                    dw, hme_values, cat, mass, setup, hme_cut=hme_cut
                )

                if np.sum(w) == 0:
                    print(f"No events in cat {cat}, continue.")
                    continue

                feature_values = {
                    "hme": hme_values,
                    "features": dw.features,
                }

                # =================================================
                # STAGE 1 OUTPUTS (multi-class)
                # =================================================
                for c in range(nClasses):
                    write_root_outputs(
                        ROOTOut,
                        preds1[:, c],
                        dw,
                        mask_dict,
                        w,
                        feature_values,
                        setup,
                        mass,
                        cat,
                        class_idx=c,
                    )

                # =================================================
                # ROC (stage 1)
                # =================================================
                y = (dw.class_target == 0).astype(int)

                fig, ax = plt.subplots()
                plot_roc(ax, y, preds1[:, 0], np.clip(w, 0, None), f"{cat} m{mass}")

                ax.plot([0, 1], [0, 1], "--")
                plt.savefig(os.path.join(output_folder, f"ROC_{cat}_{mass}.pdf"))
                plt.close()

                # =========================================================
                # CONFUSION MATRIX (STAGE 1 MULTICLASS)
                # =========================================================
                y_true_labels = np.asarray(dw.class_target, dtype=np.int64)
                y_pred_labels = np.argmax(preds1, axis=1).astype(np.int64)
                class_labels = list(range(nClasses))
                class_names = setup.get(
                    "class_names", [f"Class_{i}" for i in class_labels]
                )

                fig, ax = plt.subplots(figsize=(8, 6))
                plot_confusion_matrix(
                    ax,
                    y_true_labels,
                    y_pred_labels,
                    labels=class_labels,
                    sample_weight=np.clip(w, 0, None),
                    normalize="true",
                    title=f"Stage1 Confusion Matrix {cat} m{mass}",
                )
                ax.set_xticks(class_labels)
                ax.set_xticklabels(class_names, rotation=45, ha="right")
                ax.set_yticks(class_labels)
                ax.set_yticklabels(class_names)
                fig.tight_layout()
                plt.savefig(
                    os.path.join(output_folder, f"ConfusionMatrix_{cat}_{mass}.pdf")
                )
                plt.close()

                # =========================================================
                # FEATURE IMPORTANCE (STAGE-1 BASELINE)
                # =========================================================

                # print(
                #     f"[INFO] Running stage-1 feature importance for {cat} mass {mass}"
                # )

                # y_true = (dw.class_target == 0).astype(int)

                # feature_names = (setup["features"]).copy()
                # feature_importance_folder = os.path.join(
                #     output_folder, "feature_importance"
                # )
                # feature_importance_scan(
                #     sess1, X, y_true, w, cat, mass, feature_names, output_folder
                # )

                if getattr(setup, "train_stage_2", False):
                    # =================================================
                    # STAGE 2 (binary refinement on signal-like events)
                    # =================================================
                    if sess2 is not None:

                        signal_like = np.argmax(preds1, axis=1) == 0

                        # preds1_nested = preds1[:,0].reshape(-1, 1)
                        preds1_nested = preds1
                        X2 = X[signal_like]
                        # X2 = np.concatenate([X[signal_like], preds1_nested[signal_like]], axis=1)

                        preds2 = get_scores(sess2, X2)
                        preds2 = preds2[:, 0]  # Still a 2class technically

                        mask_dict2 = {k: v[signal_like] for k, v in mask_dict.items()}

                        feature_values2 = {
                            "hme": hme_values[signal_like],
                            "features": dw.features[signal_like],
                            "stage1_score": preds1_nested[signal_like],
                        }

                        write_root_outputs(
                            ROOTOut2,
                            preds2,
                            dw,
                            mask_dict2,
                            w[signal_like],
                            feature_values2,
                            setup,
                            mass,
                            cat,
                            class_idx=0,
                        )

                        # =========================================================
                        # FEATURE IMPORTANCE (STAGE-2)
                        # =========================================================

                        print(
                            f"[INFO] Running stage-2 feature importance for {cat} mass {mass}"
                        )

                        y_true = (dw.class_target == 0).astype(int)

                        y_true_stage2 = y_true[signal_like]
                        w_stage2 = w[signal_like]
                        # feature_names.append("stage1_score_signal")
                        # feature_names.append("stage1_score_TT")
                        # feature_names.append("stage1_score_DY")
                        # feature_names.append("stage1_score_Other")
                        feature_importance_folder = os.path.join(
                            output_folder, "feature_importance_stage2"
                        )
                        feature_importance_scan(
                            sess2,
                            X2,
                            y_true_stage2,
                            w_stage2,
                            cat,
                            mass,
                            feature_names,
                            feature_importance_folder,
                        )

    print("Done.")

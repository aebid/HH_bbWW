# python3 /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/AnaProd/NNInterface.py --inModelDir /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/config/HH_bbtautau/nn_models --inFile /tmp/prsolank/luigi-tmp-416131263.root --outFileName /tmp/prsolank/luigi-tmp-862152055.root --uncConfig /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/config/Run2_2018/weights.yaml --globalConfig /afs/cern.ch/work/p/prsolank/private/FLAF_8thJan/config/HH_bbtautau/global.yaml --EraName e2018 --Mass 400 --Spin 2 --PairType 2

from __future__ import annotations
import os, sys
import gc
import numpy as np
import awkward as ak
import onnxruntime as ort
import psutil
import yaml
import os
import ROOT
import FLAF.Common.Utilities as Utilities
import Analysis.hh_bbww as analysis


class DNNProducer:
    def __init__(self, cfg, payload_name, period):

        self.cfg = cfg
        self.payload_name = payload_name
        self.period = period

        sys.path.append(os.environ["ANALYSIS_PATH"])

        load_features = set()
        columns_to_save = set()
        self.classes_to_save = set()

        self.cfg_dict = {
            "DL_Resolved": self.cfg.get("DL_Resolved", None),
            "DL_Boosted": self.cfg.get("DL_Boosted", None),
            "SL_Resolved": self.cfg.get("SL_Resolved", None),
            "SL_Boosted": self.cfg.get("SL_Boosted", None),
        }

        self.models = {}
        self.masses = self.cfg.get("masses")
        self.dnnConfig = {}

        for channel, cfg in self.cfg_dict.items():
            if cfg == None:
                print(f"Channel {channel} does not have DNN defined, skip.")
                continue
            self.models[channel] = {}
            self.dnnConfig[channel] = {}
            parametric = cfg.get("parametric", False)

            dnnFolder = os.path.join(
                os.environ["ANALYSIS_PATH"], "config", "DNN", cfg["version"]
            )

            for mass in self.masses:
                if parametric:
                    this_mass_folder = dnnFolder
                    mass = 0
                    # Set mass to 0 since parametric should have the same config each time
                    # This helps the mass loop happening in the ApplyDNN func

                else:
                    this_mass_folder = os.path.join(dnnFolder, f"m{mass}")
                file_name = os.path.join(this_mass_folder, "dnn_config.yaml")
                with open(file_name, "r") as file:
                    self.dnnConfig[channel][f"m{mass}"] = yaml.safe_load(file)

                load_features.update(self.dnnConfig[channel][f"m{mass}"]["features"])
                self.classes_to_save.update(
                    self.dnnConfig[channel][f"m{mass}"]["class_names"]
                )

                modelname_parity = self.dnnConfig[channel][f"m{mass}"][
                    "modelname_parity"
                ]
                self.dnnConfig[channel][f"m{mass}"]["model_paths"] = [
                    [f"{os.path.join(this_mass_folder, x)}.onnx", y]
                    for x, y in modelname_parity
                ]

        columns_to_save.update(
            [f"{self.payload_name}_{col}" for col in self.cfg["columns"]]
        )

        # What to save in tmp file
        load_features.update(["FullEventId", "event", "SL", "DL", "boosted"])
        self.vars_to_save = load_features

        print(f"Initialized with these classes to save {self.classes_to_save}")

    def run(self, array):
        print("Running DNN producer")

        array = self.ApplyDNN(array)
        array = self.selectDNN(array)

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

        for channel, all_dnnConfig in self.dnnConfig.items():
            if all_dnnConfig == None:
                print(f"Channel {channel} does not have DNN defined, skip.")
                continue

            masses = self.masses if ("m0" not in all_dnnConfig.keys()) else [0]
            for mass in masses:
                dnnConfig = all_dnnConfig[f"m{mass}"]
                models = dnnConfig["model_paths"]

                features = dnnConfig["features"]

                nClasses = (
                    dnnConfig["nClasses"] if "nClasses" in dnnConfig.keys() else 3
                )
                nParity = dnnConfig["nParity"] if "nParity" in dnnConfig.keys() else 4

                use_parametric = dnnConfig["use_parametric"]
                param_mass_list = dnnConfig["parametric_list"]

                class_names_list = (
                    dnnConfig["class_names"]
                    if "class_names" in dnnConfig.keys()
                    else ["Signal", "TT", "DY"]
                )

                nEvents = len(branches)
                print(f"Running DNN Over {nEvents} events")

                event_number = np.asarray(branches.event)
                if nParity != 1:
                    event_mod = event_number % nParity

                array = np.stack(
                    [
                        np.asarray(getattr(branches, feature_name), dtype=np.float32)
                        for feature_name in features
                    ],
                    axis=1,
                )

                if use_parametric:
                    final_array = np.empty(
                        (nEvents, array.shape[1] + 1), dtype=np.float32
                    )
                    final_array[:, :-1] = array

                all_predictions = np.zeros(
                    (len(param_mass_list), nEvents, nClasses), dtype=np.float32
                )

                for parityIdx, [onnx_name, parityfunc] in enumerate(models):
                    sess = ort.InferenceSession(onnx_name)
                    for param_idx, param_mass in enumerate(param_mass_list):
                        if use_parametric:
                            final_array[:, -1] = param_mass
                            input_array = final_array
                        else:
                            input_array = array

                        prediction = sess.run(None, {"x": input_array})
                        class_prediction = np.asarray(prediction[0], dtype=np.float32)

                        if nParity != 1:
                            mask = event_mod != parityIdx
                            class_prediction[~mask, :] = 0.0

                        all_predictions[param_idx] += class_prediction
                        del prediction, class_prediction

                if nParity != 1:
                    all_predictions /= nParity - 1

                # Last save the branches
                for param_idx, param_mass in enumerate(param_mass_list):
                    this_param_prediction = all_predictions[param_idx, :, :]
                    this_param_prediction_logit = np.clip(
                        this_param_prediction, 1e-7, 1 - 1e-7
                    )
                    this_param_prediction_logit = np.log(
                        this_param_prediction_logit / (1 - this_param_prediction_logit)
                    )

                    for class_idx, class_name in enumerate(class_names_list):
                        field_name = f"{channel}_M{param_mass}_{class_name}"
                        output_fields[field_name] = this_param_prediction_logit[
                            :, class_idx
                        ].copy()

                if use_parametric:
                    del final_array
                del array
                if nParity != 1:
                    del event_mod
                del all_predictions
                gc.collect()
                print("Finishing call, memory?")
                process = psutil.Process(os.getpid())
                mem_mb = process.memory_info().rss / 1024 / 1024
                print(f"Current memory usage: {mem_mb:.2f} MB")

        for field_name, values in output_fields.items():
            branches[field_name] = values
            print(f"Added field {field_name}")

        del output_fields

        return branches

    def selectDNN(self, branches):
        # Here we will take SL and DL and choose which branch to save as final column
        output_fields = {}

        # classes_to_save = ["Signal", "TT", "DY", "ST"]

        for mass in self.masses:
            for class_name in self.classes_to_save:
                field_name = f"M{mass}_{class_name}"
                # Build the empty branches with ones
                if f"SL_Resolved_{field_name}" not in branches.fields:
                    branches[f"SL_Resolved_{field_name}"] = np.zeros_like(
                        branches.event
                    )
                if f"SL_Boosted_{field_name}" not in branches.fields:
                    branches[f"SL_Boosted_{field_name}"] = np.zeros_like(branches.event)
                if f"DL_Resolved_{field_name}" not in branches.fields:
                    branches[f"DL_Resolved_{field_name}"] = np.zeros_like(
                        branches.event
                    )
                if f"DL_Boosted_{field_name}" not in branches.fields:
                    branches[f"DL_Boosted_{field_name}"] = np.zeros_like(branches.event)
                output_fields[field_name] = np.where(
                    branches.SL,
                    np.where(
                        branches.boosted,
                        branches[f"SL_Boosted_{field_name}"],
                        branches[f"SL_Resolved_{field_name}"],
                    ),
                    np.where(
                        branches.boosted,
                        branches[f"DL_Boosted_{field_name}"],
                        branches[f"DL_Resolved_{field_name}"],
                    ),
                )

        for field_name, values in output_fields.items():
            branches[field_name] = values
        del output_fields
        return branches

import numpy as np
import Analysis.hh_bbww as analysis
import FLAF.Common.OnnxSession as OnnxSession
import FLAF.Common.Utilities as Utilities
import awkward as ak
import os
from DeepHME.src.DeepHME import DeepHME

higs_output_mapping = {
    "Hbb": {"px": 4, "py": 5, "pz": 6, "E": 7},
    "HVV": {"px": 0, "py": 1, "pz": 2, "E": 3},
}


class DeepHMEProducer:
    def __init__(self, cfg, payload_name, era):
        self.cfg = cfg
        self.payload_name = payload_name
        self.era = era

        self.cfg_dict = {
            "DL": self.cfg.get("DL", None),
            "SL": self.cfg.get("SL", None),
        }

        load_features = set()

        self.estimators = {}

        for channel, cfg in self.cfg_dict.items():
            if cfg == None:
                print(f"Channel {channel} does not have DNN defined, skip.")
                continue

            deepHME_folder = os.path.join(
                os.environ["ANALYSIS_PATH"], "config", "DeepHME", cfg["model_name"]
            )

            with open(os.path.join(deepHME_folder, "deepHME_config.txt"), "r") as file:
                load_features.update([line[:-1] for line in file.readlines()])

            self.estimators[channel] = DeepHME(
                model_name=cfg["model_name"],
                channel=channel,
                return_errors=self.cfg["return_errors"],
                session_options=OnnxSession.session_options(),
            )

        load_features.update(["FullEventId", "event", "SL", "DL"])
        self.vars_to_save = load_features

    def run(self, array, keep_all_columns=False):
        for channel, cfg in self.cfg_dict.items():
            if cfg == None:
                print(f"Channel {channel} does not have DeepHME defined, skip.")
                continue
            estimator = self.estimators[channel]

            mass = None
            mass_errors = None

            pred = estimator.predict(
                event_id=array["event"],
                lep1_pt=array["lep1_pt"],
                lep1_eta=array["lep1_eta"],
                lep1_phi=array["lep1_phi"],
                lep1_mass=array["lep1_mass"],
                lep2_pt=array["lep2_pt"] if channel == "DL" else None,
                lep2_eta=array["lep2_eta"] if channel == "DL" else None,
                lep2_phi=array["lep2_phi"] if channel == "DL" else None,
                lep2_mass=array["lep2_mass"] if channel == "DL" else None,
                met_pt=array["PuppiMET_pt"],
                met_phi=array["PuppiMET_phi"],
                jet_pt=array["centralJet_pt"],
                jet_eta=array["centralJet_eta"],
                jet_phi=array["centralJet_phi"],
                jet_mass=array["centralJet_mass"],
                jet_btagPNetB=(
                    array["centralJet_btagPNetB"]
                    if "centralJet_btagPNetB" in self.vars_to_save
                    else None
                ),
                jet_btagPNetCvB=(
                    array["centralJet_btagPNetCvB"]
                    if "centralJet_btagPNetCvB" in self.vars_to_save
                    else None
                ),
                jet_btagPNetCvL=(
                    array["centralJet_btagPNetCvL"]
                    if "centralJet_btagPNetCvL" in self.vars_to_save
                    else None
                ),
                jet_btagPNetCvNotB=(
                    array["centralJet_btagPNetCvNotB"]
                    if "centralJet_btagPNetCvNotB" in self.vars_to_save
                    else None
                ),
                jet_btagPNetQvG=(
                    array["centralJet_btagPNetQvG"]
                    if "centralJet_btagPNetQvG" in self.vars_to_save
                    else None
                ),
                jet_PNetRegPtRawCorr=(
                    array["centralJet_PNetRegPtRawCorr"]
                    if "centralJet_PNetRegPtRawCorr" in self.vars_to_save
                    else None
                ),
                jet_PNetRegPtRawCorrNeutrino=(
                    array["centralJet_PNetRegPtRawCorrNeutrino"]
                    if "centralJet_PNetRegPtRawCorrNeutrino" in self.vars_to_save
                    else None
                ),
                jet_PNetRegPtRawRes=(
                    array["centralJet_PNetRegPtRawRes"]
                    if "centralJet_PNetRegPtRawRes" in self.vars_to_save
                    else None
                ),
                fatjet_pt=array["SelectedFatJet_pt"],
                fatjet_eta=array["SelectedFatJet_eta"],
                fatjet_phi=array["SelectedFatJet_phi"],
                fatjet_mass=array["SelectedFatJet_mass"],
                fatjet_particleNet_QCD=(
                    array["SelectedFatJet_particleNet_QCD"]
                    if "SelectedFatJet_particleNet_QCD" in self.vars_to_save
                    else None
                ),
                fatjet_particleNet_XbbVsQCD=(
                    array["SelectedFatJet_particleNet_XbbVsQCD"]
                    if "SelectedFatJet_particleNet_XbbVsQCD" in self.vars_to_save
                    else None
                ),
                fatjet_particleNetWithMass_QCD=(
                    array["SelectedFatJet_particleNetWithMass_QCD"]
                    if "SelectedFatJet_particleNetWithMass_QCD" in self.vars_to_save
                    else None
                ),
                fatjet_particleNetWithMass_HbbvsQCD=(
                    array["SelectedFatJet_particleNetWithMass_HbbvsQCD"]
                    if "SelectedFatJet_particleNetWithMass_HbbvsQCD"
                    in self.vars_to_save
                    else None
                ),
                fatjet_particleNet_massCorr=(
                    array["SelectedFatJet_particleNet_massCorr"]
                    if "SelectedFatJet_particleNet_massCorr" in self.vars_to_save
                    else None
                ),
                output_format="mass",
            )

            if self.cfg["return_errors"]:
                mass, mass_errors = pred
            else:
                mass = pred

            # for now we decided to not return errors for components of p4
            pred = estimator.predict(
                event_id=array["event"],
                lep1_pt=array["lep1_pt"],
                lep1_eta=array["lep1_eta"],
                lep1_phi=array["lep1_phi"],
                lep1_mass=array["lep1_mass"],
                lep2_pt=array["lep2_pt"] if channel == "DL" else None,
                lep2_eta=array["lep2_eta"] if channel == "DL" else None,
                lep2_phi=array["lep2_phi"] if channel == "DL" else None,
                lep2_mass=array["lep2_mass"] if channel == "DL" else None,
                met_pt=array["PuppiMET_pt"],
                met_phi=array["PuppiMET_phi"],
                jet_pt=array["centralJet_pt"],
                jet_eta=array["centralJet_eta"],
                jet_phi=array["centralJet_phi"],
                jet_mass=array["centralJet_mass"],
                jet_btagPNetB=(
                    array["centralJet_btagPNetB"]
                    if "centralJet_btagPNetB" in self.vars_to_save
                    else None
                ),
                jet_btagPNetCvB=(
                    array["centralJet_btagPNetCvB"]
                    if "centralJet_btagPNetCvB" in self.vars_to_save
                    else None
                ),
                jet_btagPNetCvL=(
                    array["centralJet_btagPNetCvL"]
                    if "centralJet_btagPNetCvL" in self.vars_to_save
                    else None
                ),
                jet_btagPNetCvNotB=(
                    array["centralJet_btagPNetCvNotB"]
                    if "centralJet_btagPNetCvNotB" in self.vars_to_save
                    else None
                ),
                jet_btagPNetQvG=(
                    array["centralJet_btagPNetQvG"]
                    if "centralJet_btagPNetQvG" in self.vars_to_save
                    else None
                ),
                jet_PNetRegPtRawCorr=(
                    array["centralJet_PNetRegPtRawCorr"]
                    if "centralJet_PNetRegPtRawCorr" in self.vars_to_save
                    else None
                ),
                jet_PNetRegPtRawCorrNeutrino=(
                    array["centralJet_PNetRegPtRawCorrNeutrino"]
                    if "centralJet_PNetRegPtRawCorrNeutrino" in self.vars_to_save
                    else None
                ),
                jet_PNetRegPtRawRes=(
                    array["centralJet_PNetRegPtRawRes"]
                    if "centralJet_PNetRegPtRawRes" in self.vars_to_save
                    else None
                ),
                fatjet_pt=array["SelectedFatJet_pt"],
                fatjet_eta=array["SelectedFatJet_eta"],
                fatjet_phi=array["SelectedFatJet_phi"],
                fatjet_mass=array["SelectedFatJet_mass"],
                fatjet_particleNet_QCD=(
                    array["SelectedFatJet_particleNet_QCD"]
                    if "SelectedFatJet_particleNet_QCD" in self.vars_to_save
                    else None
                ),
                fatjet_particleNet_XbbVsQCD=(
                    array["SelectedFatJet_particleNet_XbbVsQCD"]
                    if "SelectedFatJet_particleNet_XbbVsQCD" in self.vars_to_save
                    else None
                ),
                fatjet_particleNetWithMass_QCD=(
                    array["SelectedFatJet_particleNetWithMass_QCD"]
                    if "SelectedFatJet_particleNetWithMass_QCD" in self.vars_to_save
                    else None
                ),
                fatjet_particleNetWithMass_HbbvsQCD=(
                    array["SelectedFatJet_particleNetWithMass_HbbvsQCD"]
                    if "SelectedFatJet_particleNetWithMass_HbbvsQCD"
                    in self.vars_to_save
                    else None
                ),
                fatjet_particleNet_massCorr=(
                    array["SelectedFatJet_particleNet_massCorr"]
                    if "SelectedFatJet_particleNet_massCorr" in self.vars_to_save
                    else None
                ),
                output_format="p4",
            )

            p4 = None
            p4_errors = None
            if self.cfg["return_errors"]:
                p4, p4_errors = pred
            else:
                p4 = pred

            array[f"{self.payload_name}_{channel}_mass"] = mass
            if "mass_error" in self.cfg["columns"]:
                array[f"{self.payload_name}_{channel}_mass_error"] = mass_errors

            # save energies and momenta of higgs bosons
            higgses = ["Hbb", "HVV"]
            for col in self.cfg["columns"]:
                var = col.split("_")[-1]
                for higgs in higgses:
                    if higgs in col:
                        assert var in [
                            "px",
                            "py",
                            "pz",
                            "E",
                        ], "Attempting to save non-kinematic variable from predicted p4"
                        idx = higs_output_mapping[higgs][var]
                        array[f"{self.payload_name}_{channel}_{higgs}_{var}"] = p4[
                            :, idx
                        ]

        output_fields = {}

        all_output_vars = ["mass", "mass_error"]
        for higgs in ["Hbb", "HVV"]:
            for var in ["px", "py", "pz", "E"]:
                all_output_vars.append(f"{higgs}_{var}")

        for channel in ["SL", "DL"]:
            for var in all_output_vars:
                if f"{self.payload_name}_{channel}_{var}" not in array.fields:
                    array[f"{self.payload_name}_{channel}_{var}"] = np.zeros_like(
                        array.event
                    )

        for var in all_output_vars:
            output_fields[f"{var}"] = np.where(
                array.SL,
                array[f"{self.payload_name}_SL_{var}"],
                array[f"{self.payload_name}_DL_{var}"],
            )

        for field_name, values in output_fields.items():
            array[field_name] = values
        del output_fields

        # Delete not-needed array
        for col in array.fields:
            if col not in self.cfg["columns"]:
                if col != "FullEventId":
                    if keep_all_columns:
                        continue
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

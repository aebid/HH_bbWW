import os
import uproot
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob

import ROOT
import FLAF.RunKit.grid_tools as grid_tools
from FLAF.RunKit.run_tools import ps_call

ROOT.gROOT.SetBatch(True)
ROOT.EnableThreadSafety()
ROOT.EnableImplicitMT(4)


log_variables = [
    # "lep1_pt",
    # "lep2_pt",
    # "PuppiMET_pt",
    # "HT",
    # "MT",
    # "MT2_ll",
    # "MT2_bb",
    # "MT2_blbl",
    # "MT2_blbl2",
    # "ll_mass",
    # "bjet1_pt",
    # "bjet1_mass",
    # "bjet2_pt",
    # "bjet2_mass",
    # "wjet1_pt",
    # "wjet1_mass",
    # "wjet2_pt",
    # "wjet2_mass",
    # "fatbjet_pt",
    # "fatbjet_mass_PNetCorr",
    # "DoubleLep_DeepHME_mass",
]


def add_extra_vars(rdf_tmp, class_value, X_mass):
    rdf_tmp = rdf_tmp.Define("class_value", f"{class_value}")
    rdf_tmp = rdf_tmp.Define("X_mass", f"{X_mass}")
    rdf_tmp = rdf_tmp.Define("lep1_legType", "int(channelId/10.0)")
    rdf_tmp = rdf_tmp.Define("lep2_legType", "int(channelId%10)")
    # rdf_tmp = rdf_tmp.Define(
    #     "DoubleLep_DeepHME_mass_error_rel",
    #     "float(DoubleLep_DeepHME_mass_error)/float(DoubleLep_DeepHME_mass)",
    # )
    rdf_tmp = rdf_tmp.Define(
        "b1_p4",
        f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(bjet1_pt, bjet1_eta, bjet1_phi, bjet1_mass)",
    )
    rdf_tmp = rdf_tmp.Define(
        "b2_p4",
        f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(bjet2_pt, bjet2_eta, bjet2_phi, bjet2_mass)",
    )
    rdf_tmp = rdf_tmp.Define(
        "j1_p4",
        f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(wjet1_pt, wjet1_eta, wjet1_phi, wjet1_mass)",
    )
    rdf_tmp = rdf_tmp.Define(
        "j2_p4",
        f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(wjet2_pt, wjet2_eta, wjet2_phi, wjet2_mass)",
    )
    rdf_tmp = rdf_tmp.Define(
        "fatjet_p4",
        f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(fatbjet_pt, fatbjet_eta, fatbjet_phi, fatbjet_mass)",
    )
    rdf_tmp = rdf_tmp.Define(
        "lep1_p4",
        f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(lep1_pt, lep1_eta, lep1_phi, lep1_mass)",
    )
    rdf_tmp = rdf_tmp.Define(
        "lep2_p4",
        f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(lep2_pt, lep2_eta, lep2_phi, lep2_mass)",
    )
    rdf_tmp = rdf_tmp.Define(
        "met_p4",
        f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(PuppiMET_pt, 0, PuppiMET_phi, 0)",
    )

    rdf_tmp = rdf_tmp.Define(
        "dR_b1leps",
        "TMath::Min(ROOT::Math::VectorUtil::DeltaR(b1_p4, lep1_p4), ROOT::Math::VectorUtil::DeltaR(b1_p4, lep2_p4))",
    )
    rdf_tmp = rdf_tmp.Define(
        "dR_b2leps",
        "TMath::Min(ROOT::Math::VectorUtil::DeltaR(b2_p4, lep1_p4), ROOT::Math::VectorUtil::DeltaR(b2_p4, lep2_p4))",
    )

    rdf_tmp = rdf_tmp.Define(
        "m_b1leps",
        "ROOT::Math::VectorUtil::DeltaR(b1_p4, lep1_p4) < ROOT::Math::VectorUtil::DeltaR(b1_p4, lep2_p4) ? (b1_p4 + lep1_p4).M() : (b1_p4 + lep2_p4).M()",
    )
    rdf_tmp = rdf_tmp.Define(
        "m_b2leps",
        "ROOT::Math::VectorUtil::DeltaR(b2_p4, lep1_p4) < ROOT::Math::VectorUtil::DeltaR(b2_p4, lep2_p4) ? (b2_p4 + lep1_p4).M() : (b2_p4 + lep2_p4).M()",
    )

    # rdf_tmp = rdf_tmp.Define("m_b1l1", "(b1_p4 + lep1_p4).M()")
    # rdf_tmp = rdf_tmp.Define("m_b1l2", "(b1_p4 + lep2_p4).M()")
    # rdf_tmp = rdf_tmp.Define("m_b2l1", "(b2_p4 + lep1_p4).M()")
    # rdf_tmp = rdf_tmp.Define("m_b2l2", "(b2_p4 + lep2_p4).M()")

    # rdf_tmp = rdf_tmp.Define("dR_b1l1", "ROOT::Math::VectorUtil::DeltaR(b1_p4, lep1_p4)")
    # rdf_tmp = rdf_tmp.Define("dR_b1l2", "ROOT::Math::VectorUtil::DeltaR(b1_p4, lep2_p4)")
    # rdf_tmp = rdf_tmp.Define("dR_b2l1", "ROOT::Math::VectorUtil::DeltaR(b2_p4, lep1_p4)")
    # rdf_tmp = rdf_tmp.Define("dR_b2l2", "ROOT::Math::VectorUtil::DeltaR(b2_p4, lep2_p4)")

    rdf_tmp = rdf_tmp.Define("pt_ll", "(lep1_p4 + lep2_p4).Pt()")
    rdf_tmp = rdf_tmp.Define("pt_bb", "(b1_p4 + b2_p4).Pt()")
    rdf_tmp = rdf_tmp.Define("m_llmet", "(lep1_p4 + lep2_p4 + met_p4).M()")
    rdf_tmp = rdf_tmp.Define(
        "m_bbllmet", "(b1_p4 + b2_p4 + lep1_p4 + lep2_p4 + met_p4).M()"
    )

    # Begin Run2 block
    rdf_tmp = rdf_tmp.Define("lep1_E", "(lep1_p4).E()")
    rdf_tmp = rdf_tmp.Define("lep1_px", "(lep1_p4).px()")
    rdf_tmp = rdf_tmp.Define("lep1_py", "(lep1_p4).py()")
    rdf_tmp = rdf_tmp.Define("lep1_pz", "(lep1_p4).pz()")

    rdf_tmp = rdf_tmp.Define("lep2_E", "(lep2_p4).E()")
    rdf_tmp = rdf_tmp.Define("lep2_px", "(lep2_p4).px()")
    rdf_tmp = rdf_tmp.Define("lep2_py", "(lep2_p4).py()")
    rdf_tmp = rdf_tmp.Define("lep2_pz", "(lep2_p4).pz()")

    rdf_tmp = rdf_tmp.Define("bjet1_E", "(b1_p4).E()")
    rdf_tmp = rdf_tmp.Define("bjet1_px", "(b1_p4).px()")
    rdf_tmp = rdf_tmp.Define("bjet1_py", "(b1_p4).py()")
    rdf_tmp = rdf_tmp.Define("bjet1_pz", "(b1_p4).pz()")

    rdf_tmp = rdf_tmp.Define("bjet2_E", "(b2_p4).E()")
    rdf_tmp = rdf_tmp.Define("bjet2_px", "(b2_p4).px()")
    rdf_tmp = rdf_tmp.Define("bjet2_py", "(b2_p4).py()")
    rdf_tmp = rdf_tmp.Define("bjet2_pz", "(b2_p4).pz()")

    rdf_tmp = rdf_tmp.Define("jet3_E", "(j1_p4).E()")
    rdf_tmp = rdf_tmp.Define("jet3_px", "(j1_p4).px()")
    rdf_tmp = rdf_tmp.Define("jet3_py", "(j1_p4).py()")
    rdf_tmp = rdf_tmp.Define("jet3_pz", "(j1_p4).pz()")

    rdf_tmp = rdf_tmp.Define("jet4_E", "(j2_p4).E()")
    rdf_tmp = rdf_tmp.Define("jet4_px", "(j2_p4).px()")
    rdf_tmp = rdf_tmp.Define("jet4_py", "(j2_p4).py()")
    rdf_tmp = rdf_tmp.Define("jet4_pz", "(j2_p4).pz()")

    rdf_tmp = rdf_tmp.Define("fatjet_E", "(fatjet_p4).E()")
    rdf_tmp = rdf_tmp.Define("fatjet_px", "(fatjet_p4).px()")
    rdf_tmp = rdf_tmp.Define("fatjet_py", "(fatjet_p4).py()")
    rdf_tmp = rdf_tmp.Define("fatjet_pz", "(fatjet_p4).pz()")

    rdf_tmp = rdf_tmp.Define("met_E", "(met_p4).E()")
    rdf_tmp = rdf_tmp.Define("met_px", "(met_p4).px()")
    rdf_tmp = rdf_tmp.Define("met_py", "(met_p4).py()")
    rdf_tmp = rdf_tmp.Define("met_pz", "(met_p4).pz()")

    for var in log_variables:
        rdf_tmp = rdf_tmp.Define(
            f"{var}_log", f"TMath::Log(({var} > 0 ? {var} : 0) + 1.0)"
        )

    cols_to_save = [col for col in rdf_tmp.GetColumnNames() if not col.endswith("_p4")]

    return rdf_tmp, cols_to_save


def measure_cut_datasets(config_dict, output_folder, remote=False):
    storage_folder = os.path.join(config_dict["storage_folder"])

    process_dict = {}

    iterate_cut = config_dict["iterate_cut"]
    parity_cut = config_dict["parity_cut"]

    signal_list = config_dict["signal"]

    background_list = config_dict["background"]

    for nParity in range(config_dict["nParity"]):
        output_nParity = os.path.join(output_folder, f"nParity{nParity}_Merged")
        os.makedirs(output_nParity, exist_ok=True)

        nParity_string = f"nParity_{nParity}"

        process_dict[nParity_string] = {}

        for signal_name in config_dict["signal"]:
            process_dict[nParity_string][signal_name] = {}
            signal_dict = config_dict["signal"][signal_name]
            mass_points = signal_dict["mass_points"]
            for mass_point in mass_points:
                process_dict[nParity_string][signal_name][mass_point] = {
                    "total": 0,
                    "total_cut": 0,
                    "total_cut_weighted": 0.0,
                }

        for background_name in config_dict["background"]:
            process_dict[nParity_string][background_name] = {}
            background_dict = config_dict["background"][background_name]
            dataset_names = background_dict["background_datasets"]
            for dataset_name in dataset_names:
                process_dict[nParity_string][background_name][dataset_name] = {
                    "total": 0,
                    "total_cut": 0,
                    "total_cut_weighted": 0.0,
                }

    print("Looping signal datasets")
    for signal_name in signal_list:
        signal_dict = config_dict["signal"][signal_name]
        mass_points = signal_dict["mass_points"]
        dataset_name_format = signal_dict["dataset_name_format"]
        class_value = signal_dict["class_value"]

        for mass_point in tqdm(mass_points):
            X_mass = mass_point
            dataset_name = dataset_name_format.format(mass_point)

            process_dir = os.path.join(storage_folder, dataset_name)

            treeName = "Events"

            if remote:
                input_files = f"root://cmseos.fnal.gov/{process_dir}/*.root"
                rdf = ROOT.RDataFrame(treeName, input_files)
            else:
                input_files = f"{process_dir}/*.root"
                input_file_list = glob.glob(input_files)
                rdf = ROOT.RDataFrame(treeName, input_file_list)

            total = rdf.Count().GetValue()
            rdf = rdf.Filter(iterate_cut)

            for nParity in range(config_dict["nParity"]):
                nParity_string = f"nParity_{nParity}"
                parity_cut_formatted = parity_cut.format(
                    nParity=config_dict["nParity"], parity_scan=nParity
                )
                output_nParity = os.path.join(output_folder, f"nParity{nParity}_Merged")
                output_file = os.path.join(output_nParity, f"{dataset_name}_merge.root")

                rdf_tmp = rdf.Filter(parity_cut_formatted)
                cut = rdf_tmp.Count().GetValue()
                weighted_cut = rdf_tmp.Sum("weight_Central").GetValue()

                process_dict[nParity_string][signal_name][mass_point]["total"] = total
                process_dict[nParity_string][signal_name][mass_point]["total_cut"] = cut
                process_dict[nParity_string][signal_name][mass_point][
                    "total_cut_weighted"
                ] = weighted_cut

                # rdf_tmp, cols_to_save = add_extra_vars(rdf_tmp, class_value, X_mass)
                # rdf_tmp.Snapshot(treeName, output_file, cols_to_save)
                rdf_tmp = rdf_tmp.Define("class_value", f"{class_value}")
                rdf_tmp = rdf_tmp.Define("X_mass", f"{X_mass}")
                rdf_tmp.Snapshot(treeName, output_file)

    for background_name in background_list:
        background_dict = config_dict["background"][background_name]
        dataset_names = background_dict["background_datasets"]
        class_value = background_dict["class_value"]
        X_mass = 0

        print(f"Looping background {background_name}")
        for dataset_name in tqdm(dataset_names):
            process_dir = os.path.join(storage_folder, dataset_name)

            if remote:
                input_files = f"root://cmseos.fnal.gov/{process_dir}/*.root"
            else:
                input_files = f"{process_dir}/*.root"
            treeName = "Events"
            rdf = ROOT.RDataFrame(treeName, input_files)

            total = rdf.Count().GetValue()
            rdf = rdf.Filter(iterate_cut)

            for nParity in range(config_dict["nParity"]):
                nParity_string = f"nParity_{nParity}"
                parity_cut_formatted = parity_cut.format(
                    nParity=config_dict["nParity"], parity_scan=nParity
                )
                output_nParity = os.path.join(output_folder, f"nParity{nParity}_Merged")
                output_file = os.path.join(output_nParity, f"{dataset_name}_merge.root")

                rdf_tmp = rdf.Filter(parity_cut_formatted)
                cut = rdf_tmp.Count().GetValue()
                weighted_cut = rdf_tmp.Sum("weight_Central").GetValue()

                process_dict[nParity_string][background_name][dataset_name][
                    "total"
                ] = total
                process_dict[nParity_string][background_name][dataset_name][
                    "total_cut"
                ] = cut
                process_dict[nParity_string][background_name][dataset_name][
                    "total_cut_weighted"
                ] = weighted_cut

                # rdf_tmp, cols_to_save = add_extra_vars(rdf_tmp, class_value, X_mass)
                # rdf_tmp.Snapshot(treeName, output_file, cols_to_save)
                rdf_tmp = rdf_tmp.Define("class_value", f"{class_value}")
                rdf_tmp = rdf_tmp.Define("X_mass", f"{X_mass}")
                rdf_tmp.Snapshot(treeName, output_file)

    for nParity in range(config_dict["nParity"]):
        nParity_string = f"nParity_{nParity}"
        out_yaml = f"dataset_distribution_parity{nParity}.yaml"
        with open(os.path.join(output_folder, out_yaml), "w") as outfile:
            yaml.dump(process_dict[nParity_string], outfile)


def hadd_files(config_dict, output_folder):
    for nParity in range(config_dict["nParity"]):
        # hadd the files together to make a final merged.root
        hadd_out = os.path.join(output_folder, f"nParity{nParity}_Merged.root")
        hadd_in = os.path.join(output_folder, f"nParity{nParity}_Merged/*.root")
        os.system(f"hadd {hadd_out} {hadd_in}")


def add_weight_file(output_folder, mass=None):
    inNames = [
        os.path.join(output_folder, x)
        for x in os.listdir(output_folder)
        if x.endswith(".root")
    ]
    for inName in inNames:
        if "weight" in inName:
            continue
        print(f"On file {inName}")
        in_file = uproot.open(inName)
        outName = f"{inName[:-5]}_weight.root"
        if mass != None:
            outName = f"{inName[:-5]}_weight_m{mass}.root"
        out_file = uproot.recreate(outName)

        tree = in_file["Events"]
        branches_to_load = [
            "class_value",
            "X_mass",
            "weight_Central",
        ]
        branches = tree.arrays(branches_to_load)

        X_mass = branches["X_mass"]
        class_targets = branches["class_value"]
        class_weight = branches["weight_Central"]

        # Set all signals to target 0
        class_targets = np.where(class_targets <= 0, 0, class_targets)

        # Set to binary for now actually
        class_targets_binary = np.where(class_targets <= 0, 0, 1)

        # Set any negative weight events to 0
        class_weight = np.where(class_weight <= 0, 0.0, class_weight)
        # jk

        # Clip weights to be within +- 3 std of mean
        mean_weight = np.mean(np.abs(class_weight))
        std = np.std(np.abs(class_weight))
        print(f"Normalizing from {mean_weight} +- {std}")
        class_weight = np.clip(
            class_weight, -(mean_weight + (3 * std)), (mean_weight + (3 * std))
        )

        # Set specific masses if you want
        # class_weight = np.where((class_targets == 0) & ( (X_mass < 600) | (X_mass > 1000) ), 0.0, class_weight)
        if mass != None:
            class_weight = np.where(
                (class_targets == 0) & ((X_mass != mass)), 0.0, class_weight
            )

        # Total_Signal == Total_Background
        # Scale total signal up to total background
        total_signal = np.sum(np.where(class_targets == 0, class_weight, 0.0))
        total_background = np.sum(np.where(class_targets != 0, class_weight, 0.0))

        print(f"Total signal: {total_signal}")
        print(f"Total background: {total_background}")
        norm_factor = total_background / total_signal
        class_weight = np.where(
            class_targets != 0, class_weight, class_weight * norm_factor
        )

        print(f"After reweight")
        print(
            f"Total signal: {np.sum(np.where(class_targets == 0, class_weight, 0.0))}"
        )
        print(
            f"Total background: {np.sum(np.where(class_targets != 0, class_weight, 0.0))}"
        )

        # Total_Background1 == Total_Background2 == Total_Background3
        # Scale each background to total, then reduce all to total
        ### Do not scale backgrounds to each other in binary classifier ###
        total_background = np.sum(np.where(class_targets != 0, class_weight, 0.0))
        multiclass_weight = np.copy(class_weight)
        for class_value in np.unique(class_targets):
            if class_value == 0:
                continue  # Don't do anything with signal here
            this_total = np.sum(
                np.where(class_targets == class_value, class_weight, 0.0)
            )
            rescale_factor = total_background / this_total
            multiclass_weight = np.where(
                class_targets == class_value,
                multiclass_weight * rescale_factor,
                multiclass_weight,
            )
        # current_total = np.sum(np.where(class_targets != 0, multiclass_weight, 0.0))
        # rescale_factor = total_background / current_total
        # multiclass_weight = np.where(
        #     class_targets != 0, multiclass_weight*rescale_factor, multiclass_weight
        # )

        # Scale background to nMasses being used
        # mass_cut = (class_targets == 0) * (class_weight > 0.0)
        # nMasses = len(np.unique(X_mass[mass_cut]))
        # print(f"We have {len(np.unique(X_mass[mass_cut]))} unique masses {np.unique(X_mass[mass_cut])}")
        # rescale_factor = nMasses
        # class_weight = np.where(
        #     class_targets != 0, class_weight*rescale_factor, class_weight
        # )

        print(f"Final reweight")
        print(
            f"Total signal: {np.sum(np.where(class_targets == 0, class_weight, 0.0))}"
        )
        print(
            f"Total background: {np.sum(np.where(class_targets != 0, class_weight, 0.0))}"
        )
        print(f"And multiclass reweight")
        print(
            f"Total signal: {np.sum(np.where(class_targets == 0, multiclass_weight, 0.0))}"
        )
        print(
            f"Total background: {np.sum(np.where(class_targets != 0, multiclass_weight, 0.0))}"
        )

        counts, bin_edges = np.histogram(
            branches["class_value"],
            bins=15,
            range=(-5,10),
            weights=class_weight
        )
        weighted_histogram = (counts, bin_edges)

        counts_multiclass, bin_edges_multiclass = np.histogram(
            branches["class_value"],
            bins=15,
            range=(-5,10),
            weights=multiclass_weight
        )
        weighted_histogram_multiclass = (counts_multiclass, bin_edges_multiclass)

        out_dict = {
            "weight_tree": {
                "class_weight": class_weight,
                "class_target": class_targets,
                "multiclass_weight": multiclass_weight,
                "class_targets_binary": class_targets_binary,
            },
            "weighted_class_targets": weighted_histogram,
            "weighted_class_targets_multiclass": weighted_histogram_multiclass,
        }

        print("Finished with dict")
        print(out_dict)

        for key, value in out_dict.items():
            if isinstance(value, tuple):
                val_arr = value[0].to_numpy() if hasattr(value[0], "to_numpy") else value[0]
                edge_arr = value[1].to_numpy() if hasattr(value[1], "to_numpy") else value[1]
                out_file[key] = (val_arr, edge_arr)
            else:
                out_file[key] = value

        out_file.close()


def input_feature_plots(output_folder):
    inNames = [
        os.path.join(output_folder, x)
        for x in os.listdir(output_folder)
        if x.endswith(".root")
    ]
    # color_map = plt.get_cmap("tab10").colors[:10]
    color_map = plt.get_cmap("tab20").colors

    input_features = set(
        [
            "nExtraLeps",
            "nExtraTau",
            "lep1_pt",
            "lep2_pt",

            "PuppiMET_pt",

            "HT",

            "MT",
            "MT2_ll",
            "MT2_bb",
            "MT2_blbl1",
            "MT2_blbl2",
            "total_MT",
            "lep1_MT",
            "lep2_MT",

            "ll_mass",
            "ll_pt",

            "bb_mass_PNetRegPtRawCorr_PNetRegPtRawCorrNeutrino",
            "bb_pt",
            "bb_mass",

            "llmet_mass",
            "bbllmet_mass",

            "bjet1_pt",
            "bjet1_mass",
            "bjet2_pt",
            "bjet2_mass",

            "wjet1_pt",
            "wjet1_mass",
            "wjet2_pt",
            "wjet2_mass",

            "fatbjet_pt",
            "fatbjet_mass_PNetCorr",
            "fatbjet_particleNetWithMass_HbbvsQCD",

            "ll_dR",
            "bb_dR",
            "ll_bb_dR",
            "met_ll_dphi",
            "met_bb_dphi",

            "DeepHME_mass",

            "bjet1_btagPNetB",
            "bjet2_btagPNetB",

            "fatbjet_tau1",
            "fatbjet_tau2",
            "fatbjet_tau3",
            "fatbjet_tau4",
            "fatbjet_msoftdrop",

            "bb_CosTheta",

            "ll_jj_dR",
            "ll_dphi",
            "bb_dphi",
        ]
    )
    base_branches = set(["class_value", "X_mass", "weight_Central"])
    branches_to_load = list(input_features | base_branches)

    class_names = ["bbWW_2L", "TT", "DY", "ST", "SMHiggs", "VV", "Other", "bbtautau", "bbWW_1L"]

    for inName in inNames:
        if "weight" in inName:
            continue
        print(f"On file {inName} for plots")
        in_file = uproot.open(inName)

        subfolder_name = f"{inName[:-5]}_input_features"
        os.makedirs(subfolder_name, exist_ok=True)

        tree = in_file["Events"]

        branches = tree.arrays(branches_to_load)

        X_mass = branches["X_mass"]
        class_targets = branches["class_value"]
        class_weight = branches["weight_Central"]

        for inp_feature in input_features:
            color_map_idx = 0
            # Make a plot of input features with different colors for each class_target
            for class_value in np.unique(class_targets):
                # print(f"Plotting color {color_map_idx} for class {class_value} on map {color_map}")
                feature_values = branches[inp_feature][class_targets == class_value]
                weights = class_weight[class_targets == class_value]
                mass = X_mass[class_targets == class_value]
                feature_quants = np.quantile(feature_values, [0.01, 0.99])
                mask = (feature_values >= feature_quants[0]) & (
                    feature_values <= feature_quants[1]
                )
                class_plot_name = class_names[class_value]
                if class_value == 0:
                    for x_mass in [300, 600, 800]:
                        sig_mask = (mask) & (mass == x_mass)
                        plt.hist(
                            feature_values[sig_mask],
                            bins=50,
                            weights=weights[sig_mask],
                            alpha=0.5,
                            label=f"{class_plot_name} m{x_mass}",
                            # Normalize to 1 for better comparison of shapes
                            density=True,
                            histtype="step",
                            linewidth="1.5",
                            color=color_map[color_map_idx],
                        )
                        color_map_idx += 1
                else:
                    plt.hist(
                        feature_values[mask],
                        bins=50,
                        weights=weights[mask],
                        alpha=0.5,
                        label=f"{class_plot_name}",
                        # Normalize to 1 for better comparison of shapes
                        density=True,
                        histtype="step",
                        linewidth="1.5",
                        color=color_map[color_map_idx],
                    )
                    color_map_idx += 1
            plt.xlabel(inp_feature)
            plt.ylabel("Weighted Events")
            plt.title(f"Distribution of {inp_feature}")
            plt.legend()
            plt.savefig(os.path.join(subfolder_name, f"{inp_feature}_distribution.png"))
            plt.clf()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create TrainTest Files for DNN.")
    parser.add_argument(
        "--config",
        required=False,
        type=str,
        default="default_dataset.yaml",
        help="Config YAML",
    )
    parser.add_argument(
        "--output-folder",
        required=False,
        type=str,
        default="/eos/user/d/daebi/DNN_Training_Datasets",
        help="Output folder to store dataset",
    )

    args = parser.parse_args()

    config_file = args.config
    with open(config_file, "r") as file:
        config_dict = yaml.safe_load(file)

    output_base = args.output_folder
    output_folder = os.path.join(output_base, f"Dataset")
    if os.path.exists(output_folder):
        print(f"Output folder {output_folder} exists!!!")
    os.makedirs(output_folder, exist_ok=True)
    os.system(f"cp {config_file} {output_folder}/.")

    measure_cut_datasets(config_dict, output_folder)
    hadd_files(config_dict, output_folder)
    add_weight_file(output_folder) # Option for all masses
    # for mass in config_dict["signal"]["XtoYHto2B2W_2L"]["mass_points"]:
    #     print(f"Starting mass {mass}")
    #     add_weight_file(output_folder, mass=mass)
    input_feature_plots(output_folder)

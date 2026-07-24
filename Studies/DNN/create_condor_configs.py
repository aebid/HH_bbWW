import os
import yaml
import awkward as ak

training_type = "merged"
if training_type == "resolved":
    # Resolved
    template = "config/training_setup_doubleLep_resolved.yaml"
    # output_folder = "CondorConfigs/DoubleLepton_Resolved_25May_parametric_v1"
    # output_folder = "CondorConfigs/DoubleLepton_Resolved_21June_parametric_multiclass_v1"
    # output_folder = "CondorConfigs/DoubleLepton_Resolved_22June_parametric_multiclass_v1"
    output_folder = "CondorConfigs/DoubleLepton_Resolved_22June_multiclass_v1"

    # input_file_template = "/eos/user/d/daebi/HH_bbWW/DNNDatasets/ResolvedDataset_25May_5class/Dataset/nParity{j}_Merged.root"
    # input_file_template = "/afs/cern.ch/work/d/daebi/diHiggs/HH_bbWW/Studies/DNN/Resolved_Jun17/Dataset/nParity{j}_Merged.root"
    # input_file_template = "/afs/cern.ch/work/d/daebi/diHiggs/HH_bbWW/Studies/DNN/Resolved_June21/Dataset/nParity{j}_Merged.root"
    input_file_template = "/afs/cern.ch/work/d/daebi/diHiggs/HH_bbWW/Studies/DNN/Resolved_June22/Dataset/nParity{j}_Merged.root"

    mass_specific = True
    mass_list = [300, 400, 500, 550, 600, 650, 700, 800, 900, 1000]
    # weight_file_template = "/eos/user/d/daebi/HH_bbWW/DNNDatasets/ResolvedDataset_22May/Dataset/nParity{j}_Merged_weight_m{m}.root"
    weight_file_template = "/afs/cern.ch/work/d/daebi/diHiggs/HH_bbWW/Studies/DNN/Resolved_June22/Dataset/nParity{j}_Merged_weight_m{m}.root"
    # mass_list = [0]
    # weight_file_template = "/eos/user/d/daebi/HH_bbWW/DNNDatasets/ResolvedDataset_25May_5class/Dataset/nParity{j}_Merged_weight.root"
    # weight_file_template = "/afs/cern.ch/work/d/daebi/diHiggs/HH_bbWW/Studies/DNN/Resolved_June22/Dataset/nParity{j}_Merged_weight.root"

    training_name = "DNN_DoubleLepton_Resolved_Training{i}_par{j}_m{m}"
    var_parse_dict = {
        "learning_rate": [0.0005],
        "n_epochs": [100],
        "dropout": [0.2],
        # 'parametric_list': [ [ 600 ] ],
        "parametric_list": [[-1]],
        # "parametric_list": [[300, 400, 500, 550, 600, 650, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000]],
        "l2_rate": [0.01],
        "gamma1": [1.5],
        "gamma2": [0.9],
        "n_layers": [5],
        "n_units": [512],
        "n_units_reduction_factor": [1.0],
        "signal_loss_scale": [1.0],  # Frozen 1.0
        "multiclass_loss_scale": [0.0],  # Frozen 0.5
        # "UseParametric": [True],
        "UseParametric": [False],
        "use_batch_norm": [True],
        "nClasses": [5],
        "patience": [50],
        "lr_patience": [3],
        "lr_decay": [0.8],
        "class_names": [ ["Signal", "TT", "DY", "SMHiggs", "Other"] ],
    }

elif training_type == "boosted":
    # Boosted
    template = "config/training_setup_doubleLep_boosted.yaml"
    output_folder = "CondorConfigs/DoubleLepton_Boosted_16June_v6"

    input_file_template = "/eos/user/d/daebi/HH_bbWW/DNNDatasets/Boosted_Jun15/Dataset/nParity{j}_Merged.root"

    mass_specific = True
    mass_list = [
        300,
        400,
        500,
        550,
        600,
        650,
        700,
        800,
        900,
        1000,
        1200,
        1400,
        1600,
        1800,
        2000,
    ]
    weight_file_template = "/eos/user/d/daebi/HH_bbWW/DNNDatasets/Boosted_Jun15/Dataset/nParity{j}_Merged_weight_m{m}.root"

    training_name = "DNN_DoubleLepton_Boosted_Training{i}_par{j}_m{m}"
    var_parse_dict = {
        "learning_rate": [0.0005],  # Frozen 0.005
        "n_epochs": [200],  # Frozen 100
        "dropout": [0.0],  # Frozen 0.2
        "parametric_list": [[-1]],
        # 'parametric_list': [ [ 300, 400, 500, 550, 600, 650, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000 ] ],
        "l2_rate": [0.01],  # Frozen 0.001
        "gamma1": [1.5],  # Frozen 1.5
        "gamma2": [0.9],  # Frozen 0.9
        "n_layers": [3],  # Frozen 3
        "n_units": [256],
        "n_units_reduction_factor": [1.0],  # Frozen 1
        "signal_loss_scale": [1.0],  # Frozen 1.0
        "multiclass_loss_scale": [0.5],  # Frozen 0.5
        "UseParametric": [False],
        "use_batch_norm": [True],
        "nClasses": [2],
        "patience": [50],
        "lr_patience": [3],
        "lr_decay": [0.8],
        "weight_decay": [0.001],
        "class_names": [ ["Signal", "Background"] ],
    }

elif training_type == "merged":
    # Merged
    template = "config/training_setup_doubleLep_merged.yaml"
    output_folder = "CondorConfigs/DoubleLepton_Merged_14July_Parametric"

    input_file_template = "/eos/user/d/daebi/HH_bbWW/DNNDatasets/merged_jul14_mediumBtag/Dataset/nParity{j}_Merged.root"

    mass_specific = False
    # mass_list = [
    #     300,
    #     400,
    #     500,
    #     550,
    #     600,
    #     650,
    #     700,
    #     800,
    #     900,
    #     1000,
    #     1200,
    #     1400,
    #     1600,
    #     1800,
    #     2000,
    # ]
    mass_list = [-1]
    # weight_file_template = "/eos/user/d/daebi/HH_bbWW/DNNDatasets/merged_jul14_mediumBtag/Dataset/nParity{j}_Merged_weight_m{m}.root"
    weight_file_template = "/eos/user/d/daebi/HH_bbWW/DNNDatasets/merged_jul14_mediumBtag/Dataset/nParity{j}_Merged_weight.root"

    # training_name = "DNN_DoubleLepton_Boosted_Training{i}_par{j}_m{m}"
    training_name = "DNN_DoubleLepton_Boosted_Training{i}_par{j}"
    var_parse_dict = {
        "learning_rate": [0.0005],  # Frozen 0.005
        "n_epochs": [200],  # Frozen 100
        "dropout": [0.0],  # Frozen 0.2
        # "parametric_list": [[-1]],
        'parametric_list': [ [ 300, 400, 500, 550, 600, 650, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000 ] ],
        "l2_rate": [0.01],  # Frozen 0.001
        "gamma1": [1.5],  # Frozen 1.5
        "gamma2": [0.9],  # Frozen 0.9
        "n_layers": [5],  # Frozen 3
        "n_units": [256],
        "n_units_reduction_factor": [1.0],  # Frozen 1
        "signal_loss_scale": [1.0],  # Frozen 1.0
        "multiclass_loss_scale": [0.5],  # Frozen 0.5
        "UseParametric": [True],
        "use_batch_norm": [True],
        "nClasses": [2],
        "patience": [50],
        "lr_patience": [10],
        "lr_decay": [0.8],
        "weight_decay": [0.001],
        "class_names": [ ["Signal", "Background"] ],
    }

else:
    raise RunTimeError(f"Bad training type {training_type}")

os.makedirs(output_folder, exist_ok=True)

with open(template, "r") as f:
    default_config = yaml.safe_load(f)


var_names = var_parse_dict.keys()
var_combinations_list = [x for x in var_parse_dict.values()]
var_combinations = ak.cartesian(var_combinations_list, axis=0)

for i, varset in enumerate(var_combinations):

    for m in mass_list:

        config = default_config.copy()
        for name, var in zip(var_names, varset.tolist()):
            if name == "parametric_list" and var == [-1]:
                print(f"Parametric list is empty, set to special mass {m}")
                var = [m]
            config[name] = var
        for j in range(4):
            # Set up each parity
            config["training_file"] = input_file_template.format(j=j)
            config["weight_file"] = weight_file_template.format(j=j, m=m)
            config["test_training_file"] = input_file_template.format(j=(j + 1) % 4)
            config["test_weight_file"] = weight_file_template.format(j=(j + 1) % 4, m=m)
            config["validation_file"] = input_file_template.format(j=(j + 2) % 4)
            config["validation_weight_file"] = weight_file_template.format(
                j=(j + 2) % 4, m=m
            )

            config["training_name"] = training_name.format(i=i, j=j, m=m)

            outFileName = f"{training_name.format(i = i, j = j, m = m)}.yaml"
            outFilePath = os.path.join(output_folder, outFileName)
            with open(outFilePath, "w") as f:
                yaml.dump(config, f)

import importlib
from FLAF.Common.Utilities import *
from FLAF.Common.HistHelper import *
from Corrections.Corrections import Corrections
from Corrections.CorrectionsCore import getSystName, central
from FLAF.Common.Setup import Setup

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

initialized = False
analysis = None


def Initialize():
    global initialized
    if not initialized:
        headers_dir = os.path.dirname(os.path.abspath(__file__))
        ROOT.gROOT.ProcessLine(f".include {os.environ['ANALYSIS_PATH']}")
        ROOT.gInterpreter.Declare(f'#include "FLAF/include/HistHelper.h"')
        ROOT.gInterpreter.Declare(f'#include "FLAF/include/Utilities.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/MT2.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/Lester_mt2_bisect.cpp"')
        ROOT.gROOT.ProcessLine('#include "FLAF/include/AnalysisTools.h"')
        ROOT.gROOT.ProcessLine('#include "FLAF/include/AnalysisMath.h"')
        initialized = True


def analysis_setup(setup):
    global analysis
    analysis_import = setup.global_params["analysis_import"]
    analysis = importlib.import_module(f"{analysis_import}")


def GetDfw(df, setup, dataset_name):
    global_params = setup.global_params
    isData = dataset_name == "data"
    period = global_params["era"]
    dfw = analysis.DataFrameBuilderForHistograms(df, global_params, period)
    new_dfw = analysis.PrepareDfForHistograms(dfw, isData)
    return new_dfw


central_df_weights_computed = False


def DefineWeightForHistograms(
    *,
    dfw,
    isData,
    uncName,
    uncScale,
    unc_cfg_dict,
    hist_cfg_dict,
    global_params,
    final_weight_name,
    df_is_central,
):
    global central_df_weights_computed
    is_central = uncName == central
    corrections = Corrections.getGlobal()
    if not isData and (not central_df_weights_computed or not df_is_central):
        lepton_legs = ["lep1", "lep2"]
        offline_legs = ["lep1", "lep2"]
        triggers_to_use = set()
        channels = global_params["channelSelection"]
        for channel in channels:
            trigger_list = global_params.get("triggers", {}).get(channel, [])
            for trigger in trigger_list:
                if trigger not in corrections.trigger_dict.keys():
                    raise RuntimeError(
                        f"Trigger does not exist in triggers.yaml, {trigger}"
                    )
                triggers_to_use.add(trigger)

        dfw.df, all_weights = corrections.getNormalisationCorrections(
            dfw.df,
            lepton_legs=lepton_legs,
            offline_legs=offline_legs,
            trigger_names=triggers_to_use,
            unc_source=uncName,
            unc_scale=uncScale,
            ana_caches=None,
            return_variations=is_central and global_params["compute_unc_histograms"],
            use_genWeight_sign_only=True,
        )
        if df_is_central:
            central_df_weights_computed = True

    categories = global_params["categories"]
    boosted_categories = global_params.get("boosted_categories", [])
    process_group = global_params["process_group"]
    weights_this_process = set(corrections.to_apply.keys())
    # Mode "none" still loads btag (WP-id branches) but does not create SF
    # columns. GetWeight must not multiply by weight_bTagShape_Central then.
    btag_mode = (
        corrections.to_apply.get("btag", {}).get("modes", {}).get("HistTuple", "none")
    )
    if btag_mode not in ("shape", "shape_and_norm", "wp"):
        weights_this_process.discard("btag")

    total_weight_expression = (
        analysis.GetWeight(
            weights_this_process,
            weight_base_name=global_params.get("weight_base_branch", "weight_base"),
        )
        if process_group != "data"
        else "1"
    )  # are we sure?
    weight_name = "final_weight"
    if weight_name not in dfw.df.GetColumnNames():
        dfw.df = dfw.df.Define(weight_name, total_weight_expression)
    if not is_central:
        norm_cfg = unc_cfg_dict["norm"].get(uncName, {})
        # An entry may name the correction its expression is built from.
        # HistTupleProducer asks for every norm uncertainty on every sample, but a
        # correction carrying a `processes:` list -- dy_hhbbtautau is the only one --
        # defines its branches for those processes alone, so the expression fails to
        # compile everywhere else ("use of undeclared identifier"). Where the correction
        # does not apply, the variation is the central weight: a nuisance with no effect
        # on that process, rather than an error.
        requires = norm_cfg.get("requires")
        if "expression" in norm_cfg and (
            requires is None or requires in weights_this_process
        ):
            weight_name = norm_cfg["expression"].format(scale=uncScale)
    dfw.df = dfw.df.Define(final_weight_name, weight_name)

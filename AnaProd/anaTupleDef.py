import AnaProd.baseline as AnaBaseline
import FLAF.Common.BaselineSelection as CommonBaseline
from Corrections.Corrections import Corrections

loadTF = False
loadHHBtag = False
lepton_legs = ["lep1", "lep2"]
offline_legs = ["lep1", "lep2"]


MuonObservables = [
    "mediumId",
    "tightId",
    "highPtId",
    "pfIsoId",
    "mediumPromptId",
    "looseId",
    "miniIsoId",
    "mvaMuID_WP",
    "tkRelIso",
    "pfRelIso04_all",
    "pfRelIso03_all",
    "miniPFRelIso_all",
]
ElectronObservables = [
    "mvaNoIso_WP80",
    "mvaIso_WP80",
    "mvaNoIso_WP90",
    "mvaIso_WP90",
    "pfRelIso03_all",
    "mvaIso",
    "mvaNoIso",
    "miniPFRelIso_all",
]

TauObservables = [
    "idDeepTau2018v2p5VSe",
    "idDeepTau2018v2p5VSjet",
    "idDeepTau2018v2p5VSmu",
    "decayMode",
]

EMuObservables = [
    (ElectronObservables, "e", "Electron"),
    (MuonObservables, "mu", "Muon"),
]

EMuTauObservables = EMuObservables + [(TauObservables, "tau", "Tau")]

JetObservables = [
    "PNetRegPtRawCorr",
    "PNetRegPtRawCorrNeutrino",
    "PNetRegPtRawRes",
    "rawFactor",
    "btagPNetB",
    "btagUParTAK4B",
    "btagPNetCvB",
    "btagPNetCvL",
    # "btagPNetCvNotB",
    "btagPNetQvG",
    "btagPNetTauVJet",
    "chEmEF",
    "chHEF",
    # "chMultiplicity",
    # "hfEmEF",
    # "hfHEF",
    "hfadjacentEtaStripsSize",
    "hfcentralEtaStripSize",
    "hfsigmaEtaEta",
    "hfsigmaPhiPhi",
    "jetId",
    "muEF",
    "muonSubtrFactor",
    "nConstituents",
    "nElectrons",
    "nMuons",
    "nSVs",
    "neEmEF",
    "neHEF",
    # "neMultiplicity",
    "ptRes",
    "idbtagPNetB",
    "idbtagUParTAK4B",
    "area",
]  # 2024

JetObservablesMC = ["hadronFlavour", "partonFlavour"]

FatJetObservables = [
    "area",
    # "chEmEF",
    # "chHEF",
    # "chMultiplicity",
    # "globalParT2_QCD0HF",
    # "globalParT2_QCD1HF",
    # "globalParT2_QCD2HF",
    # "globalParT2_TopW",
    # "globalParT2_TopbW",
    # "globalParT2_TopbWev",
    # "globalParT2_TopbWmv",
    # "globalParT2_TopbWq",
    # "globalParT2_TopbWqq",
    # "globalParT2_TopbWtauhv",
    # "globalParT2_Xbb",
    # "globalParT2_XbbVsQCD",
    # "globalParT2_Xcc",
    # "globalParT2_Xcs",
    # "globalParT2_Xgg",
    # "globalParT2_Xqq",
    # "globalParT2_Xtauhtaue",
    # "globalParT2_Xtauhtauh",
    # "globalParT2_Xtauhtaum",
    # "globalParT_massRes",
    # "globalParT_massVis",
    "jetId",
    "lsf3",
    "msoftdrop",
    # "muEF",
    "n2b1",
    "n3b1",
    "nConstituents",
    # "neEmEF",
    # "neHEF",
    # "neMultiplicity",
    # "particleNetLegacy_QCD",
    # "particleNetLegacy_QCDb",
    # "particleNetLegacy_QCDbb",
    # "particleNetLegacy_QCDc",
    # "particleNetLegacy_QCDcc",
    # "particleNetLegacy_QCDothers",
    # "particleNetLegacy_Xbb",
    # "particleNetLegacy_Xcc",
    # "particleNetLegacy_Xqq",
    # "particleNetLegacy_mass",
    "particleNetWithMass_H4qvsQCD",
    "particleNetWithMass_HbbvsQCD",
    "particleNetWithMass_HccvsQCD",
    "particleNetWithMass_QCD",
    "particleNetWithMass_TvsQCD",
    "particleNetWithMass_WvsQCD",
    "particleNetWithMass_ZvsQCD",
    "particleNet_QCD",
    "particleNet_QCD0HF",
    "particleNet_QCD1HF",
    "particleNet_QCD2HF",
    # "particleNet_Xbb",
    "particleNet_XbbVsQCD",
    "particleNet_XccVsQCD",
    "particleNet_XggVsQCD",
    "particleNet_XqqVsQCD",
    "particleNet_XteVsQCD",
    "particleNet_XtmVsQCD",
    "particleNet_XttVsQCD",
    "particleNet_massCorr",
    "rawFactor",
    "tau1",
    "tau2",
    "tau3",
    "tau4",
]

FatJetObservablesMC = ["hadronFlavour"]

SubJetObservables = ["btagDeepB", "eta", "mass", "phi", "pt", "rawFactor"]
SubJetObservablesMC = ["hadronFlavour", "nBHadrons", "nCHadrons"]

defaultColToSave = [
    "FullEventId",
    "luminosityBlock",
    "run",
    "event",
    "period",
    "X_mass",
    "X_spin",
    "isData",
    # "PuppiMET_covXX",
    # "PuppiMET_covXY",
    # "PuppiMET_covYY",
    # "PuppiMET_significance",
    "PuppiMET_sumEt",
    "nJet",
    "PV_npvs",
]

# Add this functionality eventually
MCObservables = [
    ("LHEPdfWeight", "LHEPdf_Weight"),
    ("LHEReweightingWeight", "LHEReweighting_Weight"),
    ("LHEScaleWeight", "LHEScale_Weight"),
    ("PSWeight", "PS_Weight"),
    "nLHEPart",
    "LHEPart_eta",
    "LHEPart_incomingpz",
    "LHEPart_mass",
    "LHEPart_pdgId",
    "LHEPart_phi",
    "LHEPart_pt",
    "LHEPart_spin",
    "LHEPart_status",
    "LHE_AlphaS",
    "LHE_HT",
    "LHE_HTIncoming",
    "LHE_Nb",
    "LHE_Nc",
    "LHE_Nglu",
    "LHE_Njets",
    "LHE_NpLO",
    "LHE_NpNLO",
    "LHE_Nuds",
    "LHE_Vpt",
]

PtEtaPhiM = ["pt", "eta", "phi", "mass"]


def getDefaultColumnsToSave(isData):
    colToSave = defaultColToSave.copy()
    if not isData:
        colToSave.extend(["Pileup_nTrueInt"])
    return colToSave


def defineLeptonVariables(dfw, leg_name, leg_idx, isData):
    def LegVar(
        var_name,
        var_expr,
        *,
        apply_cond=True,
        var_type=None,
        var_cond=None,
        default=0,
        append=True,
    ):
        define_expr = f"static_cast<{var_type}>({var_expr})" if var_type else var_expr
        if apply_cond:
            cond = f"HwwCandidate.leg_type.size() > {leg_idx}"
            if var_cond:
                cond = f"{cond} && ({var_cond})"
            full_define_expr = f"{cond} ? ({define_expr}) : {default}"
        else:
            if var_cond:
                raise ValueError("var_cond cannot be used if apply_cond is False")
            full_define_expr = define_expr
        full_name = f"{leg_name}_{var_name}"
        if append:
            dfw.DefineAndAppend(full_name, full_define_expr)
        else:
            dfw.Define(full_name, full_define_expr)

    LegVar(
        "idx",
        f"HwwCandidate.leg_index.at({leg_idx})",
        var_type="int",
        default=-1,
        append=False,
    )
    # Save the enum for now, type is used in many corrections
    LegVar("legType", f"HwwCandidate.leg_type.at({leg_idx})", default="Leg::none")
    # Save the lep* p4 and index directly to avoid using HwwCandidate in SF LUTs
    LegVar(
        "p4",
        f"HwwCandidate.leg_p4.at({leg_idx})",
        default="LorentzVectorM()",
        append=False,
    )

    for var in PtEtaPhiM:
        LegVar(var, f"{leg_name}_p4.{var}()", var_type="float", default="0.f")
    LegVar(
        "charge", f"HwwCandidate.leg_charge.at({leg_idx})", var_type="int", default="0"
    )

    for var_collection, extected_leg_type, expected_leg_name in EMuObservables:
        for var in var_collection:
            LegVar(
                f"{expected_leg_name}_{var}",
                f"{expected_leg_name}_{var}.at({leg_name}_idx)",
                var_cond=f"{leg_name}_legType == Leg::{extected_leg_type}",
                default="-1",
            )

    LegVar(
        "jetIdx",
        f"""int jet_idx = -1;
            if ({leg_name}_legType == Leg::e)
                jet_idx = Electron_jetIdx[{leg_name}_idx];
            if ({leg_name}_legType == Leg::mu)
                jet_idx = Muon_jetIdx[{leg_name}_idx];
            if (jet_idx >= Jet_p4.size())
                jet_idx = -1;
            return jet_idx;
        """,
        apply_cond=False,
        append=False,
    )

    # save pt and flavor of jet matching to leptons
    LegVar(
        "jet_pt",
        f"Jet_p4[{leg_name}_jetIdx].pt()",
        var_cond=f"{leg_name}_jetIdx >= 0",
        var_type="float",
        default="-1.f",
    )

    if not isData:
        for var in ["hadronFlavour", "partonFlavour"]:
            LegVar(
                f"jet_{var}",
                f"Jet_{var}[{leg_name}_jetIdx]",
                var_cond=f"{leg_name}_jetIdx >= 0",
                var_type="int",
                default="-1",
            )

        # save gen leptons matched to reco leptons
        # MatchGenLepton returns index of in genLetpons collection if match exists
        LegVar(
            f"gen_idx",
            f"MatchGenLepton({leg_name}_p4, genLeptons, 0.4)",
            var_type="int",
            default="-1",
            append=False,
        )
        LegVar(
            "gen_p4",
            f"LorentzVectorM(genLeptons.at({leg_name}_gen_idx).visibleP4())",
            var_cond=f"{leg_name}_gen_idx >= 0",
            default="LorentzVectorM()",
            append=False,
        )
        LegVar(
            f"gen_mother_p4",
            f"(*genLeptons.at({leg_name}_gen_idx).mothers().begin())->p4",
            var_cond=f"{leg_name}_gen_idx >= 0 && !genLeptons.at({leg_name}_gen_idx).mothers().empty()",
            default="LorentzVectorM()",
            append=False,
        )

        LegVar(
            f"gen_kind",
            f"genLeptons.at({leg_name}_gen_idx).kind()",
            var_cond=f"{leg_name}_gen_idx >= 0",
            var_type="int",
            default="-1",
        )
        LegVar(
            f"gen_motherPdgId",
            f"(*genLeptons.at({leg_name}_gen_idx).mothers().begin())->pdgId",
            var_cond=f"{leg_name}_gen_idx >= 0 && !genLeptons.at({leg_name}_gen_idx).mothers().empty()",
            var_type="int",
            default="0",
        )
        for var in PtEtaPhiM:
            LegVar(
                f"gen_mother_{var}",
                f"{leg_name}_gen_mother_p4.{var}()",
                var_type="float",
            )
            LegVar(
                f"gen_{var}",
                f"{leg_name}_gen_p4.{var}()",
                var_type="float",
            )


def defineExtraLeptonVariables(dfw):
    for var_collection, _, obj_name in EMuTauObservables:
        for var in PtEtaPhiM:
            dfw.DefineAndAppend(
                f"Extra{obj_name}_" + var,
                f"v_ops::{var}({obj_name}_p4[Extra{obj_name}_sel])",
            )

        for var in var_collection:
            dfw.DefineAndAppend(
                f"Extra{obj_name}_" + var, f"{obj_name}_{var}[Extra{obj_name}_sel]"
            )


def defineCentralJetVariables(dfw, isData):
    # save all selected reco jets
    dfw.Define("centralJet_idx", "CreateIndexes(Sum(Jet_sel))")
    existing_cols = {str(c) for c in dfw.df.GetColumnNames()}
    sort_score = (
        "Jet_btagUParTAK4B" if "Jet_btagUParTAK4B" in existing_cols else "Jet_btagPNetB"
    )
    dfw.Define(
        "centralJet_idxSorted", f"ReorderObjects({sort_score}[Jet_sel], centralJet_idx)"
    )
    for var in PtEtaPhiM:
        name = f"centralJet_{var}"
        dfw.DefineAndAppend(
            name, f"Take(v_ops::{var}(Jet_p4[Jet_sel]), centralJet_idxSorted)"
        )

    # save gen jets matched to selected reco jets
    if not isData:
        dfw.Define(
            "centralJet_matchedGenJetIdx",
            f"Take(Jet_genJetIdx[Jet_sel], centralJet_idxSorted)",
        )
        for var in PtEtaPhiM:
            dfw.DefineAndAppend(
                f"centralJet_matchedGenJet_{var}",
                f"TakeAndCast(v_ops::{var}(GenJet_p4), centralJet_matchedGenJetIdx, 0.f)",
            )

        dfw.Define(
            "GenJet_TrueBjetTag",
            "FindTwoJetsClosestToMPV(125.0, GenJet_p4, GenJet_hadronFlavour == 5)",
        )

        for var in JetObservablesMC + ["TrueBjetTag"]:
            dfw.DefineAndAppend(
                f"centralJet_matchedGenJet_{var}",
                f"Take(GenJet_{var}, centralJet_matchedGenJetIdx, std::decay_t<decltype(GenJet_{var})>::value_type(0))",
            )

    reco_jet_obs = list(JetObservables)
    if not isData:
        reco_jet_obs.extend(JetObservablesMC)
    existing_cols = {str(c) for c in dfw.df.GetColumnNames()}
    reco_jet_obs = [v for v in reco_jet_obs if f"Jet_{v}" in existing_cols]
    for jet_obs in reco_jet_obs:
        dfw.DefineAndAppend(
            f"centralJet_{jet_obs}",
            f"Take(Jet_{jet_obs}[Jet_sel], centralJet_idxSorted)",
        )


def defineFatJetVariables(dfw, isData):
    dfw.Define(
        "SelectedFatJet_idx",
        "CreateIndexes(Sum(FatJet_sel))",
    )
    dfw.Define(
        "SelectedFatJet_idxSorted",
        "ReorderObjects(FatJet_particleNetWithMass_HbbvsQCD[FatJet_sel], SelectedFatJet_idx)",
    )

    fatjet_obs = list(FatJetObservables)
    if not isData:
        fatjet_obs.extend(FatJetObservablesMC)
    existing_cols = {str(c) for c in dfw.df.GetColumnNames()}
    fatjet_obs = [v for v in fatjet_obs if f"FatJet_{v}" in existing_cols]

    for var in PtEtaPhiM:
        dfw.DefineAndAppend(
            f"SelectedFatJet_{var}",
            f"TakeAndCast(v_ops::{var}(FatJet_p4[FatJet_sel]), SelectedFatJet_idxSorted, 0.f)",
        )
    for var in fatjet_obs:
        dfw.DefineAndAppend(
            f"SelectedFatJet_{var}",
            f"Take(FatJet_{var}[FatJet_sel], SelectedFatJet_idxSorted)",
        )
    subjet_obs = list(SubJetObservables)
    if not isData:
        subjet_obs.extend(SubJetObservablesMC)
    existing_cols = {str(c) for c in dfw.df.GetColumnNames()}
    subjet_obs = [v for v in subjet_obs if f"SubJet_{v}" in existing_cols]
    for subJetIdx in [1, 2]:
        dfw.Define(
            f"SelectedFatJet_subJetIdx{subJetIdx}",
            f"Take(FatJet_subJetIdx{subJetIdx}[FatJet_sel], SelectedFatJet_idxSorted)",
        )
        dfw.DefineAndAppend(
            f"SelectedFatJet_SubJet{subJetIdx}_isValid",
            f"SelectedFatJet_subJetIdx{subJetIdx} >= 0",
        )
        for var in subjet_obs:
            dfw.DefineAndAppend(
                f"SelectedFatJet_SubJet{subJetIdx}_{var}",
                f"Take(SubJet_{var}, SelectedFatJet_subJetIdx{subJetIdx}, std::decay_t<decltype(SubJet_{var})>::value_type(0))",
            )


def defineForwardJetVariables(dfw, isData):
    for var in PtEtaPhiM:
        dfw.DefineAndAppend(
            f"ForwardJet_{var}", f"v_ops::{var}(Jet_p4[ForwardJet_sel])"
        )
    if not isData:
        for var in JetObservablesMC:
            dfw.DefineAndAppend(f"ForwardJet_{var}", f"Jet_{var}[ForwardJet_sel]")


def defineMETVariables(dfw, met_type):
    dfw.DefineAndAppend(
        f"{met_type}_pt_nano", f"static_cast<float>({met_type}_p4_nano.pt())"
    )
    dfw.DefineAndAppend(
        f"{met_type}_phi_nano", f"static_cast<float>({met_type}_p4_nano.phi())"
    )

    dfw.RedefineAndAppend(f"{met_type}_pt", f"static_cast<float>({met_type}_p4.pt())")
    dfw.RedefineAndAppend(f"{met_type}_phi", f"static_cast<float>({met_type}_p4.phi())")


def defineSignalVariables(dfw):
    # save gen H->VV
    dfw.Define(
        "H_to_VV",
        """GetGenHVVCandidate(event, genLeptons, GenPart_pdgId, GenPart_daughters, GenPart_statusFlags, GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass, GenJet_p4, true)""",
    )
    for var in PtEtaPhiM:
        dfw.DefineAndAppend(
            f"genHVV_{var}", f"static_cast<float>(H_to_VV.cand_p4.{var}())"
        )

    # save gen level vector bosons from H->VV
    for boson in [1, 2]:
        name = f"genV{boson}"
        dfw.DefineAndAppend(
            f"{name}_pdgId", f"GenPart_pdgId[ H_to_VV.legs[{boson - 1}].index ]"
        )
        for var in PtEtaPhiM:
            dfw.DefineAndAppend(
                f"{name}_{var}",
                f"static_cast<float>(H_to_VV.legs[{boson - 1}].cand_p4.{var}())",
            )

    # save gen level products of vector boson decays (prod - index of product (quark, leptons or neutrinos))
    for boson in [1, 2]:
        for prod in [1, 2]:
            name = f"genV{boson}prod{prod}"
            for var in PtEtaPhiM:
                dfw.DefineAndAppend(
                    f"{name}_{var}",
                    f"static_cast<float>(H_to_VV.legs[{boson - 1}].leg_p4[{prod - 1}].{var}())",
                )
                dfw.DefineAndAppend(
                    f"{name}_vis_{var}",
                    f"static_cast<float>(H_to_VV.legs[{boson - 1}].leg_vis_p4[{prod - 1}].{var}())",
                )
            dfw.DefineAndAppend(
                f"{name}_legType",
                f"static_cast<int>(H_to_VV.legs[{boson - 1}].leg_kind[{prod - 1}])",
            )
            dfw.DefineAndAppend(
                f"{name}_pdgId",
                f"GenPart_pdgId[ H_to_VV.legs.at({boson - 1}).leg_index.at({prod - 1}) ]",
            )

    # save gen level H->bb
    dfw.Define(
        "H_to_bb",
        """GetGenHBBCandidate(event, GenPart_pdgId, GenPart_daughters, GenPart_statusFlags, GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass, GenJet_p4, false)""",
    )
    for var in PtEtaPhiM:
        dfw.DefineAndAppend(
            f"genHbb_{var}", f"static_cast<float>(H_to_bb.cand_p4.{var}())"
        )

    # save gen level b quarks
    for b_quark in [1, 2]:
        name = f"genb{b_quark}"
        for var in PtEtaPhiM:
            dfw.DefineAndAppend(
                f"{name}_{var}",
                f"static_cast<float>(H_to_bb.leg_p4[{b_quark - 1}].{var}())",
            )
            dfw.DefineAndAppend(
                f"{name}_vis_{var}",
                f"static_cast<float>(H_to_bb.leg_vis_p4[{b_quark - 1}].{var}())",
            )


def defineMCSpecificObservables(dfw):
    for var in MCObservables:
        if isinstance(var, tuple):
            var_orig_name, var_new_name = var
            dfw.DefineAndAppend(var_new_name, var_orig_name)
        else:
            dfw.colToSave.append(var)


def defineGenTopVariables(dfw):
    """Gen-level top pT, for the top pT reweighting of ttbar.

    The reweighting needs the `isLastCopy` parton-level top -- after radiation and
    before decay (https://twiki.cern.ch/twiki/bin/view/CMS/TopPtReweighting, which is
    explicit that a reco- or particle-level proxy gives an invalid reweighting). That
    means GenPart, which is available here but is *not* written to the anaTuple, so
    the two pT values have to be reduced and saved now. The LHEPart collection that
    MCObservables does save is taken before radiation and is not a substitute.

    Defined for every MC sample rather than for ttbar alone: it is two floats per
    event, and deciding which samples are "ttbar" belongs in the correction config,
    not here. Where no last-copy top pair exists -- every non-ttbar sample, and single
    top, which has only one -- both branches stay at -1 and TopPtCorrProducer returns
    a weight of 1.
    """
    # Tested as a bitwise mask rather than via GenStatusFlags::isLastCopy: a
    # ROOT::VecOps::Map over a generic lambda deduces to void in the JIT context, and
    # a lambda with an explicit parameter type would pin the NanoAOD storage type of
    # GenPart_statusFlags, which varies between versions. The enum constant keeps the
    # bit index from being a magic number.
    dfw.Define(
        "genPart_isLastCopy",
        "(GenPart_statusFlags & (1 << GenStatusFlags::kIsLastCopy)) != 0",
    )
    for name, pdg_id in [("genTop", 6), ("genAntiTop", -6)]:
        dfw.Define(
            f"{name}_pt_lastCopy",
            f"GenPart_pt[(GenPart_pdgId == {pdg_id}) && genPart_isLastCopy]",
        )
        dfw.DefineAndAppend(
            f"{name}_pt",
            f"{name}_pt_lastCopy.size() > 0 "
            f"? static_cast<float>({name}_pt_lastCopy[0]) : -1.0f",
        )


def addAllVariables(
    dfw,
    syst_name,
    isData,
    trigger_class,
    lepton_legs,
    isSignal,
    applyTriggerFilter,
    global_params,
    channels,
    dataset_cfg,
):
    print(f"Adding variables for {syst_name}")
    dfw.Apply(
        Corrections.getGlobal().JetVetoMap.GetJetVetoMap
    )  # Must init JetVetoMap before applying
    dfw.Apply(
        CommonBaseline.ApplyJetVetoMap,
        apply_filter=True,
        defineElectronCleaning=True,
        isV12=global_params["nano_version"] == "v12",
    )
    dfw.Apply(Corrections.getGlobal().jet.getEnergyResolution)
    dfw.Apply(Corrections.getGlobal().btag.getWPid, "Jet")

    dfw.Apply(AnaBaseline.selectHWW, selected_channels=channels)

    dfw.DefineAndAppend("channelId", "static_cast<int>(HwwCandidate.channel())")
    for leg_idx, leg_name in enumerate(lepton_legs):
        defineLeptonVariables(dfw, leg_name, leg_idx, isData)

    dfw.Apply(AnaBaseline.selectExtraLeptons)
    defineExtraLeptonVariables(dfw)

    dfw.Apply(
        AnaBaseline.selectJets,
        min_n_effective_jets_SL=global_params["anaTupleSelection"][
            "min_n_effective_jets_SL"
        ],
        min_n_effective_jets_DL=global_params["anaTupleSelection"][
            "min_n_effective_jets_DL"
        ],
        era=global_params["era"],
    )

    defineCentralJetVariables(dfw, isData)
    defineFatJetVariables(dfw, isData)
    defineForwardJetVariables(dfw, isData)
    defineMETVariables(dfw, global_params["met_type"])
    if not isData:
        defineMCSpecificObservables(dfw)
        defineGenTopVariables(dfw)

    if trigger_class is not None:
        hltBranches = dfw.Apply(
            trigger_class.ApplyTriggers, lepton_legs, isData, applyTriggerFilter
        )
        dfw.colToSave.extend(hltBranches)

    if isSignal and not (
        ("XtoHHto2B2Tau" in dataset_cfg["process_name"])
        or ("XtoHHto2Tau2B" in dataset_cfg["process_name"])
    ):
        defineSignalVariables(dfw)

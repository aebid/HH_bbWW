from FLAF.Common.Utilities import *


class Channel:
    leg_names = {
        "e": "Electron",
        "mu": "Muon",
    }

    def __init__(self, *legs):
        assert len(legs) > 0
        self.legs = [self.leg_names[leg] for leg in legs]
        self.name = ""
        for leg_idx, leg in enumerate(legs):
            leg_str = leg.title() if leg_idx > 0 else leg
            self.name += leg_str


channels = [
    Channel("mu", "mu"),
    Channel("e", "mu"),
    Channel("e", "e"),
    Channel("mu"),
    Channel("e"),
]  # in order of importance during the channel selection


def selectHWW(df, selected_channels):
    df = df.Define(
        "Electron_sel",
        """
        (v_ops::pt(Electron_p4) > 10) &&
        ( (abs(Electron_superclusterEta) < 1.4442) || (abs(Electron_superclusterEta) > 1.5560) ) &&
        (
            ( (abs(Electron_superclusterEta) <= 1.479) && (abs(Electron_dxy) < 0.05) && (abs(Electron_dz) < 0.1) ) ||
            ( (abs(Electron_superclusterEta) > 1.479) && (abs(Electron_dxy) < 0.1) && (abs(Electron_dz) < 0.2) )
        ) &&
        (Electron_sip3d <= 8) && (Electron_mvaIso >= -0.9)
        """,
    )
    # Lower the muon pt threshold to 5 to check for potential improvement, done while adding low pt tight ID SF
    # Raise back to pt 10 to make simple for now
    # We cannot use no-iso because of the trigger SFs not available

    df = df.Define(
        "Muon_sel",
        """
        v_ops::pt(Muon_p4) > 10 && abs(v_ops::eta(Muon_p4)) < 2.4 && abs(Muon_dz) < 0.1 && abs(Muon_dxy) < 0.05 && abs(Muon_dxy) < 0.05 && Muon_sip3d <= 8 && Muon_pfIsoId >= 1 && Muon_looseId""",
    )
    # Can lower pT to 5 later when applying the soft muon SFs

    df = df.Define("Electron_iso", "Electron_mvaIso")
    df = df.Define("Muon_iso", "Muon_pfRelIso04_all")

    cand_columns = []
    for ch in channels:
        cand_column = f"HwwCandidates_{ch.name}"
        leg_args = []
        for leg_idx, leg in enumerate(ch.legs):
            leg_args.extend(
                [
                    f"{leg}_sel",
                    f"{leg}_p4",
                    f"{leg}_iso",
                    f"{leg}_charge",
                    f"{leg}_genMatchIdx",
                ]
            )
        leg_str = ", ".join(leg_args)
        df = df.Define(
            cand_column, f"GetHWWCandidates(Channel::{ch.name}, 0.1, {leg_str})"
        )
        cand_columns.append(cand_column)
    cand_filters = [f"{c}.size() > 0" for c in cand_columns]
    stringfilter = " || ".join(cand_filters)
    df = df.Filter(stringfilter, "Reco Baseline 2")
    cand_list_str = ", ".join(["&" + c for c in cand_columns])
    df = df.Define("HwwCandidate", f"GetBestHWWCandidate({{ {cand_list_str} }}, event)")
    df = df.Define("is_SL", "HwwCandidate.leg_type.size() == 1")

    channel_to_select = " || ".join(
        f"HwwCandidate.channel() == Channel::{ch}" for ch in selected_channels
    )
    df = df.Filter(channel_to_select, "select channels")

    return df


def selectExtraLeptons(df):
    df = df.Define(
        "ExtraElectron_sel",
        "Electron_sel && (lep1_legType != Leg::e || lep1_idx != Electron_idx) && (lep2_legType != Leg::e || lep2_idx != Electron_idx)",
    )

    df = df.Define(
        "ExtraMuon_sel",
        "Muon_sel && (lep1_legType != Leg::mu || lep1_idx != Muon_idx) && (lep2_legType != Leg::mu || lep2_idx != Muon_idx)",
    )

    df = df.Define(
        "Tau_sel",
        f"""v_ops::pt(Tau_p4) > 20 && abs(v_ops::eta(Tau_p4)) < 2.5
            && abs(Tau_dz) < 0.2 && Tau_decayMode != 5 && Tau_decayMode != 6
            && Tau_idDeepTau2018v2p5VSe >= {WorkingPointsTauVSe.VVLoose.value}
            && Tau_idDeepTau2018v2p5VSmu >= {WorkingPointsTauVSmu.Tight.value}
            && Tau_idDeepTau2018v2p5VSjet >= {WorkingPointsTauVSjet.VVLoose.value}
        """,
    )

    df = df.Define(
        "ExtraTau_sel", "RemoveOverlaps(Tau_p4, Tau_sel, HwwCandidate.getLegP4s(), 0.5)"
    )
    return df


def selectJets(df, *, min_n_effective_jets_SL, min_n_effective_jets_DL, era=None):
    df = df.Define(
        "Jet_Incl",
        "v_ops::pt(Jet_p4) > 20 && abs(v_ops::eta(Jet_p4)) < 2.5 && Jet_passJetIdTight",
    )
    df = df.Define(
        "ForwardJet_sel",
        "v_ops::pt(Jet_p4) > 20 && abs(v_ops::eta(Jet_p4)) > 2.5 && Jet_passJetIdTight",
    )
    fatjet_id = (
        ""
        if era in ("Run3_2024", "Run3_2025", "Run3_2026")
        else " && ( FatJet_jetId & 2 )"
    )
    df = df.Define(
        "FatJet_Incl",
        f"v_ops::pt(FatJet_p4) > 200 && abs(v_ops::eta(FatJet_p4)) < 2.5{fatjet_id}",
    )
    df = df.Define(
        "Jet_sel",
        "Jet_Incl && Jet_idx != lep1_jetIdx && Jet_idx != lep2_jetIdx",
    )
    df = df.Define(
        "FatJet_sel",
        """FatJet_Incl && (is_SL || RemoveOverlaps(FatJet_p4, FatJet_Incl, HwwCandidate.getLegP4s(), 0.8))""",
    )
    df = df.Define(
        "Jet_cleaned",
        "RemoveOverlaps(Jet_p4, Jet_sel, FatJet_p4[FatJet_sel], 0.8)",
    )
    df = df.Define(
        "FatJet_cleaned",
        "RemoveOverlaps(FatJet_p4, FatJet_sel, Jet_p4[Jet_sel], 0.8)",
    )

    df = df.Define(
        "n_eff_jets",
        "std::max(Sum(FatJet_cleaned) * 2 + Sum(Jet_sel), Sum(FatJet_sel) * 2 + Sum(Jet_cleaned))",
    )

    df = df.Define(
        "min_n_eff_jets",
        f"is_SL ? {min_n_effective_jets_SL} : {min_n_effective_jets_DL}",
    )
    return df.Filter("n_eff_jets >= min_n_eff_jets", "Reco bjet candidates")

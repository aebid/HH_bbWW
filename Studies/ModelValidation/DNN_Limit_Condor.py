import argparse
import threading
import yaml
from FLAF.RunKit.kinit import cond as kInit_cond, update_kinit_thread
import os
import shutil
from FLAF.RunKit.run_tools import ps_call
from FLAF.RunKit.envToJson import get_cmsenv
import uproot
import json
import ROOT
import uproot
import numpy as np
import awkward as ak  # only needed if your file has complex objects; here it's harmless
import array

cmssw_env = get_cmsenv(cmssw_path=os.getenv("FLAF_CMSSW_BASE"))


def limit_to_json(input_root, output_json):
    t = uproot.open(f"{input_root}:limit", branches=["limit"])
    arrays = t.arrays()
    limits = {
        "exp-2": arrays.limit[0],
        "exp-1": arrays.limit[1],
        "exp": arrays.limit[2],
        "exp+1": arrays.limit[3],
        "exp+2": arrays.limit[4],
        "observed": arrays.limit[5],
    }
    with open(output_json, "w") as fp:
        json.dump(limits, fp)


def _compute_rebin_edges_from_right_all_nonempty(hists, eps=0.0):
    """
    Build new bin edges such that *each* final bin is non‑empty
    in *every* histogram, merging from the RIGHT.

    Condition for a final bin [j..i] (bin indices, 1-based):
        For every histogram h in hists:
            sum_{k=j..i} h.GetBinContent(k) > eps
    """
    if not hists:
        return None

    h0 = hists[0]
    nbins = h0.GetNbinsX()
    if nbins <= 0:
        return None

    # Check same binning
    for h in hists[1:]:
        if (
            h.GetNbinsX() != nbins
            or h.GetXaxis().GetXmin() != h0.GetXaxis().GetXmin()
            or h.GetXaxis().GetXmax() != h0.GetXaxis().GetXmax()
        ):
            raise RuntimeError("Histograms do not share the same binning")

    edges = []
    i = nbins  # 1-based index of last bin

    while i >= 1:
        j = i
        while j >= 1:
            all_ok = True
            for h in hists:
                s = 0.0
                for k in range(j, i + 1):
                    s += h.GetBinContent(k)
                if s <= eps:
                    all_ok = False
                    break
                # if h.GetBinContent(k+1) > 0:
                #     # Next bin is still empty, we can keep going
                #     all_ok = False
                #     break
            if all_ok:
                break
            j -= 1

        # place a boundary at the right edge of bin i
        right_edge = h0.GetBinLowEdge(i + 1)
        edges.append(right_edge)

        # next bin group ends left of j
        i = j - 1

    # global left edge
    global_left = h0.GetBinLowEdge(1)
    edges.append(global_left)

    # sort and remove duplicates
    edges = sorted(set(edges))
    if len(edges) < 2:
        return None

    return edges


def RebinShape(file_name, mass=None):
    """
    Rebin each mass point in `file_name` so that for each mass mXXX:
      - processes: Signal, TT, DY, ST
      - are rebinned with a common variable binning
      - such that *each process* has no empty bins in any final bin,
        with bins merged from the RIGHT.

    Operates in-place on the ROOT file.
    """
    ROOT.TH1.AddDirectory(False)  # avoid automatic ownership issues

    f = ROOT.TFile.Open(file_name, "UPDATE")
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open {file_name}")

    # Collect histogram names
    keys = [k.GetName() for k in f.GetListOfKeys()]
    mass_points = sorted({name.split("_")[0] for name in keys if "_" in name})
    processes = ["Signal", "TT", "DY", "Other"]
    rebin_processes = ["Signal", "TT", "Other"]

    print(f"Found mass points: {mass_points}")

    if mass != None:
        mass_points = [mass]

    for mass in mass_points:
        # Collect histograms in memory
        hists = []
        for proc in rebin_processes:
            # hname = f"{mass}_{proc}"
            hname = f"m{mass}_{proc}_class0"
            h = f.Get(hname)
            if h and isinstance(h, ROOT.TH1):
                # With TH1.AddDirectory(False), what Get(), Clone() and Rebin() return is
                # in no directory and not owned by PyROOT: owned by nobody, never freed
                ROOT.SetOwnership(h, True)
                h_clone = h.Clone()  # work on a clone in memory
                h_clone.SetDirectory(0)
                ROOT.SetOwnership(h_clone, True)
                hists.append(h_clone)

        print(
            f"For mass {mass}, found histograms for processes: {[h.GetName() for h in hists]}"
        )
        if not hists:
            continue

        # Compute common binning
        edges = _compute_rebin_edges_from_right_all_nonempty(hists)
        print(f"Computed new edges for mass {mass}: {edges}")
        if not edges:
            continue

        edges_arr = array.array("d", edges)
        nbins_new = len(edges_arr) - 1

        # Rebin each process and overwrite in file
        for proc in processes:
            # hname = f"{mass}_{proc}"
            hname = f"m{mass}_{proc}_class0"
            h_orig = f.Get(hname)
            if not h_orig or not isinstance(h_orig, ROOT.TH1):
                continue
            ROOT.SetOwnership(h_orig, True)

            # Clone original into memory, detach from file
            h_tmp = h_orig.Clone(hname + "_tmp")
            h_tmp.SetDirectory(0)
            ROOT.SetOwnership(h_tmp, True)

            # Rebin to variable binning; this propagates errors correctly
            print(f"Rebinning proc {proc} at mass {mass} with bins {edges_arr}")
            h_reb = h_tmp.Rebin(nbins_new, hname + "_reb", edges_arr)
            h_reb.SetDirectory(0)
            ROOT.SetOwnership(h_reb, True)

            # Delete all old versions from file, then write new
            f.Delete(hname + ";*")
            h_reb_final = h_reb.Clone(hname)
            h_reb_final.SetDirectory(f)
            h_reb_final.Write()

            # cleanup
            del h_tmp
            del h_reb
            del h_reb_final

        # cleanup in-memory hists
        for h in hists:
            del h

    f.Close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create TrainTest Files for DNN.")
    parser.add_argument(
        "--output_folder", required=True, type=str, help="Output Folder"
    )
    parser.add_argument("--resolved", required=True, type=int)
    parser.add_argument("--mass", required=True, type=int)
    parser.add_argument("--validation_paths", required=True, nargs="+", type=str)

    args = parser.parse_args()

    try:

        thread = threading.Thread(target=update_kinit_thread)
        thread.start()

        os.makedirs(args.output_folder, exist_ok=True)

        resolved = args.resolved
        mass = args.mass
        output_folder = args.output_folder

        cats = ["res1b", "res2b"] if resolved else ["boosted"]

        shutil.copy(
            os.path.join(
                os.environ["ANALYSIS_PATH"],
                "Studies",
                "ModelValidation",
                "config",
                "call_combine_benchmark.sh",
            ),
            ".",
        )
        shutil.copy(
            os.path.join(
                os.environ["ANALYSIS_PATH"],
                "Studies",
                "ModelValidation",
                "config",
                "Run3card_boosted_noDY.txt",
            ),
            ".",
        )
        shutil.copy(
            os.path.join(
                os.environ["ANALYSIS_PATH"],
                "Studies",
                "ModelValidation",
                "config",
                "Run3card_res2b.txt",
            ),
            ".",
        )
        shutil.copy(
            os.path.join(
                os.environ["ANALYSIS_PATH"],
                "Studies",
                "ModelValidation",
                "config",
                "Run3card_recovery.txt",
            ),
            ".",
        )
        shutil.copy(
            os.path.join(
                os.environ["ANALYSIS_PATH"],
                "Studies",
                "ModelValidation",
                "config",
                "Run3card_combined.txt",
            ),
            ".",
        )

        for cat in cats:
            hadd_list = []
            for val_path in args.validation_paths:
                hadd_list.append(os.path.join(val_path, f"validation_{cat}.root"))
            ps_call(["hadd", f"run3_{cat}.root", *hadd_list])

            RebinShape(f"run3_{cat}.root", mass)

            shutil.copy(
                f"run3_{cat}.root",
                os.path.join(output_folder, f"run3_{cat}_m{mass}.root"),
            )

        if resolved:
            ps_call(
                [
                    "combine",
                    "-M",
                    "AsymptoticLimits",
                    "Run3card_res2b.txt",
                    "--rMax",
                    "1",
                    "-t",
                    "-1",
                    "-n",
                    f"_m{mass}_res2b",
                    "-m",
                    f"{mass}",
                ],
                env=cmssw_env,
            )
            limit_to_json(
                f"higgsCombine_m{mass}_res2b.AsymptoticLimits.mH{mass}.root",
                os.path.join(output_folder, f"m{mass}_res2b.json"),
            )

            ps_call(
                [
                    "combine",
                    "-M",
                    "AsymptoticLimits",
                    "Run3card_recovery.txt",
                    "--rMax",
                    "1",
                    "-t",
                    "-1",
                    "-n",
                    f"_m{mass}_recovery",
                    "-m",
                    f"{mass}",
                ],
                env=cmssw_env,
            )
            limit_to_json(
                f"higgsCombine_m{mass}_recovery.AsymptoticLimits.mH{mass}.root",
                os.path.join(output_folder, f"m{mass}_recovery.json"),
            )
        else:
            ps_call(
                [
                    "combine",
                    "-M",
                    "AsymptoticLimits",
                    "Run3card_boosted_noDY.txt",
                    "--rMax",
                    "1",
                    "-t",
                    "-1",
                    "-n",
                    f"_m{mass}_boosted",
                    "-m",
                    f"{mass}",
                ],
                env=cmssw_env,
            )
            limit_to_json(
                f"higgsCombine_m{mass}_boosted.AsymptoticLimits.mH{mass}.root",
                os.path.join(output_folder, f"m{mass}_boosted.json"),
            )

        # ps_call(["combine", "-M", "AsymptoticLimits", "Run3card_combined.txt", "--rMax", "1", "-t", "-1", "-n", f"_m{mass}_combined", "-m", f"{mass}"], env=cmssw_env)
        # limit_to_json("higgsCombine_m600_combined.AsymptoticLimits.mH600.root", os.path.join(args.output_folder, f"m{mass}_combined.json"))

        # Clean working directory for next mass point
        for cat in cats:
            ps_call(["rm", f"run3_{cat}.root"])

    finally:
        kInit_cond.acquire()
        kInit_cond.notify_all()
        kInit_cond.release()
        thread.join()

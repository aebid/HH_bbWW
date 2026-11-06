#!/usr/bin/env python3
"""The histograms that the Studies scripts read, clone, rebin or project must be freed.

PyROOT does not own what TDirectory::Get, TH1::Clone, TH1::Rebin(n, newname, ...) and
TH2::ProjectionY return, nor a histogram constructed while a file directory is current, and
SetDirectory(0) does not give it back: detached, such a histogram would be owned by nobody and
stay allocated until the job ends. The functions are called as the scripts call them: the
DY_fit loaders and projections with gROOT current, make_hist_from_fit.fill_th2 with a directory
of the output file current, DNN_Limit_Condor.RebinShape under its TH1.AddDirectory(False). The
live histograms are counted with ROOT's object table, which counts only the objects made after
it is switched on, so the calls run in a child process that switches it on first.

Studies/DY_fit is published from HH_bbWW_dev (hh_bbww/share), where a change to it is made.
Needs ROOT: run inside the analysis environment (source env.sh).
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import pytest

pytestmark = pytest.mark.needs_root

ana_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = """
import gc, json, os, sys
ana_path, tmp = sys.argv[1:3]
sys.path.insert(0, ana_path)
sys.path.insert(0, os.path.join(ana_path, "Studies", "DY_fit"))
sys.path.insert(0, os.path.join(ana_path, "Studies", "ModelValidation"))
import numpy as np
import ROOT
ROOT.TObject.SetObjectStat(True)
ROOT.gROOT.SetBatch(True)
# Both ModelValidation scripts build a CMSSW environment on import; nothing called here uses it.
import FLAF.RunKit.envToJson
FLAF.RunKit.envToJson.get_cmsenv = lambda *args, **kwargs: {}
import core, grid_merge, make_hist_from_fit, plot_fit_slices, prepare_shapes
from DNN_Limit_Condor import RebinShape

def live():
    gc.collect()
    ROOT.gObjectTable.UpdateInstCount()
    return ROOT.TH1D.Class().GetInstanceCount() + ROOT.TH2D.Class().GetInstanceCount()

leaked = {}
results = {}

def step(name, fn):
    start = live()
    results[name] = fn()
    leaked[name] = live() - start

# In functions, so that no name keeps the proxy of a histogram that Close() deleted: cppyy
# would hand that proxy out again for a histogram made at the same address.
def write_dy_input(path):
    f = ROOT.TFile(path, "RECREATE")
    f.mkdir("plots_2d").cd()
    h = ROOT.TH2D("DY_2d", "", 15, -7.5, 7.5, 25, 0.0, 2500.0)
    for ix in range(1, 16):
        for iy in range(1, 26):
            h.SetBinContent(ix, iy, ix + iy)
            h.SetBinError(ix, iy, 1.0)
    f.Write()
    f.Close()

def write_mv_input(path, mass):
    f = ROOT.TFile(path, "RECREATE")
    for proc in ("Signal", "TT", "DY", "Other"):
        h = ROOT.TH1D(f"m{mass}_{proc}_class0", "", 20, -10.0, 10.0)
        for ib in range(1, 21):
            h.SetBinContent(ib, ib)
    f.Write()
    f.Close()

def integral(path, name):
    f = ROOT.TFile.Open(path)
    value = f.Get(name).Integral()
    f.Close()
    return value

data_dir = os.path.join(tmp, "dy")
os.makedirs(data_dir)
hadd = os.path.join(data_dir, "hadd_m650_res2b.root")
write_dy_input(hadd)

step("core.load_hist_2d", lambda: [core.load_hist_2d(650, data_dir).integral for _ in range(3)])
step(
    "grid_merge.load_th2",
    lambda: [
        float(grid_merge.th2_to_arrays(grid_merge.load_th2(hadd, "plots_2d/DY_2d"))[0].sum())
        for _ in range(3)
    ],
)
step(
    "make_hist_from_fit.load_binning_from_hist",
    lambda: [
        len(make_hist_from_fit.load_binning_from_hist(hadd + ":plots_2d/DY_2d")[0])
        for _ in range(3)
    ],
)

def write_templates():
    path = os.path.join(tmp, "templates.root")
    x_edges = np.linspace(-7.5, 7.5, 16)
    y_edges = np.linspace(0.0, 2500.0, 26)
    Z = np.ones((15, 25))
    fout = ROOT.TFile.Open(path, "RECREATE")
    for i in range(3):
        make_hist_from_fit.fill_th2("nominal_%d" % i, "", x_edges, y_edges, Z).Write()
    fout.Close()
    return [integral(path, "nominal_%d" % i) for i in range(3)]

step("make_hist_from_fit.fill_th2", write_templates)

h2 = ROOT.TH2D("fit_model", "", 15, -7.5, 7.5, 25, 0.0, 2500.0)
for ix in range(1, 16):
    for iy in range(1, 26):
        h2.SetBinContent(ix, iy, 1.0)
step(
    "plot_fit_slices._project_hme",
    lambda: [plot_fit_slices._project_hme(h2, 3, 5, "p%d" % i).Integral() for i in range(3)],
)

def hadd_parity():
    for par in range(4):
        write_mv_input(os.path.join(tmp, f"par{par}_m300_res2b.root"), 300)
    fp = os.path.join(tmp, "par{par}_m{mass}_{cat}.root")
    out = prepare_shapes.hadd_parity_files(
        fp, fp, "m{mass}_{proc}_class0", [300], ["res2b"], os.path.join(tmp, "hadd")
    )
    return integral(out.format(mass=300, cat="res2b"), "signal")

step("prepare_shapes.hadd_parity_files", hadd_parity)

def rebin_shape():
    path = os.path.join(tmp, "run3_res2b.root")
    write_mv_input(path, 300)
    for _ in range(3):
        RebinShape(path, 300)
    # RebinShape switches it off for the whole process; integral() needs it on
    ROOT.TH1.AddDirectory(True)
    return integral(path, "m300_TT_class0")

step("DNN_Limit_Condor.RebinShape", rebin_shape)
print(json.dumps({"leaked": leaked, "results": results}))
"""


class TestStudiesHistOwnership(unittest.TestCase):
    maxDiff = None

    def test_histograms_are_freed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-c", CHILD, ana_path, tmp],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            out = json.loads(result.stdout.strip().splitlines()[-1])
        dy_total = float(sum(ix + iy for ix in range(1, 16) for iy in range(1, 26)))
        self.assertEqual(
            out["results"],
            {
                "core.load_hist_2d": [dy_total] * 3,
                "grid_merge.load_th2": [dy_total] * 3,
                "make_hist_from_fit.load_binning_from_hist": [16] * 3,
                "make_hist_from_fit.fill_th2": [375.0] * 3,
                "plot_fit_slices._project_hme": [75.0] * 3,
                "prepare_shapes.hadd_parity_files": 840.0,
                "DNN_Limit_Condor.RebinShape": 210.0,
            },
        )
        self.assertEqual(
            out["leaked"],
            dict.fromkeys(out["results"], 0),
            "histograms left owned by nobody, per function",
        )


if __name__ == "__main__":
    unittest.main()

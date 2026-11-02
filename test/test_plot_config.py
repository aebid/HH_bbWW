#!/usr/bin/env python3
"""HH_bbWW's control-plot configuration, through FLAF's own plot rules.

For every era, the meta-eras of era_groups (Run3_22_23) and both DY models (Run3_Model,
Run3_Model_DYSplit), the processes HistPlotter plots are built from the Setups as HistPlotter
builds them, then grouped, blinded and filtered by FLAF's plot_settings / control_regions /
want_data / want_signals / GroupProcesses (Common/PlotGroups.py), and checked against what the
plots must show:

- the plot settings: signals normalised to the background (signal_plot_scale: bkg), and the
  control regions exactly TT_CR and DY_CR -- a misspelt name is an error here, not a warning;
- no plot group raises (malformed group, background mixed with signal, ...);
- every background lands in a plot group, the stack sums its inputs, and the groups are
  stacked in the intended order (STACK, from the bottom);
- in a signal region: one X->HH (Radion) entry per plotted mass (400, 700, 1000 GeV) in
  increasing order, each the sum of the bbWW 2L, bbWW 1L and bbtautau samples of that mass -- a
  typo in a group's process names would instead give the single samples -- and no data for a
  blinded variable;
- in the control regions: no signal, and data also for a blinded variable.

ROOT is not needed. Run: python3 test/test_plot_config.py
"""

import importlib.util
import os
import sys
import tempfile
import unittest
import warnings
from unittest import mock

ana_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ana_path)
os.environ.setdefault("ANALYSIS_PATH", ana_path)
# FLAF imports ROOT and nothing tested here uses it, so where ROOT is missing a stand-in serves
# the import only and is withdrawn again: test modules share one process (FLAF test/conftest.py).
_root_stand_in = None
if "ROOT" not in sys.modules and importlib.util.find_spec("ROOT") is None:
    _root_stand_in = sys.modules["ROOT"] = mock.MagicMock()

from FLAF.Common.PlotGroups import (  # noqa: E402
    GroupProcesses,
    control_regions,
    plot_settings,
    want_data,
    want_signals,
)
from FLAF.Common.Setup import Setup, meta_eras  # noqa: E402

if _root_stand_in is not None and sys.modules.get("ROOT") is _root_stand_in:
    del sys.modules["ROOT"]

ERAS = [
    "Run3_2022",
    "Run3_2022EE",
    "Run3_2023",
    "Run3_2023BPix",
    "Run3_2024",
    "Run3_2025",
    "Run3_2026",
]
MODELS = ["Run3_Model", "Run3_Model_DYSplit"]
SIGNAL_REGION = "SR"
CONTROL_REGIONS = ["TT_CR", "DY_CR"]
# the spin-0 samples summed into one X->HH entry per mass
RADION = ("GluGluToRadion_bbWW_2L", "GluGluToRadion_bbWW_1L", "GluGluToRadion_bbTauTau")
# the background groups from the bottom of the stack, per DY model
STACK = {
    "Run3_Model": [
        "Diboson + Triboson",
        "Single Higgs",
        "W#rightarrow#ell#nu + Jets",
        "DY",
        "Single top",
        "t#bar{t} + t#bar{t}V",
    ],
    "Run3_Model_DYSplit": [
        "Diboson + Triboson",
        "Single Higgs",
        "W#rightarrow#ell#nu + Jets",
        "DY m_{#ell#ell}#in[10,50] GeV",
        "DY m_{#ell#ell}>50 GeV",
        "Single top",
        "t#bar{t} + t#bar{t}V",
    ],
}


class Hist:
    """The part of a ROOT TH1 GroupProcesses uses; one bin, so sums can be checked."""

    def __init__(self, value):
        self.value = value

    def Clone(self, name):
        return Hist(self.value)

    def SetDirectory(self, directory):
        pass

    def Add(self, other):
        self.value += other.value


def setup_for(era, model, tmp_dir):
    user_custom = os.path.join(tmp_dir, f"user_custom_{model}.yaml")
    with open(user_custom, "w") as f:
        f.write(f"phys_model: {model}\n")
    return Setup(ana_path, era, "test_plot_config", user_custom_file=user_custom)


def plotted_processes(setups, channel, region, plot_control_regions):
    """{process: (hist, plot name, color, process group)} as HistPlotter builds it for one
    plot of the eras of ``setups`` (data left out), each background with its own value so
    sums can be checked."""
    show_signals = want_signals(True, region, plot_control_regions)
    hists = {}
    for setup in setups:
        for dataset in setup.datasets.values():
            group = dataset["process_group"]
            if group == "data":
                continue
            name = setup.base_processes[dataset["process_name"]]["parent_process"]
            process = setup.parent_processes[name]
            if not process.get("to_plot", True):
                continue
            if group == "signals":
                if not show_signals or channel not in process.get(
                    "channels", [channel]
                ):
                    continue
            value = float(len(hists) + 1)
            hists[name] = (
                Hist(value),
                process.get("name", name),
                process["color"],
                group,
            )
    return hists


def radion_masses(setups, channel, samples=RADION):
    """The masses of the Radion `samples` the configuration plots in `channel`."""
    masses = set()
    for setup in setups:
        for name, process in setup.parent_processes.items():
            sample, _, mass = name.rpartition("_")
            if sample in samples and process.get("to_plot"):
                if channel in process.get("channels", [channel]):
                    masses.add(int(mass))
    return sorted(masses)


class PlotConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.setups = {
            (era, model): setup_for(era, model, cls.tmp.name)
            for model in MODELS
            for era in ERAS
        }
        # every plotted era: each era alone, and each meta-era as the sum of its eras
        cls.plotted = {
            (era, model): [cls.setups[(era, model)]] for model in MODELS for era in ERAS
        }
        cls.meta_eras = meta_eras(ana_path)
        for meta, eras in cls.meta_eras.items():
            for model in MODELS:
                cls.plotted[(meta, model)] = [cls.setups[(era, model)] for era in eras]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def settings(self, setups):
        """The plot settings of the eras of ``setups`` and their control regions; a control
        region that is not one of the regions fails instead of warning."""
        settings = plot_settings({s.era: s.global_params for s in setups})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            crs = control_regions(settings["plot_control_regions"], settings["regions"])
        return settings, crs

    def plots(self):
        """(era, model, setups, settings, control regions, channel, region) of every plot."""
        for (era, model), setups in self.plotted.items():
            settings, crs = self.settings(setups)
            for channel in setups[0].global_params["channelSelection"]:
                for region in [SIGNAL_REGION] + crs:
                    yield era, model, setups, settings, crs, channel, region

    def test_the_plot_settings(self):
        self.assertIn("Run3_22_23", self.meta_eras)
        for (era, model), setups in self.plotted.items():
            with self.subTest(era=era, model=model):
                settings, crs = self.settings(setups)
                self.assertEqual(crs, CONTROL_REGIONS)
                self.assertEqual(settings["signal_plot_scale"], "bkg")
                # a blinded variable shows data in the control regions only
                for cr in crs:
                    self.assertTrue(want_data(True, True, cr, crs))
                self.assertFalse(want_data(True, True, SIGNAL_REGION, crs))

    def test_groups_stack_every_background_in_order_and_keep_the_total(self):
        for era, model, setups, settings, crs, channel, region in self.plots():
            with self.subTest(era=era, model=model, channel=channel, region=region):
                hists = plotted_processes(setups, channel, region, crs)
                grouped = GroupProcesses(hists, settings["plot_process_groups"])
                backgrounds = [e for e in grouped.values() if e[3] == "backgrounds"]
                self.assertEqual([e[1] for e in backgrounds], STACK[model])
                self.assertEqual(
                    sum(e[0].value for e in backgrounds),
                    sum(e[0].value for e in hists.values() if e[3] == "backgrounds"),
                )

    def test_one_radion_entry_per_mass_in_the_signal_region_none_in_control_regions(
        self,
    ):
        for era, model, setups, settings, crs, channel, region in self.plots():
            with self.subTest(era=era, model=model, channel=channel, region=region):
                hists = plotted_processes(setups, channel, region, crs)
                grouped = GroupProcesses(hists, settings["plot_process_groups"])
                signals = [(k, e) for k, e in grouped.items() if e[3] == "signals"]
                if region in crs:
                    self.assertEqual(signals, [])
                    continue
                masses = radion_masses(setups, channel)
                self.assertEqual(len(signals), len(masses), [k for k, _ in signals])
                for mass, (key, entry) in zip(masses, signals):
                    self.assertRegex(entry[1], rf"=\s*{mass} GeV")
                    self.assertEqual(
                        sorted(key.split("+")),
                        sorted(
                            f"{sample}_{mass}"
                            for sample in RADION
                            if f"{sample}_{mass}" in hists
                        ),
                    )

    def test_the_early_eras_plot_the_three_masses_of_each_radion_sample(self):
        for era in ERAS[:4] + ["Run3_22_23"]:
            for model in MODELS:
                setups = self.plotted[(era, model)]
                for channel in ("e", "mu", "eE", "eMu", "muMu"):
                    for sample in RADION:
                        with self.subTest(
                            era=era, model=model, channel=channel, sample=sample
                        ):
                            self.assertEqual(
                                radion_masses(setups, channel, (sample,)),
                                [400, 700, 1000],
                            )


if __name__ == "__main__":
    unittest.main()

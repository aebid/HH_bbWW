# Running the analysis

The pipeline — anaTuples, observables, histograms, plots — is the **standard FLAF chain**; follow
the **[FLAF full-workflow walkthrough](https://cms-flaf.github.io/FLAF/workflow/walkthrough/)** for
the commands. This page collects what is **specific to HH→bb̄WW**.

```sh
ERA=Run3_2022
VER=dev

law run FLAF.Analysis.tasks.HistPlotTask --era $ERA --version $VER --workflow local --test 1000
```

## The b-tag-shape cache runs first

HH→bb̄WW computes **b-tag shape** weights in a dedicated caching stage,
`AnalysisCacheTask` (aggregated by `AnalysisCacheAggregationTask`), which LAW runs automatically
before histogramming — see
[FLAF → Task reference](https://cms-flaf.github.io/FLAF/reference/tasks/#analysiscachetask).

!!! warning "Budget time for the cache on a cold start"
    On a cold cache `AnalysisCacheTask` is **slow** (roughly an hour per branch), even for simple
    variables. When iterating, reuse an existing cache instead of recomputing it, using a
    [per-task version override](https://cms-flaf.github.io/FLAF/workflow/arguments/#per-task-version-overrides),
    e.g. `--AnalysisCacheTask-version <existing> --AnalysisCacheAggregationTask-version <existing>`.

## Mass reconstruction: DeepHME

The HH mass is reconstructed with **DeepHME** (the bb̄WW counterpart to SVfit in bb̄ττ). It is part
of the observable computation and requires no special command — it runs as part of the standard
producer chain.

DeepHME and the DNN producers run their ONNX models with one thread per session
(`FLAF.Common.OnnxSession`), so their outputs do not depend on the core count of the batch node.
They still differ in the last digits between `x86_64-v3` and `x86_64-v4` nodes: compare them with a
tolerance ([NN outputs and the CPU of the node](https://cms-flaf.github.io/FLAF/concepts/environment/#nn-outputs-and-the-cpu-of-the-node)).

## Stored gen-level t̄t information

Processes with `genInfo: [ TT ]` store, next to the usual observables:

| Column | Content | Read by |
|---|---|---|
| `TTInfo_nLeptonicW` | number of leptonically decaying W bosons (0, 1 or 2; τ counts as leptonic) | — (t̄t is not stitched: `TT` uses only the three decay-channel samples) |
| `genTop_{pt,eta,phi,mass}` | last-copy top and anti-top, in this order | top-p<sub>T</sub> reweighting at `AnaTupleFileTask` (`genTop_pt`; configured for Run3_2022–2023BPix) |
| `genTop_b_{pt,eta,phi}`, `genTop_lep_{pt,eta,phi,mass}` | the b quark and the W's charged lepton of each top | — |
| `genTop_lep_gen_kind` | `GenLepton::Kind` of that lepton, -1 for a hadronic W | — |

`TTInfo_*` columns must stay scalars: the anaTuple stores all columns that share a prefix as one
collection, so an array under `TTInfo_` would turn the count into an array as well.

## Corrections in Run3_2024 and later

`config/Run3_2024/global.yaml` (and 2025, 2026) inherits the analysis-wide `corrections:` block
through the anchor `*corrections_default` and overrides only `btag` (UParTAK4, no shape
calibration yet) and `dy_hhbbtautau`. A correction added to `config/global.yaml` — the shape
weights `pu`, `parton_shower`, `top_pt`, `pdf` and `qcd_scale` included — therefore applies to
those eras too; an entry that has to differ there is overridden as a whole.

## Resonant mass grid

The bb̄WW signal is `GluGluTo{Radion,BulkGraviton}` at 40 mass points from 250 GeV to 5 TeV in each
of `Run3_2022` through `Run3_2023BPix`, with the single-lepton (`2B2JLNu`) and dilepton (`2B2L2Nu`)
final states as separate datasets. In 2023 and 2023BPix the whole grid comes from the custom
production (DSProd, read from private storage with `fs_nanoAOD:` + `dirName:`); in 2022 and 2022EE
18 points per final state do and the other 22 are central `Run3Summer22` datasets. The central
X→YH samples are not selected. The bb̄ττ signal (`2B2Tau`, processes
`GluGluTo{Radion,BulkGraviton}_bbTauTau`) has the same 40 masses: in 2023 and 2023BPix 29 points
per spin come from the custom production and 11 are central, in 2022 and 2022EE 7 per spin come
from the custom production and 33 are central. `Run3_Model` carries every hypothesis through
anaTuple and histogram production; a fit picks one later, in the datacard. See the
[physics model](setup.md#production-model).

Two networks consume that grid, and they do not cover the same part of it:

- the **dilepton DNN** is parametric — one model per fold, evaluated at whatever masses
  `DNN.columns` in `config/global.yaml` asks for. It is configured over the produced points up to
  1.5 TeV, and the datacard `param_values` in `config/Datacards/x_hh_bbww_DL_run3.yaml` follow
  the same 25 points. The four lowest of them, 250 to 280 GeV, are below the lowest
  `signal_mass_points` of the models currently in `config/DNN/DoubleLepton_Parametric_*_v1`, so
  those scores extrapolate and stay provisional until the network is retrained over this grid.
- the **single-lepton two-stage DNN** has a binary stage that is parametric too, evaluated at the
  `TwoStageDNN.masses` in `config/global.yaml`: the 30 produced points from 250 GeV to 2 TeV. Its
  multiclass stage does not depend on the mass, so neither do the SL signal and control regions
  (see [below](#single-lepton-two-stage-dnn)). The SL datacard
  (`config/Datacards/x_hh_bbww_SL_run3.yaml`) has not followed yet: it still reads the per-mass
  regions `SR_SL_M${MX}` that no longer exist, at ten masses from 300 GeV to 1 TeV.

Every mass point named in a `variables:` list also needs its binning in `config/plot/histograms.yaml`
— a variable with no matching entry there stops `HistProducerFromNTuple` with a `KeyError`.

## Single-lepton two-stage DNN

The `TwoStageDNN` payload producer (`TwoStageDNNProducer` in `Analysis/DNN_Application.py`,
configured under `payload_producers` in `config/global.yaml`) applies two networks per category
(boosted, resolved) and fold. The models and their feature lists are in
`config/DNN/SingleLepton_AllMassMulticlass_ParametricBinary/`: `<category>/multiclass_nParity<p>.onnx`
and `<category>/parametric_binary_nParity<p>.onnx`, next to `{multiclass,binary}_<category>_dnn_config.yaml`.
Model `p` is applied to the events with `event % 4 == (p + 3) % 4`.

1. **Multiclass**, one model for all masses. Its 7 outputs, in the order of `class_names` (Signal,
   TT, ST, WJets, DY, H, VV), are stored as `TwoStageDNN_<class>`. In `Analysis/hh_bbww.py` the
   class with the highest score selects the SL region: `SR_SL` for Signal, `CR_SL_<class>` for the
   others. The regions are built only when one of them is in `QCDRegions`.
2. **Binary**, parametrized by the resonance mass, which is the **last** input column. The producer
   evaluates it at each of `TwoStageDNN.masses`, filling that column with the hypothesis mass in
   GeV (the input normalization is part of the ONNX graph), and stores `TwoStageDNN_M<mass>_binary`.
   The last entry of the binary feature list must be `mass_feature` (`X_mass`), or the producer
   stops; the `X_mass` branch of the event is not read.

What the models return and what is stored is configured per network in `multiclass_output` and
`binary_output`:

| Key | Meaning |
|---|---|
| `num_nodes` | number of model outputs, checked when the model runs (7 multiclass, 2 binary) |
| `type` | what the model returns: `logit` or `probability` |
| `activation` | what turns its logits into probabilities: `softmax` or `sigmoid` |
| `save_as` | what is stored: `logit` (raw), `probability`, or `recomputed_logit` |
| `signal_node` | binary model with more than one output: the signal output (1; 0 is background) |

The current models return raw logits. Both networks store `recomputed_logit`: log(p / (1 − p)) of
the softmax probability of each output, with p clipped to [`logit_clip_eps`, 1 − `logit_clip_eps`].
For the 2-output binary model this is the logit difference z<sub>1</sub> − z<sub>0</sub>. The raw
signal logit z<sub>1</sub> alone has no absolute scale: softmax does not change when the same
constant is added to both outputs. With `logit_clip_eps: 1.0e-10` every stored score is within
±23.03, and the TwoStageDNN entries of `config/plot/histograms.yaml` use the range [−23.5, 23.5]
(94 bins in 1D, 47 on the DNN axis of the plots against DeepHME). The range has to contain the
bound strictly, as plots drop the overflow; changing `logit_clip_eps` means changing it too.
Storing raw logits cannot be configured for a model that returns probabilities.

An event that no fold scores stops the producer. A non-finite model output only prints a warning,
with the event numbers, and is stored as it is.

## Columns taken from the central tree

`config/global.yaml` lists in `anaTuple_shift_invariant_columns` the anaTuple columns that no
systematic shift changes: event numbers and dataset metadata, generator weights, the pileup, parton-
shower, PDF, scale and top-p<sub>T</sub> weights, pileup truth, the LHE record and the generator truth
of the signal (`gen*`) and of t̄t (`TTInfo_*`, `genTop_*`). FLAF stores them in the central tree only
and fills them in for events that only a shift selected, after checking that every variation agrees.

- Only event-level generator quantities belong there. Generator information attached to a
  reconstructed object — `lepN_gen_*`, `centralJet_matchedGenJet_*`, the jet flavour labels — follows
  the selected object and changes under JES, JER and the lepton scales; the fuse step stops with
  `Column '…' is declared shift-invariant but differs in …` if one is listed.
- `HLT_*` is left out: `Analysis/hh_bbww.py` defines a trigger column that `GetColumnNames()` does not
  list, and a column taken from the central tree is not listed there.
- Changing the list changes the anaTuple layout, so it goes with a new anaTuple production.

## Categories

HH→bb̄WW is analysed in **resolved** and **boosted** categories (low- and high-p_T HH topologies).
Category and channel selection is driven by `config/global.yaml`; narrow or extend it there or via
your [`user_custom.yaml`](https://cms-flaf.github.io/FLAF/configuration/user-custom/).

## Choosing which variables to histogram

As for any FLAF analysis, the variable set is controlled by the `variables:` list in
`user_custom.yaml` (or `--variables`). A short list keeps test runs fast:

```yaml
variables:
  - lep1_pt
  - ggF_DNN_HH
```

## Control plots

`HistPlotTask` draws the stacked control plots with the HH→bb̄WW settings below; the keys are
FLAF's, described in its [HistPlotTask reference](https://cms-flaf.github.io/FLAF/reference/tasks/#histplottask).

- **Backgrounds** are grouped into one stack entry each (`plot_process_groups` in
  `config/global.yaml`), from the bottom: Diboson + Triboson, Single Higgs, W→ℓν + Jets, DY (or,
  with `Run3_Model_DYSplit`, DY m_ℓℓ ∈ [10, 50] GeV and DY m_ℓℓ > 50 GeV), Single top,
  t̄t + t̄tV.
- **Signal:** only the spin-0 (Radion) X→HH signal is drawn, at M_X = 400, 700 and 1000 GeV, in
  every channel of the eras with these samples (Run3_2022 to Run3_2023BPix, and Run3_22_23): per
  mass, one curve summing its bb̄WW dilepton (which includes HH→bb̄ZZ→2ℓ2ν),
  bb̄WW single-lepton and bb̄ττ samples. Each sample is normalised to σ(X→HH) = 1 pb times the
  branching fraction of its own final state, and the final states do not overlap. The merged
  histograms keep the samples apart (see [Production model](setup.md#production-model)).
  Signals are normalised to the total background (`signal_plot_scale: bkg`), as the legend says.
- **Control regions** (`plot_control_regions: [ TT_CR, DY_CR ]`): no signal is drawn, and the
  data is shown even for variables blinded in the signal regions (`blind: true` in
  `config/plot/histograms.yaml`).
- **Labels:** "CMS Preliminary" ("CMS Simulation Preliminary" where the data are blinded); the
  luminosity of each era is the one the simulation is normalised to; axis titles carry their
  units.
- **Tests:** `test/test_plot_config.py` (CI: Unit Tests, through FLAF's reusable workflow) checks,
  with FLAF's own plot rules, in every era, the Run3_22_23 meta-era and both DY models: the
  control regions (exactly TT_CR and DY_CR) with the data they show, the signal normalisation,
  the grouping and stack order of the backgrounds, and the X→HH entries. It does not render the
  plots.

## Statistical interpretation

Continue to [Statistical inference](stat_inference.md) for datacards, limits and diagnostics.

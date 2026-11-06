# HH → bb̄WW

This is the documentation for the **HH→bb̄WW** analysis — the search for Higgs-boson pair
production in the bb̄WW final state, built on the
[FLAF framework](https://cms-flaf.github.io/FLAF/).

!!! abstract "The common workflow lives in the FLAF docs"
    HH→bb̄WW runs on FLAF, so installation, the task pipeline (NanoAOD → anaTuples → histograms →
    plots), the configuration system, storage, eras and CI are **shared with every FLAF analysis**
    and are documented once, in the **[FLAF documentation](https://cms-flaf.github.io/FLAF/)**:

    - [Prerequisites & installation](https://cms-flaf.github.io/FLAF/getting-started/installation/)
    - [Your first run](https://cms-flaf.github.io/FLAF/getting-started/first-run/)
    - [Full workflow walkthrough](https://cms-flaf.github.io/FLAF/workflow/walkthrough/)
    - [Command arguments](https://cms-flaf.github.io/FLAF/workflow/arguments/)

    **This site covers only what is specific to HH→bb̄WW.**

## What this analysis adds on top of FLAF

| Ingredient | Purpose |
|---|---|
| **DeepHME** | Neural mass reconstruction of the HH system (the bb̄WW counterpart to SVfit). |
| **B-tag-shape cache** | Per-event b-tag shape weights, pre-computed via `AnalysisCacheTask`. |
| Resolved & boosted categories | Two topologies analysed (low- and high-p_T HH). |
| **StatInference** | Datacards and limits (cards listed in `x_hh_bbww_run3.yaml`: `DL/dl_spin0.yaml`, `DL/dl_spin2.yaml`; `x_hh_bbww_SL_run3.yaml`). |

Setup is in [Setup](setup.md); analysis-specific run notes in
[Running the analysis](analysis.md); the statistics step in
[Statistical inference](stat_inference.md).

## Quickstart

```sh
git clone --recursive git@github.com:cms-flaf/HH_bbWW.git
cd HH_bbWW
git lfs pull                                   # HH_bbWW ships large files via git LFS
source env.sh                                  # first time builds the environment
voms-proxy-init -voms cms -rfc -valid 192:00
law index --verbose
```

!!! warning "Run `git lfs pull` after cloning"
    HH→bb̄WW stores some large files with [git LFS](https://git-lfs.com/). If `git lfs pull` is
    skipped, those files are placeholder pointers and the analysis will fail in confusing ways.

Then smoke-test the chain (see
[FLAF → first run](https://cms-flaf.github.io/FLAF/getting-started/first-run/)):

```sh
law run FLAF.Analysis.tasks.HistPlotTask \
  --version my_first_run --era Run3_2022 --workflow local --test 1000
```

New to FLAF? Read [Key terms](https://cms-flaf.github.io/FLAF/getting-started/key-terms/) and
[Concepts](https://cms-flaf.github.io/FLAF/concepts/architecture/) first.

## Eras

HH→bb̄WW runs over the Run 3 eras `Run3_2022`, `Run3_2022EE`, `Run3_2023`, `Run3_2023BPix`,
`Run3_2024`, `Run3_2025` and `Run3_2026`. 2025 and 2026 reuse the 2024 Summer24 MC
(with that year's corrections); see
[FLAF → Eras](https://cms-flaf.github.io/FLAF/concepts/eras/). 2024+ AK4
b-tagging uses UParTAK4 scores and working points (ParticleNet is kept for 2022/2023).

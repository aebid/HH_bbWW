# HH_bbWW — instructions for Copilot code review

The HH→bb̄WW analysis, built on [FLAF](https://github.com/cms-flaf/FLAF). It differs from the other
analyses in using DeepHME for mass reconstruction and in running the statistical inference chain
(datacards and limits) as part of its integration test.

**Read `FLAF/.github/copilot-instructions.md` first.** It carries the framework invariants — law
task semantics, bundles, remote-storage caching, processor stages, concurrency — and the rules on
what a useful comment looks like and what not to flag. The rule that documentation ships in the same PR applies here too, and is restated below with the pages that matter for this repository. Everything there applies here. This file
adds only what is specific to this analysis.

## Analysis-specific invariants

### Shift-invariant columns

- `anaTuple_shift_invariant_columns` in `config/global.yaml` lists only event-level generator and
  event quantities. Generator information attached to reconstructed objects (`lepN_gen_*`,
  `centralJet_matchedGenJet_*`, `*_hadronFlavour`) changes under shifts and must not be listed; nor
  `HLT_*` while `Analysis/hh_bbww.py` checks trigger columns with `GetColumnNames()`.
- Every input of the merge-stage `weight_base` (generator, luminosity, cross-section and shape
  weights, the stitching variables) must be listed: shifted trees take `weight_base` from the central
  tree, and a placeholder row only gets a real weight if all of its inputs are filled.

### Corrections reach the hooks as an argument

FLAF builds one `Corrections` object per dataset, era and stage and passes it to the hooks as the
required keyword `corrections`: `addAllVariables(..., *, corrections)`, `GetDfw(df, setup,
dataset_name, *, corrections, ...)` and `DefineWeightForHistograms(..., corrections)`. There is no
global instance: code that keeps one in a module-level variable, or builds its own, mixes eras
when a process handles several. The object owns C++ providers that the dataframe refers to by
address, so it must stay reachable while the dataframe is used (FLAF keeps it on the wrapper as
`dfw.corrections`). The era is called `era` (`--era`); `period` is only the AnaTuple column.

### Stitching processors

- The shared anchors in `config/processes.yaml` (`.DY_processors`,
  `.DY_processors_allFlavors`, …) differ **only** in the stitching config file they point at.
  Every one of them must declare `stages: [ AnaTuple, AnaTupleMerge ]`. A stitcher present at
  `AnaTuple` alone writes an anaCache denominator that nothing can combine, and every merge of
  every process using that anchor dies with
  `combineAnaCaches: processor Stitcher not provided for combining anaCaches`. This shipped
  undetected in the 2024–2026 configs because no CI process ran DY.
- Every anchor also declares `dependency_level: {AnaTuple: file, AnaTupleMerge: process}`. The
  merge has to see the whole process: where an inclusive sample sits next to exclusive ones, a
  per-dataset merge normalises each to the full cross-section and counts the overlap twice.
- t̄t is not stitched: `TT` takes only the three decay-channel samples (`TTto2L2Nu`, `TTtoLNu2Q`,
  `TTto4Q`) in every era, which do not overlap. The inclusive `TT`/`TT_ext1` samples of Run3_2022
  and Run3_2023BPix are not used — they carry no parton-shower weights (`PSWeight` has a single
  entry), on which the `parton_shower` producer stops. A diff that brings them back needs the t̄t
  stitcher back as well.
- Every `TTInfo_*` column must stay a scalar. FuseAnaTuples stores all columns of one prefix as one
  collection, so an array under `TTInfo_` would turn the scalars into one copy per entry. The
  per-top arrays therefore live in `genTop`.
- Which anchor an era's DY process uses is deliberate — `allFlavors` for 2022–2023BPix, the
  single-flavour one for 2024 onwards. Do not propose harmonising them.

### The CI datacard

`config/ci_custom.yaml` selects the datacard configuration of the `test_multi_era` job:
`config/Datacards/CI_card_SL.yaml` (single lepton) today, `config/Datacards/CI_card.yaml` (double
lepton, with the preprocess step) before. Two things about them:

- They declare `eras: Run3_2022 … Run3_2023BPix` only. The job passes every CI era in `--era`
  (StatInference takes several): the four behind the cards are carded, the later ones are only
  plotted. A card naming a later era would fail on a missing signal histogram; that is the
  card's scope, not a bug.
- **The DY background is deliberately not in it.** The bbWW DY samples are amc@NLO, so they carry
  negative event weights, and under `--test 1000` too few events remain for those to cancel: the
  histogram comes out negative but statistically compatible with zero. `resolveNegativeBins`
  returns as soon as the integral is negative, so including it would need
  `allow_negative_integral: true` — disabling a real safety check — and would be flaky either
  way. Do not suggest adding it back, and treat any new `allow_negative_*` flag as needing an
  explicit justification.

### Integration test

`TestModel` runs `custom_CI_Background_TT`, `custom_CI_Background_DY` and
`custom_CI_Background_W` plus one signal and one data process, and each CI background must carry
the same `processors:` as the real `TT` / DY / W process **of that era** — that is what exercises the stitching end to end. A diff that changes a
real process's processors and leaves the CI counterpart behind silently removes the coverage.

The process names are also listed in `cms-flaf/FLAF_ci`, a **different repository**; renaming or
adding one here needs that updated in step.

### Era corrections

- `config/Run3_2024/global.yaml`, 2025 and 2026 inherit `corrections:` with
  `<<: *corrections_default` and override only `btag` and `dy_hhbbtautau`. Do not turn them back
  into full copies: a correction added at top level then silently disappears from those eras,
  which is how the 2024–2026 configs lost `pdf`, `qcd_scale` and `top_pt`. The merge is one
  level deep, so an overridden entry must be complete.

### ONNX sessions

- DeepHME and the DNN producers get their ONNX Runtime sessions, or their session options, from
  `FLAF.Common.OnnxSession` (one thread; DeepHME through its `session_options` argument). A bare
  `ort.InferenceSession` with the default options makes the outputs depend on the core count of
  the batch node (cms-flaf/FLAF#357).

### Cost

`AnalysisCacheTask` (BtagShape) runs before `HistTupleProducerTask` even for simple variables and
dominates the runtime. A change that adds work to the per-branch path there is expensive; say so.

## Documentation must ship with the change

A PR must update the documentation **in the same PR** whenever it changes anything a user of the
framework can observe. Treat this as a review item of the same weight as correctness — docs
drifting from the code is the failure that motivated the current documentation, and a PR that
lands without them is not complete.

Ask, for every diff: does it add, rename or remove any of these?

- a task or DAG node, or the arguments/parameters of one;
- a command, a CLI flag, or the meaning of an existing one;
- a configuration key — `global.yaml`, `user_custom.yaml`, `processes.yaml`, `phys_models.yaml`,
  cross-sections, `fs_*` storage keys, bundle flavours, processor entries;
- a dataset, era, process or physics-model name;
- the environment, installation or setup steps;
- storage locations, output paths or log locations;
- a CI workflow, or how the integration test is triggered or configured;
- any behaviour a user relies on, including a default that changes.

If the answer is yes and the diff touches **no** documentation file, say so and name the page that
should have changed. If the author states the change is internal-only, that is a legitimate
answer — a pure refactor or bugfix with no user-visible effect is exempt — but it should be
stated in the PR, not left implicit.

Also flag the inverse: documentation edited to describe behaviour the diff does not implement, and
new pages added without being wired into `mkdocs.yml`'s `nav` (the build fails on that, but the
review should catch it first).

Where it goes:

- `docs/` in this repository for analysis-specific material (`analysis.md`, `setup.md`, `stat_inference.md`).
- **`FLAF/docs/` for anything framework-wide.** If the change alters shared behaviour, the
  documentation belongs there, in a companion PR to `cms-flaf/FLAF` — flag that it is missing
  rather than accepting an analysis-local description of a framework change.
- New pages must be added to `nav:` in `mkdocs.yml`; verified with `mkdocs build --strict`.

## Repository facts

Verified 2026-10-23; re-check before relying on any of it.

| | |
|---|---|
| Layout | `AnaProd/` (`anaTupleDef.py`, `baseline.py`), `Analysis/` (`hh_bbww.py`, `histTupleDef.py`, DeepHME/DNN producers), `Studies/`, `config/`, `include/`, `docs/`, `test/` |
| Submodules | `FLAF`, `Corrections`, `StatInference`, `inference`, `DeepHME`, `SyncTool` |
| Eras | Run 3: 2022, 2022EE, 2023, 2023BPix, 2024, 2025, 2026 |
| Configs | `config/global.yaml`, `config/processes.yaml` (processor anchors), `config/phys_models.yaml`, `config/<era>/{datasets,processes}.yaml`, `config/Datacards/` (`x_hh_bbww_{DL,SL}_run3.yaml`, CI: `CI_card.yaml`, `CI_card_SL.yaml`) |
| Large files | DNN `.onnx` payloads are tracked with Git LFS; never commit a binary directly |
| Tests | `test/test_plot_config.py`: the control-plot configuration, checked through FLAF's plot rules (`FLAF/Common/PlotGroups.py`) in every era, the Run3_22_23 meta-era and both DY models, without ROOT. A `test/test_*.py` added here runs in `unit-tests` without a workflow change (FLAF's reusable workflow, against the pinned FLAF); a test that needs ROOT, CVMFS or a grid proxy carries the `needs_root`/`needs_cvmfs`/`needs_proxy` marker, and a ROOT stand-in is withdrawn after the import it serves (FLAF's `docs/contributing.md`, Unit tests). Otherwise validation is the integration test and physics checks; the framework's suites live in `FLAF/test/` |
| Workflows | `formatting-check`, `unit-tests` (FLAF's, over `test/` with the pinned FLAF, `test/test_plot_config.py` included), `repo-sanity-checks` (FLAF's sanity checks and FLAF's `run_tools/check_era_rename.py` for leftovers of the `period` → `era` rename), `test-setup-loading`, `test-era-dimension` (FLAF's `test/test_era_dimension.py` over the seven Run 3 eras: every stage built over all eras at once and taken through what the scheduler does before anything runs, and the settings that must agree between eras), `deploy-docs`, `trigger-flaf-integration`. Formatting, era loading, the multi-era construction and the control-plot configuration are checked automatically — do not comment on them |
| Integration test | Triggered by `@cms-flaf-bot please test`; its configuration lives in `cms-flaf/FLAF_ci`, **not** in this repo. There is no `.github/integration_cfg.yaml` here |
| Docs | `docs/`, plus the shared framework docs in `FLAF/docs/` |

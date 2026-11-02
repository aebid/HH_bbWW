# Statistical inference

The final step turns the merged histograms into **datacards** and runs limits with
[Combine](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/), via the `StatInference` and
`inference` submodules. See
[FLAF → walkthrough, stage 5](https://cms-flaf.github.io/FLAF/workflow/walkthrough/#stage-5-statistical-inference)
for where this sits in the pipeline.

These commands run inside CMSSW/Combine, so prefix them with `cmsEnv` (or open one subshell):

```sh
cmsEnv /bin/zsh        # a CMSSW+Combine subshell
```

## 1. Make datacards and run limits

One law chain takes the merged histograms all the way to the overlay limit plots. The
datacard step itself consumes 1D shapes and does no rebinning of its own; the shapes this
analysis feeds it are cut from the 2D DNN-vs-HME plane by a configurable preprocessing
step the chain runs first — see [Where the binning is decided](#where-the-binning-is-decided):

```sh
law run PlotResonantLimitsTask \
  --version dev \
  --hists-version VERSION_OF_THE_MERGED_HISTS \
  --era Run3_2022,Run3_2022EE,Run3_2023,Run3_2023BPix \
  --workflow local
```

It runs, in order:

| Task | Does |
| --- | --- |
| `PreprocessShapesTask` | runs the configuration's `preprocess:` step, if it declares one |
| `CreateDatacardsTask` | builds the datacards from the resulting shapes |
| `ResonantLimitsTask` | runs combine and combines the per-era cards per mass point |
| `PlotResonantLimitsTask` | draws the plots declared in the configuration's `limit_plots` |

Each `limit_plots` entry becomes one overlay of its datacard globs. An entry that also
sets `bands: true` gets, in addition, the standard single-curve plot with the ±1σ/±2σ
bands for each of its curves — the overlay draws expected lines only. Note the dhi task
of almost the same name below (`PlotResonantLimits`, no `Task`): that is the one this
task shells out to for the band plots.

`--version` names what the chain *writes*; `--hists-version` names the `Hists_merged`
tree it *reads*, so a re-binning or a re-fit does not require the input histograms to be
reproduced under a new name.

The analysis configuration is [`config/Datacards/x_hh_bbww_DL_run3.yaml`](https://github.com/cms-flaf/HH_bbWW/blob/main/config/Datacards/x_hh_bbww_DL_run3.yaml),
selected by `StatInference.config` in `config/global.yaml`. It declares the eras,
channels, categories, mass points, processes and uncertainties — the chain reads them
from there, not from `global.yaml`'s own variable lists.

!!! warning "`--era` must hold every era behind the datacards"
    Which eras get datacards and limits comes from the configuration's `eras:` and
    `era_groups:` blocks, not from `--era`. A real era listed inside an `era_groups:` entry
    is covered by that meta-era and is not built standalone, so with the Run 3
    configuration the chain builds the `Run3_Early` combination of all four eras. `--era`
    must name every real era those cards read — here the four of `Run3_Early` — or the
    run stops with `the cards of … need Run3_Early: …, which --era does not include`.
    Further eras are allowed and are not carded. `CreateDatacardsTask` and
    `PreprocessShapesTask` are one workflow each, with one branch per card, and their
    outputs (`Datacards/<card>`, `Hists_preprocessed/<card>`) do not depend on `--era`
    ([FLAF → Eras](https://cms-flaf.github.io/FLAF/concepts/eras/#which-tasks-take-several-eras)).

### Where the shape uncertainties come from

A `type: shape` entry in the configuration's `uncertainties:` list is only a *declaration*
— it names a histogram the merger must already have written. What actually produces that
histogram is the corresponding entry in `config/Run3_<era>/weights.yaml`, whose `name:`
field is the datacard nuisance name plus a `_{}` placeholder for `Up`/`Down`. Adding a
nuisance to the datacard configuration without adding it there fails with
`Cannot find histogram ...`; adding it to `weights.yaml` without an `expression:` is worse,
because the variation is then produced but is byte-identical to the nominal shape, giving a
nuisance that constrains nothing and looks fine.

A `shape:` entry costs more than a `norm:` one, and in a way that is easy to get wrong. A
`norm:` entry is a weight expression evaluated on the central tree, so registering one and
re-running the histTuple step is enough. A `shape:` entry names a *shifted tree*, and
`HistTupleProducer` opens that tree in the anaTuple **and in every AnalysisCache it reads**
(DNN, DeepHME). A source can therefore be present in the anaTuples and still fail, because
the caches were produced before it was registered — the symptom is
`ERROR: tree Events__<source>__Up not found in file ...` pointing at a `luigi-tmp` cache
file, not at the anaTuple. Checking the anaTuple alone is not sufficient. Adding a new
`shape:` source means regenerating the caches, so it belongs with an anaTuple production
rather than a weights-only re-run.

An event that only a shift moves into the selection (typically JES or JER) enters that shift's
template with its real weight. The generator-level inputs of `weight_base` are listed in
`anaTuple_shift_invariant_columns` (see [Analysis](analysis.md#columns-taken-from-the-central-tree)),
so the central tree carries them for that event too and the merge computes its `weight_base`. In
anaTuples produced before that list existed such events had weight 0, so they were missing from the
shifted templates.

#### Correlating and decorrelating across eras

There is no `correlated:` switch. Combine correlates two entries when they carry the
**same nuisance name** and treats them as independent when the names differ, so the whole
correlation model is expressed by naming — and, because the nuisance name *is* the
histogram name, it is `weights.yaml` that decides it, not the datacard configuration.

A source decorrelated per era carries the era in its producer name, so the four eras write
four different histograms:

```yaml
# config/Run3_2022/weights.yaml          config/Run3_2023/weights.yaml
name: CMS_scale_j_2022_{}                name: CMS_scale_j_2023_{}
```

and the datacard configuration then declares one entry per era, each scoped with `eras:`:

```yaml
- name: CMS_scale_j_2022
  type: shape
  eras: [ Run3_2022 ]
```

A correlated source keeps one name in every era's `weights.yaml` (`CMS_btag_LF_{}`) and
one entry in the datacard configuration **with no `eras:` field at all** — an empty era
list matches everything, which is what makes it apply to all four.

Two consequences worth knowing:

- Renaming a source to decorrelate it changes a histogram name, so it costs a HistMerger
  re-run. It does not need a new anaTuple production: the shifted trees are unchanged.
- The two files must change together. Splitting `weights.yaml` without splitting the
  datacard configuration leaves the card asking for a histogram nobody writes any more
  (`Cannot find histogram ... CMS_scale_j_Up`), and it fails at datacard time rather than
  at merge time. Cross-check every card's `type: shape` entries against the producers of
  each era it declares before merging a correlation change.

For a meta-era such as `Run3_Early`, a nuisance scoped to one sub-era still applies: it
varies that sub-era's contribution to the summed shape while the others contribute their
nominal. The same holds for an `lnN` whose value differs per sub-era — `lumi_1_13p6TeV` is
written into the card as a shape for exactly that reason, since one log-normal on the
summed yield cannot express a factor that changes between eras. An `lnN` that does not
depend on the era stays an `lnN`.

#### b-tagging shape calibration

The BTV shape calibration contributes eight nuisances — `CMS_btag_LF`, `CMS_btag_HF`,
`CMS_btag_{lf,hf}stats{1,2}` and `CMS_btag_cferr{1,2}` — registered under `norm:` in each
era's `weights.yaml`. Two things about them are deliberate:

*Boosted events are excluded at weight level.* `GetWeight` in `Analysis/hh_bbww.py` folds
`weight_bTagShape_Central` into the resolved branch only of its `boosted ? ... : ...`
expression; boosted events carry `weight_FatJetSF_Central` instead. The variation
expressions therefore read `(boosted ? 1.0f : weight_bTagShape_<src>{scale}_rel) *
final_weight`, so a boosted event's varied weight equals its nominal weight and
`DatacardMaker`'s `canIgnore` threshold drops the nuisance from boosted categories on its
own. Do not reach for `unc_to_not_consider_boosted` for this — that mechanism is commented
out in `FLAF/Analysis/HistMergerFromHists.py`, so no code reads the key.

*The four `*stats*` sources are decorrelated per era, the other four are not.* That
follows the BTV prescription: `LF`, `HF` and the two `cferr` sources describe a common
calibration and stay correlated; the `{lf,hf}stats{1,2}` sources are statistical and get
one nuisance per era.

How the split is expressed is worth understanding, because it is not a datacard-only
change. The nuisance name is also the histogram name — `DatacardMaker` reads
`<process>_<name>_<Up|Down>` — so **an era-specific nuisance has to be named by the
producer**. Each era's `weights.yaml` writes `name: CMS_btag_lfstats1_2022_{}` and the
datacard declares one entry per era scoped with `eras:`. A source that stays correlated
keeps a single unsuffixed name in all four files. There is no separate "decorrelate" switch:
whether a source is split is visible from what the merger writes.

Two consequences to keep in mind. Renaming here is a **merge-stage** change only —
`HistTupleProducer` keys off the `weights.yaml` *keys* (`bTagShape_lfstats1`, `JER`), and
`name:` is read in exactly one place, `FLAF/Analysis/HistMergerFromHists.py`, so
re-producing the histograms does not mean re-producing the shifted trees. And the two
halves cannot drift silently: suffix the producer without the datacard, or the reverse,
and the build stops with `Cannot find histogram ...`.

On the `dc_make` side this needs two things, both of which treat a real era as a
one-element meta-era so configurations without `era_groups:` are untouched.
`DatacardMaker.uncAppliesTo` registers the nuisance on the meta-era bin when any
sub-era matches — `Uncertainty.appliesTo` compares against the meta-era name and would
otherwise drop the entry silently, which is why the `CMS_pileup_<era>` block in the
datacard configuration used to be commented out. `getCombinedShape` then varies only the
matching sub-eras and takes the rest at nominal, which is what the lnN path has always
done in `_getSubEraLnNVariedShapes`.

lnN uncertainties needed neither change and can be split with no producer involvement at
all: `lumi_13p6TeV` is two entries, `eras: [Run3_2022, Run3_2022EE]` and
`eras: [Run3_2023, Run3_2023BPix]`, because the luminosity calibration is a per-year
measurement rather than a per-era one.

Note also that `config/Run3_2024/weights.yaml` has **no** btag entries on purpose:
`config/Run3_2024/global.yaml` overrides btag to `HistTuple: none`, so the
`weight_bTagShape_*_rel` columns do not exist for that era.

#### AK8 (fatbjet) calibration

`Corrections/fatjet.py` supplies the mirror image for boosted events: three sources
`Hbb`, `Hcc` and `tau21` (`FatJetCorrProducer.fatjet_Sources`), registered as
`CMS_bbww_ak8_{Hbb,Hcc,tau21}`. Because `GetWeight` puts `weight_FatJetSF_Central` in the
*boosted* branch, the guard runs the other way — `(boosted ?
weight_FatJetSF_<src>{scale}_rel : 1.0f) * final_weight` — so it is the resolved events
that are neutralised, and `canIgnore` drops these nuisances from resolved categories.

Two differences from the btag block are worth knowing when reading the resulting
nuisances rather than fixing them:

- There is no renormalisation step. btagShape has `UpdateBtagWeight` restoring the
  per-(channel, nJet) yield; the AK8 SFs have no equivalent and need none, so these
  nuisances legitimately carry a normalisation component.
- `Hbb` applies only to `hadronFlavour == 5` and `Hcc` only to `== 4`
  (`FatJetCorrProvider::sourceApplies`, `Corrections/fatjet.h`). `Hcc` is therefore tiny
  for most processes and will fall under the `canIgnore` threshold in many categories.

The calibration files are per-era and cover the four 2022/2023 eras only, so as with btag
there is nothing to register for `Run3_2024`. They are decorrelated per era for the same
reason the btag `*stats*` sources are: four separate files means four independent fits.

### Where the binning is decided

By a **preprocessing step the datacard configuration declares**, not by anything in the
chain's own logic. `PreprocessShapesTask` runs whatever `preprocess:` names, supplying
`--input`, `--output`, `--era` and `--config`; it knows nothing about what the step does.
A configuration with no `preprocess:` block skips the task entirely and the datacards are
built from the merged histograms unchanged, so an analysis that needs no preprocessing is
unaffected.

This analysis plugs in `StatInference/bin_opt_2d/rebin_2d.py`, which derives the binning
from the shapes themselves. The DL card uses its `window` strategy (`window_axis: y`): for
each mass, channel and base category, one HME window around the resonance with the DNN
score binned inside it; events outside the window are dropped. Each base category
`SR/res2b` becomes the single datacard bin `SR/res2b_hmebox0`.

```yaml
preprocess:
  script: StatInference/bin_opt_2d/rebin_2d.py
  args:
    - --binning-config
    - config/Datacards/binning_hmebox.yaml
```

The alternative, `config/Datacards/binning_2d.yaml`, cuts each base category into DNN
slices (`SR/res2b_dnn0…dnn3`) with HME bins inside each; the window gives a ~5% better
expected limit with about a third fewer fit bins.

The two halves live in different places on purpose. The **knobs** — strategy, slice count,
bin budget, and the minimum signal/background yields and effective-entry floors a bin must
satisfy — are analysis configuration, versioned with the card in
[`config/Datacards/binning_hmebox.yaml`](https://github.com/cms-flaf/HH_bbWW/blob/main/config/Datacards/binning_hmebox.yaml).
The **derived** `binning.json` is a product, not configuration, and is written into the
task's output on EOS beside the shapes it produced.

Every era in `eras:` is binned on its own statistics and gets its own limit; a group era
from `era_groups:` is binned on its members' summed statistics, and its output keeps the
sub-eras separate (`Run3_Early/{Run3_2022,Run3_2022EE,…}`) so the per-era uncertainties
survive to be combined in the datacard maker.

`rebin_2d.py` is a plain executable, so a binning can be inspected without going through
the chain by calling it with the same four arguments the task passes:

```sh
python3 StatInference/bin_opt_2d/rebin_2d.py \
  --input  "$ANALYSIS_BIG_DATA_PATH/VERSION_OF_THE_MERGED_HISTS/Hists_merged" \
  --output /tmp/$USER/rebin_Run3_Early \
  --era    Run3_Early \
  --config config/Datacards/x_hh_bbww_DL_run3.yaml \
  --binning-config config/Datacards/binning_hmebox.yaml
```

The datacard configuration then lists the written names in `categories:`
(`SR/res2b_hmebox0`, …) and repeats the `category_pattern` used to write them, which is
how the per-category limits group the bins of one base category back together.

There is also `StatInference/bin_opt/`, an offline combine-driven search over candidate
binnings feeding the `hist_bins` option. This analysis does not use it, and leaves
`hist_bins` unset.

### Datacards on their own

To build cards outside the chain — a quick check on shapes that are already rebinned:

```sh
cmsEnv python3 StatInference/dc_make/create_datacards.py \
  --input  PATH_TO_REBINNED_SHAPES \
  --output PATH_TO_CARDS \
  --config config/Datacards/x_hh_bbww_DL_run3.yaml
```

## 2. Run limits on existing datacards

For cards you already have on disk, the dhi task can be called directly:

```sh
law run PlotResonantLimits --version dev --datacards 'PATH_TO_CARDS/*.txt' --xsec fb --y-log
```

Hints:

- add `--workflow htcondor` to submit to the batch system (local by default);
- add `--remove-output 4,a,y` to clear previous outputs;
- add `--print-status 0` to get the workflow status and the output file name;
- options and background: the [cms-hh inference documentation](https://cms-hh.web.cern.ch/tools/inference/).

## 3. Pulls & impacts

Declared in the datacard configuration's `impact_plots` block and drawn by
`PlotPullsAndImpactsTask`, which writes to `<version>/ImpactPlots/<mass>/` on `fs_default`
beside the limit plots:

```sh
law run PlotPullsAndImpactsTask \
  --version dev \
  --hists-version VERSION_OF_THE_MERGED_HISTS \
  --era Run3_2022,Run3_2022EE,Run3_2023,Run3_2023BPix
```

```yaml
impact_plots:
  - name: combined
    masses: [ 300, 600, 800 ]
    poi_value: limit
    plot_params:
      method: default
      order_by_impact: true
      mc_stats: false
```

`poi_value` is the signal strength the Asimov dataset is built at, and it has to be near
the limit to mean anything. dhi's default is `r=1`, which on this scan is an almost
invisible signal at 300 (limit ~6.8) and ten times more than can be excluded at 800
(limit ~0.065) — the first gives a background-only ranking, the second pins `r` and
collapses every impact to ~0. `limit` reads the expected limit for each mass from the
limits already merged for those cards; a number sets it directly.

!!! danger "Changing `poi_value` does not move the outputs"
    It reaches combine as `--expectSignal` through `PullsAndImpacts`' `custom_args`, which
    is significant for task identity but **not** part of the store path — every value
    writes to the same file. Clear `PullsAndImpacts`, `MergePullsAndImpacts` and
    `PlotPullsAndImpacts` for that version under `inference/data/store` before re-running
    with a different one, or law reports the old fit as complete and you get the previous
    ranking. It cannot go through dhi's `parameter_values` either: with `hh_model=NO_STR`,
    `POITask` hard-codes both the joined values and the postfix, dropping it silently.

Each entry becomes one dhi `PlotPullsAndImpacts` per era and mass, and the plots land in
`<version>/ImpactPlots/<name>/<era>/<mass>/`.

With no `glob:`, an entry uses the combined card `ResonantLimitsTask` writes at
`data/<version>/Datacards/combined/combined_<mass>.txt` — every era and category in one
fit, which is the ranking that describes the result. It is drawn once and labelled
`combined`.

To rank a single category instead, give the entry a `glob:`, resolved against that era's
datacard directory with `${ERA}` substituted, the same idiom `limit_plots` uses:

```yaml
  - name: res2b
    glob: "${ERA}/categories/SR_res2b/*.txt"
    masses: [ 300, 600 ]
```

The glob and the mass must select **exactly one** card: `PullsAndImpacts` fits a single
workspace, one combine job per parameter, so it cannot be handed a set. If it matches none
or several, the error names the directory and lists what is in it. Note dhi draws one plot
per card — it cannot split a card into per-category panels — so per-category rankings come
from pointing at the per-category cards, which the chain already writes.

`plot_params` accepts any `PlotPullsAndImpacts` parameter and is checked against them, so
a typo is refused rather than ignored. `hh_model` is pinned to `NO_STR`: this is a
resonant search, and the dhi default would otherwise fit `r` alongside `kl`, `kt`, `CV`
and `C2V`, against a workspace other than the one the chain built.

!!! warning "What a mass costs depends on `method`"
    With `method: robust` — what this analysis uses — dhi runs **one** fit per mass and
    reads every impact off the inverted Hessian, so all three masses take minutes.
    `method: default` instead fits each nuisance separately, roughly two combine jobs per
    nuisance per mass (~160 here), and only that case is worth sending to a batch system.
    `create_branch_map` in `dhi/tasks/pulls_impacts.py` is where the difference lives.
    Pass `--redraw` to redraw without refitting.

!!! danger "`method: robust` can silently drop a nuisance"
    robustHesse removes parameters it cannot invert, logging `Dropping <name> from the
    hessian` and then exiting successfully. The dropped nuisance is simply **absent** from
    the plot and the merged JSON, with nothing marking its absence — on the Run3_Early
    cards this happens to `CMS_res_j`. The task diffs the fitted parameters against the
    card's own nuisance lines afterwards and warns, naming what went missing; take that
    warning seriously before reading a ranking as complete.

!!! warning "`mc_stats` needs `parameters_per_page`"
    The combined cards carry a few hundred `autoMCStats` bins, so `mc_stats: true` puts
    ~460 parameters on the plot. `parameters_per_page` defaults to `-1`, meaning a single
    page, and the result is an unreadable hairline strip rather than an error — so the
    task refuses the combination. Set `parameters_per_page: 25` alongside it.

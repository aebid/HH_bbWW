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

## 1. Rebin, make datacards and run limits

One law chain takes the merged histograms all the way to the overlay limit plots:

```sh
law run PlotResonantLimitsTask \
  --version dev \
  --hists-version VERSION_OF_THE_MERGED_HISTS \
  --period Run3_2022 \
  --workflow local
```

It runs, in order:

| Task | Does |
| --- | --- |
| `HistRebinTask` | rebins the 2D DNN×HME shapes into significance-sliced 1D categories |
| `CreateDatacardsTask` | builds the datacards from those shapes |
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

!!! warning "`--period` does not choose the era"
    Which eras get datacards and limits comes from the configuration's `eras:` and
    `era_groups:` blocks. A real era listed inside an `era_groups:` entry is covered by
    that meta-era and is not built standalone, so with the Run 3 configuration the chain
    always builds the `Run3_Early` combination of all four eras regardless of `--period`.
    `--period` is only used to construct a valid FLAF `Setup`.

### Where the shape uncertainties come from

A `type: shape` entry in the configuration's `uncertainties:` list is only a *declaration*
— it names a histogram the merger must already have written. What actually produces that
histogram is the corresponding entry in `config/Run3_<era>/weights.yaml`, whose `name:`
field is the datacard nuisance name plus a `_{}` placeholder for `Up`/`Down`. Adding a
nuisance to the datacard configuration without adding it there fails with
`Cannot find histogram ...`; adding it to `weights.yaml` without an `expression:` is worse,
because the variation is then produced but is byte-identical to the nominal shape, giving a
nuisance that constrains nothing and looks fine.

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
out in `FLAF/Analysis/HistMergerFromHists.py` and is live only in the offline
`ShapeOrLogNormal.py`.

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
`DatacardMaker.uncAppliesInEra` registers the nuisance on the meta-era bin when any
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
there is nothing to register for `Run3_2024`. Unlike the btag `*stats*` sources these are
still correlated across eras even though the calibration is derived per era — they are the
second stage of the decorrelation, to be decided on impacts once the sources with an
explicit POG prescription have been split.

### Where the binning is decided

`HistRebinTask` runs `StatInference/dc_make/hist_rebin_2d.py`, which derives the DNN slice
boundaries and the HME mass-bin edges from the shapes themselves. Everything it does is
controlled by the annotated `binning:` block of the configuration above — slice count,
mass-bin budget, and the minimum signal/background yields and effective-entry floors a
bin must satisfy. Each base category `SR/res2b` becomes the datacard bins
`SR/res2b_dnn0…dnn3`.

This is separate from `StatInference/bin_opt/`, which is an offline, combine-driven search
over candidate binnings feeding the `hist_bins` option. This analysis does not use it, and
leaves `hist_bins` unset.

### Running on the 1D DNN shapes instead

The `binning:` block is also the switch that decides which kind of input the chain reads.
[`config/Datacards/x_hh_bbww_DL_run3_1D.yaml`](https://github.com/cms-flaf/HH_bbWW/blob/main/config/Datacards/x_hh_bbww_DL_run3_1D.yaml)
is the same analysis reading the per-mass 1D DNN score variables
(`Hists_merged/<era>/DNN_M<mass>_Signal/`). It declares no `binning:` block, so
`HistRebinTask` drops out of the graph entirely and `CreateDatacardsTask` reads the merged
histograms directly; the datacard bins are then the categories exactly as listed
(`SR/res2b`, not `SR/res2b_dnn0`), coarsened by its own `hist_bins:` edge list.

```sh
law run PlotResonantLimitsTask \
  --version limits_1D \
  --hists-version VERSION_OF_THE_MERGED_HISTS \
  --period Run3_2022 \
  --user-custom config/user_custom_1D.yaml
```

!!! warning "Give the 1D run its own `--version`"
    The datacard and limit output paths do not encode which configuration produced them,
    so reusing a 2D run's version overwrites its cards.

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

`ResonantLimitsTask` already writes the input this needs: one card per mass combining all
eras, at `data/<version>/Datacards/combined/combined_<mass>.txt`. The shape paths in those
cards are absolute, which is what makes them usable here at all — `PullsAndImpacts` runs
combine in a temporary directory.

```sh
law run PlotPullsAndImpacts --version dev \
  --datacards data/<version>/Datacards/combined/combined_500.txt \
  --hh-model NO_STR --parameter-values r=1 --parameter-ranges r,-100,100 \
  --method robust --PlotPullsAndImpacts-order-by-impact True \
  --PullsAndImpacts-custom-args="--expectSignal=1"
```

That gives one readable page of the real nuisances. Without `--unblinded` the fit is to
Asimov, so every pull sits at zero by construction and only the impacts carry information.

!!! warning "One mass point at a time"
    Run pulls & impacts on a **single** datacard, not a glob. Use `--print-status 0` to find the
    output file and `--remove-output 4,a,y` to clear previous outputs.

!!! warning "`--mc-stats True` needs `--parameters-per-page`"
    The combined cards carry a few hundred `autoMCStats` bins, so `--mc-stats True` puts
    ~460 parameters on the plot. `parameters_per_page` defaults to `-1`, meaning one page,
    and the result is an unreadable hairline strip. Add
    `--PlotPullsAndImpacts-parameters-per-page 25` to get a paginated PDF instead. Since
    that option is `significant=False` it does not change the output filename, so a replot
    also needs `--remove-output 0,a,y`.

!!! danger "`--method robust` can silently drop a nuisance"
    robustHesse removes parameters it cannot invert, logging `Dropping <name> from the
    hessian` and then continuing successfully. The dropped nuisance is simply **absent**
    from the plot and the merged JSON — nothing in the plot marks its absence. On the
    Run3_Early cards this happens to `CMS_res_j`. Always diff the parameter names in the
    merged JSON against the card's own `^\S+\s+(shape|lnN)` lines before reading a ranking
    as complete.

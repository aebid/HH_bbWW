# ShapeYields

What is actually in the bins combine is fitting.

`shape_yields.py` reads the rebinned shapes -- the output of the datacard configuration's
`preprocess:` step, which for HH->bbWW is `StatInference/bin_opt_2d/rebin_2d.py`, and which
is the same thing `CreateDatacardsTask` reads as its input -- and reports every bin's
content and MC-statistical error, per process, per channel, per era.

Nominal shapes only. The Up/Down variations are in the same files and are left alone.

## Quick start

Run from the repository root. All three work as written.

**Everything, for one production.** Reads the shapes off EOS -- a few minutes, that read is
what takes the time -- and writes the CSV plus one PDF per era:

```
python3 Studies/ShapeYields/shape_yields.py \
    --input /eos/user/d/daebi/HH_bbWW/uncv2/Hists_preprocessed/Run3_Early \
    --config config/Datacards/x_hh_bbww_DL_run3.yaml \
    --binning-config config/Datacards/binning_2d.yaml \
    --output Studies/ShapeYields/output/uncv2
```

For another production, swap `uncv2` in the `--input` and `--output` paths.

**One page for a slide.** Once a CSV exists, `--from-csv` redraws from it without touching
EOS, so this is instant:

```
python3 Studies/ShapeYields/shape_yields.py \
    --from-csv Studies/ShapeYields/output/uncv2/yields.csv \
    --config config/Datacards/x_hh_bbww_DL_run3.yaml \
    --binning-config config/Datacards/binning_2d.yaml \
    --era Run3_Early --mass 500 --channel muMu \
    --output /tmp/slide
```

That gives one page per base category; add `--category SR/res2b_dnn1` to narrow to one.
Point `--from-csv` at `/eos/user/d/daebi/HH_bbWW/uncv2/ShapeYields/yields.csv` and you can
skip the first command entirely.

**Confirm the numbers.** Compares the CSV against the datacards, exits non-zero on any
disagreement:

```
python3 Studies/ShapeYields/verify_against_datacards.py \
    --csv Studies/ShapeYields/output/uncv2/yields.csv \
    --datacards data/uncv2/Datacards/Run3_Early \
    --config config/Datacards/x_hh_bbww_DL_run3.yaml \
    --era Run3_Early
```

## The options

`--input` is the era-group directory, holding one sub-directory per source era. The four
sub-eras are read individually and their sum is reported alongside them as `Run3_Early`,
which is the thing the datacards are built from.

`--era`, `--mass`, `--channel` and `--category` are repeatable and each narrows the run.
`--category` takes the sliced name, e.g. `SR/res2b_dnn1`.

`--from-csv` redraws from an existing `yields.csv` instead of reading the shapes, with the
selectors still applying. Use it for anything to do with how the page looks, or to cut a
subset: reading the files is the slow part, and the numbers do not change once written.

`--no-pdf` writes only the CSV.

`--binning-config` supplies `category_pattern` and `slice_var`; without it both fall back to
the datacard configuration's own.

`--era-group` names the group whose members are read and whose sum is reported. It defaults
to the single entry of the configuration's `eras:`, so it is only needed if that list ever
holds more than one.

## What comes out

`yields.csv` -- one row per (era, mass, channel, category, bin, process), carrying the
bin's HME edges, its content and its error. Tidy, so it sorts and greps and reads into
pandas without reshaping.

`yields_<era>.pdf` -- the same numbers as tables, one page per
(era, mass, channel, base category) with the four DNN slices stacked down the page. Each
page is a slide. Cell shade is log magnitude within the slice; negative content is boxed
and printed in red.

## The shape book

`shape_book.py` draws the same shapes as plots, one page per mass: the channels as rows and
the categories as columns. That is everything the fit sees at that mass, on one page.
Each panel has the stacked backgrounds with their MC-stat band and the summed signal (log
scale), and is labelled with the selection its category stands for, read from the
`binning.json` beside the shapes: "670 < HME < 1170" for an HME window, the DNN range for a
DNN slice.

```
python3 Studies/ShapeYields/shape_book.py \
    --input /eos/user/d/daebi/HH_bbWW/<production>/Hists_preprocessed/Run3_Early \
    --config config/Datacards/x_hh_bbww_DL_run3.yaml \
    --output Studies/ShapeYields/output/<production>
```

It writes `shapes_<era-group>.pdf`, for the era group only, since that sum is what the
fit sees. `--from-csv yields.csv --binning-json <path>` redraws from a CSV without reading
the shapes, and `--mass` and `--channel` narrow the run. `--x-name` and `--y-name` name the
2D input's axes (defaults "DNN score" and "HME [GeV]"); each panel is labelled with
whichever one it is binned along.

## The uncv2 run

Already made, kept beside the shapes it was read from:

```
/eos/user/d/daebi/HH_bbWW/uncv2/ShapeYields/
```

`yields.csv` (54,995 rows) and one 87-page PDF per era -- the four sub-eras and their sum.

## Checking it

`verify_against_datacards.py` compares the CSV against the datacard shapes
`CreateDatacardsTask` wrote, in both directions: every nominal histogram in the datacard
must be reproducible from the CSV, and every background in the CSV must be consumed by the
datacard. It exits non-zero on any unmatched histogram or any bin difference above
tolerance.

On uncv2: 1684 histograms over all ten masses, worst relative bin difference 3.9e-16,
nothing unmatched in either direction.

Both directions matter. Checking only "is every datacard histogram reproducible" lets the
merged `TotalBkg` categories be skipped as explained by the merge, and that is exactly
where a real defect hid -- see the note in `merged_scopes()`.

What the check does and does not establish: it establishes that this tool reads and sums
the shapes the way the datacard chain does. It cannot establish that the shapes are
physically right, since both sides read the same files -- an error upstream in the
rebinning would be reproduced identically by both.

## What it does not do

It applies no thresholds and passes no judgement on which bins are acceptable. It does not
consult the binning that produced the shapes, and it re-derives nothing from
`rebin_2d.py` -- the numbers in the files are the report, and the reading is meant to be
independent of the code that wrote them.

Everything but the file layout comes from the datacard configuration: the processes and
their channel/category restrictions, the masses, the channels, the categories, and the
`input_file_pattern` the paths are built from.

`data_obs` is deliberately absent: the observation is Asimov (`is_asimov_data`), so it
carries nothing the summed background does not, and the `Total bkg` row is that sum.

The merged `TotalBkg` template is assembled by the datacard maker and is not a histogram in
these files. Its four constituents are, and they are reported individually, so the boosted
pages show what goes into the merge rather than the merge itself. Note the rule that
implies, since it is easy to get wrong: the maker collects those constituents by path and
ignores a constituent's own `channels` list, so DY -- restricted to `[eE, muMu]` by its own
entry -- is nevertheless part of the eMu boosted background, and is reported there. It
stays out of eMu res2b and recovery, which the merge does not claim.

Signal rows are the raw histograms. Whether those are also the datacard rates depends on
whether the configuration declares a process `scale`, which the maker applies when it reads
the histogram; each page's footer states which case it is. The committed configuration
declares none, and there signal matches the datacard rate exactly.

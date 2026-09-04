# msoftdrop cut closure

Compares the efficiency of the boosted `msoftdrop > 30` requirement between
data and simulation, per era, lepton channel and process.

```bash
python3 Studies/msoftdrop_closure/msoftdrop_closure.py \
    --input-dir /eos/user/d/daebi/HH_bbWW/v2605a_msoftdrop/Hists_merged \
    --per-era --csv out.csv
```

Reads the merged `fatbjet_msoftdrop` histograms
(`<hists>/<era>/fatbjet_msoftdrop/fatbjet_msoftdrop.root`, layout
`<channel>/<region>/<category>/<process>`) and prints, for each channel:

* an efficiency matrix, process x era, with a `Total MC` row and a `Data/MC` row;
* a yield table with `nTotalEvents`, the failing and passing yields, the
  efficiency with its binomial error, and the `0 < msd < 30` count.

Signal is excluded by default; backgrounds are `TT, DY, ST, VV, W, SingleHiggs`
and data is `Data_Full`.

Results for `v2605a_msoftdrop` are checked in as
`msoftdrop_closure_v2605a.txt` / `.csv`.

## What the current input can and cannot measure

`baseline_boosted` (`SelectedFatJet_pt.size() >= 1`) carries no msoftdrop cut,
but the *variable* does. In `Analysis/hh_bbww.py`:

```python
df = df.Define("FatBJet_Sel",
    "SelectedFatJet_particleNetWithMass_HbbvsQCD > 0.92 && SelectedFatJet_msoftdrop > 30")
...
df = df.Define("fatbjet_isValid", "(Nfatbjets > 0)")
df = df.Define(f"fatbjet_{var}",
    f"fatbjet_isValid ? FatBJet_{var}[0] : std::decay_t<decltype(FatBJet_{var})>::value_type()")
```

So `fatbjet_msoftdrop` is only filled from a jet that already passed
`msoftdrop > 30`; everything else gets the default 0. In the histograms this
shows up as a delta at 0 with a hard, exactly-empty gap above it:

* the underflow bin is 0 in every era / channel / process;
* bins covering 4-28 GeV are *exactly* 0.00000 (bin error 0 too) everywhere;
* in the dilepton channels `total(baseline_boosted) - bin[0,4)` equals
  `total(boosted)` to ~1e-13, i.e. the first bin is precisely the set of events
  with `fatbjet_isValid == false`;
* `boosted` itself has nothing in bin `[0,4)` in the dilepton channels.

The `msd<30` column is therefore the combined *no valid FatBJet* rate: it mixes
`HbbvsQCD <= 0.92` failures, `msoftdrop <= 30` failures, and (in the
single-lepton channels) events that are boosted only through the W-jet leg.
The `0<msd<30` column is identically zero for the same reason.

To isolate msoftdrop alone, re-produce with the msoftdrop term dropped from
`FatBJet_Sel` (keeping the tagger cut), e.g.

```python
df = df.Define("FatBJet_Sel",
    "SelectedFatJet_particleNetWithMass_HbbvsQCD > 0.92")
```

Then `0 < msd < 30` becomes the real numerator and this script reports the
msoftdrop efficiency without any change.

## Binning caveat

The histogram uses 4 GeV bins on [0, 200], so the threshold falls inside
`[28, 32)`. That bin is counted as passing and reported separately; the script
warns when the threshold is interior to a bin. A production intended for this
measurement should place a bin edge at 30.

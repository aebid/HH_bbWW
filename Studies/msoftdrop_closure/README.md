# msoftdrop cut closure

Compares the efficiency of the boosted `msoftdrop > 30` requirement between
data and simulation, per era, lepton channel and process.

## Why `fatbjet_msoftdrop` cannot be used

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
`msoftdrop > 30`; everything else gets the default 0. Loosening the *category*
does not reach the *variable*. In the `v2605a_msoftdrop` histograms this shows
up as a delta at 0 with a hard, exactly-empty gap above it:

* the underflow bin is 0 in every era / channel / process;
* bins covering 4-28 GeV are *exactly* 0.00000, bin error 0 too, everywhere;
* in the dilepton channels `total(baseline_boosted) - bin[0,4)` equals
  `total(boosted)` to ~1e-13, i.e. the first bin is precisely the set of events
  with `fatbjet_isValid == false`;
* `boosted` itself has nothing in bin `[0,4)` in the dilepton channels.

That first bin is therefore a sentinel mixing three different failures:
`HbbvsQCD <= 0.92`, `msoftdrop <= 30`, and (in the single-lepton channels)
events boosted only through the W-jet leg. Reading it as "fails msoftdrop"
gives a wrong answer.

## The probe variables

`Analysis/hh_bbww.py` now also defines two variables that keep the same jet
ordering but drop the mass requirement, with `-1` (underflow) as the "no
candidate" sentinel so it cannot collide with the physical range:

| Variable | Denominator | Use |
|---|---|---|
| `fatbjetProbe_msoftdrop` | fat jets with `HbbvsQCD > 0.92` | efficiency in the selection sequence actually used |
| `leadfatjet_msoftdrop` | leading-Hbb-score fat jet, no cuts | avoids the mass-aware tagger correlating with the denominator |

Both are additive: no existing variable, category or selection changes
behaviour. They are registered in `config/global.yaml` and
`config/plot/histograms.yaml` with 5 GeV bins on `[0, 200]` so that a bin edge
falls exactly on 30.

`test_probe_definitions.py` is a standalone RDataFrame check of the two C++
expressions, including the empty-collection case:

```bash
python3 Studies/msoftdrop_closure/test_probe_definitions.py
```

## Running

Once a production defines the probes:

```bash
python3 Studies/msoftdrop_closure/msoftdrop_closure.py \
    --input-dir <hists>/Hists_merged \
    --variable fatbjetProbe_msoftdrop --sentinel-value -1 \
    --per-era --csv out.csv
```

Reads `<hists>/<era>/<variable>/<variable>.root` with layout
`<channel>/<region>/<category>/<process>` and prints, per channel:

* an efficiency matrix, process x era, with `Total MC` and `Data/MC` rows;
* a yield table splitting `nTotalEvents` into `no candidate` + `candidates`,
  and `candidates` into `msd<30` + `msd>30`, with the cut efficiency
  `msd>30 / candidates` and its binomial error.

Signal is excluded by default; backgrounds are `TT, DY, ST, VV, W, SingleHiggs`
and data is `Data_Full`.

The script warns if the threshold falls inside a bin rather than on an edge,
and prints an explicit note if nothing populates below the cut — the signature
of running it on `fatbjet_msoftdrop` by mistake.

## Checked-in results

`msoftdrop_closure_v2605a.txt` / `.csv` are the `v2605a_msoftdrop` numbers.
Every cut efficiency there is a trivial 100% for the reason above; the tables
are kept because the `no candidate` / `candidates` split still records the
combined "has a valid FatBJet" rate (tagger AND msoftdrop), which ran 0.83
(`e`) to 1.23 (`muMu`) in Data/MC. That combined number is not a msoftdrop
closure test and should not be quoted as one.

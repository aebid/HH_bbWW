# v2610 parametric DNN: studies for the analysis-note appendix

Record of the studies behind the dilepton parametric DNN (pDNN) trained on the v2610 production:
what was tested, how, the result, and the decision taken. Figures are collected in
`/eos/user/d/daebi/HH_bbWW/DNNTraining/pytorch/appendix/` (file names quoted below); every trained
model is in `/eos/user/d/daebi/HH_bbWW/DNNTraining/pytorch/<model>/` with its code snapshot,
logs, limits and checks. Code: `Studies/DNN_pytorch`, `Studies/DNN_Dataset` (branch
`pdnn-v2610-training`).

## 1. Training dataset

- **Inputs.** v2610 HistTuples (`histTuple_flavor: DNN_training`) of the four 2022–2023 eras,
  read from two sites: Run3_2022/2022EE on LPC (`/store/user/lpcflaf/HH_bbWW/v2610_training`),
  Run3_2023/2023BPix on CERN EOS (`/eos/user/a/abolshov/HH_bbWW/v2610_training`). Every
  HistTuple was checked against its anaTuple input (all present, readable, same 178 branches).
- **Selection.** `SR`, `res2b || recovery || boosted`, dilepton channels (ee, eμ, μμ).
- **Samples and classes** (`class_value`): Radion / BulkGraviton signals in 2L (0 / −3), 1L
  (−1 / −4) and bbττ (−2 / −5) at 30 masses, 250–2000 GeV; backgrounds TT 1, DY 2, ST 3,
  single H 4, VV 5, W 6, ttV 7, VVV 8 (all analysis background processes, one class each). 250
  datasets, every one with inputs in all four eras.
- **Folds.** `event % 4` defines four parity files, about 6.2 M events each, 180 branches.
- **Validation of the dataset maker.** The new `create_dataset.py` reproduces the July25 dataset
  used by the previous pDNN (same branches, weight tree and per-parity yields, identical event
  sets).
- Location: `/eos/user/d/daebi/HH_bbWW/DNNDatasets/v2610_training/Dataset`.

## 2. Input-variable study (before training)

- **Method.** On parity 0, each of the 128 numeric HistTuple variables: weighted single-variable
  separation (2·|AUC−0.5|) of signal (2L + bbττ, both spins, flat in mass) vs class-balanced
  background in three mass bands, background-pair separations (TT/DY/Other), correlation with
  DeepHME_mass, fraction of default values. Table: `01_input_variable_study.txt`.
- **Findings.** Low mass (250–400 GeV) has no strong single variable (best: b-tag scores ~0.4);
  the strongest variables at higher mass (`MX_reduced`, `Pzeta`, `bbllmet_mass`, `HT_total`,
  MT2) correlate 0.8–0.9 with DeepHME_mass; about 35 semileptonic variables are constant in
  dilepton events; several exact duplicates.
- **Decision.** First feature set = previous production (v6) inputs without DeepHME_mass, plus
  `MX_reduced`, `Pzeta`, `MT2_bb`, `MT2_ll` and `DeepHME_mass_error_rel`
  (= error / DeepHME_mass, uncorrelated with the mass). The fat-jet HbbvsQCD score is not used:
  only its 0.92 working point is calibrated.

## 3. Training setup (model v1)

- **Signal.** Radion + BulkGraviton, 2L + bbττ; 1L signal events are not trained on.
- **Outputs.** Signal / TT / DY / Other (softmax); the signal log-odds is the analysis score.
- **Parametric in mass.** Input `X_mass`; signal at its own mass, backgrounds drawn uniformly from
  the 30 masses each epoch; `ll_dR` and `bb_dR` enter as `*_scaled = dR · X_mass`.
- **Weights.** From `weight_Central`: negative weights set to 0, clipped at mean + 3σ; signal
  flattened to equal total weight per mass point (except masses with < 1000 signal events, i.e.
  boosted 250–400 GeV, which keep their own weight); each class scaled to the signal total, per
  regime.
- **Networks.** Separate resolved and boosted networks; MLP 512-256-128-64, SiLU, batch norm,
  dropout 0.35; AdamW (lr 2e-3, weight decay 0.01), batch 8192; LR halved after 6 epochs without
  improvement; early stopping after 10, at most 150 epochs.
- **Folds.** Cycle i trains on parity i, early-stops on parity i+1, is validated on parity i+2
  and applied to parity i+3. Every event is scored once by a model that never saw it.
- **Hardware.** lxplus-gpu (Tesla T4), ~10 s per resolved epoch; one model is 4 jobs of
  ~45 minutes. Loss curves: `02_loss_curve_*.png`. Typical behaviour: the resolved network
  improves slowly for 90–150 epochs with test loss at or below training loss; the boosted
  network stops at 19–40 epochs as its test loss turns up.

## 4. Limit metric

`limit_pdnn.py`: for each mass and spin hypothesis, templates of the signal log-odds in res2b,
recovery and boosted from the validation scores of all four cycles (full statistics, physical
weights), binned in equal-signal quantiles merged until each bin has ≥ 1 background event and
≥ 10 background MC events; stat-only expected limits on r (`combine -M AsymptoticLimits -t -1`,
`--rAbsAcc 1e-6 --rRelAcc 1e-3`), per category and combined. Signal hypothesis: 2L + bbττ of
one spin (spin 0 Radion, spin 2 Graviton) at that mass; backgrounds TT, DY, Other.
Result for v1: `02_limits_v1_combined.png`.

## 5. Overtraining

- **Method.** `overtraining_check.py`: each cycle's model on its training parity vs its
  validation parity, at 5 masses (and both spins for the spin model), weighted KS test per class
  and signal-vs-background AUC.
- **Result** (all four cycles):

  | Model | Regime | AUC(train) − AUC(val): mean | range |
  |---|---|---|---|
  | v1 | resolved | +0.0002 | −0.0008 … +0.0022 |
  | v1 | boosted | +0.0026 | −0.088 … +0.067 |
  | v1_spin | resolved | +0.0003 | −0.0009 … +0.0038 |
  | v1_spin | boosted | +0.0020 | −0.088 … +0.085 |

  KS p < 0.01 in 4 of 400 tests. No significant overtraining. The boosted spread comes from its
  small samples (~17k background events per parity), and early stopping limits the slight
  fitting of the training events. Figures: `03_overtraining_*.pdf`.

## 6. Spin: parametric vs inclusive

- **Test.** v1 (spin-inclusive) vs v1_spin (same model with the spin hypothesis as an input:
  signal carries its spin, backgrounds a random one).
- **Result.** Limit ratio v1_spin / v1, combined categories: spin 0 average 0.997 (1.03 below
  450 GeV, 1.00 at 500–1000, 0.96 above 1 TeV); spin 2 average 0.991. Point-to-point
  differences are within the training noise (section 7).
  Figures: `04_limits_spin_inclusive_vs_parametric_*.png`.
- **Decision.** The production model is **spin-inclusive**.

## 7. Training-to-training noise

- **Test.** v1 retrained with another random seed (v1_seed2).
- **Result.** Ratio of expected limits (combined): RMS deviation ~3% at 250–400 GeV, 2–3% at
  450–900 GeV, 4–6% above 1 TeV (largest single points 6%, 7%, 12%); res2b alone 1–3% at
  intermediate mass; boosted alone up to 7.5% RMS. Figure: `05_limits_seed_noise_combined.png`.
- **Use.** A model change is significant only beyond this band.

## 8. Feature importance (v1)

- **Method.** `feature_importance.py`: on each cycle's validation parity, at 10 masses, each input
  is shuffled (mass-scaled inputs recomputed), as is each group of inputs correlated with
  |r| > 0.8 in background. Measured: relative increase of the AUC deficit,
  (AUC − AUC_shuffled)/(1 − AUC), and of the class-balanced loss; mean and spread over cycles.
  Tables and figures: `06_importance_v1_{resolved,boosted}.{csv,png}`,
  `06_importance_vs_mass_*.png`.
- **Resolved.** The mass-like group (`MX_reduced`, `bbllmet_mass`, `MT2_blbl1/2`, `bb_pt`)
  dominates (AUC deficit ×62 when shuffled together, ×179 above 1 TeV), then `HT_total`+`Pzeta`;
  b-tag scores and `ll_met_MT_WithMass` are the most important at low mass. Negligible:
  `ll_dphi`, `met_ll_dphi`, `ll_bb_pt_ratio`, `DeepHME_mass_error_rel` (≤ 0.02).
- **Boosted.** Most important: the mass-like group, the transverse-mass group, `ll_bb_dR`,
  `ll_dR_scaled`, fat-jet mass. Negligible: `met_ll_dphi`, `fatbjet_tau4`, `lep1_pt`,
  `DeepHME_mass_error_rel`, `ll_dphi`, `MT`, `fatbjet_tau3`.
- `DeepHME_mass_error_rel` is not used by either network and is dropped.
- **v2b at all trained masses.** Same method on v2b at all 30 masses (boosted: 350–2000 GeV;
  250–320 GeV have < 200 boosted signal events per parity and are skipped), with all boosted
  background events (10% in resolved). Line plots of the top 12 inputs and the correlated groups
  vs mass, band = uncertainty of the mean over cycles:
  `06_importance_evolution_v2b_{resolved,boosted}.png` (absolute AUC loss, AUC − AUC_shuffled,
  log scale), `06_importance_evolution_loss_v2b_*.png` (multiclass class-balanced loss, all output
  nodes). Inputs are ranked by their share of the AUC loss at each mass (sum over single inputs
  = 1), averaged over masses. The relative AUC loss, ΔAUC / (1 − AUC), is not plotted: 1 − AUC
  falls to 2·10⁻⁴ at 2 TeV and inflates it at high mass. The AUC loss of every input falls by
  10–1000× from 300 GeV to 2 TeV (less left to lose as AUC → 1). Resolved: `bjet1_btagPNetB`
  leads at 260–400 GeV, `ll_pt`, the b-tag scores and `bb_mass` at 450–750 GeV, `MX_reduced` and
  `bjet1_pt` above 800 GeV; the mass-like group (shuffled together) is the largest at every mass.
  Boosted: `ll_bb_dR` and the `MX_reduced`+`bbllmet_mass` group lead below 600 GeV, `MT2_bb` and
  `MT2_ll` at 600–1000 GeV, the MT2 and MET inputs above 1.2 TeV.
- **b-tag vs `ll_pt` across models.** Same method on the production model (v2610 parity files,
  its own inputs, 10 masses; `production_v1_on_v2610/feature_importance`). Absolute AUC loss
  at 300 / 400 / 600 GeV, production | v1 | v2b: `bjet1_btagPNetB` 0.163/0.088/0.019 |
  0.146/0.064/0.014 | 0.151/0.066/0.014; `ll_pt` 0.029/0.021/0.007 | 0.044/0.020/0.005 |
  0.085/0.050/0.018. The b-tag scores contribute the same in all three models (baseline AUCs
  within 0.008). `bjet1_btagPNetB` is the leading single input at 250–500 GeV in the production
  model but only at 260–400 GeV in v2b, where `ll_pt` ranks above it from 500 GeV up. The production model
  uses the lepton pT too (`lep1_pt` above `ll_pt` and the b-tag scores at 1 TeV). v2b, without
  `MT`, `min_lmet_mt` and `ll_bb_pt_ratio` (|r| = 0.1–0.3 with `ll_pt`), puts 2–4× more of it
  on `ll_pt` at the same AUC: a shift of attribution, not new information. Shuffling `ll_pt`
  alone costs more than shuffling it with `lep1_pt`, so its single-input value is inflated by
  inconsistent lepton momenta. The b-tag dominance in the earlier production plots came from
  the `train_pdnn.py` scan below, which favours mass-independent inputs.
- **Not used: the scan built into `train_pdnn.py`** (`feature_importance_evolution_lines_*`,
  `per_mass_feature_importance_*`). At each hypothesis it labels the signal of all masses as
  signal, unweighted (boosted: 96% of the events), so the loss mostly measures the rejection of
  off-mass signal; shuffling a mass-sensitive input lowers it, and the clip at 0 hides this
  (v2b boosted at 1.5 TeV: 90% of the inputs clipped). Each `_foldsN` file is one cycle.

## 9. Pruned models

- **Models.** v2a (moderate, 31 resolved / 27 boosted inputs) and v2b (aggressive, 26 / 21),
  from section 8; the removed inputs are listed in `model_configs/v2610_pDNN_v2{a,b}.yaml`.
- **Result.** Mean ratio of expected limits to the geometric mean of v1 and v1_seed2, in the
  bands 250–400 | 450–900 | 1000–2000 GeV:

  | Category | v1_seed2 (noise) | v2a | v2b |
  |---|---|---|---|
  | combined, spin 0 | 1.010 / 0.994 / 0.987 | 1.001 / 0.996 / 0.971 | 1.013 / 0.963 / 0.954 |
  | combined, spin 2 | 1.009 / 0.988 / 0.995 | 1.003 / 0.994 / 0.989 | 1.020 / 0.952 / 0.954 |
  | res2b, spin 0 | 1.011 / 0.999 / 1.011 | 1.000 / 1.020 / 0.974 | 1.014 / 1.012 / 1.005 |
  | boosted, spin 0 | 0.989 / 0.990 / 0.985 | 0.990 / 0.922 / 0.970 | 0.920 / 0.876 / 0.949 |
  | boosted, spin 2 | 0.979 / 0.984 / 0.995 | 0.941 / 0.926 / 0.988 | 0.951 / 0.867 / 0.949 |

  Removing the inputs costs no sensitivity. v2b is better by 5–13% in boosted, beyond the seed
  noise, consistent with less overfitting of the small boosted background sample, and 4–5% better
  combined above 450 GeV. Figures: `07_limits_pruned_models_*.png`.
- **Decision (proposed).** v2b is the optimized input set.

## 10. DeepHME_mass: DNN input vs 2D fit

- **Comparison**, all with the same limit chain and the v2b input set:
  1. 1D fit of v2b (no DeepHME_mass input), the reference;
  2. 2D fit of the v2b score × DeepHME_mass (`limit_pdnn.py --observable dnn_hme`: each bin of the
     1D DNN binning split further in DeepHME_mass at the signal quantiles, merged to ≥ 1
     background event and ≥ 10 MC events per bin, unrolled);
  3. 1D fit of v3 = v2b + DeepHME_mass as an input (same training otherwise).
- **Binning check.** A first 2D binning with coarser DNN slices (to leave room for HME bins) lost
  the best 1D bins at high score and was 10–20% worse than 1D at 600–1500 GeV. Making the 2D
  binning a refinement of the 1D binning fixed this; the 2D fit is then never worse than 1D.
- **Result.** Mean ratio of expected limits to the v2b 1D limit, bands 250–400 | 450–900 |
  1000–2000 GeV:

  | Category | 2D v2b × HME | 1D v3 (HME input) |
  |---|---|---|
  | combined, spin 0 | 0.942 / 0.964 / 0.995 | 0.984 / 1.025 / 0.973 |
  | combined, spin 2 | 0.935 / 0.965 / 0.999 | 0.981 / 1.022 / 0.987 |
  | res2b, spin 0 | 0.943 / 0.956 / 0.989 | 0.984 / 1.033 / 0.967 |
  | boosted, spin 0 | 0.834 / 0.976 / 0.996 | 0.901 / 0.996 / 0.976 |

  Figures: `08_limits_deephme_input_vs_2d_*.png`.
- **Correlation with DeepHME_mass** (res2b background, weighted Pearson r of the score with
  DeepHME_mass at 300 / 500 / 800 / 1000 / 1500 GeV): v2b −0.15 / 0.46 / 0.57 / 0.58 / 0.58;
  v3 −0.15 / 0.47 / 0.58 / 0.60 / 0.60. The DNN without HME already carries most of the
  HME-like information through the mass-like inputs; adding HME as an input changes little.
- **Conclusion.** Using DeepHME_mass as a second fit dimension improves the expected limits by
  ~6% at 250–400 GeV and ~3.5% at 450–900 GeV (beyond the training noise of ~3%); feeding it to
  the DNN does not reproduce this gain (within ±3% of v2b). Above 1 TeV all three agree within
  noise, because the best DNN bins are already nearly background-free. Proposed production setup:
  v2b without HME input, in a 2D DNN × DeepHME fit.

## 12. Comparison with the production model and with Run 2

- **Inputs.** Run 2 DL expected limits (spin 0, 138 fb⁻¹, `run2_spin0_DL_external_limits.json`);
  the latest production DL expected limits (deployed model
  `config/DNN/DoubleLepton_Parametric_{Resolved,Boosted}_v1`, full analysis with systematics and
  TT/DY control regions, 2D DNN × DeepHME fit, 62.3 fb⁻¹, `slides_update_v8/limits_table.csv`).
  The signal samples are normalised to σ(pp → X → HH) = 1 pb times the branching ratios, so the
  quick limits of section 4 are on σ(pp → X → HH) in pb as well.
- **Method.** The quick limits are stat-only and are not compared with the full analysis directly.
  The deployed model is applied to the v2610 dataset exactly as in the analysis
  (`score_deployed.py`: the model of fold f scores parity (f + 3) mod 4, never its own training
  events) and run through the same quick chain as v2b. The production limit is projected with
  the ratio: projected(m) = production(m) × quick_v2b(m) / quick_production(m)
  (`production_comparison.py`). Spin 0, combined categories, 2D DNN × DeepHME fit for both models.
- **Result** (pb):

  | m_X | Run 2 DL | production DL | v2b / production (quick, 2D) | projected with v2b | projected / Run 2 |
  |---|---|---|---|---|---|
  | 300 | 6.63 | 5.84 | 0.91 | 5.34 | 0.81 |
  | 400 | 2.07 | 1.53 | 0.89 | 1.36 | 0.66 |
  | 500 | 0.397 | 0.308 | 0.84 | 0.259 | 0.65 |
  | 600 | 0.188 | 0.149 | 0.91 | 0.136 | 0.72 |
  | 700 | 0.106 | 0.084 | 0.87 | 0.073 | 0.68 |
  | 800 | 0.078 | 0.060 | 0.92 | 0.055 | 0.71 |
  | 900 | 0.057 | 0.044 | 0.93 | 0.040 | 0.71 |
  | 1000 | – | 0.034 | 0.95 | 0.032 | – |
  | 1200 | – | 0.027 | 0.98 | 0.026 | – |
  | 1400 | – | 0.029 | 1.01 | 0.029 | – |

  The new model improves the expected limits by 8–16% at 300–900 GeV (1D fit: 8–21%), beyond the
  ~3% training noise, and matches the production model at 1.2–1.4 TeV. The production DL limits
  are 12–27% below Run 2 DL; with v2b the projection is 19–35% below, with 45% of the luminosity.
  The quick chain reproduces the production limits of the deployed model within ~10% at
  400–1000 GeV. Figures and tables: `09_production_and_run2_comparison_{2D,1D}.{png,csv}`.
- **All 30 masses.** The production model (trained on 15 Radion masses, 300–2000 GeV) evaluated at
  all 30 v2b masses, both spins, same quick chain (2D fit). Ratio v2b / production:
  250 / 260 / 270 / 280 GeV (below the production model's training range): 0.25 / 0.36 / 0.62 / 0.70
  (spin 0; spin 2 the same); masses between its training points (320–850 GeV) 0.84–0.92, the same
  gain as at the neighbouring trained masses; above 1 TeV 0.90–1.07 for both. The production model
  interpolates well between its training masses but extrapolates poorly below 300 GeV (limits up to
  4× worse, 11× in the 1D fit), which the v2b training range removes. For spin 2 (not in the
  production model's training) v2b gains a few percent more than for spin 0. Figure and table:
  `10_production_vs_v2b_all_masses.{png,csv}`.

- **Assumption.** The projection assumes the systematics and control-region constraints of the
  full analysis change the limits of both models by the same factor.

## 11. Run-2-like MEDIUM b-tagging vs LOOSE (deferred)

Run 2 required MEDIUM b-tagged jets; this analysis uses LOOSE (`idbtagPNetB >= 1`), which
increases the acceptance. Planned study, deferred for now:
- The Run-2-like categories are a subset of the training dataset: res2b = both b-jet candidates
  pass the MEDIUM PNet working point, recovery = only the leading one; boosted unchanged.
- The MEDIUM working point is era dependent (2022 0.245, 2022EE 0.2605, 2023 0.1917,
  2023BPix 0.1919), and the dataset has no era column: add one in `create_dataset.py`
  (`DefinePerSample` from the input file path) and re-produce the dataset.
- Check that `weight_Central` includes the b-tag shape reweighting (`weight_bTagShape_Central`),
  which stays valid for any cut on the score.
- Retrain v2b on the MEDIUM selection and compare 1D and 2D expected limits with LOOSE v2b.

## Commits

`da45ca3` dataset maker; `b09ee5c` training, limits, overtraining; `5e4fcf5` seed; `2dad9b1`,
`786328a` feature importance; `6f4e26b` pruned configs.

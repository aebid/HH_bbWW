"""Standalone check of the C++ expressions added to defineJetSelections."""
import ROOT

ROOT.gROOT.SetBatch(True)

# Build a tiny frame mimicking SelectedFatJet_* RVec columns, including the
# empty-collection case that the ternary has to survive.
df = ROOT.RDataFrame(4)
df = df.Define("evt", "rdfentry_")
df = df.Define(
    "SelectedFatJet_particleNetWithMass_HbbvsQCD",
    """
    if (evt == 0) return ROOT::VecOps::RVec<float>{0.99f, 0.50f};   // one tagged
    if (evt == 1) return ROOT::VecOps::RVec<float>{0.10f, 0.20f};   // none tagged
    if (evt == 2) return ROOT::VecOps::RVec<float>{};               // no fat jets
    return ROOT::VecOps::RVec<float>{0.95f, 0.93f};                 // two tagged
    """,
)
df = df.Define(
    "SelectedFatJet_msoftdrop",
    """
    if (evt == 0) return ROOT::VecOps::RVec<float>{12.0f, 140.0f};
    if (evt == 1) return ROOT::VecOps::RVec<float>{88.0f, 91.0f};
    if (evt == 2) return ROOT::VecOps::RVec<float>{};
    return ROOT::VecOps::RVec<float>{125.0f, 25.0f};
    """,
)

# --- exactly the expressions added to Analysis/hh_bbww.py ---
df = df.Define("FatBJetProbe_Sel",
               "SelectedFatJet_particleNetWithMass_HbbvsQCD > 0.92")
df = df.Define("FatBJetProbe_msoftdrop",
               "SelectedFatJet_msoftdrop[FatBJetProbe_Sel]")
df = df.Define("fatbjetProbe_msoftdrop",
               "FatBJetProbe_msoftdrop.size() > 0 ? FatBJetProbe_msoftdrop[0] : -1.f")
df = df.Define("leadfatjet_msoftdrop",
               "SelectedFatJet_msoftdrop.size() > 0 ? SelectedFatJet_msoftdrop[0] : -1.f")

cols = ["evt", "fatbjetProbe_msoftdrop", "leadfatjet_msoftdrop"]
res = df.AsNumpy(cols)
print(f"{'evt':>4} {'probe':>10} {'lead':>10}   expected")
expected = [
    (0, 12.0, 12.0, "tagged jet has msd=12 -> below the 30 cut"),
    (1, -1.0, 88.0, "no tagged jet -> probe sentinel; lead still filled"),
    (2, -1.0, -1.0, "no fat jets at all -> both sentinel"),
    (3, 125.0, 125.0, "leading-score tagged jet, msd=125 -> passes"),
]
ok = True
for i in range(4):
    e = expected[i]
    got = (res["evt"][i], res["fatbjetProbe_msoftdrop"][i], res["leadfatjet_msoftdrop"][i])
    match = abs(got[1] - e[1]) < 1e-5 and abs(got[2] - e[2]) < 1e-5
    ok = ok and match
    print(f"{got[0]:>4} {got[1]:>10.1f} {got[2]:>10.1f}   {e[3]}  {'OK' if match else 'MISMATCH'}")
print("\nALL OK" if ok else "\nFAILED")

# EtaB→J/ψJ/ψ (Di-J/ψ) Validation Workflow

## Overview

Generate EtaB→J/ψJ/ψ events using `EtaBToJpsiJpsi_14TeV_TuneCP5_pythia8_cfi.py`, run reconstruction, and produce di-J/ψ mass spectra.

## Physics Motivation

- EtaB (η_b) is the pseudoscalar bottomonium ground state
- Mass fixed to 9.4 GeV in generator
- Decay: η_b → J/ψ J/ψ (both J/ψ → μ⁺μ⁻)
- Signal in 4-muon final state with characteristic di-J/ψ mass peak

## Environment Setup (on lxplus)

```bash
ssh username@lxplus.cern.ch

cmsrel CMSSW_14_0_13
cd CMSSW_14_0_13/src
cmsenv

# Clone and build fork
git clone https://github.com/simulationstation/cmssw fork_cmssw
cp -r fork_cmssw/PhysicsTools/BPHNano PhysicsTools/
cp -r fork_cmssw/PhysicsTools/NanoAOD/python/custom_bph_cff.py PhysicsTools/NanoAOD/python/
cp fork_cmssw/Configuration/Generator/python/EtaBToJpsiJpsi_14TeV_TuneCP5_pythia8_cfi.py Configuration/Generator/python/
scram b -j8
```

## Step 1: GEN-SIM with Filters

The generator fragment includes filters:
- `etafilter`: Require η_b in event
- `etatojpsipairfilter`: Require η_b → J/ψ J/ψ with |η| < 2.6
- `jpsifilter`: Require J/ψ → μμ with pT(μ) > 1.8 GeV, |η(μ)| < 2.5

```bash
cmsDriver.py Configuration/Generator/python/EtaBToJpsiJpsi_14TeV_TuneCP5_pythia8_cfi.py \
  --fileout file:dijpsi_gensim.root \
  --mc \
  --eventcontent RAWSIM \
  --datatier GEN-SIM \
  --conditions 130X_mcRun3_2022_realistic_postEE_v6 \
  --beamspot Realistic25ns13p6TeVEarly2022Collision \
  --step GEN,SIM \
  --geometry DB:Extended \
  --era Run3 \
  --python_filename dijpsi_gensim_cfg.py \
  --no_exec \
  -n 5000

cmsRun dijpsi_gensim_cfg.py
```

## Step 2: DIGI-RECO

```bash
cmsDriver.py \
  --filein file:dijpsi_gensim.root \
  --fileout file:dijpsi_reco.root \
  --mc \
  --eventcontent MINIAODSIM \
  --datatier MINIAODSIM \
  --conditions 130X_mcRun3_2022_realistic_postEE_v6 \
  --step DIGI,L1,DIGI2RAW,HLT:GRun,RAW2DIGI,L1Reco,RECO,PAT \
  --geometry DB:Extended \
  --era Run3 \
  --python_filename dijpsi_reco_cfg.py \
  --no_exec \
  -n -1

cmsRun dijpsi_reco_cfg.py
```

## Step 3: NanoAOD with BPH Customization

```bash
cmsDriver.py \
  --filein file:dijpsi_reco.root \
  --fileout file:dijpsi_nanoaod.root \
  --mc \
  --eventcontent NANOAODSIM \
  --datatier NANOAODSIM \
  --conditions 130X_mcRun3_2022_realistic_postEE_v6 \
  --step NANO \
  --era Run3 \
  --customise PhysicsTools/NanoAOD/custom_bph_cff.nanoAOD_customizeBPH \
  --customise PhysicsTools/NanoAOD/custom_bph_cff.nanoAOD_customizeMC \
  --python_filename dijpsi_nano_cfg.py \
  --no_exec \
  -n -1

cmsRun dijpsi_nano_cfg.py
```

## Alternative: GEN-Only Quick Workflow

```bash
cmsDriver.py Configuration/Generator/python/EtaBToJpsiJpsi_14TeV_TuneCP5_pythia8_cfi.py \
  --fileout file:dijpsi_gen.root \
  --mc \
  --eventcontent GENRAW \
  --datatier GEN \
  --conditions 130X_mcRun3_2022_realistic_postEE_v6 \
  --step GEN \
  --era Run3 \
  --python_filename dijpsi_gen_cfg.py \
  --no_exec \
  -n 50000

cmsRun dijpsi_gen_cfg.py
```

## Step 4: Run Histogram Analysis

```bash
# After NanoAOD production
python3 dijpsi_analysis_cfg.py \
  --input dijpsi_nanoaod.root \
  --output dijpsi_hist.root \
  --csv dijpsi_hist.csv

# For GEN-only analysis
python3 dijpsi_analysis_cfg.py \
  --input dijpsi_gen.root \
  --output dijpsi_hist.root \
  --csv dijpsi_hist.csv \
  --gen-only
```

## Expected Outputs

| File | Description |
|------|-------------|
| `dijpsi_gensim.root` | GEN-SIM output |
| `dijpsi_reco.root` | RECO/MINIAOD output |
| `dijpsi_nanoaod.root` | NanoAOD with BPH collections |
| `dijpsi_hist.root` | TH1 histograms |
| `dijpsi_hist.csv` | Binned histogram data |

## Mass Spectrum Details

1. **Single J/ψ mass** (μ⁺μ⁻ pairs near 3.1 GeV)
   - Range: [2.5, 3.5] GeV
   - Bins: 100

2. **Di-J/ψ mass** (m(J/ψ J/ψ) ≈ m(η_b) = 9.4 GeV)
   - Range: [6, 15] GeV
   - Bins: 180

3. **4-muon invariant mass**
   - Range: [6, 15] GeV
   - Bins: 180

## Physics Expectations

- J/ψ peak at 3.097 GeV
- Di-J/ψ peak at 9.4 GeV (η_b mass)
- Filter efficiency: ~40% (from generator)
- Double J/ψ threshold: 2 × 3.097 = 6.194 GeV

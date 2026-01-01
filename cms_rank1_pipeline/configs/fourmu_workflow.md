# FourMu Validation Workflow

## Overview

Generate 4-muon events using `FourMuPt_1_200_pythia8_cfi.py`, run minimal reconstruction, and produce mass spectra.

## Environment Setup (on lxplus)

```bash
# SSH to lxplus
ssh username@lxplus.cern.ch

# Setup CMSSW
cmsrel CMSSW_14_0_13
cd CMSSW_14_0_13/src
cmsenv

# Clone and build fork
git clone https://github.com/simulationstation/cmssw fork_cmssw
cp -r fork_cmssw/PhysicsTools/BPHNano PhysicsTools/
cp -r fork_cmssw/PhysicsTools/NanoAOD/python/custom_bph_cff.py PhysicsTools/NanoAOD/python/
cp fork_cmssw/Configuration/Generator/python/FourMuPt_1_200_pythia8_cfi.py Configuration/Generator/python/
scram b -j8
```

## Step 1: GEN-SIM (Full Simulation)

```bash
cmsDriver.py Configuration/Generator/python/FourMuPt_1_200_pythia8_cfi.py \
  --fileout file:fourmu_gensim.root \
  --mc \
  --eventcontent RAWSIM \
  --datatier GEN-SIM \
  --conditions 130X_mcRun3_2022_realistic_postEE_v6 \
  --beamspot Realistic25ns13p6TeVEarly2022Collision \
  --step GEN,SIM \
  --geometry DB:Extended \
  --era Run3 \
  --python_filename fourmu_gensim_cfg.py \
  --no_exec \
  -n 1000

cmsRun fourmu_gensim_cfg.py
```

## Step 2: DIGI-RECO (Minimal)

```bash
cmsDriver.py \
  --filein file:fourmu_gensim.root \
  --fileout file:fourmu_reco.root \
  --mc \
  --eventcontent MINIAODSIM \
  --datatier MINIAODSIM \
  --conditions 130X_mcRun3_2022_realistic_postEE_v6 \
  --step DIGI,L1,DIGI2RAW,HLT:GRun,RAW2DIGI,L1Reco,RECO,PAT \
  --geometry DB:Extended \
  --era Run3 \
  --python_filename fourmu_reco_cfg.py \
  --no_exec \
  -n -1

cmsRun fourmu_reco_cfg.py
```

## Step 3: NanoAOD with BPH Customization

```bash
cmsDriver.py \
  --filein file:fourmu_reco.root \
  --fileout file:fourmu_nanoaod.root \
  --mc \
  --eventcontent NANOAODSIM \
  --datatier NANOAODSIM \
  --conditions 130X_mcRun3_2022_realistic_postEE_v6 \
  --step NANO \
  --era Run3 \
  --customise PhysicsTools/NanoAOD/custom_bph_cff.nanoAOD_customizeBPH \
  --customise PhysicsTools/NanoAOD/custom_bph_cff.nanoAOD_customizeMC \
  --python_filename fourmu_nano_cfg.py \
  --no_exec \
  -n -1

cmsRun fourmu_nano_cfg.py
```

## Alternative: GEN-Only Quick Workflow

For faster validation without full detector simulation:

```bash
cmsDriver.py Configuration/Generator/python/FourMuPt_1_200_pythia8_cfi.py \
  --fileout file:fourmu_gen.root \
  --mc \
  --eventcontent GENRAW \
  --datatier GEN \
  --conditions 130X_mcRun3_2022_realistic_postEE_v6 \
  --step GEN \
  --era Run3 \
  --python_filename fourmu_gen_cfg.py \
  --no_exec \
  -n 10000

cmsRun fourmu_gen_cfg.py
```

Then run the analysis script:

```bash
python3 fourmu_analysis_cfg.py --input fourmu_gen.root --output fourmu_hist.root
```

## Step 4: Run Histogram Analysis

```bash
# After NanoAOD production
python3 fourmu_analysis_cfg.py \
  --input fourmu_nanoaod.root \
  --output fourmu_hist.root \
  --csv fourmu_hist.csv

# For GEN-only analysis
python3 fourmu_analysis_cfg.py \
  --input fourmu_gen.root \
  --output fourmu_hist.root \
  --csv fourmu_hist.csv \
  --gen-only
```

## Expected Outputs

| File | Description |
|------|-------------|
| `fourmu_gensim.root` | GEN-SIM output |
| `fourmu_reco.root` | RECO/MINIAOD output |
| `fourmu_nanoaod.root` | NanoAOD with BPH collections |
| `fourmu_hist.root` | TH1 histograms (dimuon, 4μ masses) |
| `fourmu_hist.csv` | Binned histogram data for rank-1 test |

## Mass Spectrum Details

The analysis produces:

1. **Dimuon mass** (all opposite-sign pairs)
   - Range: [0, 15] GeV
   - Bins: 150

2. **4-muon invariant mass** (all 4μ combinations)
   - Range: [0, 50] GeV
   - Bins: 100

3. **Di-dimuon mass** (pairing into 2 dimuons)
   - Range: [0, 50] GeV
   - Bins: 100

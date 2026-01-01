# Ready to Run on LXPLUS

## Prerequisites

This pipeline requires the CMS software environment (CMSSW) and grid access.
These are **NOT available** in the current environment.

## Quick Start on LXPLUS

### 1. SSH to LXPLUS

```bash
ssh -Y username@lxplus.cern.ch
```

### 2. Setup VOMS Proxy (Grid Access)

```bash
# Initialize grid proxy
voms-proxy-init -voms cms -rfc -valid 192:00

# Verify proxy
voms-proxy-info --all
```

### 3. Setup CMSSW Environment

```bash
# Create working area
cd /afs/cern.ch/work/${USER:0:1}/$USER/
mkdir -p cms_rank1_pipeline && cd cms_rank1_pipeline

# Setup CMSSW (Run3 compatible release)
cmsrel CMSSW_14_0_13
cd CMSSW_14_0_13/src
cmsenv

# Clone and integrate the fork
git clone https://github.com/simulationstation/cmssw fork_cmssw

# Copy relevant modules
cp -r fork_cmssw/PhysicsTools/BPHNano PhysicsTools/
cp -r fork_cmssw/PhysicsTools/NanoAOD/python/custom_bph_cff.py PhysicsTools/NanoAOD/python/
cp fork_cmssw/Configuration/Generator/python/FourMuPt_1_200_pythia8_cfi.py Configuration/Generator/python/
cp fork_cmssw/Configuration/Generator/python/EtaBToJpsiJpsi_14TeV_TuneCP5_pythia8_cfi.py Configuration/Generator/python/

# Build
scram b -j8
```

## Real Data Workflow

### 4. Query DAS for Data Files

```bash
# Query ParkingDoubleMuonLowMass dataset
dasgoclient -query="file dataset=/ParkingDoubleMuonLowMass0/Run2022G-22Sep2023-v1/MINIAOD" --limit=20 > files_2022G.txt

# Or for 2023 data
dasgoclient -query="file dataset=/ParkingDoubleMuonLowMass0/Run2023C-22Sep2023_v4-v1/MINIAOD" --limit=20 > files_2023C.txt

# Check file availability
head -5 files_2022G.txt
```

### 5. Run NanoAOD+BPH Production

```bash
# Copy config from this pipeline
cp /path/to/cms_rank1_pipeline/configs/realdata_bphnano_cfg.py .

# Run on a single file (for testing)
cmsRun realdata_bphnano_cfg.py \
  inputFiles=/store/data/Run2022G/ParkingDoubleMuonLowMass0/MINIAOD/22Sep2023-v1/xxx.root \
  globalTag=124X_dataRun3_v15 \
  maxEvents=10000

# Or use file list
cmsRun realdata_bphnano_cfg.py \
  inputFiles_load=files_2022G.txt \
  maxEvents=-1
```

### 6. Run Histogram Analysis

```bash
# After NanoAOD production
python3 realdata_hist_cfg.py \
  --input bphnano_output.root \
  --output realdata_hist.root \
  --csv-prefix realdata

# Outputs:
# - realdata_hist.root (TH1 histograms)
# - realdata_jpsi.csv
# - realdata_dijpsi.csv
# - realdata_4mu.csv
```

## MC Validation Workflow

### 7. Generate FourMu Sample

```bash
# GEN-only (fast)
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

# Run analysis
python3 fourmu_analysis_cfg.py \
  --input fourmu_gen.root \
  --output fourmu_hist.root \
  --csv fourmu_hist.csv \
  --gen-only
```

### 8. Generate EtaB→J/ψJ/ψ Sample

```bash
# GEN-only (fast)
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

# Run analysis
python3 dijpsi_analysis_cfg.py \
  --input dijpsi_gen.root \
  --output dijpsi_hist.root \
  --csv dijpsi_hist.csv \
  --gen-only
```

## Full Production Chain (with detector simulation)

For a complete chain with detector simulation:

```bash
# Step 1: GEN-SIM
cmsDriver.py Configuration/Generator/python/FourMuPt_1_200_pythia8_cfi.py \
  --fileout file:step1_gensim.root \
  --mc --eventcontent RAWSIM --datatier GEN-SIM \
  --conditions 130X_mcRun3_2022_realistic_postEE_v6 \
  --beamspot Realistic25ns13p6TeVEarly2022Collision \
  --step GEN,SIM --geometry DB:Extended --era Run3 \
  -n 1000 --no_exec --python_filename step1_cfg.py

# Step 2: DIGI-HLT
cmsDriver.py --filein file:step1_gensim.root \
  --fileout file:step2_digi.root \
  --mc --eventcontent RAWSIM --datatier GEN-SIM-RAW \
  --conditions 130X_mcRun3_2022_realistic_postEE_v6 \
  --step DIGI,L1,DIGI2RAW,HLT:GRun \
  --geometry DB:Extended --era Run3 \
  -n -1 --no_exec --python_filename step2_cfg.py

# Step 3: RECO
cmsDriver.py --filein file:step2_digi.root \
  --fileout file:step3_reco.root \
  --mc --eventcontent MINIAODSIM --datatier MINIAODSIM \
  --conditions 130X_mcRun3_2022_realistic_postEE_v6 \
  --step RAW2DIGI,L1Reco,RECO,PAT \
  --geometry DB:Extended --era Run3 \
  -n -1 --no_exec --python_filename step3_cfg.py

# Step 4: NanoAOD+BPH
cmsDriver.py --filein file:step3_reco.root \
  --fileout file:step4_nano.root \
  --mc --eventcontent NANOAODSIM --datatier NANOAODSIM \
  --conditions 130X_mcRun3_2022_realistic_postEE_v6 \
  --step NANO --era Run3 \
  --customise PhysicsTools/NanoAOD/custom_bph_cff.nanoAOD_customizeBPH \
  --customise PhysicsTools/NanoAOD/custom_bph_cff.nanoAOD_customizeMC \
  -n -1 --no_exec --python_filename step4_cfg.py
```

## CRAB Submission (Large-Scale Production)

For processing full datasets:

```bash
# Use the fork's CRAB submission script
cd PhysicsTools/BPHNano/production/

# Edit samples.yml to select dataset
# Then submit:
python3 submit_on_crab.py --samples data_Run2022G_part0 --era Run3
```

## Global Tags Reference

| Era | Data | MC |
|-----|------|-----|
| Run2022 (pre-reprocessing) | 124X_dataRun3_v15 | 130X_mcRun3_2022_realistic_postEE_v6 |
| Run2023 | 130X_dataRun3_Prompt_v4 | 130X_mcRun3_2023_realistic_v14 |
| Run2024 | 140X_dataRun3_Prompt_v4 | 140X_mcRun3_2024_realistic_v8 |

## Troubleshooting

### Proxy Issues
```bash
# Check proxy validity
voms-proxy-info --timeleft

# Renew if expired
voms-proxy-init -voms cms -rfc -valid 192:00
```

### XRootD Access Issues
```bash
# Test file access
xrdfs root://cms-xrd-global.cern.ch/ ls /store/data/Run2022G/

# Set fallback redirector
export CMS_XROOTD_REDIRECTOR=root://cmsxrootd.fnal.gov/
```

### SCRAM Build Issues
```bash
# Clean and rebuild
scram b clean
scram b -j8
```

## Expected Outputs

After running the full pipeline:

| File | Description |
|------|-------------|
| `fourmu_hist.root` | FourMu mass histograms |
| `fourmu_hist_dimu.csv` | Dimuon spectrum CSV |
| `fourmu_hist_4mu.csv` | 4-muon spectrum CSV |
| `dijpsi_hist.root` | Di-J/psi mass histograms |
| `dijpsi_hist_jpsi.csv` | J/psi spectrum CSV |
| `dijpsi_hist_dijpsi.csv` | Di-J/psi spectrum CSV |
| `realdata_hist.root` | Real data histograms |
| `realdata_jpsi.csv` | Real data J/psi CSV |
| `realdata_dijpsi.csv` | Real data di-J/psi CSV |

These CSV files can then be used with the rank-1 bottleneck test framework.

# CMSSW Fork Information

## Repository Details

| Property | Value |
|----------|-------|
| **Repository URL** | https://github.com/simulationstation/cmssw |
| **Git Commit (HEAD)** | `161f87df7a9b83b583468969bf352f6a852d6db3` |
| **Branch** | `master` |
| **Clone Date** | 2025-12-31 |

## CMSSW Release Detection

### Detection Results

| Check | Result |
|-------|--------|
| CMSSW_VERSION file | Not found |
| .SCRAM directory | Not found |
| Git tags | None |
| config/SCRAM | Not found |

### Inferred Release Series

Based on the codebase structure and configurations found:

1. **samples.yml globaltags** indicate Run3 compatibility:
   - `124X_dataRun3_v15` (Data 2022+2023)
   - `130X_mcRun3_2022_realistic_postEE_v6` (MC)
   - `140X_dataRun3_Prompt_v2` (Data 2024)

2. **Generator configurations** use:
   - `Pythia8ConcurrentGeneratorFilter` (Run3 era)
   - `TuneCP5` settings (Run3 standard)
   - 14 TeV center-of-mass energy for EtaB samples

3. **NanoAOD customizations** indicate compatibility with:
   - CMSSW_13_X or CMSSW_14_X series
   - Run3 NanoAOD format

### Recommended CMSSW Release

For full compatibility with this fork:
```
CMSSW_14_0_X or CMSSW_13_3_X
```

For running the configurations in this pipeline:
```bash
# Recommended setup on lxplus
cmsrel CMSSW_14_0_13
cd CMSSW_14_0_13/src
cmsenv
git cms-merge-topic simulationstation:master  # or manual merge
scram b -j8
```

## Fork Contents

This fork extends standard CMSSW with:

1. **BPHNano module** (`PhysicsTools/BPHNano/`)
   - Custom NanoAOD producers for B-physics
   - DiMuon, DiTrack, V0, B meson reconstruction
   - Trigger matching for low-mass double muon paths

2. **Custom generators**
   - FourMu gun configurations
   - EtaB→J/ψJ/ψ generator with filters

3. **Heavy Flavor Analysis tools**
   - `BPHHistoSpecificDecay` for mass histogramming
   - Vertex analysis utilities

## Environment Status

| Tool | Available |
|------|-----------|
| scramv1 | NO |
| cmsenv | NO |
| cmsDriver.py | NO |
| ROOT | NO |
| dasgoclient | NO |
| voms-proxy | NO |

**Mode: CONFIG GENERATION** - Configs and READMEs generated for lxplus execution.

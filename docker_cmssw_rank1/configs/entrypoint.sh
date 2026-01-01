#!/bin/bash
set -e

# ============================================================
# CMS Rank-1 Pipeline - GEN-ONLY Container Entrypoint
# No NanoAOD, No BPHNano - Pure generator-level MC
# ============================================================

LOGFILE="/outputs/CONTAINER_RUNLOG.txt"
exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================================"
echo "CMS Rank-1 Pipeline - GEN-ONLY Container Execution"
echo "Started: $(date)"
echo "============================================================"

# CMSSW scripts use unbound variables - disable strict checking
set +u

# ============================================================
# Step 1: Verify CVMFS is mounted and functional
# ============================================================
echo ""
echo "=== Checking CVMFS mount ==="

if [ ! -d "/cvmfs/cms.cern.ch" ]; then
    echo "BLOCKER: /cvmfs/cms.cern.ch is not mounted!"
    echo "Fix: Ensure CVMFS is running on host and mount with -v /cvmfs:/cvmfs:ro"
    exit 1
fi

if [ ! -f "/cvmfs/cms.cern.ch/cmsset_default.sh" ]; then
    echo "BLOCKER: /cvmfs/cms.cern.ch/cmsset_default.sh not found!"
    echo "CVMFS mount may be incomplete. Check: cvmfs_config probe cms.cern.ch"
    exit 1
fi

echo "CVMFS mount verified: /cvmfs/cms.cern.ch exists"
ls -la /cvmfs/cms.cern.ch/ | head -5

# ============================================================
# Step 2: Source CMS environment
# ============================================================
echo ""
echo "=== Sourcing CMS environment ==="
export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
export SCRAM_ARCH=el9_amd64_gcc12
source /cvmfs/cms.cern.ch/cmsset_default.sh
echo "CMS environment sourced successfully"
echo "SCRAM_ARCH: $SCRAM_ARCH"

# Verify cmsrel is available
if ! command -v cmsrel &> /dev/null; then
    echo "BLOCKER: cmsrel command not found after sourcing cmsset_default.sh"
    echo "This indicates CVMFS is not properly configured"
    exit 1
fi
echo "cmsrel command available: $(which cmsrel)"

# ============================================================
# Step 3: Create CMSSW work area
# ============================================================
echo ""
echo "=== Creating CMSSW_14_0_13 work area ==="
cd /work

# Clean any previous work area
rm -rf CMSSW_14_0_13

cmsrel CMSSW_14_0_13
cd CMSSW_14_0_13/src
eval $(scramv1 runtime -sh)
echo "CMSSW_BASE: $CMSSW_BASE"
echo "SCRAM_ARCH: $SCRAM_ARCH"

# Verify cmsRun is available
if ! command -v cmsRun &> /dev/null; then
    echo "BLOCKER: cmsRun command not found after cmsenv"
    exit 1
fi
echo "cmsRun command available: $(which cmsRun)"

# ============================================================
# Step 4: Skip generator fragment files (use inline configs)
# ============================================================
echo ""
echo "=== Using inline generator configurations (no local fragments) ==="
echo "Generator configs will be created inline in cmsRun config files"

# ============================================================
# Step 5: Skip CMSSW build (nothing local to build)
# ============================================================
echo ""
echo "=== Skipping scram b (no local code) ==="
echo "Using purely central CMSSW configuration"

# ============================================================
# Step 6: FourMu GEN-only workflow
# ============================================================
echo ""
echo "============================================================"
echo "=== FourMu GEN-only (5000 events) ==="
echo "============================================================"

# Create minimal GEN-only config with INLINE Pythia8 settings
cat > fourmu_gen_cfg.py << 'CFGEOF'
import FWCore.ParameterSet.Config as cms

process = cms.Process("GEN")

# Message logger
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 500

# Number of events
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(5000))

# Source (empty for generator)
process.source = cms.Source("EmptySource")

# Random number service
process.load("Configuration.StandardSequences.Services_cff")
process.RandomNumberGeneratorService.generator = cms.PSet(
    initialSeed = cms.untracked.uint32(12345),
    engineName = cms.untracked.string('TRandom3')
)

# Generator - FourMu (Z->mumu) with inline Pythia8 settings
# Import the common and CP5 settings from central CMSSW
from Configuration.Generator.Pythia8CommonSettings_cfi import pythia8CommonSettingsBlock
from Configuration.Generator.MCTunesRun3ECM13p6TeV.PythiaCP5Settings_cfi import pythia8CP5SettingsBlock

process.generator = cms.EDFilter("Pythia8ConcurrentGeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(13600.),
    PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        processParameters = cms.vstring(
            'WeakSingleBoson:ffbar2gmZ = on',
            '23:onMode = off',
            '23:onIfAny = 13',
            '23:mMin = 3.0',
        ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CP5Settings',
                                    'processParameters')
    )
)

# GenParticles producer - use generator output directly (no smearing needed for GEN-only)
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
process.genParticles = cms.EDProducer("GenParticleProducer",
    src = cms.InputTag("generator", "unsmeared"),
    abortOnUnknownPDGCode = cms.untracked.bool(False),
    saveBarCodes = cms.untracked.bool(True)
)

# GEN particles output
process.GENoutput = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:fourmu_gen.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_generator_*_*',
        'keep *_genParticles_*_*',
    ),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring())
)

# Path: generator -> genParticles (no VtxSmeared for GEN-only)
process.generation_step = cms.Path(process.generator * process.genParticles)
process.GENoutput_step = cms.EndPath(process.GENoutput)

# Schedule
process.schedule = cms.Schedule(process.generation_step, process.GENoutput_step)
CFGEOF

echo "Running cmsRun fourmu_gen_cfg.py..."
time cmsRun fourmu_gen_cfg.py 2>&1

if [ -f "fourmu_gen.root" ]; then
    FOURMU_SIZE=$(du -h fourmu_gen.root | cut -f1)
    echo "FourMu GEN complete: fourmu_gen.root ($FOURMU_SIZE)"
    cp fourmu_gen.root /outputs/
else
    echo "ERROR: fourmu_gen.root not created!"
    echo "Checking for error messages..."
    exit 1
fi

# ============================================================
# Step 7: DiJpsi GEN-only workflow
# ============================================================
echo ""
echo "============================================================"
echo "=== DiJpsi (EtaB->JpsiJpsi) GEN-only (5000 events) ==="
echo "============================================================"

# Create minimal GEN-only config with INLINE Pythia8 settings
cat > dijpsi_gen_cfg.py << 'CFGEOF'
import FWCore.ParameterSet.Config as cms

process = cms.Process("GEN")

# Message logger
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 500

# Number of events
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(5000))

# Source (empty for generator)
process.source = cms.Source("EmptySource")

# Random number service
process.load("Configuration.StandardSequences.Services_cff")
process.RandomNumberGeneratorService.generator = cms.PSet(
    initialSeed = cms.untracked.uint32(67890),
    engineName = cms.untracked.string('TRandom3')
)

# Generator - EtaB->JpsiJpsi (bottomonium) with inline Pythia8 settings
# Import the common and CP5 settings from central CMSSW
from Configuration.Generator.Pythia8CommonSettings_cfi import pythia8CommonSettingsBlock
from Configuration.Generator.MCTunesRun3ECM13p6TeV.PythiaCP5Settings_cfi import pythia8CP5SettingsBlock

process.generator = cms.EDFilter("Pythia8ConcurrentGeneratorFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(14000.),
    PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        processParameters = cms.vstring(
            'Bottomonium:all = on',
            '553:onMode = off',
            '553:onIfAny = 443',
        ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CP5Settings',
                                    'processParameters')
    )
)

# GenParticles producer - use generator output directly (no smearing needed for GEN-only)
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
process.genParticles = cms.EDProducer("GenParticleProducer",
    src = cms.InputTag("generator", "unsmeared"),
    abortOnUnknownPDGCode = cms.untracked.bool(False),
    saveBarCodes = cms.untracked.bool(True)
)

# GEN particles output
process.GENoutput = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:dijpsi_gen.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_generator_*_*',
        'keep *_genParticles_*_*',
    ),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring())
)

# Path: generator -> genParticles (no VtxSmeared for GEN-only)
process.generation_step = cms.Path(process.generator * process.genParticles)
process.GENoutput_step = cms.EndPath(process.GENoutput)

# Schedule
process.schedule = cms.Schedule(process.generation_step, process.GENoutput_step)
CFGEOF

echo "Running cmsRun dijpsi_gen_cfg.py..."
time cmsRun dijpsi_gen_cfg.py 2>&1

if [ -f "dijpsi_gen.root" ]; then
    DIJPSI_SIZE=$(du -h dijpsi_gen.root | cut -f1)
    echo "DiJpsi GEN complete: dijpsi_gen.root ($DIJPSI_SIZE)"
    cp dijpsi_gen.root /outputs/
else
    echo "ERROR: dijpsi_gen.root not created!"
    exit 1
fi

# ============================================================
# Step 8: Run histogram analysis on GEN ROOT files
# ============================================================
echo ""
echo "============================================================"
echo "=== Running FourMu histogram analysis (GEN-only mode) ==="
echo "============================================================"
python3 /configs/fourmu_analysis_cfg.py \
  --input /outputs/fourmu_gen.root \
  --output /outputs/fourmu_hist.root \
  --csv /outputs/fourmu_hist.csv \
  --gen-only

echo ""
echo "============================================================"
echo "=== Running DiJpsi histogram analysis (GEN-only mode) ==="
echo "============================================================"
python3 /configs/dijpsi_analysis_cfg.py \
  --input /outputs/dijpsi_gen.root \
  --output /outputs/dijpsi_hist.root \
  --csv /outputs/dijpsi_hist.csv \
  --gen-only

# ============================================================
# Step 9: Run Rank-1 Bottleneck Test
# ============================================================
echo ""
echo "============================================================"
echo "=== Running Rank-1 Bottleneck Test ==="
echo "============================================================"

python3 /configs/cms_rank1_test.py \
  --channel-a /outputs/dijpsi_hist_dijpsi.csv \
  --channel-b /outputs/fourmu_hist_4mu.csv \
  --output /outputs/RANK1_RESULT.md \
  --bootstrap 100

# ============================================================
# Final Summary
# ============================================================
echo ""
echo "============================================================"
echo "=== PIPELINE COMPLETE ==="
echo "============================================================"

echo ""
echo "=== Output files ==="
ls -lh /outputs/

echo ""
echo "=== First 10 lines of FourMu 4mu CSV ==="
head -10 /outputs/fourmu_hist_4mu.csv 2>/dev/null || echo "File not found"

echo ""
echo "=== First 10 lines of DiJpsi dijpsi CSV ==="
head -10 /outputs/dijpsi_hist_dijpsi.csv 2>/dev/null || echo "File not found"

echo ""
echo "=== Rank-1 Result ==="
cat /outputs/RANK1_RESULT.md 2>/dev/null || echo "File not found"

echo ""
echo "Completed: $(date)"
echo "============================================================"

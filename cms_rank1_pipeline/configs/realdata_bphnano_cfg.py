#!/usr/bin/env python3
"""
Real Data BPHNano Production Configuration

Produces NanoAOD with BPH customizations from ParkingDoubleMuonLowMass MINIAOD.

Usage on lxplus:
  cmsenv
  cmsRun realdata_bphnano_cfg.py inputFiles=file:input.root
"""

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

# Parse command-line options
options = VarParsing('analysis')
options.register('globalTag', '124X_dataRun3_v15',
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    'Global tag for conditions')
options.register('maxEvents', -1,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.int,
    'Maximum number of events to process')
options.parseArguments()

# Initialize process
from Configuration.StandardSequences.Eras import eras
process = cms.Process('NANO', eras.Run3)

# MessageLogger
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

# Source
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring(options.inputFiles),
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

# Global tag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

# Geometry and conditions
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

# NanoAOD
process.load('PhysicsTools.NanoAOD.nano_cff')

# BPH customizations
from PhysicsTools.NanoAOD.custom_bph_cff import nanoAOD_customizeBPH, nanoAOD_customizeDiMuonBPH

# Apply customizations
process = nanoAOD_customizeBPH(process)

# Output
process.NANOAODoutput = cms.OutputModule('NanoAODOutputModule',
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(9),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('NANOAOD'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('bphnano_output.root'),
    outputCommands = process.NANOAODEventContent.outputCommands,
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('*')
    ),
)

# Path
process.nanoAOD_step = cms.Path(process.nanoSequence)
process.NANOAODoutput_step = cms.EndPath(process.NANOAODoutput)

# Schedule
process.schedule = cms.Schedule(
    process.nanoAOD_step,
    process.NANOAODoutput_step
)

# Multithreading
process.options.numberOfThreads = 4
process.options.numberOfStreams = 0

print(f"[BPHNano Config] Global tag: {options.globalTag}")
print(f"[BPHNano Config] Max events: {options.maxEvents}")
print(f"[BPHNano Config] Input files: {options.inputFiles}")

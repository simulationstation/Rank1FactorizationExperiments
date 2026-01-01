#!/bin/bash
# Status script for calibration diagnostics

echo "=== CALIBRATION STATUS ==="
echo ""

# Check if calibration is running
PIDS=$(pgrep -f "run_calibration.py" 2>/dev/null)
if [ -n "$PIDS" ]; then
    echo "Calibration RUNNING (PIDs: $PIDS)"
    echo ""
else
    echo "Calibration NOT RUNNING"
    echo ""
fi

# Show log tail
LOGFILE="/home/primary/DarkBItParticleColiderPredictions/sim_rank_sweep_calib_v3/logs/calibration.log"
if [ -f "$LOGFILE" ]; then
    echo "=== Last 30 lines of log ==="
    tail -30 "$LOGFILE"
else
    echo "No log file yet"
fi

echo ""
echo "=== Output files ==="
ls -la /home/primary/DarkBItParticleColiderPredictions/sim_rank_sweep_calib_v3/out/ 2>/dev/null || echo "No output files yet"

echo ""
echo "=== Diagnostic traces ==="
ls -la /home/primary/DarkBItParticleColiderPredictions/sim_rank_sweep_calib_v3/diagnostics/ 2>/dev/null || echo "No diagnostic files yet"

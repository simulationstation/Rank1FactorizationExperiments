#!/bin/bash
# Status script for smoke calibration

BASEDIR="/home/primary/DarkBItParticleColiderPredictions/sim_rank_sweep_calib_v4"

echo "=== SMOKE CALIBRATION STATUS ==="
echo ""

# Check if calibration is running
PIDS=$(pgrep -f "smoke_calibrate.py" 2>/dev/null)
if [ -n "$PIDS" ]; then
    echo "Status: RUNNING (PIDs: $PIDS)"
    echo ""
else
    echo "Status: NOT RUNNING"
    echo ""
fi

# Show log tail
LOGFILE="$BASEDIR/logs/smoke.log"
if [ -f "$LOGFILE" ]; then
    echo "=== Last 25 lines of smoke.log ==="
    tail -25 "$LOGFILE"
else
    echo "No log file yet"
fi

echo ""
echo "=== SMOKE STATUS FILES ==="
for f in "$BASEDIR/out/SMOKE_STATUS_"*.md; do
    if [ -f "$f" ]; then
        echo ""
        echo "--- $(basename $f) ---"
        head -30 "$f"
    fi
done 2>/dev/null || echo "No status files yet"

echo ""
echo "=== DIAGNOSTIC FILES ==="
ls -la "$BASEDIR/diagnostics/"*.md 2>/dev/null || echo "No diagnostic files yet"

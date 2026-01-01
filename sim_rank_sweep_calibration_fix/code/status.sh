#!/bin/bash
# status.sh - Monitor sweep progress

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$SCRIPT_DIR/../logs/sweep.log"
CHECKPOINT="$SCRIPT_DIR/../out/checkpoint.pkl"

echo "========================================"
echo "RANK-1 POWER SWEEP STATUS"
echo "========================================"
echo ""

# Check if process is running
if pgrep -f "sim_sweep.py" > /dev/null; then
    echo "STATUS: RUNNING"
    PID=$(pgrep -f "sim_sweep.py")
    echo "PID: $PID"
    RUNTIME=$(ps -o etime= -p $PID 2>/dev/null | tr -d ' ')
    echo "Runtime: $RUNTIME"
else
    echo "STATUS: NOT RUNNING"
fi
echo ""

# Check log file
if [ -f "$LOG_FILE" ]; then
    echo "--- Last 20 lines of log ---"
    tail -20 "$LOG_FILE"
    echo ""

    # Count completed trials
    echo "--- Progress Summary ---"
    M0_DONE=$(grep -c "M0 @" "$LOG_FILE" 2>/dev/null || echo 0)
    M1_DONE=$(grep -c "M1 @" "$LOG_FILE" 2>/dev/null || echo 0)
    M4_DONE=$(grep -c "M4 dr=" "$LOG_FILE" 2>/dev/null || echo 0)
    echo "M0 conditions completed: $M0_DONE"
    echo "M1 conditions completed: $M1_DONE"
    echo "M4 grid points completed: $M4_DONE"

    # Get latest Type I and Power
    echo ""
    echo "--- Latest Results ---"
    grep -E "Type I|Power \(Wilks\)" "$LOG_FILE" | tail -6
else
    echo "Log file not found: $LOG_FILE"
fi
echo ""

# Check checkpoint
if [ -f "$CHECKPOINT" ]; then
    echo "--- Checkpoint ---"
    ls -la "$CHECKPOINT"
    SIZE=$(stat -c%s "$CHECKPOINT" 2>/dev/null || stat -f%z "$CHECKPOINT" 2>/dev/null)
    echo "Size: $SIZE bytes"
else
    echo "No checkpoint file yet"
fi
echo ""

# Check output files
OUT_DIR="$SCRIPT_DIR/../out"
if [ -d "$OUT_DIR" ]; then
    echo "--- Output Files ---"
    ls -la "$OUT_DIR"/*.md "$OUT_DIR"/*.csv 2>/dev/null || echo "No output files yet"
fi

echo ""
echo "========================================"

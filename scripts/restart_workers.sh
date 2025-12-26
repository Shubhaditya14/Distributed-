#!/bin/bash

# Restart all workers after failure detection
# Workers will automatically load from the last checkpoint

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"
PID_FILE="$PROJECT_ROOT/.pids"

# Activate conda environment
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate myproject 2>/dev/null

if [ ! -f "$PID_FILE" ]; then
    echo "Error: PID file not found. Is the master running?"
    exit 1
fi

# Source the PID file to get MASTER_PID
source "$PID_FILE"

# Check if master is still running
if ! kill -0 "$MASTER_PID" 2>/dev/null; then
    echo "Error: Master is not running. Start the cluster with: ./scripts/run.sh"
    exit 1
fi

# Kill any remaining workers
echo "Killing any remaining workers..."
for PID in "${WORKER_PIDS[@]}"; do
    kill -9 "$PID" 2>/dev/null
done

# Wait for master to enter recovery mode (should already be in recovery if workers died)
echo "Waiting for DDP cleanup..."
sleep 3

# Change to src directory
cd "$SRC_DIR"

# Start 4 new workers
echo "Starting 4 new workers (will resume from checkpoint)..."
NEW_WORKER_PIDS=()
for i in {1..4}; do
    python worker.py &
    NEW_WORKER_PIDS+=($!)
    sleep 0.5
done

# Update PID file with new worker PIDs
echo "MASTER_PID=$MASTER_PID" > "$PID_FILE"
echo "WORKER_PIDS=(${NEW_WORKER_PIDS[@]})" >> "$PID_FILE"

echo ""
echo "New worker PIDs: ${NEW_WORKER_PIDS[@]}"
echo "Workers will load checkpoint and resume training."

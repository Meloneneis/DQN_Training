#!/bin/bash
set -u  # Treat unset variables as an error

# --- Configuration ---
MONITOR_INTERVAL=3  # Seconds to wait between checks
TEMP_FILE="wandb_listener_initial_pids.tmp"

# Clean up the temporary file when the script exits
trap "rm -f $TEMP_FILE; echo 'Listener stopped. Temp file cleaned.'" EXIT

## 1. Capture initial PIDs (The "OLD" PIDs to IGNORE)
echo "--- Starting wandb Process Spawn Listener ---"
echo "Capturing currently running 'wandb' processes. These PIDs will be IGNORED."

# Get PID and CMD, filtering for 'wandb' processes, storing only the PIDs in the temp file.
ps -eo pid,cmd | grep wandb | grep -v grep | awk '{print $1}' > "$TEMP_FILE"

echo "Initial wandb PIDs captured: $(cat "$TEMP_FILE" | tr '\n' ' ')"
echo "------------------------------------------------"
echo "Monitoring for new spawns every ${MONITOR_INTERVAL} seconds... (Press Ctrl+C to stop)"

## 2. Main Monitoring Loop
while true; do

    # Get all current wandb processes (PID, PPID, CMD)
    # The 'while read -r line' construct ensures we process the list line by line.
    ps -eo pid,ppid,cmd | grep wandb | grep -v grep | while read -r line; do

        CURRENT_PID=$(echo "$line" | awk '{print $1}')
        CURRENT_PPID=$(echo "$line" | awk '{print $2}')
        CURRENT_CMD=$(echo "$line" | awk '{$1=$2=""; print $0}' | xargs) # Get command line part

        # Check if the current PID is NOT in our initial ignore list
        if ! grep -q "^${CURRENT_PID}$" "$TEMP_FILE"; then

            echo ""
            echo "🚨 NEW WANDB SPAWN DETECTED! 🚨"
            echo "PID: $CURRENT_PID, PPID: $CURRENT_PPID"
            echo "CMD: $CURRENT_CMD"

            ## 3. Kill the new spawn (PID) and its parent (PPID)
            echo "Attempting to KILL -9 PID $CURRENT_PID and its PPID $CURRENT_PPID..."

            # Kill the PID first (the new wandb process itself)
            if kill -9 "$CURRENT_PID" 2>/dev/null; then
                echo "✅ Successfully killed new process (PID $CURRENT_PID)."
            else
                echo "⚠️ Warning: PID $CURRENT_PID already terminated or kill failed."
            fi

            # Kill the PPID (the parent process that spawned the new agent/run)
            if kill -9 "$CURRENT_PPID" 2>/dev/null; then
                echo "✅ Successfully killed parent process (PPID $CURRENT_PPID)."
            else
                echo "⚠️ Warning: PPID $CURRENT_PPID already terminated or kill failed."
            fi

            # Add the PID to the ignore list to prevent double-processing in the next loop
            # and to handle cases where the kill failed temporarily.
            echo "$CURRENT_PID" >> "$TEMP_FILE"
            echo "------------------------------------------------"
        fi
    done

    ## 4. Wait before checking again
    sleep $MONITOR_INTERVAL
done
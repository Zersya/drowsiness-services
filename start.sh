#!/bin/bash

echo "Starting Landmark-based Drowsiness Detection System..."
echo "Container optimized for CPU-based facial landmark detection"

# Check Python and required packages
python -c "import cv2, dlib, numpy, scipy, flask; print('All required packages available')" || echo "Warning: Some packages may be missing"

# Ensure data directories exist and have the right permissions
mkdir -p /app/data
mkdir -p /app/logs
chmod -R 777 /app/data
chmod -R 777 /app/logs
echo "Data directories created and permissions set"

echo "Starting landmark system with worker support (LANDMARK_MAX_WORKERS=${LANDMARK_MAX_WORKERS:-1})..."

# Set environment variables to optimize for landmark processing
export PYTHONTHREADDEBUG=1
export PYTHONFAULTHANDLER=1

# PID of the landmark process
PID_LANDMARK=""

# Function to handle signals and pass them to the landmark process
function handle_signal() {
    SIGNAL_TYPE=$1 # e.g., SIGINT, SIGTERM
    echo "Received signal $SIGNAL_TYPE. Forwarding to landmark process..."

    # Identify PIDs that were active at the time of signal
    # These are the processes we will attempt to terminate gracefully
    PIDS_TO_SIGNAL_AND_MONITOR=()
    # Check if PID_LANDMARK is set and the process is running
    [ ! -z "$PID_LANDMARK" ] && kill -0 "$PID_LANDMARK" 2>/dev/null && PIDS_TO_SIGNAL_AND_MONITOR+=("$PID_LANDMARK")

    if [ ${#PIDS_TO_SIGNAL_AND_MONITOR[@]} -eq 0 ]; then
        echo "No active landmark process to signal."
        return # Exit the handler
    fi

    echo "Signaling PIDs: ${PIDS_TO_SIGNAL_AND_MONITOR[@]} with signal $SIGNAL_TYPE"
    # Send the received signal (e.g., SIGINT for graceful shutdown) to each process
    for pid in "${PIDS_TO_SIGNAL_AND_MONITOR[@]}"; do
        kill "-$SIGNAL_TYPE" "$pid" 2>/dev/null # Send the signal (e.g. SIGINT, SIGTERM)
    done

    # Wait for processes to finish with a timeout (30 seconds)
    echo "Waiting up to 30 seconds for processes [${PIDS_TO_SIGNAL_AND_MONITOR[@]}] to exit gracefully..."
    for i in {1..30}; do # Loop for 30 seconds
        all_exited_this_iteration=1
        # Check if all processes we signaled have exited
        for pid in "${PIDS_TO_SIGNAL_AND_MONITOR[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then # Check if process is still running
                all_exited_this_iteration=0 # At least one is still running
                break
            fi
        done

        if [ "$all_exited_this_iteration" -eq 1 ]; then
            echo "All signaled processes have exited gracefully."
            break # Exit the 30-second wait loop
        fi
        
        if [ $i -eq 30 ]; then # If loop finishes after 30 seconds
            echo "Timeout reached (30 seconds). Not all processes exited gracefully."
            # The force kill will happen below
            break 
        fi
        
        # Provide an update on which PIDs are still being waited on
        CURRENTLY_RUNNING_PIDS=()
        for pid_check in "${PIDS_TO_SIGNAL_AND_MONITOR[@]}"; do
             if kill -0 "$pid_check" 2>/dev/null; then
                CURRENTLY_RUNNING_PIDS+=("$pid_check")
             fi
        done
        echo "Waiting... ($i/30). Still running: [${CURRENTLY_RUNNING_PIDS[@]}]"
        sleep 1 # Wait for 1 second before checking again
    done

    # After the wait period, force kill any of the initially signaled processes that are still running
    echo "Final check: Force killing any remaining signaled processes..."
    any_force_killed=0
    for pid in "${PIDS_TO_SIGNAL_AND_MONITOR[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then # If process is still running
            echo "Process PID $pid did not exit gracefully, force killing (SIGKILL)..."
            kill -9 "$pid" 2>/dev/null # Force kill with SIGKILL
            any_force_killed=1
        fi
    done
    
    if [ "$any_force_killed" -eq 0 ] && [ "$all_exited_this_iteration" -eq 1 ]; then
        # This means all processes exited within the 30s grace period
        echo "All processes shut down gracefully or were already gone."
    elif [ "$any_force_killed" -eq 1 ]; then
        echo "Some processes were force-killed."
    else
        # This case implies timeout was reached but somehow no processes were running to be force-killed,
        # or they exited between the loop end and this check.
        echo "Processes appear to have exited after the grace period, before force kill was applied, or were not running."
    fi
    # The main script's 'wait' command will handle the final exit synchronization,
    # or the script itself will terminate if the signal was fatal (like an unhandled SIGINT outside a trap).
}

# Set up signal handlers to call the 'handle_signal' function
trap 'handle_signal SIGTERM' SIGTERM # Handle termination signal
trap 'handle_signal SIGINT' SIGINT   # Handle interrupt signal (Ctrl+C)

# Start landmark system process in the background and store its PID
echo "Starting landmark system process..."

echo "Starting landmark system..."
python start_landmark_system.py --port ${LANDMARK_PORT:-8003} --workers ${LANDMARK_MAX_WORKERS:-1} &
PID_LANDMARK=$!
echo "landmark system started with PID: $PID_LANDMARK"


# Wait for landmark process to finish
# Collect PID that was successfully assigned
PIDS_TO_WAIT_FOR=()
[ ! -z "$PID_LANDMARK" ] && PIDS_TO_WAIT_FOR+=("$PID_LANDMARK")

if [ ${#PIDS_TO_WAIT_FOR[@]} -gt 0 ]; then
    echo "Waiting for landmark process to complete: [${PIDS_TO_WAIT_FOR[@]}]"
    # The 'wait' command will block until the specified PID has terminated.
    # If a signal handler terminates it, 'wait' will then return.
    wait "${PIDS_TO_WAIT_FOR[@]}"
    echo "Landmark process has completed."
else
    echo "No landmark process was started or it exited immediately."
fi

echo "Script finished."

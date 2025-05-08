#!/bin/bash

echo "Checking CUDA version..."
echo "Container is configured for CUDA 12.1, compatible with host CUDA 12.4"

# Check for nvcc and print its version if found
nvcc --version || echo "NVCC not found, but container will still run with CPU support"

# Check PyTorch and CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('PyTorch version:', torch.__version__)"

# Ensure data directories exist and have the right permissions
mkdir -p /app/data
mkdir -p /app/logs
chmod -R 777 /app/data
chmod -R 777 /app/logs
echo "Data directories created and permissions set"

echo "Starting application with ThreadPool support (MAX_WORKERS=$MAX_WORKERS)..."

# Set environment variables to optimize for worker threads
export PYTHONTHREADDEBUG=1
export PYTHONFAULTHANDLER=1

# PIDs of the Python processes
PID_DROWSINESS=""
PID_WEBSERVER=""
PID_SIMPLIFY=""

# Function to handle signals and pass them to the Python processes
function handle_signal() {
    SIGNAL_TYPE=$1 # e.g., SIGINT, SIGTERM
    echo "Received signal $SIGNAL_TYPE. Forwarding to Python processes..."

    # Identify PIDs that were active at the time of signal
    # These are the processes we will attempt to terminate gracefully
    PIDS_TO_SIGNAL_AND_MONITOR=()
    # Check if PID_DROWSINESS is set and the process is running
    [ ! -z "$PID_DROWSINESS" ] && kill -0 "$PID_DROWSINESS" 2>/dev/null && PIDS_TO_SIGNAL_AND_MONITOR+=("$PID_DROWSINESS")
    # Check if PID_WEBSERVER is set and the process is running
    [ ! -z "$PID_WEBSERVER" ] && kill -0 "$PID_WEBSERVER" 2>/dev/null && PIDS_TO_SIGNAL_AND_MONITOR+=("$PID_WEBSERVER")
    # Check if PID_SIMPLIFY is set and the process is running
    [ ! -z "$PID_SIMPLIFY" ] && kill -0 "$PID_SIMPLIFY" 2>/dev/null && PIDS_TO_SIGNAL_AND_MONITOR+=("$PID_SIMPLIFY")

    if [ ${#PIDS_TO_SIGNAL_AND_MONITOR[@]} -eq 0 ]; then
        echo "No active Python processes to signal."
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

# Start Python processes in the background and store their PIDs
echo "Starting Python processes..."

echo "Starting drowsiness_detector.py..."
python drowsiness_detector.py &
PID_DROWSINESS=$! # Get PID of the last backgrounded process
echo "drowsiness_detector.py started with PID: $PID_DROWSINESS"

echo "Starting web_server.py..."
python web_server.py &
PID_WEBSERVER=$!
echo "web_server.py started with PID: $PID_WEBSERVER"

echo "Starting simplify.py..."
python simplify.py &
PID_SIMPLIFY=$!
echo "simplify.py started with PID: $PID_SIMPLIFY"


# Wait for all Python processes to finish
# Collect all PIDs that were successfully assigned
PIDS_TO_WAIT_FOR=()
[ ! -z "$PID_DROWSINESS" ] && PIDS_TO_WAIT_FOR+=("$PID_DROWSINESS")
[ ! -z "$PID_WEBSERVER" ] && PIDS_TO_WAIT_FOR+=("$PID_WEBSERVER")
[ ! -z "$PID_SIMPLIFY" ] && PIDS_TO_WAIT_FOR+=("$PID_SIMPLIFY")

if [ ${#PIDS_TO_WAIT_FOR[@]} -gt 0 ]; then
    echo "Waiting for all Python processes to complete: [${PIDS_TO_WAIT_FOR[@]}]"
    # The 'wait' command will block until all specified PIDs have terminated.
    # If a signal handler terminates them, 'wait' will then return.
    wait "${PIDS_TO_WAIT_FOR[@]}"
    echo "All Python processes have completed."
else
    echo "No Python processes were started or they exited immediately."
fi

echo "Script finished."

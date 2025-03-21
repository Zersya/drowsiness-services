#!/bin/bash

# Directory and virtual environment
WORK_DIR="$HOME/drowsiness-services"
VENV_PATH="$WORK_DIR/.venv/bin/activate"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "tmux not found. Installing..."
    sudo apt install tmux -y
fi

# Function to start a service in a tmux session
start_service() {
    SESSION_NAME=$1
    COMMAND=$2
    tmux has-session -t "$SESSION_NAME" 2>/dev/null
    if [ $? != 0 ]; then
        echo "Starting $SESSION_NAME..."
        tmux new-session -d -s "$SESSION_NAME"
        tmux send-keys -t "$SESSION_NAME" "cd $WORK_DIR" C-m
        tmux send-keys -t "$SESSION_NAME" "source $VENV_PATH" C-m
        tmux send-keys -t "$SESSION_NAME" "$COMMAND" C-m
    else
        echo "$SESSION_NAME already running."
    fi
}

# Define services
start_service "service1" "python drowsiness_detector.py"
start_service "service2" "python web_server.py"
start_service "service3" "cloudflared tunnel run"

echo "All services started. Use 'tmux ls' to see sessions."
echo "To attach: 'tmux attach -t service1' (or service2, service3)"
echo "To detach: Ctrl+B, then D"
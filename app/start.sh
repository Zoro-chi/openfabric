#!/bin/bash

# Function to start the server
start_server() {
    echo "Starting server on port 8888..."
    python ignite.py
}

# Main execution
start_server

# Keep the script running
echo "Server started. Press Ctrl+C to stop."
while true; do
    sleep 60
done
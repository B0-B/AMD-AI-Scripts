#!/bin/bash

SERVICE_NAME="vbench" # The name defined in your docker-compose.yml

# Print Start Message
echo "[$(date +'%T')] vbench: Starting Docker Benchmarking Pipeline..."

# Ensure results directory exists
mkdir -p ./results
touch ./results/benchmark.log

# Start the container
docker compose run --rm vbench

echo "[vbench] Streaming logs from ./results/benchmark.log ..."
tail -f ./results/benchmark.log & 
TAIL_PID=$!

# Wait for the container to finish
echo "[$(date +'%T')] MONITORING: Waiting for script to complete..."
while [ $(docker compose ps --status running -q ${SERVICE_NAME}) ]; do
    # Optional: print a dot or status every 5 seconds to show it's working
    sleep 5
done

# Kill the tail process
kill $TAIL_PID

# Print Stop Message
echo "------------------------------------------"
echo "[$(date +'%T')] STOPPED: Container has terminated."
echo "Final results available in ./results/"
echo "------------------------------------------"

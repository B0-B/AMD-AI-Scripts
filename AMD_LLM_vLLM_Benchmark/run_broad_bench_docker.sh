#!/bin/bash

SERVICE_NAME="vbench" # The name defined in your docker-compose.yml

# 1. Print Start Message
echo "[$(date +'%T')] vbench: Starting Docker Benchmarking Pipeline..."

# 2. Ensure results directory exists
mkdir -p ./results

# 3. Start the container in background
docker compose up

# 4. Wait for the container to finish
echo "[$(date +'%T')] MONITORING: Waiting for script to complete..."
while [ $(docker compose ps --status running -q ${SERVICE_NAME}) ]; do
    # Optional: print a dot or status every 5 seconds to show it's working
    sleep 5
done

# 5. Print Stop Message
echo "------------------------------------------"
echo "[$(date +'%T')] STOPPED: Container has terminated."
echo "Final results available in ./results/"
echo "------------------------------------------"

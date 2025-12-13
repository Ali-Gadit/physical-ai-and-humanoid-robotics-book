#!/bin/bash

# Kill all background jobs on exit
trap 'kill $(jobs -p)' EXIT

# Function to wait for a port to be open
wait_for_port() {
  local port=$1
  local timeout=60
  echo "Waiting for port $port to be open..."
  for i in $(seq 1 $timeout); do
    nc -z localhost $port >/dev/null 2>&1
    if [ $? -eq 0 ]; then
      echo "Port $port is open."
      return 0
    fi
    sleep 1
  done
  echo "Error: Port $port did not open within $timeout seconds."
  return 1
}

echo "Starting User Auth Service (Port 3000)..."
(cd user-auth && npm run dev) &
PID_AUTH=$!
wait_for_port 3000 || exit 1

echo "Starting Backend API (Port 8000)..."
(cd backend && venv/bin/python -m uvicorn src.api.main:app --reload --port 8000) &
PID_BACKEND=$!
wait_for_port 8000 || exit 1

echo "Starting Docusaurus (Port 3001)..."
(cd docusaurus-book && npm start -- --port 3001 --no-open) &
PID_FRONTEND=$!
wait_for_port 3001 || exit 1

echo "Services are started!"
echo "--------------------------------------------------"
echo "User Auth Service: http://localhost:3000"
echo "Backend API:       http://localhost:8000"
echo "Docusaurus Book:   http://localhost:3001"
echo "--------------------------------------------------"
echo "Press Ctrl+C to stop all services."

wait
#!/usr/bin/env bash
set -euo pipefail

PORT=5000
NEW_PORT=5001
COMPOSE_FILE="docker-compose.yml"
BACKUP="${COMPOSE_FILE}.bak"

usage(){ echo "Usage: $0 [kill|rebind]"; exit 1; }

if [ $# -ne 1 ]; then usage; fi
ACTION=$1

pids=$(lsof -nP -iTCP:${PORT} -sTCP:LISTEN -t || true)

if [ -z "$pids" ]; then
  echo "port ${PORT} free"
else
  echo "port ${PORT} in use by PIDs: $pids"
  ps -p $pids -o pid,user,comm,args || true
fi

if [ "$ACTION" = "kill" ]; then
  if [ -z "$pids" ]; then
    echo "nothing to kill"
  else
    echo "killing $pids"
    sudo kill -9 $pids
    sleep 1
    lsof -nP -iTCP:${PORT} -sTCP:LISTEN || echo "port ${PORT} freed"
  fi
elif [ "$ACTION" = "rebind" ]; then
  if [ ! -f "$COMPOSE_FILE" ]; then echo "compose file not found: $COMPOSE_FILE"; exit 1; fi
  cp "$COMPOSE_FILE" "$BACKUP"
  if sed --version >/dev/null 2>&1; then
    sed -i 's/5000:5000/'"${NEW_PORT}"':5000/g' "$COMPOSE_FILE"
  else
    sed -i '' 's/5000:5000/'"${NEW_PORT}"':5000/g' "$COMPOSE_FILE"
  fi
  echo "updated $COMPOSE_FILE (backup at $BACKUP)"
else
  usage
fi

echo "Starting docker compose..."
docker compose up -d
docker compose ps

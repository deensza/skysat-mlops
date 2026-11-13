#!/bin/bash
# Self-healing on the server (docs/server.md), run by cron every 5 minutes: after 3 failed health checks in a row,
# restart the API. A failed check is logged even when nothing is restarted, so a stall can be found afterwards.
state=/var/tmp/skysat-watchdog-fails
fails=$(cat "$state" 2>/dev/null || echo 0)
project="label=com.docker.compose.project=skysat-mlops"

# Anything of the stack that has stopped (Nginx, Caddy, Grafana, ...) is started again right away
stopped=$(docker ps -aq --filter "$project" --filter status=exited)
if [ -n "$stopped" ]; then
  echo "$(date -u +%FT%TZ) starting stopped containers"
  docker start $stopped
fi

if curl -fs --max-time 25 -o /dev/null http://127.0.0.1:8080/health; then
  echo 0 > "$state"
  exit 0
fi

fails=$((fails + 1))
echo "$(date -u +%FT%TZ) health check failed ($fails)"
if [ "$fails" -ge 3 ]; then
  echo "$(date -u +%FT%TZ) restarting the API"
  docker restart $(docker ps -aq --filter "$project" --filter label=com.docker.compose.service=api)
  fails=0
fi
echo "$fails" > "$state"

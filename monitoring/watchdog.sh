#!/bin/bash
# Self-healing on the server (docs/server.md), run by cron every 5 minutes: after 3 failed health checks in a row,
# restart the API. A failed check is logged even when nothing is restarted, so a stall can be found afterwards.
# A 503 from /health is not a failure: the API is alive but has no model yet and loads it again by itself, so a
# restart would help nothing. Only no answer at all (000) or an error from Nginx (502, 504) counts.
# Pause switch: while the file /var/tmp/skysat-watchdog-off exists the script does nothing. `touch` it before stopping
# a container on purpose (the script would start it again), `rm` it to resume.
# WATCHDOG_STATE, WATCHDOG_OFF and WATCHDOG_URL replace the three paths and the URL below, for the test.
state=${WATCHDOG_STATE:-/var/tmp/skysat-watchdog-fails}
off=${WATCHDOG_OFF:-/var/tmp/skysat-watchdog-off}
url=${WATCHDOG_URL:-http://127.0.0.1:8080/health}

if [ -e "$off" ]; then
  echo "$(date -u +%FT%TZ) paused ($off exists), doing nothing"
  exit 0
fi

fails=$(cat "$state" 2>/dev/null || echo 0)
project="label=com.docker.compose.project=skysat-mlops"

# Anything of the stack that has stopped (Nginx, Caddy, Grafana, ...) is started again right away
stopped=$(docker ps -aq --filter "$project" --filter status=exited)
if [ -n "$stopped" ]; then
  echo "$(date -u +%FT%TZ) starting stopped containers"
  docker start $stopped
fi

code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 25 "$url")
if [ "$code" = 200 ]; then
  echo 0 > "$state"
  exit 0
fi
if [ "$code" = 503 ]; then
  echo "$(date -u +%FT%TZ) the API answers but has no model yet"
  echo 0 > "$state"
  exit 0
fi

fails=$((fails + 1))
echo "$(date -u +%FT%TZ) health check failed ($fails, status $code)"
if [ "$fails" -ge 3 ]; then
  echo "$(date -u +%FT%TZ) restarting the API"
  docker restart $(docker ps -aq --filter "$project" --filter label=com.docker.compose.service=api)
  fails=0
fi
echo "$fails" > "$state"

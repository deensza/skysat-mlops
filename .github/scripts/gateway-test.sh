#!/usr/bin/env bash
# The public way into the server (docs/server.md): Caddy -> Nginx with its login -> API, intake and Grafana.
# smoke-test.sh only calls the API on the server's own port, with no login. So the login, the queue's rule and
# Grafana could all be broken while a deploy was green: on 18 Nov 2026 every Grafana panel answered "permission
# denied for table llm_scores" and nothing noticed.
# Usage: SMOKE_LOGIN=user:password bash .github/scripts/gateway-test.sh https://<server>
# The login is any gateway login (the drift check's `github` is meant). It changes nothing on the server.
set -euo pipefail
url="${1:?the server, e.g. https://<server>}"
: "${SMOKE_LOGIN:?set SMOKE_LOGIN=user:password (a gateway login)}"

# status [curl arguments]: the HTTP status of one request sent with the login (kept out of the process list)
status() { curl -s -o /dev/null -w '%{http_code}' --max-time 30 -K <(printf 'user = "%s"\n' "$SMOKE_LOGIN") "$@" || true; }
expect() {  # expect <what> <wanted status> <got status>
  if [ "$3" = "$2" ]; then echo "ok    $1 ($3)"; else echo "FAIL  $1: wanted $2, got $3"; failed=1; fi
}
failed=0

expect "no login is refused" 401 "$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 "$url/health" || true)"
expect "the login reaches the API" 200 "$(status "$url/health")"
if [ "${SMOKE_LOGIN%%:*}" != trainer ]; then
  expect "the queue is closed to this login" 403 "$(status "$url/queue/stats")"
fi
expect "Grafana answers" 200 "$(status "$url/grafana/api/health")"
# The query a panel sends: Grafana's read-only database user reading the prediction log
expect "Grafana can read the prediction log" 200 "$(status -X POST "$url/grafana/api/ds/query" \
  -H "Content-Type: application/json" -d '{"from": "now-1h", "to": "now", "queries": [{"refId": "A",
  "datasource": {"uid": "skysat-pg", "type": "grafana-postgresql-datasource"}, "format": "table",
  "rawSql": "SELECT count(*) FROM llm_scores"}]}')"

[ "$failed" = 0 ] || { echo "Gateway test FAILED"; exit 1; }
echo "Gateway test passed"

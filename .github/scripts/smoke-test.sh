#!/usr/bin/env bash
# Smoke test after a deploy: the API answers /health, /model and POST /score through Nginx.
# Usage: bash .github/scripts/smoke-test.sh [base url, default http://localhost:8080] [tries, 3 s apart, default 60]
# Needs a champion adapter (make llm-train) and the base model loaded. A review on CPU takes seconds.
# The review below is made up (not a dataset row).
set -euo pipefail
url="${1:-http://localhost:8080}"
tries="${2:-60}"

echo "Waiting for $url/health (needs MLflow up, a champion adapter and the base model loaded) ..."
for _ in $(seq "$tries"); do
  if curl -fs "$url/health"; then echo; break; fi
  sleep 3
done
curl -fsS "$url/health" > /dev/null || { echo "FAIL: /health never answered 200"; exit 1; }

curl -fsS "$url/model"; echo
answer=$(curl -fsS --max-time 120 -X POST "$url/score" -H "Content-Type: application/json" -d '{
  "airline": "Imaginary Air", "cabin": "Economy Class", "travel_type": "Solo Leisure", "title": "Cold food",
  "review": "The crew were friendly but the meal was cold and the seat was cramped on a six hour flight."}')
echo "$answer"
grep -q '"model_version"' <<< "$answer" || { echo "FAIL: /score gave no model_version"; exit 1; }
grep -q '"score"' <<< "$answer" || { echo "FAIL: /score gave no score"; exit 1; }
echo "Smoke test passed"

#!/usr/bin/env bash
# Smoke test after a deploy (step 8): the API answers /health and /predict through Nginx.
# Usage: bash .github/scripts/smoke-test.sh [base url, default http://localhost:8080]
# The passenger below is made up (not a dataset row).
set -euo pipefail
url="${1:-http://localhost:8080}"

echo "Waiting for $url/health (needs MLflow up and a champion model) ..."
for _ in $(seq 60); do
  if curl -fs "$url/health"; then echo; break; fi
  sleep 3
done
curl -fsS "$url/health" > /dev/null || { echo "FAIL: /health never answered 200"; exit 1; }

curl -fsS "$url/model"; echo
answer=$(curl -fsS -X POST "$url/predict" -H "Content-Type: application/json" -d '{
  "passenger_id": 1, "gender": "Female", "customer_type": "Loyal Customer", "type_of_travel": "Business travel",
  "travel_class": "Business", "age": 40, "flight_distance": 1800, "departure_delay_minutes": 0,
  "arrival_delay_minutes": 0, "inflight_wifi_service": 4, "departure_arrival_time_convenient": 4,
  "ease_of_online_booking": 4, "gate_location": 3, "food_and_drink": 4, "online_boarding": 5, "seat_comfort": 5,
  "inflight_entertainment": 5, "on_board_service": 4, "leg_room_service": 4, "baggage_handling": 4,
  "checkin_service": 4, "inflight_service": 4, "cleanliness": 5}')
echo "$answer"
grep -q '"probability"' <<< "$answer" || { echo "FAIL: /predict gave no probability"; exit 1; }
echo "Smoke test passed"

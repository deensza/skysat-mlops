// Step 5 – k6 load test for POST /predict (through Nginx).
// Passengers are random but valid (same value ranges as skysat/schema.py); no dataset rows are used.
//
//   docker compose --profile loadtest run --rm --no-deps k6                  # 50 users, 30 s
//   docker compose --profile loadtest run --rm --no-deps -e VUS=100 k6
// --no-deps matters: without it `run` also re-applies replicas: 3 to the API and spoils a 1-copy test.
import http from 'k6/http';
import { check } from 'k6';

const BASE_URL = __ENV.BASE_URL || 'http://nginx';

export const options = {
  vus: Number(__ENV.VUS || 50),
  duration: __ENV.DURATION || '30s',
  thresholds: {
    http_req_failed: ['rate<0.01'],     // fewer than 1% errors
    http_req_duration: ['p(95)<500'],   // 95% of requests under 500 ms
  },
};

const CATEGORICAL = {
  gender: ['Male', 'Female'],
  customer_type: ['Loyal Customer', 'disloyal Customer'],
  type_of_travel: ['Personal Travel', 'Business travel'],
  travel_class: ['Business', 'Eco', 'Eco Plus'],
};
const RATINGS = [
  'inflight_wifi_service', 'departure_arrival_time_convenient', 'ease_of_online_booking',
  'gate_location', 'food_and_drink', 'online_boarding', 'seat_comfort', 'inflight_entertainment',
  'on_board_service', 'leg_room_service', 'baggage_handling', 'checkin_service',
  'inflight_service', 'cleanliness',
];

const randInt = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
const pick = (values) => values[randInt(0, values.length - 1)];

function passenger() {
  const p = { passenger_id: randInt(1, 1000000) };
  for (const [col, values] of Object.entries(CATEGORICAL)) p[col] = pick(values);
  p.age = randInt(7, 85);
  p.flight_distance = randInt(31, 4983);
  p.departure_delay_minutes = Math.random() < 0.6 ? 0 : randInt(1, 180);
  p.arrival_delay_minutes = Math.max(0, p.departure_delay_minutes + randInt(-10, 10));
  for (const col of RATINGS) p[col] = randInt(0, 5);
  return p;
}

export default function () {
  const res = http.post(`${BASE_URL}/predict`, JSON.stringify(passenger()), {
    headers: { 'Content-Type': 'application/json' },
  });
  check(res, {
    'status 200': (r) => r.status === 200,
    'has probability': (r) => r.status === 200 && r.json('probability') !== undefined,
  });
}

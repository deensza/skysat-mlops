// k6 load test for POST /score (through Nginx). The model answers one review at a time and takes seconds on CPU,
// so this is a gentle test: 1 user for 1 minute by default, not 50 users. It checks that the queue works and
// that nothing fails, not that it is fast.
//
//   docker compose --profile loadtest run --rm --no-deps k6
//   docker compose --profile loadtest run --rm --no-deps -e VUS=2 -e DURATION=2m k6
// The reviews below are made up (no dataset text). They carry no true_score, so they do not move the live MAE.
import http from 'k6/http';
import { check } from 'k6';

const BASE_URL = __ENV.BASE_URL || 'http://nginx';

export const options = {
  vus: Number(__ENV.VUS || 1),
  duration: __ENV.DURATION || '1m',
  thresholds: {
    http_req_failed: ['rate<0.01'],      // fewer than 1% errors
    http_req_duration: ['p(95)<30000'],  // 95% under 30 s (CPU generation + waiting in the queue)
  },
};

const REVIEWS = [
  ['Cold food', 'The crew were friendly but the meal was cold and the seat was cramped for a six hour flight.'],
  ['Smooth flight', 'Boarding was quick, the cabin was clean and we landed early. Would fly again.'],
  ['Never again', 'Our bags were lost, nobody at the ground desk could help, and the wifi did not work at all.'],
  ['Fine for the price', 'Nothing special: small screen, ok legroom, snacks were average. The price was fair.'],
];

export default function () {
  const [title, review] = REVIEWS[Math.floor(Math.random() * REVIEWS.length)];
  const body = { airline: 'Imaginary Air', cabin: 'Economy Class', travel_type: 'Solo Leisure', title, review };
  const res = http.post(`${BASE_URL}/score`, JSON.stringify(body), {
    headers: { 'Content-Type': 'application/json' },
    timeout: '120s',
  });
  check(res, {
    'status 200': (r) => r.status === 200,
    'has model_version': (r) => r.status === 200 && r.json('model_version') !== undefined,
  });
}

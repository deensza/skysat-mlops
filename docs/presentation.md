# SkySat – Presentation

SE4458 Assignment 1 · Topic: **MLOps – build a simple ML model and demonstrate its complete lifecycle from
training to production** · Team: Bermal Deniz Akkaya, Osman Şahin Güler · Code: https://github.com/deensza/skysat-mlops

What the assignment asks for, and where it is answered:

| Requirement (Demo Instructions) | Where |
|---|---|
| Motivation for the topic | slides 2-3 |
| Architecture | slides 4-5, [architecture.md](architecture.md) |
| Brief document / presentation with the public GitHub link | this file (slides are built from it) |
| A RUNNING code example | slide 10 + the video, [demo/video.md](../demo/video.md) |

Open points (tick when done):

- [ ] Build the slides from this file (one `##` = one slide; speaker notes under each)
- [ ] Re-measure the numbers on the final stack before presenting; the server numbers (slide 8) are not measured yet
- [ ] Record the video ([demo/video.md](../demo/video.md)) and put its link on slide 10
- [ ] Enable the two schedules (`traffic.yml`, `drift.yml`) for the final run, and say so on slide 9
- [ ] Upload slides (with the GitHub link) and code to Moodle
- [ ] Rehearse once with a timer (target: ~10 min slides + 3 min video + questions)

---

## 1. SkySat: a complaint box for airline reviews, from notebook to production

- An LLM reads a **real airline review** and answers: score out of 10, recommended yes/no, main complaint
- The model is the easy part; the project is **everything around it**
- Public code: github.com/deensza/skysat-mlops

> Notes: introduce the team and the one-sentence goal. The topic is the lifecycle, not the algorithm.

## 2. Motivation: why do we need MLOps?

- A model in a notebook helps nobody; it must be **served, watched and replaced**
- Typical failures without MLOps: "it worked on my machine", nobody knows which model is in production, the world
  changes (**drift**) and accuracy drops silently, retraining is manual so it does not happen
- MLOps = DevOps ideas (automation, CI/CD, monitoring) **plus** data and models as versioned things

> Notes: one example: a model tuned on pre-2019 reviews meets a survey whose raters are harsher.

## 3. Why an LLM changes the picture (and what stays the same)

- Same lifecycle: data, train, gate, register, promote, serve, monitor, retrain
- Different: the 3 GB base model is **frozen**; we train a **17 MB LoRA adapter** (~0.3% of the weights), so a "new
  model" is a small download
- Training needs a **GPU** (5 min on an RTX 4070) so it is not on GitHub; serving is on CPU and takes seconds per review

## 4. The lifecycle we built

| Stage | Tool |
|---|---|
| Data | Kaggle *128k airline reviews* (DbCL-1.0), split by date and airline |
| Training + gate | LoRA (r=16, 2,000 reviews) on Qwen2.5-1.5B-Instruct; gate on MAE, recommended, complaint |
| Tracking + registry | **MLflow** (`skysat-reviews@champion`) |
| Packaging | **Docker**, Docker Compose |
| Serving | **FastAPI** `POST /score`, behind **Nginx** |
| Monitoring | **PostgreSQL** `llm_scores` + **Grafana** |
| Drift | live MAE and recommended accuracy against the gate (`llm/drift_check.py`) |
| CI / CD | **GitHub Actions**: tests, image, deploy; hourly replay and drift check |

## 5. Architecture

See the diagram in [architecture.md](architecture.md): GPU PC → MLflow (adapter) → API on CPU → PostgreSQL → Grafana;
GitHub replays traffic, checks drift and opens an issue; a person retrains.

> Notes: two loops: the training loop (GPU → MLflow) and the serving loop (traffic → API → log → Grafana). Drift
> detection connects them. Monitoring stores numbers, never review text.

## 6. Training, the gate and the champion

| Adapter | Reviews | MAE | Recommended | Gate |
|---|---|---|---|---|
| v1 (2,000 reviews flown <= 2018) | validation | 0.91 | 94.5% (complaint 68%) | passes: champion |
| v1 | real 2019-2022 | 0.75-0.83 | | passes |
| v1 | 2019, "harsh" survey | 1.18 | 89.5% | **fails** |
| v2 (retrained on harsh 2019) | harsh 2019 | 0.41 | 95.5% | passes, promoted |

- Gate limits: MAE <= 1.0, recommended >= 0.90, complaint >= 0.65; a new adapter must also beat the champion on the same reviews
- Real 2019-2022 drift does **not** hurt the model, so the drift is simulated (`simulator/scenarios.py`)

## 7. Serving and monitoring

- `POST /score` → score, recommended, complaint, model version, latency; one review at a time on CPU
- A new champion is a new adapter loaded next to the old one on the same base model: no restart
- Grafana: requests per hour, latency p50/p95, live MAE and recommended accuracy (where true labels were sent),
  complaint shares, model version over time
- `GET /scores/recent`: aggregates only

## 8. The server

- One always-on VM (>= 8 GB RAM): Caddy (HTTPS), Nginx (rate limit, login), `api`, PostgreSQL, Grafana ([server.md](server.md))
- Server measurements with the LLM: *(not measured yet: restart time, latency per review)*

## 9. CI / CD and the automatic loop

- **CI**: tests (no GPU, no model), compose files valid, image built without any model download
- **CD**: on `main` only, never for pull requests; image to ghcr.io, deploy on the server's runner once it is switched on
- **traffic.yml** replays live reviews (optionally `harsh`); **drift.yml** checks them and opens the issue "Drift detected:
  retrain the LoRA adapter (make llm-cycle)". Schedules written, enabled for the final run
- **Retraining is not on GitHub**: no GPU. A person runs `make llm-cycle`

## 10. Live demo (video, ~3 min)

1. MLflow: v1 champion and its gate scores
2. One review through Nginx; Grafana with normal traffic
3. `harsh` traffic → live MAE and accuracy cross the limits → `make llm-drift` says DRIFT
4. Retrain on harsh 2019 → v2 promoted → the API loads it → Grafana recovers

Video: *(link)* · Script: [demo/video.md](../demo/video.md)

## 11. What we learned

- The gate limits come from measured runs, not guesses (constant answers score MAE 1.2-3.2)
- A registry version must be loadable by the production code: the API refuses an adapter for another base model or
  prompt version and keeps the old one
- Real drift did not hurt, so we measured that and simulated the rest instead of claiming a result
- Monitoring that stores no text is enough to see the model degrade

## 12. Summary and questions

- One model, the whole lifecycle: LoRA training, gate, registry, API, monitoring, drift check, CI/CD
- Reproduce: `make install && make test`, then `make llm-cycle` on a GPU, `make d-up`
- Data: Kaggle joelljungstrom/128k-airline-reviews (DbCL-1.0), scraped from airlinequality.com by the dataset author; never committed

### Likely questions (prepare an answer)

- Why LoRA and not fine-tuning everything? (0.3% of the weights: 5 minutes on one GPU, a 17 MB file to promote and ship)
- Why does the gate have those limits? (measured: v1 passes, constant answers fail, v1 under the harsh survey fails; `llm/gate.py`)
- Real drift did not hurt, so is the drift demo fake? (The scenario is simulated and we say so; the monitoring and the retrain path are real)
- What happens if MLflow is down? (the API keeps the adapter it has and retries; a logging failure never breaks serving)
- How do you stop a worse adapter from going to production? (the gate, and it must beat the champion on the same reviews)
- Why is retraining not automatic? (it needs a GPU, which GitHub's machines do not have; the drift check opens an issue)
- Where does production run, and what does it store? (an always-on VM, [server.md](server.md); numbers only, never review text)

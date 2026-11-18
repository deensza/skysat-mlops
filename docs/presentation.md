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
- [ ] Re-measure the numbers on the final stack before presenting; the server numbers (slide 8) are partly measured
      (2026-11-16, docs/server.md), the restart time is not
- [ ] Record the video ([demo/video.md](../demo/video.md)) and put its link on slide 10
- [ ] The repository owner adds the secrets and variables ([server.md](server.md#switching-it-on))
- [ ] Run the loop once for real (drift.yml, then train.yml) and note the date on slide 9
- [ ] Freeze it before presenting: `AUTO_RETRAIN` and `DEPLOY_TO_RUNNER` to `false` ([server.md](server.md))
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
- Training needs a **GPU** (5 min on an RTX 4070) so it is not on GitHub's machines (a workflow rents one); serving is on CPU and takes seconds per review

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
| CI / CD | **GitHub Actions**: tests, image, deploy; hourly drift check; a training workflow (built, not yet run) |

## 5. Architecture

See the diagram in [architecture.md](architecture.md): GPU PC → MLflow (adapter) → API on CPU → PostgreSQL → Grafana;
GitHub checks drift and opens an issue; the Train workflow can rent the GPU (built, not yet switched on).

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
- Server measurements with the LLM (2026-11-16): a review takes about one minute in the live loop; restart time: *(not measured yet)*

## 9. CI / CD and the automatic loop

- **CI**: tests (no GPU, no model), `nginx -t` on both gateway configs, compose files valid, image built without any
  model download; `tests/test_pipeline.py` runs the real trainer with a stub for the GPU step against a real PostgreSQL
- **CD**: on `main` only, never for pull requests; image to ghcr.io, deploy on the server's runner once it is switched
  on; if the smoke test fails it goes back to the running image (the image only)
- **drift.yml** runs hourly, judges only the serving model, opens one issue on DRIFT and, only if `AUTO_RETRAIN` is
  `true`, calls **train.yml**
- **train.yml** has a cooldown (default 6 h), rents nothing if fewer than 300 reviews are pending, trains on the cheapest
  free RunPod GPU for at most 45 minutes (at most about $0.26), then checks that no pod is left
- Built, but **not yet switched on** (the repository owner adds the secrets) and **not yet run on GitHub**

> Notes: say plainly that the loop is code and configuration; it has not run on GitHub yet. A full training run on
> RunPod has not been done yet either (a boot test pod has).

## 9b. MLOps level 2, in small

- Level 0: every step by hand. Level 1: training is a pipeline that runs by itself (continuous training). Level 2: the
  pipeline's own code is also built, tested and deployed automatically
- SkySat: queue -> trainer -> gate -> registry (level 1); `ci.yml` and `drift.yml` -> `train.yml` (level 2)
- Missing for a real level 2: an orchestrator such as Kubeflow or Airflow (GitHub Actions plays that part), a feature
  store, data validation beyond the intake's checks, a staging environment
- The loop is built but must be switched on by the repository owner

> Notes: the three levels are from Google's "MLOps: Continuous delivery and automation pipelines in machine learning".
> Detail: [architecture.md](architecture.md).

## 10. Live demo (video, ~3 min)

1. MLflow: v1 champion and its gate scores
2. One review through Nginx; Grafana with normal traffic
3. `harsh` traffic → live MAE and accuracy cross the limits → `make llm-drift` says DRIFT
4. Retrain on harsh 2019 → v2 promoted → the API loads it → Grafana recovers

Videos: [server and retraining loop](https://drive.google.com/file/d/1Ulbq2-DxDoxo9LLP3ISe-qSe07x_d64b/view) · [Kaggle competition, the first model](https://drive.google.com/file/d/1dPYeIG6mb0o7lfYjoIIgoqsWV4Z0CbQd/view) · Script: [demo/video.md](../demo/video.md)

## 11. What we learned

- The gate limits come from measured runs, not guesses (constant answers score MAE 1.2-3.2)
- A registry version must be loadable by the production code: the API refuses an adapter for another base model or
  prompt version and keeps the old one
- Real drift did not hurt, so we measured that and simulated the rest instead of claiming a result
- Monitoring that stores no text is enough to see the model degrade

## 12. Summary and questions

- One model, the whole lifecycle: LoRA training, gate, registry, API, monitoring, drift check, CI/CD
- Reproduce: `make install && make test`, then `make train-remote` (or `make llm-train` on a local GPU), `make d-up`
- Data: Kaggle joelljungstrom/128k-airline-reviews (DbCL-1.0), scraped from airlinequality.com by the dataset author; never committed

### Likely questions (prepare an answer)

- Why LoRA and not fine-tuning everything? (0.3% of the weights: 5 minutes on one GPU, a 17 MB file to promote and ship)
- Why does the gate have those limits? (measured: v1 passes, constant answers fail, v1 under the harsh survey fails; `llm/gate.py`)
- Real drift did not hurt, so is the drift demo fake? (The scenario is simulated and we say so; the monitoring and the retrain path are real)
- What happens if MLflow is down? (the API keeps the adapter it has and retries; a logging failure never breaks serving)
- How do you stop a worse adapter from going to production? (the gate, and it must beat the champion on the same reviews)
- Is retraining automatic? (built: a drift alarm can start the Train workflow, which rents a RunPod GPU; it needs `AUTO_RETRAIN=true` and the secrets, and has not run on GitHub yet)
- Where does production run, and what does it store? (an always-on VM, [server.md](server.md); numbers only, never review text)

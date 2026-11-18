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
- [x] The repository owner adds the secrets and variables ([server.md](server.md#switching-it-on)): done 17 Nov 2026
- [x] Run each part once for real: done 17 Nov 2026 (drift.yml by hand; train.yml by hand on a RunPod RTX 3070)
- [x] The automatic chain end to end: done 17 Nov 2026, 15:09 UTC (drift alarm -> Train -> RTX 3080 Ti -> version 6
      promoted); the drift check itself was started by hand
- [ ] GitHub's hourly schedule started no drift check on 17 Nov until after 17:35 UTC; two scheduled runs came later
      (21:53 UTC on 17 Nov, 03:49 UTC on 18 Nov) and failed because the server was off: find out why slots are missing,
      or start the check by hand
- [x] Version 1 serves again: the champion alias was moved back from version 6 by hand on 17 Nov 2026 (by 17:12
      UTC). Version 6 was trained on drifted reviews only and was not measured on healthy ones
- [ ] Check `GET /model` before presenting: a later training can move the alias again
      (`make llm-promote VERSION=1 SHARED=yes` moves it back)
- [ ] Freeze it before presenting: `AUTO_RETRAIN` to `false` ([server.md](server.md)); the deploy job is already off
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
| Data | Kaggle *128k airline reviews* (DbCL-1.0), split by date and airline; a simulator posts them as passengers |
| Intake + queue | **FastAPI** `POST /reviews`, **PostgreSQL** `queue.reviews`; each live review is scored first, trained on after |
| Training + gate | LoRA (r=16) on Qwen2.5-1.5B-Instruct, on a rented **RunPod** GPU; gate on MAE, recommended, complaint |
| Tracking + registry | **MLflow** (`skysat-reviews@champion`) |
| Packaging | **Docker**, Docker Compose |
| Serving | **FastAPI** `POST /score`, behind **Nginx** |
| Monitoring | **PostgreSQL** `llm_scores` + **Grafana** |
| Drift | live MAE and recommended accuracy against the gate (`llm/drift_check.py`) |
| CI / CD | **GitHub Actions**: tests, image (the deploy job ran once, on 13 Nov 2026, and is off since); a drift check that calls the training workflow, which rents the GPU (on since 17 Nov 2026) |

## 5. Architecture

See the diagram in [architecture.md](architecture.md): simulator → intake → queue → trainer on a rented RunPod GPU →
gate → MLflow champion → API on CPU → PostgreSQL → Grafana. GitHub checks drift every hour, opens an issue and calls
the Train workflow, which rents the GPU.

> Notes: two loops: the training loop (queue → rented GPU → gate → MLflow) and the serving loop (reviews → API → log →
> Grafana). Drift detection connects them. Monitoring stores numbers, never review text. The server has no GPU and
> GitHub has none either: the GPU exists only while a pod trains.

## 6. Training, the gate and the champion

Measured through the server, 16 and 17 Nov 2026 ([llm.md](llm.md#through-the-servers-queue-16-and-17-nov-2026)):

| What | Reviews | MAE | Recommended | Result |
|---|---|---|---|---|
| Version 1, live: healthy reviews | 100 | 0.950 | 95.0% | drift check: no drift |
| Version 1, live: "harsh" survey | 300 | 1.150 | 89.0% | drift check: **DRIFT** |
| Retraining 1: new reviews + 500 replayed old ones | 100 held back | 1.07 | 87.0% | **fails the gate**, not registered |
| Retraining 2: the new reviews only, 3 epochs | 100 held back | 0.60 | 93.0% | passes; champion on the same reviews 1.13, 90.0%; becomes **version 4** |
| Version 4, live: new harsh reviews | 165 | 0.551 | 97.0% | drift check: no drift |
| RunPod by hand, 17 Nov (RTX 3070, about 5 min, about one cent) | 100 held back | 0.33 | 93.0% | passes, registered as version 5; combined 0.858 against the champion's 0.861, so version 4 stays |
| Version 4, live, after reviews with a stronger label shift | 591 | 1.230 | 81.2% | drift check: **DRIFT**, and it calls the Train workflow |
| RunPod by the drift alarm, 17 Nov (RTX 3080 Ti, 5.5 min, about 2 cents) | 100 held back (batch of 405) | 0.23 | 100% (of 100) | passes and beats version 4: **version 6**, loaded by the server |
| Version 6, live, until the alias was moved back to version 1 by hand | 58 | not judged | not judged | the drift check needs 100 requests of one version |

- Gate limits: MAE <= 1.0, recommended >= 0.90, complaint >= 0.65; a new adapter must also beat the champion on the same reviews
- Real 2019-2022 drift does **not** hurt the model, so the drift is simulated (`simulator/scenarios.py`)
- Retrainings 1 and 2 ran on a Colab T4 started by hand, before the RunPod path was tried; the RunPod rows are the
  path the project uses
- Not measured: versions 4 and 6 on healthy reviews. Version 6's gate numbers (DagsHub, run a62a1127, 17 Nov 2026
  15:13 UTC): 305 of the batch's 405 reviews trained on, 100 kept back, 3 epochs, 80 s of training; on those 100:
  MAE 0.23, recommended 1.00 (on 100 reviews only), complaint 0.71, within 1 point 0.94, combined 0.895; the champion,
  version 4, scored combined 0.735 on the same 100

> Notes: the earlier offline numbers of 13 Nov (v1 0.91 / 94.5%, harsh 1.18 / 89.5%, v2 0.41 / 95.5%) were measured
> from files on a PC; they are in [llm.md](llm.md). The slide shows what the server measured.

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
- **CD**: on `main` only, never for pull requests; image to ghcr.io. The deploy job (server's runner, smoke test, back
  to the running image if it fails) ran once, on 13 Nov 2026 (run 43416289152), and is **off since**; the version with
  the rollback has never run. The server is updated by hand
- **drift.yml** runs hourly, judges only the serving model, opens one issue on DRIFT and, only if `AUTO_RETRAIN` is
  `true`, calls **train.yml**
- **train.yml** has a cooldown (default 6 h), rents nothing if fewer than 300 reviews are ready (pending and already
  scored by the model), trains on the cheapest free RunPod GPU for at most 45 minutes (at most about $0.26; measured:
  about 5 minutes on an RTX 3070, about one cent), then checks that no pod is left
- **Switched on since 17 Nov 2026.** That day: train.yml rented a GPU three times by hand and registered version 5;
  the drift check found drift on version 4 and called train.yml by itself
- The alarm called train.yml four times: three times RunPod had no free GPU on our list (nothing rented); the fourth
  time it rented an RTX 3080 Ti for 5.5 minutes (about 2 cents), trained on a batch of 405 reviews and **promoted
  version 6**; the server loaded it
- Not seen on 17 Nov by 17:35 UTC: GitHub's hourly schedule starting the drift check. Every check of the working day was
  started by hand; two scheduled runs came later (21:53 UTC, 03:49 UTC on 18 Nov) and failed because the server was off

> Notes: say plainly how far the automatic chain got ([server.md](server.md#what-has-run-17-nov-2026)).
> The chain ran end to end once on 17 Nov (15:09 UTC), but the drift check was started by hand: GitHub's schedule had
> not fired by 17:35 UTC that day (two scheduled runs came later and failed: the server was off). Version 6 was not judged on live traffic: after 58 requests (the check needs 100) the alias was moved back to
> version 1 by hand. Also say: on GitHub a Train run looks failed unless it registered a version, because RunPod does not
> let our key read the pod's log; the outcome comes from the server.

## 9b. MLOps level 2, in small

- Level 0: every step by hand. Level 1: training is a pipeline that runs by itself (continuous training). Level 2: the
  pipeline's own code is also built, tested and deployed automatically
- SkySat: queue -> trainer -> gate -> registry (level 1); `ci.yml` and `drift.yml` -> `train.yml` (level 2)
- Missing for a real level 2: an orchestrator such as Kubeflow or Airflow (GitHub Actions plays that part), a feature
  store, data validation beyond the intake's checks, a staging environment
- The loop has been on since 17 Nov 2026; the deploy step is still done by hand. The chain ran end to end
  once on 17 Nov 2026; the hourly schedule had not fired by 17:35 UTC that day, so each check was started by hand (two scheduled runs came
  later and failed: the server was off)

> Notes: the three levels are from Google's "MLOps: Continuous delivery and automation pipelines in machine learning".
> Detail: [architecture.md](architecture.md).

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
- After drift, mixing old reviews into the retraining kept the old labels alive: it failed the gate; the new reviews
  alone passed
- Rented GPUs are cheap but not always there: on 17 Nov every GPU type on our list was out of stock for two hours
- A schedule is not a guarantee: GitHub's hourly cron started no drift check on 17 Nov until after 17:35 UTC (2 runs in 22 slots by 03:35 UTC on 18 Nov)

## 12. Summary and questions

- One model, the whole lifecycle: LoRA training, gate, registry, API, monitoring, drift check, CI/CD
- Reproduce: `make install && make test`, `make llm-up`, `make simulate-seed N=2000`, then
  `make llm-train LLM_MLFLOW=http://localhost:5050` on a local GPU (on the server: the Train workflow, on RunPod)
- Data: Kaggle joelljungstrom/128k-airline-reviews (DbCL-1.0), scraped from airlinequality.com by the dataset author; never committed

### Likely questions (prepare an answer)

- Why LoRA and not fine-tuning everything? (0.3% of the weights: 5 minutes on one GPU, a 17 MB file to promote and ship)
- Why does the gate have those limits? (measured: v1 passes, constant answers fail, v1 under the harsh survey fails; `llm/gate.py`)
- Real drift did not hurt, so is the drift demo fake? (The scenario is simulated and we say so; the monitoring and the retrain path are real)
- What happens if MLflow is down? (the API keeps the adapter it has and retries; a logging failure never breaks serving)
- How do you stop a worse adapter from going to production? (the gate, and it must beat the champion on the same reviews)
- Is retraining automatic? (yes, since 17 Nov 2026: a drift alarm starts the Train workflow, which rents a RunPod GPU. It ran end to end once on 17 Nov: drift on version 4, a rented RTX 3080 Ti, version 6 promoted, about 2 cents. The drift check was started by hand, because GitHub's hourly schedule had not fired by 17:35 UTC)
- Where does production run, and what does it store? (an always-on VM, [server.md](server.md); numbers only, never review text)

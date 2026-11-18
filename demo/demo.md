# SkySat demo by hand, in 5 minutes

The same eight steps as [`demo.sh`](demo.sh), typed by hand or shown in the browser, for when a script that runs
everything is not wanted. Every command here was run against the server on 18 Nov 2026; the output blocks are what
came back that day (the server's address is replaced by `<server>`).

- **With the script:** `bash demo/demo.sh` (Enter between steps), or `bash demo/demo.sh --timed` (each pause ends by
  itself after 12 s; the whole run is about 3 minutes). Machine time measured on 18 Nov: 1 min 34 s.
- **By hand:** this page. About 4 min 30 s if you follow the time plan.

Both show two things: what the model can do (step 2), and what makes this a level 2 MLOps project in small (steps 1
and 3 to 8).

## Before the audience arrives

1. The server is on (`az vm start -g skysat-server -n skysat-server`, about 4 minutes) and answers.
2. One terminal with a large font, in the repository folder. Put the address and the logins into variables **without
   showing them**:

   ```bash
   S="https://$(cat ~/.config/skysat/server-address)"
   U="team:$(cat ~/.config/skysat/login-team)"
   T="trainer:$(cat ~/.config/skysat/login-trainer)"
   M="https://dagshub.com/HawkOsm/skysat-mlops.mlflow"
   ask() { curl -s -u "$U" -H 'Content-Type: application/json' -X POST "$S/score" -d "{\"airline\":\"Imaginary Air\",\"review\":\"$1\"}"; echo; }
   clear
   ```

3. Four browser tabs, already logged in:
   - the API's own page: `<server>/docs`
   - Grafana: `<server>/grafana/`
   - the registry: <https://dagshub.com/HawkOsm/skysat-mlops.mlflow> (Models → `skysat-reviews`)
   - the loop's run: <https://github.com/deensza/skysat-mlops/actions/runs/44246174603>
4. A rehearsal that changes nothing: `bash demo/demo.sh --no-pause --skip-post`. Every line of its summary must say
   `shown` or `listed`.

What the drift check shows in step 4 depends on the traffic of the last 24 hours. On 18 Nov the serving version had
almost no labelled requests yet, so it said "not enough labelled requests". For a table with real numbers, post about
100 reviews an hour before (`make simulate ...`, see [README.md](README.md)); they take about 20 minutes to score.

## Time plan

| # | Step | Time | On screen |
|---|---|---|---|
| 1 | A model is serving, and it is the registry's | 0:30 | terminal, then the registry tab |
| 2 | What the model can do | 1:00 | terminal (or the `/docs` tab) |
| 3 | A review becomes a measurement | 0:40 | terminal, then Grafana |
| 4 | Monitoring decides: the drift check | 0:25 | terminal |
| 5 | The pipeline's code is tested and built | 0:20 | GitHub Actions |
| 6 | Training starts by itself | 0:45 | the run's page |
| 7 | What keeps it cheap and safe | 0:25 | the same page |
| 8 | What is missing | 0:25 | nothing: say it |
| | **Total** | **4:30** | |

If you are late after step 4, skip step 5's commands and only say its sentence.

---

## 1. A model is serving, and it is the one the registry says (0:30)

**Say:** "The server does not serve a file someone copied. It serves whatever version the model registry calls the
champion."

```bash
curl -s -o /dev/null -w '%{http_code}\n' "$S/model"          # no login
curl -s -u "$U" "$S/model" | python3 -m json.tool | head -6
curl -s "$M/api/2.0/mlflow/registered-models/alias?name=skysat-reviews&alias=champion" | python3 -m json.tool | grep '"version"'
```

```
401
{
    "name": "skysat-reviews",
    "alias": "champion",
    "version": "7",
    "source": "models:/skysat-reviews@champion",
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "version": "7",
```

Then the registry tab: seven versions, each with the numbers it got at the gate.

| Version | Score MAE | Recommended | Complaint | Combined | Serving model on the same reviews | Trained on |
|---|---|---|---|---|---|---|
| 1 | 0.91 | 94.5% | 67.0% | 0.838 | - | RTX 4070 laptop |
| 2 | 0.41 | 95.5% | 74.0% | 0.883 | 0.842 | RTX 4070 laptop |
| 3 | 0.97 | 90.5% | 67.5% | 0.824 | 0.830 | T4 |
| 4 | 0.60 | 93.0% | 72.0% | 0.861 | 0.831 | T4 |
| 5 | 0.33 | 93.0% | 68.0% | 0.858 | 0.861 | RunPod RTX 3070 |
| 6 | 0.23 | 100.0% | 71.0% | 0.895 | 0.735 | RunPod RTX 3080 Ti |
| **7** (champion) | 0.59 | 93.0% | 73.0% | 0.865 | 0.795 | RunPod RTX 3080 Ti |

Versions 3 and 5 passed the gate but did not beat the serving model, so they were registered and never served.

## 2. What the model can do (1:00)

**Say:** "It reads a review and writes three things: a score out of ten, would this passenger recommend the airline,
and the main complaint out of seven, or none. It is a 1.5 billion parameter model with a 17 MB adapter, on a CPU:
about seven seconds a review."

```bash
ask "Everything went smoothly from check-in to landing. The crew were warm, the meal was hot and tasty, and we arrived early. I will fly with them again."
ask "The lasagne was still frozen in the middle and the bread was stale. The seat was fine, but for a nine hour flight the food was not acceptable."
ask "My knees touched the seat in front for seven hours and the seat would not recline. Crew and food were fine."
```

```
{"score":8,"recommended":"yes","complaint":"none","model_version":"7","latency_ms":6484}
{"score":2,"recommended":"no","complaint":"food","model_version":"7","latency_ms":6511}
{"score":2,"recommended":"no","complaint":"entertainment","model_version":"7","latency_ms":6362}
```

**Say about the third:** "That one is wrong: the complaint is the seat. The model gets the complaint right about three
times in four, and the gate measures exactly that before a version may serve."

The same in the browser: `/docs` → `POST /score` → "Try it out". Let someone in the room dictate a review.

What the script's nine reviews gave on 18 Nov (version 7):

| The review is about | Score | Recommended | Main complaint | A person would say | |
|---|---|---|---|---|---|
| a happy passenger | 8/10 | yes | none | none | agrees |
| bad food | 2/10 | no | food | food | agrees |
| rude cabin crew | 2/10 | no | service | service | agrees |
| typos, no details, HTML | 1/10 | no | ground service | ground service | agrees |
| wifi that never worked | 2/10 | no | wifi | wifi | agrees |
| broken screens | 2/10 | no | entertainment | entertainment | agrees |
| no legroom | 1/10 | no | ground service | seat | differs |
| trouble on the ground | 2/10 | no | value | ground service | differs |
| too expensive | 2/10 | no | food | value | differs |

6 of 9; at its gate version 7 had 73% on 100 held-back reviews. An empty review is refused with HTTP 422 before it
reaches the model.

What it cannot do: other languages were not tested, it reads only the first 250 words of a review, and it knows only
these eight complaint labels.

## 3. A passenger's review becomes a measurement (0:40)

**Say:** "A real review arrives with the passenger's own score. The intake queues it, the model answers it as live
traffic, and prediction and truth are logged as numbers. Never the text. That log is all that monitoring reads."

```bash
curl -s -u "$U" -H 'Content-Type: application/json' -X POST "$S/reviews" -d '{"review_id": "hand:'$(date +%s)'", "airline": "Imaginary Air", "review": "The crew were kind but my suitcase arrived two days late and nobody at the desk could help.", "score": 3, "recommended": "no", "ratings": {"ground service": 1, "service": 4}}'; echo
curl -s -u "$T" "$S/queue/stats" | python3 -m json.tool | head -8
sleep 10; curl -s -u "$U" "$S/scores/recent?hours=1" | python3 -m json.tool | head -9
```

```
{"id":"hand:1794983241","queued":true,"complaints":["ground service"]}
{
    "pending": 4,
    "claimed": 0,
    "trained": 3519,
    "last_received": "2026-11-18T06:25:34.319168+00:00",
    "total": 3523,
    "unscored": 1,
    "ready": 3,
{
    "hours": 1.0,
    "version": null,
    "count": 50,
    "mean_score": 2.56,
    "labelled_score": 5,
    "mae": 0.6,
    "mae_sd": 0.5477,
    "labelled_recommended": 5,
```

Point at three things: the intake worked out the complaint label from the sub-ratings; `unscored: 1` is the review
just posted, and `ready` means "scored, may now be trained on"; the last answer has totals only. Then the Grafana tab: the same log as a dashboard.

## 4. Monitoring decides: the drift check (0:25)

**Say:** "It judges only the version that is serving, over 24 hours, against the same limits as the gate, and it calls
it drift only when the miss is bigger than chance on at least 100 labelled requests."

```bash
API_USER=github API_PASSWORD="$(cat ~/.config/skysat/login-github)" .venv/bin/python -m llm.drift_check --url "$S" --hours 24 | head -9
```

On 18 Nov, before the retraining, for version 1:

```
## Drift check: DRIFT
Last 24 h, model version 1 only: 372 requests, mean predicted score 3.62
| Measure | Live (± 1 standard error) | Gate limit | Verdict |
| score MAE | 1.374 ± 0.076 | <= 1.0 | DRIFT (n=364) |
| recommended accuracy | 0.838 ± 0.019 | >= 0.9 | DRIFT (n=364) |
```

And right after it, for the new version 7:

```
## Drift check: no drift
Last 24 h, model version 7 only: 30 requests, mean predicted score 2.4
| score MAE | 0.000 | <= 1.0 | not enough labelled requests (2 < 100) |
| recommended accuracy | 1.000 | >= 0.9 | not enough labelled requests (2 < 100) |
```

**Say:** "On GitHub the same check runs as a workflow. On drift it opens an issue and calls the training."

## 5. The pipeline's own code is tested and built on every change (0:20)

**Say:** "Level 2 is about the pipeline, not only the model. Every change to its code goes through tests and a build
before it is used. One of the tests runs the real trainer against a real database, with only the GPU part replaced."

```bash
gh run list -R deensza/skysat-mlops --workflow ci.yml --branch main --limit 3
```

```
completed  success  Merge pull request #23 from deensza/hunt-fixes               CI/CD  main  push  44243841329  2m32s
completed  success  Merge pull request #20 from deensza/docs-after-runpod-test   CI/CD  main  push  44240967200  3m24s
completed  success  Merge pull request #22 from deensza/grafana-reader-grant     CI/CD  main  push  44240961191  2m44s
```

In run 44243841329: job `test` passed (128 passed, 9 skipped), job `image` built the Docker image the server runs, job
`deploy` was skipped: it is switched off and the server is updated by hand.

## 6. Training starts by itself and ends in a new serving model (0:45)

**Say:** "This is the run of 18 Nov. One drift check was started. Everything after it happened without anyone
touching anything: the alarm, the issue, a GPU rented on RunPod, the training, the gate, the registry, and the server
switching to the new version."

Show the run's page, or:

```bash
gh run view 44246174603 -R deensza/skysat-mlops --log | grep -a "Drift check: \|Pod [a-z0-9]* \|From the server\|left behind" | cut -f3- | cut -c30-
```

```
## Drift check: DRIFT
Pod 3s96b7znrmvs12 created on NVIDIA GeForce RTX 3080 Ti (branch main, trainer args '--replay 0 --epochs 3')
Pod 3s96b7znrmvs12 is gone
The pod's log could not be read. From the server: reviews marked trained 3099 -> 3519, serving model 1 -> 7
Pod 3s96b7znrmvs12 was rented for this run
Trainer pods left behind: 0
```

Numbers to have ready: the run took 9 min 39 s; the pod existed 8 min 43 s on an RTX 3080 Ti at $0.18 an hour, about
3 cents; 420 reviews were trained on; version 7 scored 0.865 against version 1's 0.795 on the same 100 held-back
reviews. The issue it opened is number 24.

Doing this live takes about 10 minutes and needs a free GPU, so it does not fit a 5-minute demo:
`bash demo/demo.sh --live` (the conditions are in the script's header).

## 7. What keeps it from costing money or serving a bad model (0:25)

**Say, pointing at the steps of the same run:**

| Guard | Where it shows |
|---|---|
| No GPU without 300 reviews that are ready | step "Queue before" |
| No automatic run within 6 hours of a rental | step "Cooldown" |
| A pod never outlives its job: it deletes itself, a 45-minute timer, the launcher, a last check | "Trainer pods left behind: 0" |
| Cheapest GPU first; at most about $0.26 a run | "Pod ... created on ..." after the cheaper types were refused |
| A new model must pass the gate (MAE ≤ 1.0, recommended ≥ 90%, complaint ≥ 65%) on reviews it never trained on | the registry's numbers |
| And beat the serving model on the same reviews | versions 3 and 5 did not, and never served |
| Going back is one command | `make llm-promote VERSION=<n> SHARED=yes`; the server follows within a minute |

## 8. What a full level 2 system has that this one does not (0:25)

**Say it before anyone asks:**

- GitHub's hourly schedule for the drift check fires only now and then; the runs shown were started by hand.
- The deploy job is off: a new image reaches the server by hand.
- No orchestrator (GitHub Actions plays that part), no feature store, one server and no staging environment.
- The loop has ended in a new serving model twice (17 and 18 Nov 2026), both times on simulated drift.

So the claim is "level 2 in small, with these gaps", not "a level 2 system".

---

## If something goes wrong

| Problem | What to do |
|---|---|
| The server does not answer | Show this page: every output block is a real one from 18 Nov. Steps 5 and 6 need only GitHub |
| A review takes very long | Keep talking; the API answers one review at a time. After 100 s it gives up with a 503: send it again |
| The model gives a wrong complaint | Say so. It is right about three times in four, and that is what the gate measures |
| The drift check says "not enough labelled requests" | Say why (fewer than 100 for this version) and show the two tables in step 4 |
| `gh` asks you to sign in | Open the links in the browser instead |
| A password appears on screen | Never `cat` the files in `~/.config/skysat/`; use the variables from "Before the audience arrives" |

# SkySat live demo plan

One speaker, about 9 minutes, after the slides (slide 21 hands over to it). Every number below was measured on
18 Nov 2026. The server's address never appears on screen: it is `<server>` here and `$S` in the terminal.

---

## 1. Before the demo (morning)

1. In the repository folder, start the server and wait about 4 minutes:

   ```bash
   az vm start -g skysat-server -n skysat-server
   ```

2. Check it, all read-only:

   ```bash
   S="https://$(cat ~/.config/skysat/server-address)"; U="team:$(cat ~/.config/skysat/login-team)"; T="trainer:$(cat ~/.config/skysat/login-trainer)"; gh variable get AUTO_RETRAIN -R deensza/skysat-mlops; gh run list -R deensza/skysat-mlops --limit 3; curl -s -u "$U" "$S/model" | python3 -c 'import json,sys;print("serving version",json.load(sys.stdin)["version"])'; curl -s -u "$T" "$S/queue/stats" | python3 -c 'import json,sys;d=json.load(sys.stdin);print("ready",d["ready"],"claimed",d["claimed"])'
   ```

   Expected: `false`, no Train run since 18 Nov, serving version 7, ready about 774, claimed 0. If `/model` answers
   503, the model is still loading: wait a minute. The RunPod Pods page should be empty.

3. In the terminal you will show, set the hidden variables, then clear the screen:

   ```bash
   S="https://$(cat ~/.config/skysat/server-address)"
   U="team:$(cat ~/.config/skysat/login-team)"
   T="trainer:$(cat ~/.config/skysat/login-trainer)"
   clear
   ```

   Never `cat` the files in `~/.config/skysat/` on screen.

4. Open the tabs, left to right, logged in:

   | # | Tab |
   |---|---|
   | 1 | GitHub Actions → Train: <https://github.com/deensza/skysat-mlops/actions/workflows/train.yml> |
   | 2 | The API's page: `<server>/docs` |
   | 3 | The terminal (its own window) |
   | 4 | `docker-compose.yml`: <https://github.com/deensza/skysat-mlops/blob/main/docker-compose.yml> |
   | 5 | Issue #31 "Training runs": <https://github.com/deensza/skysat-mlops/issues/31> |
   | 6 | RunPod Pods: <https://console.runpod.io/pods> |
   | 7 | DagsHub MLflow: <https://dagshub.com/HawkOsm/skysat-mlops.mlflow> (Experiments → `skysat-llm`, Models → `skysat-reviews`) |
   | 8 | Code: [gate.py L24-26](https://github.com/deensza/skysat-mlops/blob/main/llm/gate.py#L24-L26), [train_lora.py L266](https://github.com/deensza/skysat-mlops/blob/main/llm/train_lora.py#L266), [drift.yml L112-113](https://github.com/deensza/skysat-mlops/blob/main/.github/workflows/drift.yml#L112-L113), [serve.py L173](https://github.com/deensza/skysat-mlops/blob/main/llm/serve.py#L173), [nginx.server.conf L40-69](https://github.com/deensza/skysat-mlops/blob/main/nginx/nginx.server.conf#L40-L69) |
   | 9 | Grafana: `<server>/grafana/d/skysat` |
   | 10 | The recorded automatic run: <https://github.com/deensza/skysat-mlops/actions/runs/44246174603> |
   | 11 | PR #32 green: <https://github.com/deensza/skysat-mlops/pull/32> |
   | 12 | PR #33 red: <https://github.com/deensza/skysat-mlops/pull/33> |

5. Decide: should version 7 stay champion for sure? Then add `--no-promote` to the trainer arguments below. Without
   it, a better new version is promoted and the server switches to it about a minute after the run.

---

## 2. Start the training (about 1 minute before slide 21)

Tab 1 → **Run workflow** → branch `main`, reason `by hand`, trainer arguments:

```
--replay 0 --epochs 3 --max-train-seconds 60
```

(or `--replay 0 --epochs 3 --max-train-seconds 60 --no-promote`). The same from the terminal:

```bash
gh workflow run train.yml -R deensza/skysat-mlops -f reason="by hand" -f trainer_args="--replay 0 --epochs 3 --max-train-seconds 60"
```

What happens after you press it (run 44311668226, 18 Nov):

| After | What appears | Where |
|---|---|---|
| about 1:00 | the pod; comment "A GPU was rented" | RunPod, issue #31 |
| 2:48 | the MLflow run, with its link as a comment | DagsHub, issue #31 |
| 2:48-4:30 | base model download, 60 s of training, the loss every 25 steps | DagsHub run page |
| 5:01 | gate numbers and the new registry version | DagsHub |
| 6:53 | pod deleted, run ended, ending comment | Actions, issue #31 |

---

## 3. The demo

| Time | Step | On screen | Say |
|---|---|---|---|
| 0:00 | **It's alive** | tab 2: `GET /health` → Try it out → Execute | "This request went through HTTPS, the gateway and the login to the model. It's up, and version 7 is serving." |
| 0:30 | **What it does** | tab 2: `POST /ask` → Try it out → a review from the room → Execute; then `GET /model` | "Score, recommended, main complaint, and which registry version answered. The server never gets a model file copied onto it: it serves whatever the registry calls champion." |
| 1:45 | **Security** | tab 3: the two commands below; tab 8: the nginx snippet | "One gateway: HTTPS, a login on every request, a rate limit, and the training queue only for the trainer's own login." |
| 2:30 | **The server** | the request path below; tab 4; optional `docker ps` | "Six containers, all defined in code. Only ports 80 and 443 are open. Every container restarts itself, and a watchdog checks the API every 5 minutes." |
| 3:30 | **The GPU** | tab 5: "A GPU was rented"; tab 6: the pod | "I started a real training before the demo. This pod was rented for these minutes only, and it deletes itself." |
| 4:15 | **Training live** | tab 5: the MLflow link; tab 7: the run | "The run is in the registry from the moment training starts. This is the loss going down right now." |
| 5:00 | **The gate** | tab 8: gate.py L24-26, then train_lora.py L266 | "A new model must pass these three limits, on reviews it never trained on, and beat the serving model on the same reviews. Otherwise it never serves." |
| 5:45 | **Monitoring → training** | tab 9 briefly; tab 8: drift.yml L112-113 | "Every live answer is logged next to the passenger's own score. When the live error goes above the gate's limits, these two lines start a training." |
| 6:30 | **The automatic chain** | tab 10 | "Here nobody pressed anything: the drift check found the model too weak, opened an issue, rented a GPU, trained, and version 7 replaced version 1. The server switched by itself." |
| 7:30 | **CI** | tab 11: green checks; tab 12: Files changed, then the red checks | "Every change to the pipeline is tested and its image is built. This one weakens the gate, so the tests fail and the image is never built. main only changes through pull requests." |
| 8:30 | **The result** | tab 5: the ending comment; tab 7: the new version; tab 2: `GET /model` | the sentence in section 4 that matches |
| 9:15 | **Questions** | this page on the side | |

### Commands and snippets for the steps

Security (tab 3):

```bash
curl -s -o /dev/null -w '%{http_code}\n' "$S/model"
```

prints **401**: no login, no access.

```bash
curl -s -o /dev/null -w '%{http_code}\n' -u "$U" "$S/queue/stats"
```

prints **403**: the `team` login may use the model, not the training queue.

The request path (step "The server"):

```
browser → Caddy (HTTPS, Let's Encrypt) → Nginx (rate limit, login, routing)
        → api (FastAPI + the model) / intake (the queue) / Grafana
                         ↘ PostgreSQL (queue + prediction log)
```

Optional live proof (about 30 s; start it as you begin the step and talk while it runs; prints names and status
only):

```bash
az vm run-command invoke -g skysat-server -n skysat-server --command-id RunShellScript --scripts 'docker ps --format "{{.Names}}  {{.Status}}"' --query "value[0].message" -o tsv
```

Good reviews for `POST /ask` if nobody in the room gives one (version 7, 18 Nov):

- "The screen at my seat was broken and so was the one next to it. No films and no music for ten hours." →
  1/10, no, entertainment
- "Dinner was a cold, soggy sandwich and breakfast never came. The crew were kind and we left on time, but the food
  was terrible." → food
- Avoid the lasagne review from slide 12 here: through `/ask` (no airline sent) it answers `value`.

---

## 4. The result sentence (step 8:30)

| The ending comment on issue #31 says | Say |
|---|---|
| "registered and promoted" | "It passed the gate, beat version 7 on the same held-back reviews, and the server loaded it by itself." Show `GET /model` with the new version |
| "registered but the champion stays" (always with `--no-promote`) | "It passed the gate but wasn't better than version 7, so production didn't change. That's the gate doing its job." |
| "Nothing was registered" | "It didn't pass the gate, so it was thrown away and the reviews went back to the queue." |
| "No GPU type in the list was free" | "No GPU was free, so nothing was rented and nothing was paid. That's one of the guards." Show run 44311668226 |
| no ending yet | Say which step it is in (the run's page shows it), and show yesterday's versions 8 and 9 on DagsHub |

---

## 5. If something goes wrong

| Problem | Do |
|---|---|
| The server doesn't answer | skip 0:00-2:30; everything else is on GitHub, RunPod and DagsHub |
| `/ask` takes long | keep talking: one review at a time, about 7 s; after 100 s it answers 503, send it again |
| A wrong complaint | "It's right about 3 times in 4, and that is exactly what the gate measures." |
| The Train run waits at "Queue before" or is skipped | check `ready` with the check command; it needs 300 |
| An outside service is down (GitHub, RunPod, DagsHub) | "This part runs on X, which is down right now; here is the recorded run." |
| A password or the address appears on screen | stop sharing the terminal; only use the variables from section 1 |

---

## 6. Numbers to have ready

| | |
|---|---|
| Model | Qwen2.5-1.5B-Instruct (3 GB, frozen) + LoRA adapter (17 MB) |
| Speed | about 7 s a review on the server's CPU |
| Gate | MAE ≤ 1.0, recommended ≥ 90%, complaint ≥ 65%, and beat the champion on the same held-back reviews |
| Drift | MAE > 1.0 or recommended < 90% by more than chance, on ≥ 100 labelled answers in 24 h |
| Training run | 300+ ready reviews; 6 h cooldown when automatic; 45-minute limit; usually 1-3 cents |
| Version 7 at its gate | MAE 0.59, recommended 93%, complaint 73%, combined 0.865 (version 1 on the same reviews: 0.795) |
| The recorded automatic run | 44246174603: 9 min 39 s, RTX 3080 Ti, about 3 cents, issue #24 |
| CI | 166 tests; `test` 36 s and `image` 1 min 41 s on PR #30 |

---

## 7. After the demo

- Set the loop back if you changed it: `gh variable get AUTO_RETRAIN -R deensza/skysat-mlops` must say `false`.
- Check that RunPod's Pods page is empty.
- Switch the server off: `az vm deallocate -g skysat-server -n skysat-server`.

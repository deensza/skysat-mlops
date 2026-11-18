# The team's server

Serving and monitoring run on one always-on VM, so they do not depend on a team PC being on. It is the same stack as on a
PC (the complaint-box `api`, Nginx, PostgreSQL, Grafana), with [docker-compose.server.yml](../docker-compose.server.yml) on
top. Training happens on a rented RunPod GPU or a local NVIDIA card; the adapter and the `champion` alias are in DagsHub.

| | |
|---|---|
| Machine | Ubuntu 24.04 VM, Azure `Standard_B4as_v2` (4 vCPU, 16 GB RAM; resized on 2026-11-17 from `Standard_B2as_v2`, 2 vCPU and 8 GB). On this size the API runs in float32 (`LLM_DTYPE` and `mem_limit: 11g` in [docker-compose.server.yml](../docker-compose.server.yml)): about 7 s a review and about 9.3 GB of memory. A server with 8 GB must set `LLM_DTYPE=bfloat16` in its `.env` (the model then needs ~4 GB and a review takes 34-60 s); less than 8 GB cannot host it. The first version ran on a `Standard_B2ats_v2` (1 GB RAM), which cannot host it |
| Runs | Caddy (HTTPS), Nginx (the gateway), 1 `api` container (the LLM, CPU), the `intake` (`POST /reviews` and the queue), PostgreSQL, Grafana |
| Does not run | local MLflow (the API loads the champion from DagsHub), the k6 load test, any training (worker 2 runs on a RunPod pod) |
| Holds | the review queue (`queue.reviews`, review text readable only by the `skysat` database user) and the prediction log (numbers only). The base model (~3 GB) is downloaded into a Docker volume at the first start |
| Addresses | `https://<server>/docs` (API: `POST /score`, and `POST /ask` to try the model with one text box), `https://<server>/grafana/d/skysat` (dashboard). The name is not in this public repo on purpose |

## The gateway: the only way in

```
internet -> Azure firewall (only 80, 443) -> Caddy (HTTPS) -> Nginx (rate limit, then login) -> API or Grafana
```

| Protection | How | Checked from outside |
|---|---|---|
| Only a few accounts | HTTP Basic login in Nginx, `tutor` and `team`, 128-bit random passwords; `github` for the drift check and `trainer` for the RunPod trainer ([switching it on](#switching-it-on)). The browser asks once; Swagger and Grafana then just work | No login or a wrong one: 401 on every path |
| Encryption | Let's Encrypt certificate, renewed by Caddy; port 80 only redirects | HTTP answers 308 to HTTPS |
| Password guessing and flooding | 20 requests/s per address (burst 100), checked **before** the login; addresses a client makes up are ignored | 429 under load, the tutor still gets in right after |
| Oversized requests | 64 kB body limit (a review is a few kB; the API accepts 50,000 characters) | 100 kB: 413 *(re-check: the old 16 kB limit was the one tested)* |
| Nothing else reachable | Azure firewall: 80 and 443 only, SSH closed. PostgreSQL and the health port listen on the server itself; Nginx forwards only `/score`, `/ask`, `/model`, `/health`, `/docs`, `/openapi.json`, `/scores/recent`, `/grafana/`, `/reviews` (to the intake, any login), and `/queue/claim|done|stats` (to the intake, **only for the `trainer` login**: 403 for anyone else) | 22, 3000, 5432, 8001, 8080 closed from outside |
| A viewer cannot damage the log | Grafana's viewers can send their own SQL, so its data source is a read-only database user, `grafana_reader` | `DROP`, `DELETE`, reading server files, running programs: all denied |

Not protected against: someone who gets a password can use the API within the limits, and the DagsHub token sits in the
server's `.env`. To lock someone out, delete their line in `gateway-users`.

## Staying alive

- **Crash:** Docker restarts a container by itself (`restart: unless-stopped`), also after a reboot.
- **Hang or stopped container:** [monitoring/watchdog.sh](../monitoring/watchdog.sh), from cron every 5 minutes. It
  starts stopped containers at once and restarts the API after 3 failed health checks.
- **Disk:** container logs are capped at 20 MB each.
- **Dashboard:** opens on the last 30 days. Fresh traffic comes from the simulator (`make simulate` with
  `QUEUE=https://<server>` and the `team` login in `API_USER` / `API_PASSWORD`): each review is scored as it arrives.

| Failure, tested for real on 2026-11-05 with the previous (tabular) API on the 1 GB VM: re-test with the LLM | Back after |
|---|---|
| Whole VM rebooted | 218 s |
| Nginx stopped | next watchdog run (at most 5 min) |
| API process killed | 100 s |
| API hung | 67 s after the third failed check (about 15 min in all) |

## The automatic loop

```
simulator (a PC) -> POST /reviews -> queue.reviews -> drift.yml -> train.yml -> trainer pod (RunPod, through /queue/*)
  -> DagsHub champion -> api (here)
```

- **Intake** is always on: every review posted to `/reviews` is queued and sent to the API with its true labels.
- **The drift check** is [drift.yml](../.github/workflows/drift.yml), scheduled hourly at :35; since 18 Nov 2026 the
  server also starts it every 10 minutes from cron (`monitoring/drift-trigger.sh`), because the schedule rarely fires
  (see [Starting the drift check from the server](#starting-the-drift-check-from-the-server)). It reads `GET /model` and
  `GET /scores/recent?version=N` (aggregates only) and judges only the serving model. On DRIFT it opens one issue
  ("Drift detected: the serving model fell below the gate") and, only if the repository variable `AUTO_RETRAIN` is
  `true`, calls the Train workflow with `--replay 0 --epochs 3` (the new reviews only: after drift the old labels are
  the problem). A scheduled run with missing secrets is skipped with a warning. The issue is closed by the Train
  run that promotes a new version.
- **Training** is [train.yml](../.github/workflows/train.yml) ("Train"), started by the drift check or by hand
  (`gh workflow run train.yml`). It checks its settings, respects a cooldown (`TRAIN_COOLDOWN_HOURS`, default 6: no
  automatic run if a GPU was rented less than that long ago; a run that found no free GPU rented nothing and does not
  count; a run by hand skips the cooldown), rents nothing if fewer than 300 reviews are `ready` (pending and already
  scored by the model as live traffic; `GET /queue/stats`), runs `llm.runpod_pod --wait --branch main --max-minutes 45`,
  then `--sweep` (fails the run if a trainer pod is left), and writes a summary table. One run at a time.
  `make train-remote` from a PC is the same launcher. `AUTO_TRAIN=1` in this server's `.env` (then also
  `RUNPOD_API_KEY`, `QUEUE_URL`, `QUEUE_USER`, `QUEUE_PASSWORD`) makes the intake start pods itself; it is off and has
  not been tried on the server.
- **Deploy** is the `deploy` job of [ci.yml](../.github/workflows/ci.yml) (see CD below). It is off: the server has no
  self-hosted runner and `DEPLOY_TO_RUNNER` is unset. It ran once, on 13 Nov 2026 (run 43416289152), and has been off
  since; the version with the rollback has never run. The server is updated by hand with the `az vm run-command` line
  in step 1 below.

### What has run (17 Nov 2026)

The loop was switched on on 17 Nov 2026: the secrets and variables below are set. `AUTO_RETRAIN` is the switch: `true`
when a drift alarm may rent a GPU, anything else to pause it (for example while testing on the server).

- **The drift check, by hand.** 08:06 UTC: version 4, 267 requests, "no drift" (MAE 0.820 ± 0.080, recommended
  0.899 ± 0.018). 13:12 UTC, after 492 more reviews with a strong label shift were posted: version 4, 511 requests,
  "DRIFT" (MAE 1.209 ± 0.075, recommended 0.816 ± 0.017). It opened issue 18 and called the Train workflow.
- **The Train workflow, by hand**, three times between 06:23 and 06:45 UTC: each rented one RTX 3070 for about
  5 minutes. The first trained on a batch of 300 reviews and registered version 5; the champion stayed version 4.
  The other two were tests of the launcher that train nothing. Table: [llm.md](llm.md#what-ran-on-runpod-17-nov-2026).
- **The Train workflow, called by the drift check** at 13:12 UTC: the cooldown and the queue check passed (319 reviews
  ready), then RunPod had none of the six GPU types then in the list free ("No GPU had capacity"). Nothing was
  rented, the job failed, and the cooldown did not start. The RTX 4090 was then added as a seventh type. Two more
  drift checks by hand, at 13:38 UTC (528 requests, MAE 1.203 ± 0.074, recommended 0.818 ± 0.017) and 14:24 UTC (560
  requests, MAE 1.211 ± 0.072, recommended 0.820 ± 0.016), said "DRIFT" and called Train again: all seven types were
  out of stock, nothing was rented.
- **The whole chain, once.** Drift check 44111118456, started by hand at 15:09 UTC: version 4, 591 requests, MAE
  1.230 ± 0.070, recommended 0.812 ± 0.016, "DRIFT". Its Train job: cooldown and queue check passed; RTX 3070 and RTX
  A4000 out of stock; a pod "created on NVIDIA GeForce RTX 3080 Ti" at 15:10:17 UTC and "is gone" at 15:15:49 UTC
  (5 min 32 s; at $0.18/h about 2 cents). "From the server: reviews marked trained 2694 -> 3099, serving model 4 ->
  6"; `registered: True`, version 6, `promoted: True`; "Trainer pods left behind: 0". The run ended green. This
  rental starts the 6 hour cooldown. Version 6's gate numbers are in its DagsHub run a62a1127: 100 gate reviews, MAE 0.23, recommended 1.00 (on 100
  reviews only), complaint 0.71, combined 0.895 (version 4: 0.735 on the same reviews); the pod's log could not be read.
- **After the promotion:** version 6 was not judged on live traffic. By 17:12 UTC the champion alias was moved back
  to version 1 by hand, to run the whole cycle again from the start. Drift check 44138054792 (by hand, 17:32 UTC)
  read version 1 only: 71 requests, MAE 1.845 ± 0.255, recommended 0.775 ± 0.050, each "not enough labelled requests
  (71 < 100)", so "no drift" and no training. In its 24 hour window: version 1, 71 requests; version 4, 594;
  version 6, 58.
- **Not done by the workflow then:** closing the issue. Issue 18 was closed by hand at 17:12 UTC. Since 18 Nov 2026
  a Train run that promotes a new version closes the open drift issue itself
  ([close-drift-issue.sh](../.github/scripts/close-drift-issue.sh)); a run that registers nothing, or a version
  that does not beat the champion, leaves it open.
- **The hourly schedule.** `drift.yml` reached `main` at 06:17 UTC with the schedule `35 * * * *`. GitHub started no
  scheduled run in the seven hourly slots from 06:35 to 12:35 UTC, although the workflow was listed as active. At
  13:08 UTC the workflow was disabled and enabled again.
  By 17:35 UTC no scheduled run had appeared: twelve hourly slots in a row since the workflow reached `main`, five of
  them after it was enabled again. Two scheduled runs came later: run 44184538814 at 21:53:45 UTC on 17 Nov and run
  44233113971 at 03:49:43 UTC on 18 Nov, neither at :35. That is 2 runs in the 22 hourly slots from 06:35 UTC on 17 Nov
  to 03:35 UTC on 18 Nov; the first came 15 h 36 min after the schedule reached `main` (06:17 UTC). Both failed in the
  check step with "Could not read the server's /model: <urlopen error timed out>", because the VM was switched off at
  those times. No issue was opened and no training was started (the retrain job was skipped). The cause of the missing
  slots is not known. Not proven yet: a scheduled check that reaches the server and completes. Until the schedule is
  reliable, the drift check has to be started by hand (`gh workflow run drift.yml`) or by something else that has a
  clock.
- **On GitHub the Train runs look failed** even when they trained: RunPod answers 403 to this project's API key on
  the pod-log endpoint, so the launcher reads the outcome from this server (`/queue/stats`, `/model`). A run that
  registered a version ends green; a run that registered nothing ends red, whether the gate failed, there were too few
  reviews, or the trainer crashed. The summary page of the run says which numbers changed.

### Starting the drift check from the server

GitHub started 2 scheduled runs in 22 hourly slots (see "The hourly schedule" above), so the drift check cannot rely on
its own clock. The VM has one: [monitoring/drift-trigger.sh](../monitoring/drift-trigger.sh) runs from cron every 10
minutes and starts `drift.yml` on `main` through the GitHub REST API (`POST .../actions/workflows/drift.yml/dispatches`),
the same as `gh workflow run drift.yml`.

- **The token** is a classic personal access token (GitHub, Settings, Developer settings, Personal access tokens, Tokens
  (classic)) with the scopes `repo` and `workflow`, created by a person who can start workflows in
  `deensza/skysat-mlops` (a fine-grained token cannot be issued for a repository one is only a collaborator on). Make it
  short-lived (90 days). It goes to the VM only: never into git, an issue or a comment.
- **Install**, from a PC with `az` (SSH is closed). First the token, from a local file the person made with
  `(umask 077; printf '%s' '<token>' > ~/.config/skysat/github-dispatch-token)`. Not with `az vm run-command invoke`:
  the agent on the VM keeps every script it ran, so a token inside a script would stay on the disk. A *managed* run
  command passes it as a **protected parameter** instead (Azure: "Use ProtectedParameter to pass any sensitive inputs
  to script such as passwords, keys"; on Linux "Named Parameters and its values are set to environment config", so the
  script reads `$token`, with `$1` as the fallback should it come as an argument), writes the file, and is deleted
  again at once:

  ```bash
  az vm run-command create -g skysat-server --vm-name skysat-server --run-command-name set-dispatch-token --location austriaeast --protected-parameters token="$(cat ~/.config/skysat/github-dispatch-token)" --script 'umask 077; printf "%s" "${token:-$1}" > /opt/skysat/github-dispatch-token && chmod 600 /opt/skysat/github-dispatch-token && echo "written: $(wc -c < /opt/skysat/github-dispatch-token) bytes"' --query "instanceView.output" -o tsv
  az vm run-command delete -g skysat-server --vm-name skysat-server --run-command-name set-dispatch-token --yes
  ```

  Expect `written: 40 bytes` (a classic token is 40 characters). What was checked (18 Nov 2026, without running it):
  `az vm run-command create --help` (az 2.90.0) lists `--protected-parameters` ("Usage: --protected-parameters
  credentials=somefoo secret=somebar") and `--script`, and Azure's page on managed run commands says the sentences
  quoted above and that `delete` removes the run command from the VM. Assumed, not checked: that the named protected
  parameter is exactly `$token` in the script (the page does not name the variable; hence the `$1` fallback), and
  that nothing of a deleted run command stays on the VM. Still true: the token is in the `az` process's arguments on
  the PC for the seconds the command runs (the help shows no `@file` form for `--protected-parameters`), and in the
  shell's history only as the text `$(cat ...)`, not as the value. If `written: 0 bytes` comes back, the parameter
  did not reach the script by either name: do not fall back to `invoke`; ask on the Azure page which name it uses.

  Then the checkout, the script and the cron entry (no secret in this one, so `invoke` is fine):

  ```bash
  az vm run-command invoke -g skysat-server -n skysat-server --command-id RunShellScript --scripts 'cd /opt/skysat && git fetch -q origin main && git checkout -q origin/main && install -m 755 monitoring/drift-trigger.sh /usr/local/bin/skysat-drift-trigger && echo "*/10 * * * * root /usr/local/bin/skysat-drift-trigger >> /var/log/skysat-drift-trigger.log 2>&1" > /etc/cron.d/skysat-drift-trigger && chmod 644 /etc/cron.d/skysat-drift-trigger' --query "value[0].message" -o tsv
  ```

  The cron line is `*/10 * * * * root /usr/local/bin/skysat-drift-trigger >> /var/log/skysat-drift-trigger.log 2>&1`.
- **Check:** `tail /var/log/skysat-drift-trigger.log` through the same `az vm run-command invoke` shows one line every 10
  minutes, `<UTC time> dispatch drift.yml: HTTP 204 (ok)`; GitHub, Actions, Drift check lists runs with the event
  "workflow_dispatch". HTTP 401, 403 or 404 mean the token or its scopes, 422 a disabled workflow or a wrong ref.
- **Why a run every 10 minutes is safe**, from the files as they are today: `drift.yml` opens an issue only if none with
  that title is open ("Open an issue on drift", lines 87-96) and has `concurrency: group: drift` (lines 34-35), so runs
  go one at a time. `train.yml`'s "Cooldown" step (line 119 on) counts the runs of the last `TRAIN_COOLDOWN_HOURS`
  (default 6; line 94) whose "A GPU was rented" step ran and skips the training otherwise; it reads the last 200 runs
  (line 141) so that a run every 10 minutes does not push a rental out of the list. `train.yml` has
  `concurrency: group: train-gpu` (lines 74-76), one run at a time, and its queue check rents nothing with fewer than
  300 ready reviews (line 172); a call that finds no free GPU rents nothing. The one cost: a drift check that finds
  drift inside the cooldown does nothing, but the first one after the cooldown rents again, so at most 4 rentals a
  day, about the limit already in [What it can cost](#what-it-can-cost).
- **Switch off:** delete the cron entry (the same `az vm run-command invoke` with `--scripts 'rm /etc/cron.d/skysat-drift-trigger'`)
  or revoke the token on GitHub (the script then logs HTTP 401 every 10 minutes until the entry is removed).

### Switching it on

1. Merge into `main` and put the new image on the server: **done on 2026-11-16** (`main` runs here with the intake and
   the queue). For reference:

   ```bash
   az vm run-command invoke -g skysat-server -n skysat-server --command-id RunShellScript --scripts 'cd /opt/skysat && git fetch -q origin main && git checkout -q origin/main && docker compose pull api intake && docker compose up -d --no-build && docker compose restart nginx' --query "value[0].message" -o tsv
   ```

2. Two gateway logins for robots, `github` (the drift check) and `trainer` (the RunPod pod): **done on 2026-11-16**.
   The passwords are made and kept on the PC that created them (`~/.config/skysat/login-github` and `login-trainer`);
   only their hashes are on the server. For reference:

   ```bash
   (umask 077; openssl rand -hex 16 > ~/.config/skysat/login-github; openssl rand -hex 16 > ~/.config/skysat/login-trainer)
   ```
   ```bash
   az vm run-command invoke -g skysat-server -n skysat-server --command-id RunShellScript --scripts "echo 'github:$(openssl passwd -apr1 -in ~/.config/skysat/login-github)' >> /opt/skysat/gateway-users && echo 'trainer:$(openssl passwd -apr1 -in ~/.config/skysat/login-trainer)' >> /opt/skysat/gateway-users && cd /opt/skysat && docker compose exec -T nginx nginx -s reload" --query "value[0].message" -o tsv
   ```

3. Check the queue from the PC (expect counts, then `"pending"` growing):

   ```bash
   curl -s -u "trainer:$(cat ~/.config/skysat/login-trainer)" "https://$(cat ~/.config/skysat/server-address)/queue/stats"
   ```

4. The settings in GitHub, Settings, Secrets and variables, Actions: **done on 2026-11-17** (all except
   `TRAIN_COOLDOWN_HOURS` and `DEPLOY_TO_RUNNER`, which are unset; the RunPod key is stored as `RUNPOD_API`). The
   passwords for `github` and `trainer` are on the PC that created them; they go to the repository owner privately,
   never in an issue, a comment or a commit: the repository is public.

   | Secrets | Value |
   |---|---|
   | `RUNPOD_API_KEY` | from RunPod, Settings, API Keys (a secret named `RUNPOD_API` is read too) |
   | `SKYSAT_SERVER` | the server's host name, no `https://` |
   | `SKYSAT_LOGIN` | `github:` + the password of `github` |
   | `SKYSAT_TRAINER_PASSWORD` | the password of `trainer` |
   | `MLFLOW_TRACKING_PASSWORD` | exists |

   | Variables | Value |
   |---|---|
   | `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` | exist |
   | `AUTO_RETRAIN` | `true` lets a drift alarm start the Train workflow |
   | `TRAIN_COOLDOWN_HOURS` | optional, default 6 |
   | `DEPLOY_TO_RUNNER` | `true` for the deploy job (needs the self-hosted runner, see CD below) |

5. Try each part by hand and read its log (`gh workflow run drift.yml`, then `gh workflow run train.yml`): **done on
   2026-11-17**, see [What has run](#what-has-run-17-nov-2026).

**Switching it off**: `AUTO_RETRAIN` not `true` (a drift alarm then only opens the issue); `gh workflow disable
drift.yml` stops the drift check; `DEPLOY_TO_RUNNER` not `true` stops the deploy job; `AUTO_TRAIN=0` (the default)
stops pods started by the intake; `rm /etc/cron.d/skysat-drift-trigger` on the VM stops the trigger. To lock a robot out, delete its line in `gateway-users`.

### Before a demo: freeze it

Set `AUTO_RETRAIN` and `DEPLOY_TO_RUNNER` to `false` (or disable the two workflows with `gh workflow disable`), so
nothing retrains or redeploys while presenting. If the demo needs them, run them by hand (`gh workflow run train.yml`).
Enable them again afterwards.

Before you stop a container on purpose (for example `docker compose stop intake` to freeze traffic), pause the
watchdog with `touch /var/tmp/skysat-watchdog-off`, or it starts the container again within 5 minutes; `rm` the file to
resume. The server runs a copy of the script, so after a change to `monitoring/watchdog.sh` copy it again with
`install -m 755 /opt/skysat/monitoring/watchdog.sh /usr/local/bin/skysat-watchdog` (cron line:
`*/5 * * * * root /usr/local/bin/skysat-watchdog >> /var/log/skysat-watchdog.log 2>&1`).

## Cost and switching it off

### What it can cost

- A training run: the Train workflow stops the pod after 45 minutes; at the dearest GPU in the list (an RTX 4090 at
  $0.34/h, tried only when the six cheaper types are all out of stock) that is at most about $0.26 per run. With the
  6 h cooldown there are at most 4 automatic runs a day. Measured on 17 Nov 2026, by the price list (the bill itself
  was not read): each of the three hand-started runs held an RTX 3070 ($0.13/h) for about 5 minutes, about one cent;
  the run started by the drift check held an RTX 3080 Ti ($0.18/h) for 5 min 32 s, about 2 cents. A run that finds no
  free GPU costs nothing.
- The VM: `Standard_B4as_v2` in austriaeast is $0.173 per hour on Azure's price list, about $4.15 a day (the 8 GB
  `Standard_B2as_v2` is $0.0864 per hour, about $2 a day), from the Azure
  for Students credit. No card is charged: when the credit ends, Azure disables the subscription. Check Cost
  Management. Deallocate the VM when it is not needed:

```bash
az vm deallocate -g skysat-server -n skysat-server    # stop; the disk and IP still cost about $6 a month
az vm start -g skysat-server -n skysat-server         # start again; the stack comes back by itself
az group delete -n skysat-server                      # remove everything, including the server's prediction log
```

## How it was built

```bash
az group create -n skysat-server -l austriaeast
az vm create -g skysat-server -n skysat-server --image Ubuntu2404 --size <now Standard_B4as_v2, 16 GB; Standard_B2as_v2, 8 GB, works with LLM_DTYPE=bfloat16; the first version used Standard_B2ats_v2, 1 GB> \
  --admin-username skysat --ssh-key-values ~/.ssh/skysat-server.pub --public-ip-sku Standard \
  --storage-sku StandardSSD_LRS --os-disk-size-gb 32 --nsg-rule SSH
az network public-ip update -g skysat-server -n skysat-serverPublicIP --dns-name <name>
az network nsg rule delete -g skysat-server --nsg-name skysat-serverNSG -n default-allow-ssh
az network nsg rule create -g skysat-server --nsg-name skysat-serverNSG -n web --priority 1010 --destination-port-ranges 80 443
```

Azure for Students only allows a few regions. A new subscription may first need
`az provider register -n Microsoft.Network --wait` (and the same for `Microsoft.Compute`).

On the VM: 2 GB of swap, Docker, a clone of this repo in `/opt/skysat`, the watchdog in cron, the read-only database
user, and these two files, which are never in git:

```
/opt/skysat/gateway-users    the logins: tutor:<hash>, team:<hash>, github:<hash> and trainer:<hash> for the loop (openssl passwd -apr1)
/opt/skysat/.env (mode 600)
  MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD   the team's DagsHub
  POSTGRES_PASSWORD, GRAFANA_ADMIN_PASSWORD, GRAFANA_DB_PASSWORD            random, made for this server
  SITE_ADDRESS=<name>.austriaeast.cloudapp.azure.com
  (MODEL_RELOAD_SECONDS is 60 on the server, set in docker-compose.server.yml: a promotion goes live within a minute)
  optional, for AUTO_TRAIN=1: RUNPOD_API_KEY, QUEUE_URL=https://<SITE_ADDRESS>, QUEUE_USER=trainer, QUEUE_PASSWORD
  API_IMAGE=ghcr.io/deensza/skysat-mlops-api:latest                         nothing is built on the server
  COMPOSE_FILE=docker-compose.yml:docker-compose.server.yml
  COMPOSE_PROJECT_NAME=skysat-mlops
```

Then `docker compose pull && docker compose up -d --no-build`. `API_IMAGE` keeps its name: it is now the complaint-box image
(`llm/Dockerfile`).

**The read-only database user.** Grafana reads through the role `grafana_reader`. Create it by hand, with
`docker compose exec postgres psql -U skysat -d skysat`:

```sql
CREATE ROLE grafana_reader LOGIN PASSWORD '<GRAFANA_DB_PASSWORD from .env>';
GRANT SELECT ON llm_scores TO grafana_reader;
```

The table must exist first: the API creates it with its first logged answer, or `monitoring/init.sql` does on a new
volume. The role gets nothing on the schema `queue`, where the review text is. Read from the server's database on 18 Nov
2026: `grafana_reader` had SELECT on the old table `predictions` only and no grant on `llm_scores`, so all 11 dashboard
panels failed there with `permission denied for table llm_scores (SQLSTATE 42501)`. The role was created by hand on 5 Nov,
before `llm_scores` existed; `monitoring/init.sql` runs only when the database volume is created, the server's volume is
from 5 Nov, and a new table gives other roles nothing.

**Reaching the VM.** SSH is closed. `az vm run-command invoke` works from anywhere, over HTTPS (the university Wi-Fi
blocks SSH anyway: its Palo Alto firewall answers the handshake). To open SSH for a while:
`az vm open-port -g skysat-server -n skysat-server --port 22 --priority 1000`, then delete that rule again.

## CD: deploy on every push to main (written, not switched on)

This job ran once. CI/CD run 43416289152 (push to `main`, commit be607ef, 13 Nov 2026) ran it on the self-hosted runner
`skysat-server` (labels self-hosted, skysat) from 13:09 to 13:11 UTC and it succeeded: it pulled the new image,
restarted the stack and the smoke test passed. A second deploy the same day (commit 9aabbbd, 14:56 UTC) was cancelled.
After that the runner was removed and the variable `DEPLOY_TO_RUNNER` deleted, and every CI run on `main` since has
`deploy: skipped`. The rollback was added on 16 Nov 2026 (commit 317f0fb) and has never run. CI pushes the image to
ghcr.io on every push to `main`, and the server is updated by hand with the `az vm run-command` line in
[Switching it on](#switching-it-on), step 1.

The `deploy` job in [ci.yml](../.github/workflows/ci.yml) is written to run on a self-hosted runner on this VM: it
pulls the image CI just built, restarts the stack and runs the smoke test on the server's own port. That port has no
login and reaches only the API, so the job's last step checks the public way in
(`.github/scripts/gateway-test.sh`: the login, the queue's rule, and Grafana reading the prediction log as its
read-only user). From a PC: `SMOKE_LOGIN="github:$(cat ~/.config/skysat/login-github)" bash
.github/scripts/gateway-test.sh https://<server>`. Only the repository owner can switch the job on: steps 1, 2 and 4
need a repository admin (Deniz, `deensza`), because a collaborator cannot see Settings or Runners.

1. Settings → Actions → General → *Fork pull request workflows from outside collaborators* → **Require approval for
   all external contributors**. The repo is public: never approve runs from strangers, because their code would run
   on this server, which holds the DagsHub token.
2. Settings → Actions → Runners → **New self-hosted runner**, Linux x64: copy the token from the page (valid 1 hour)
   and the version number it shows.
3. On the VM, through the same `az vm run-command invoke` as in [Switching it on](#switching-it-on) step 1 (it runs as
   root, so the runner's own files are made as user `skysat` with `sudo -u skysat`), with `<version>` and
   `<token from the page>` taken from the page of step 2 (the runner refuses to be configured as root):

   ```bash
   mkdir -p /opt/actions-runner && chown skysat /opt/actions-runner && cd /opt/actions-runner
   sudo -u skysat bash -c 'curl -o actions-runner-linux-x64.tar.gz -L https://github.com/actions/runner/releases/download/v<version>/actions-runner-linux-x64-<version>.tar.gz && tar xzf actions-runner-linux-x64.tar.gz && ./config.sh --url https://github.com/deensza/skysat-mlops --token <token from the page> --labels skysat --unattended --name skysat-server'
   ./svc.sh install skysat && ./svc.sh start
   ```

   The user `skysat` must be in the `docker` group (`usermod -aG docker skysat`), and `/opt/skysat/.env` (mode 600,
   the job's `COMPOSE_ENV_FILES`) must be readable by it: `chown skysat /opt/skysat/.env`. These lines were written
   from GitHub's runner page and have not been run on this VM.
4. Settings → Secrets and variables → Actions → Variables: `DEPLOY_TO_RUNNER` = `true`.

The deploy job remembers the image that was running and goes back to it if the smoke test fails (only the image: the
compose and gateway files stay as they are). The image it goes back to is the name `docker inspect` reports for the running container, which on this server is
`ghcr.io/deensza/skysat-mlops-api:latest`, so the rollback returns to whatever the local `latest` tag was. It has not
been tried on the server.

A runner uses memory on a VM and is the one thing that runs outside code here, so stop it when it is not needed.

## Measured

On 2026-11-16, on the `Standard_B2as_v2` VM with champion version 1:

| What | Result |
|---|---|
| One review in the live loop | about one minute (34 s for the short smoke-test review) |
| 100 healthy reviews (2019-01) | live MAE 0.950 ± 0.083, recommended accuracy 95.0%, drift check "no drift" |
| then 300 harsh reviews (2019-02) | MAE 1.150 ± 0.061, recommended accuracy 89.0% on those 300, drift check "DRIFT" (on MAE) |
| all 400 together | MAE 1.100 ± 0.050, accuracy 90.5%, "DRIFT" |

The ± is one standard error.

On 2026-11-18, on the `Standard_B4as_v2` VM with float32: one review takes 6.2 to 6.6 s through `POST /score` or
`POST /ask`; after an update the API answered `/health` again 10 s and 45 s after its container was replaced (two
updates, base model already in the volume).

Not yet measured on the server: the failure table above, which is from 2026-11-05. The numbers of 2026-11-05 were
for the previous tabular API on a 1 GB VM and no longer apply.

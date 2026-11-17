# The team's server

Serving and monitoring run on one always-on VM, so they do not depend on a team PC being on. It is the same stack as on a
PC (the complaint-box `api`, Nginx, PostgreSQL, Grafana), with [docker-compose.server.yml](../docker-compose.server.yml) on
top. Training happens on a rented RunPod GPU or a local NVIDIA card; the adapter and the `champion` alias are in DagsHub.

| | |
|---|---|
| Machine | Ubuntu 24.04 VM, Azure `Standard_B2as_v2` (2 vCPU, 8 GB RAM; resized on 2026-11-16). The LLM needs **at least 8 GB** (the `api` container has `mem_limit: 8g`; the 1.5B-parameter model alone needs ~4 GB in bfloat16). The first version ran on a `Standard_B2ats_v2` (1 GB RAM), which cannot host it |
| Runs | Caddy (HTTPS), Nginx (the gateway), 1 `api` container (the LLM, CPU), the `intake` (`POST /reviews` and the queue), PostgreSQL, Grafana |
| Does not run | local MLflow (the API loads the champion from DagsHub), the k6 load test, any training (worker 2 runs on a RunPod pod) |
| Holds | the review queue (`queue.reviews`, review text readable only by the `skysat` database user) and the prediction log (numbers only). The base model (~3 GB) is downloaded into a Docker volume at the first start |
| Addresses | `https://<server>/docs` (API, `POST /score`), `https://<server>/grafana/d/skysat` (dashboard). The name is not in this public repo on purpose |

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
| Nothing else reachable | Azure firewall: 80 and 443 only, SSH closed. PostgreSQL and the health port listen on the server itself; Nginx forwards only `/score`, `/model`, `/health`, `/docs`, `/openapi.json`, `/scores/recent`, `/grafana/`, `/reviews` (to the intake, any login), and `/queue/claim|done|stats` (to the intake, **only for the `trainer` login**: 403 for anyone else) | 22, 3000, 5432, 8001, 8080 closed from outside |
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
- **The drift check** is [drift.yml](../.github/workflows/drift.yml), scheduled hourly at :35. It reads `GET /model` and
  `GET /scores/recent?version=N` (aggregates only) and judges only the serving model. On DRIFT it opens one issue
  ("Drift detected: the serving model fell below the gate") and, only if the repository variable `AUTO_RETRAIN` is
  `true`, calls the Train workflow with `--replay 0 --epochs 3` (the new reviews only: after drift the old labels are
  the problem). A scheduled run with missing secrets is skipped with a warning.
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
  self-hosted runner and `DEPLOY_TO_RUNNER` is unset, so the job has never run. The server is updated by hand with the
  `az vm run-command` line in step 1 below.

### What has run (17 Nov 2026)

The loop was switched on on 17 Nov 2026: the secrets and variables below are set and `AUTO_RETRAIN` is `true`.

- **The drift check, by hand.** 08:06 UTC: version 4, 267 requests, "no drift" (MAE 0.820 ± 0.080, recommended
  0.899 ± 0.018). 13:12 UTC, after 492 more reviews with a strong label shift were posted: version 4, 511 requests,
  "DRIFT" (MAE 1.209 ± 0.075, recommended 0.816 ± 0.017). It opened issue 18 and called the Train workflow.
- **The Train workflow, by hand**, three times between 06:23 and 06:45 UTC: each rented one RTX 3070 for about
  5 minutes. The first trained on a batch of 300 reviews and registered version 5; the champion stayed version 4.
  The other two were tests of the launcher that train nothing. Table: [llm.md](llm.md#what-ran-on-runpod-17-nov-2026).
- **The Train workflow, called by the drift check** at 13:12 UTC: the cooldown and the queue check passed (319 reviews
  ready), then RunPod had none of the six GPU types then in the list free ("No GPU had capacity"). Nothing was
  rented, the job failed, and the cooldown did not start. The RTX 4090 was then added as a seventh type. A second
  drift check by hand at 13:38 UTC (version 4, 528 requests, MAE 1.203 ± 0.074, recommended 0.818 ± 0.017, "DRIFT")
  called Train again: all seven types were out of stock, nothing was rented.
- **Not seen yet:** a training started by a drift alarm, the promotion after it, and a drift check that says
  "no drift" for the new version.
- **The hourly schedule.** `drift.yml` reached `main` at 06:17 UTC with the schedule `35 * * * *`. GitHub started no
  scheduled run in the seven hourly slots from 06:35 to 12:35 UTC, although the workflow was listed as active. At
  13:08 UTC the workflow was disabled and enabled again. No scheduled run had appeared by 13:40 UTC (the 13:35 slot included). The cause is not known.
- **On GitHub the Train runs look failed** even when they trained: RunPod answers 403 to this project's API key on
  the pod-log endpoint, so the launcher reads the outcome from this server (`/queue/stats`, `/model`). A run that
  registered a version ends green; a run that registered nothing ends red, whether the gate failed, there were too few
  reviews, or the trainer crashed. The summary page of the run says which numbers changed.

### Switching it on

1. Merge into `main` and put the new image on the server: **done on 2026-11-16** (`main` runs here with the intake and
   the queue). For reference:

   ```bash
   az vm run-command invoke -g skysat-server -n skysat-server --command-id RunShellScript --scripts 'cd /opt/skysat && git pull -q && docker compose pull api intake && docker compose up -d --no-build' --query "value[0].message" -o tsv
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
stops pods started by the intake. To lock a robot out, delete its line in `gateway-users`.

### Before a demo: freeze it

Set `AUTO_RETRAIN` and `DEPLOY_TO_RUNNER` to `false` (or disable the two workflows with `gh workflow disable`), so
nothing retrains or redeploys while presenting. If the demo needs them, run them by hand (`gh workflow run train.yml`).
Enable them again afterwards.

## Cost and switching it off

### What it can cost

- A training run: the Train workflow stops the pod after 45 minutes; at the dearest GPU in the list (an RTX 4090 at
  $0.34/h, tried only when the six cheaper types are all out of stock) that is at most about $0.26 per run. With the
  6 h cooldown there are at most 4 automatic runs a day. Measured on 17 Nov 2026: each of the three hand-started runs
  held an RTX 3070 ($0.13/h) for about 5 minutes, about one cent by the price list (the bill itself was not read). A
  run that finds no free GPU costs nothing.
- The VM: `Standard_B2as_v2` in austriaeast is $0.0864 per hour on Azure's price list, about $2 a day, from the Azure
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
az vm create -g skysat-server -n skysat-server --image Ubuntu2404 --size <a size with >= 8 GB RAM, now Standard_B2as_v2; the first version used Standard_B2ats_v2, 1 GB> \
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

**Reaching the VM.** SSH is closed. `az vm run-command invoke` works from anywhere, over HTTPS (the university Wi-Fi
blocks SSH anyway: its Palo Alto firewall answers the handshake). To open SSH for a while:
`az vm open-port -g skysat-server -n skysat-server --port 22 --priority 1000`, then delete that rule again.

## CD: deploy on every push to main (written, not switched on)

As of 17 Nov 2026 this job has never run: the VM has no runner and `DEPLOY_TO_RUNNER` is unset. CI pushes the image to
ghcr.io on every push to `main`, and the server is updated by hand with the `az vm run-command` line in
[Switching it on](#switching-it-on), step 1.

The `deploy` job in [ci.yml](../.github/workflows/ci.yml) is written to run on a self-hosted runner on this VM: it
pulls the image CI just built, restarts the stack and runs the smoke test on the server's own port. Only the
repository owner can switch it on:

1. Settings → Actions → General → *Fork pull request workflows from outside collaborators* → **Require approval for
   all external contributors**. The repo is public: never approve runs from strangers, because their code would run
   on this server, which holds the DagsHub token.
2. Settings → Actions → Runners → **New self-hosted runner**, Linux x64: copy the token from the page (valid 1 hour).
3. On the VM, as user `skysat`, install the runner with the label `skysat`, and start it as a service.
4. Settings → Secrets and variables → Actions → Variables: `DEPLOY_TO_RUNNER` = `true`.

The deploy job remembers the image that was running and goes back to it if the smoke test fails (only the image: the
compose and gateway files stay as they are). The rollback has not been tried on the server.

A runner uses memory on a VM and is the one thing that runs outside code here, so stop it when it is not needed.

## Measured

On 2026-11-16, on the `Standard_B2as_v2` VM with champion version 1:

| What | Result |
|---|---|
| One review in the live loop | about one minute (34 s for the short smoke-test review) |
| 100 healthy reviews (2019-01) | live MAE 0.950 ± 0.083, recommended accuracy 95.0%, drift check "no drift" |
| then 300 harsh reviews (2019-02) | MAE 1.150 ± 0.061, recommended accuracy 89.0% on those 300, drift check "DRIFT" (on MAE) |
| all 400 together | MAE 1.100 ± 0.050, accuracy 90.5%, "DRIFT" |

The ± is one standard error. Not yet measured on the server: the restart time (until `/score` answers, including loading
the base model from the volume) and the failure table above, which is from 2026-11-05. The numbers of 2026-11-05 were
for the previous tabular API on a 1 GB VM and no longer apply.

# The team's server

Serving and monitoring run on one always-on VM, so they do not depend on a team PC being on. It is the same stack as on a
PC (the complaint-box `api`, Nginx, PostgreSQL, Grafana), with [docker-compose.server.yml](../docker-compose.server.yml) on
top. Training happens on a GPU PC or Colab; the adapter and the `champion` alias are in DagsHub.

| | |
|---|---|
| Machine | Ubuntu 24.04 VM with **at least 8 GB of RAM** (the `api` container has `mem_limit: 8g`; the 1.5B-parameter model alone needs ~4 GB in bfloat16). The first version ran on an Azure `Standard_B2ats_v2` (1 GB RAM), which cannot host the LLM: resize it before the first deploy |
| Runs | Caddy (HTTPS), Nginx (the gateway), 1 `api` container (the LLM, CPU), PostgreSQL, Grafana |
| Does not run | local MLflow (the API loads the champion from DagsHub), the k6 load test, any training |
| Holds no data | no dataset file is ever copied here; the replay runs on GitHub or a laptop and sends reviews over HTTPS. The base model (~3 GB) is downloaded into a Docker volume at the first start |
| Addresses | `https://<server>/docs` (API, `POST /score`), `https://<server>/grafana/d/skysat` (dashboard). The name is not in this public repo on purpose |

## The gateway: the only way in

```
internet -> Azure firewall (only 80, 443) -> Caddy (HTTPS) -> Nginx (rate limit, then login) -> API or Grafana
```

| Protection | How | Checked from outside |
|---|---|---|
| Only 2 accounts | HTTP Basic login in Nginx, `tutor` and `team`, 128-bit random passwords (a third, `github`, only for [the automatic loop](#the-automatic-loop)). The browser asks once; Swagger and Grafana then just work | No login or a wrong one: 401 on every path |
| Encryption | Let's Encrypt certificate, renewed by Caddy; port 80 only redirects | HTTP answers 308 to HTTPS |
| Password guessing and flooding | 20 requests/s per address (burst 100), checked **before** the login; addresses a client makes up are ignored | 429 under load, the tutor still gets in right after |
| Oversized requests | 64 kB body limit (a review is a few kB; the API accepts 50,000 characters) | 100 kB: 413 *(re-check: the old 16 kB limit was the one tested)* |
| Nothing else reachable | Azure firewall: 80 and 443 only, SSH closed. PostgreSQL and the health port listen on the server itself; Nginx forwards only `/score`, `/model`, `/health`, `/docs`, `/openapi.json`, `/scores/recent`, and `/grafana/` | 22, 3000, 5432, 8080 closed from outside |
| A viewer cannot damage the log | Grafana's viewers can send their own SQL, so its data source is a read-only database user, `grafana_reader` | `DROP`, `DELETE`, reading server files, running programs: all denied |

Not protected against: someone who gets a password can use the API within the limits, and the DagsHub token sits in the
server's `.env`. To lock someone out, delete their line in `gateway-users`.

## Staying alive

- **Crash:** Docker restarts a container by itself (`restart: unless-stopped`), also after a reboot.
- **Hang or stopped container:** [monitoring/watchdog.sh](../monitoring/watchdog.sh), from cron every 5 minutes. It
  starts stopped containers at once and restarts the API after 3 failed health checks.
- **Disk:** container logs are capped at 20 MB each.
- **Dashboard:** opens on the last 30 days, so a visitor sees data even if no traffic was sent today. Once
  [the automatic loop](#the-automatic-loop) is on, GitHub sends reviews every hour; without it, before showing it, from a PC
  with `data/reviews/AirlineReviews.csv`: `API_USER=team API_PASSWORD=$(cat ~/.config/skysat/login-team) make llm-replay URL=https://$(cat ~/.config/skysat/server-address) MONTHS=3 PER_MONTH=50`
  (a review takes seconds on CPU, one at a time).

| Failure, tested for real on 2026-11-05 with the previous (tabular) API on the 1 GB VM: re-test with the LLM | Back after |
|---|---|
| Whole VM rebooted | 218 s |
| Nginx stopped | next watchdog run (at most 5 min) |
| API process killed | 100 s |
| API hung | 67 s after the third failed check (about 15 min in all) |

## The automatic loop

Two GitHub workflows use the server like a real airline would: [traffic.yml](../.github/workflows/traffic.yml) replays
live reviews with their true labels (a scenario such as `harsh` can be chosen), and
[drift.yml](../.github/workflows/drift.yml) reads `GET /scores/recent` (aggregates only) through the gateway and opens a
GitHub issue when the live MAE or recommended accuracy is past the gate limits. **Nothing trains on GitHub or on this
machine**: a person retrains on a GPU (`make llm-cycle`), and the API checks DagsHub every `MODEL_RELOAD_SECONDS` for the
new champion. Both schedules are commented out in the workflow files: enable them for the final run.

`GET /scores/recent` sits behind the same login as Grafana. It returns counts, means and shares, never review text. The
replay sends review text from the GitHub runner to the server over HTTPS.

**Switching it on**, once, in this order:

1. `osm` merged into `main`: scheduled workflows only run from `main`, and CI builds the image there.
2. The new image on the server (the first start downloads the 3 GB base model; the VM needs the RAM above):

   ```bash
   az vm run-command invoke -g skysat-server -n skysat-server --command-id RunShellScript --scripts 'cd /opt/skysat && git pull -q && docker compose pull api && docker compose up -d --no-build' --query "value[0].message" -o tsv
   ```

3. A gateway login only for GitHub, so the tutor's and the team's passwords never leave this PC and the robot can
   be locked out alone. The password is made and kept on the PC, and only its hash goes to the server:

   ```bash
   (umask 077; openssl rand -hex 16 > ~/.config/skysat/login-github)
   ```
   ```bash
   az vm run-command invoke -g skysat-server -n skysat-server --command-id RunShellScript --scripts "echo 'github:$(openssl passwd -apr1 -in ~/.config/skysat/login-github)' >> /opt/skysat/gateway-users && cd /opt/skysat && docker compose exec -T nginx nginx -s reload" --query "value[0].message" -o tsv
   ```

4. The repository owner adds the secrets `SKYSAT_SERVER` (the address), `SKYSAT_LOGIN` (`github:` + the password from
   step 3) and `KAGGLE_API_TOKEN` (the traffic workflow downloads the dataset onto the temporary runner), sent to them
   privately, never in an issue or a comment: the repository is public.
5. First runs by hand: `gh workflow run traffic.yml -f scenario=harsh`, then `gh workflow run drift.yml`, and read both
   runs' logs. Then remove the `#` before `schedule:` and its `cron` line in both files.

**Switching it off**: put the `#` back, or `gh workflow disable traffic.yml` and `gh workflow disable drift.yml`.
To lock GitHub out of the server too, delete the `github` line in `gateway-users`.

## Cost and switching it off

About $3 to $7 for 15 days from the Azure for Students credit, depending on whether the free VM hours apply: check
Cost Management. No card is charged: when the credit ends, Azure disables the subscription.

```bash
az vm deallocate -g skysat-server -n skysat-server    # stop; the disk and IP still cost about $6 a month
az vm start -g skysat-server -n skysat-server         # start again; the stack comes back by itself
az group delete -n skysat-server                      # remove everything, including the server's prediction log
```

## How it was built

```bash
az group create -n skysat-server -l austriaeast
az vm create -g skysat-server -n skysat-server --image Ubuntu2404 --size <a size with >= 8 GB RAM; the first version used Standard_B2ats_v2, 1 GB> \
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
/opt/skysat/gateway-users    the logins: tutor:<hash>, team:<hash>, and github:<hash> for the loop (openssl passwd -apr1)
/opt/skysat/.env (mode 600)
  MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD   the team's DagsHub
  POSTGRES_PASSWORD, GRAFANA_ADMIN_PASSWORD, GRAFANA_DB_PASSWORD            random, made for this server
  SITE_ADDRESS=<name>.austriaeast.cloudapp.azure.com
  (MODEL_RELOAD_SECONDS is 60 on the server, set in docker-compose.server.yml: a promotion goes live within a minute)
  API_IMAGE=ghcr.io/deensza/skysat-mlops-api:latest                         nothing is built on the server
  COMPOSE_FILE=docker-compose.yml:docker-compose.server.yml
  COMPOSE_PROJECT_NAME=skysat-mlops
```

Then `docker compose pull && docker compose up -d --no-build`. `API_IMAGE` keeps its name: it is now the complaint-box image
(`llm/Dockerfile`).

**Reaching the VM.** SSH is closed. `az vm run-command invoke` works from anywhere, over HTTPS (the university Wi-Fi
blocks SSH anyway: its Palo Alto firewall answers the handshake). To open SSH for a while:
`az vm open-port -g skysat-server -n skysat-server --port 22 --priority 1000`, then delete that rule again.

## CD: deploy on every push to main

The `deploy` job in [ci.yml](../.github/workflows/ci.yml) runs on a self-hosted runner on this VM: it pulls the image
CI just built, restarts the stack and runs the smoke test on the server's own port. Only the repository owner can
switch it on:

1. Settings → Actions → General → *Fork pull request workflows from outside collaborators* → **Require approval for
   all external contributors**. The repo is public: never approve runs from strangers, because their code would run
   on this server, which holds the DagsHub token.
2. Settings → Actions → Runners → **New self-hosted runner**, Linux x64: copy the token from the page (valid 1 hour).
3. On the VM, as user `skysat`, install the runner with the label `skysat`, and start it as a service.
4. Settings → Secrets and variables → Actions → Variables: `DEPLOY_TO_RUNNER` = `true`.

A runner uses memory on a VM and is the one thing that runs outside code here, so stop it when it is not needed.

## Measured

Not yet measured with the LLM on the server: fill in the restart time (until `/score` answers, including loading the base
model from the volume), the latency of one review on the server's CPU, and the smoke test. The numbers of 2026-11-05 were for the previous
tabular API on a 1 GB VM and no longer apply.

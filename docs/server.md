# The team's server

Serving and monitoring run on one small always-on VM, so they no longer depend on a team PC being on. It is the same
stack as on a PC, with [docker-compose.server.yml](../docker-compose.server.yml) on top. Training stays where it was:
CT on GitHub, models and the champion in DagsHub.

| | |
|---|---|
| Machine | Azure `Standard_B2ats_v2` (2 vCPU, 1 GB RAM + 2 GB swap), Ubuntu 24.04, Austria East |
| Runs | Caddy (HTTPS), Nginx (the gateway), 1 API copy, PostgreSQL, Grafana |
| Does not run | local MLflow (the API loads the champion from DagsHub), the k6 load test |
| Holds no data | no dataset file is ever copied here; the simulator runs on a laptop and sends passengers over HTTPS |
| Addresses | `https://<server>/docs` (API), `https://<server>/grafana/d/skysat` (dashboard). The name is not in this public repo on purpose |

## The gateway: the only way in

```
internet -> Azure firewall (only 80, 443) -> Caddy (HTTPS) -> Nginx (rate limit, then login) -> API or Grafana
```

| Protection | How | Checked from outside |
|---|---|---|
| Only 2 accounts | HTTP Basic login in Nginx, `tutor` and `team`, 128-bit random passwords. The browser asks once; Swagger and Grafana then just work | No login or a wrong one: 401 on every path |
| Encryption | Let's Encrypt certificate, renewed by Caddy; port 80 only redirects | HTTP answers 308 to HTTPS |
| Password guessing and flooding | 20 requests/s per address (burst 100), checked **before** the login; addresses a client makes up are ignored | 429 under load, the tutor still gets in right after |
| Oversized requests | 16 kB body limit | 40 kB: 413 |
| Nothing else reachable | Azure firewall: 80 and 443 only, SSH closed. PostgreSQL and the health port listen on the server itself | 22, 3000, 5432, 8080 closed from outside |
| A viewer cannot damage the log | Grafana's viewers can send their own SQL, so its data source is a read-only database user, `grafana_reader` | `DROP`, `DELETE`, reading server files, running programs: all denied |

Not protected against: someone who gets a password can use the API within the limits, and the DagsHub token sits in the
server's `.env`. To lock someone out, delete their line in `gateway-users`.

## Staying alive

- **Crash:** Docker restarts a container by itself (`restart: unless-stopped`), also after a reboot.
- **Hang or stopped container:** [monitoring/watchdog.sh](../monitoring/watchdog.sh), from cron every 5 minutes. It
  starts stopped containers at once and restarts the API after 3 failed health checks.
- **Disk:** container logs are capped at 20 MB each.
- **Dashboard:** opens on the last 30 days, so a visitor sees data even if no traffic was sent today. Before showing it:
  `make simulate-server N=2000` (about 15 minutes, from a PC with `data/processed/live.csv`).

| Failure, tested for real on 2026-11-05 | Back after |
|---|---|
| Whole VM rebooted | 218 s |
| Nginx stopped | next watchdog run (at most 5 min) |
| API process killed | 100 s |
| API hung | 67 s after the third failed check (about 15 min in all) |

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
az vm create -g skysat-server -n skysat-server --image Ubuntu2404 --size Standard_B2ats_v2 \
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
/opt/skysat/gateway-users    the 2 logins: tutor:<hash>, team:<hash> (openssl passwd -apr1)
/opt/skysat/.env (mode 600)
  MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD   the team's DagsHub
  POSTGRES_PASSWORD, GRAFANA_ADMIN_PASSWORD, GRAFANA_DB_PASSWORD            random, made for this server
  SITE_ADDRESS=<name>.austriaeast.cloudapp.azure.com
  MODEL_RELOAD_SECONDS=60                                                   not 10: this runs all day against DagsHub
  API_IMAGE=ghcr.io/deensza/skysat-mlops-api:latest                         nothing is built on this small VM
  COMPOSE_FILE=docker-compose.yml:docker-compose.server.yml
  COMPOSE_PROJECT_NAME=skysat-mlops
```

Then `docker compose pull && docker compose up -d --no-build`.

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

A runner uses memory on a small VM and is the one thing that runs outside code here, so stop it when it is not needed.

## Measured (2026-11-05)

| | 2 API copies | 1 API copy (in use) |
|---|---|---|
| Restart until `/predict` answers | about 6.5 min (heavy swapping) | about 2 min |
| Typical response from a PC in Turkey | 180 ms | 192 ms |
| Slowest response | 15.9 s (1 of 30) | 19 s (1 of 60) |

300 simulated passengers over the internet: 0 failures, accuracy 96.67%, 23 ms median inside the API. The occasional
stall is the price of 1 GB: on a machine with enough RAM the same requests do not have to wait for the disk. Stalls
were seen around restarts: once warmed up, 659 predictions in 10 minutes took at most 58 ms inside the API.

2,000 passengers through the gateway: 1,253 answered with no server errors; the other 747 failed because the sending
PC went to sleep. The server did not notice, which is the point of having one. Accuracy over all logged predictions: 96.40%.

# The team's server

Serving and monitoring run on one small always-on VM, so they no longer depend on a team PC being on. It is the same
stack as on a PC, with [docker-compose.server.yml](../docker-compose.server.yml) on top. Training stays where it was:
CT on GitHub, models and the champion in DagsHub.

| | |
|---|---|
| Machine | Azure `Standard_B2ats_v2` (2 vCPU, 1 GB RAM + 2 GB swap), Ubuntu 24.04, Austria East |
| Runs | 1 API copy, Nginx (port 80), PostgreSQL (closed to the outside), Grafana (port 3000, login needed) |
| Does not run | local MLflow (the API loads the champion from DagsHub), the k6 load test |
| Holds no data | no dataset file is ever copied here; the simulator runs on a laptop and sends passengers over HTTP |
| Addresses | `http://<server-ip>/docs`, `http://<server-ip>:3000` (the IP is not in this public repo on purpose) |

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
az vm open-port -g skysat-server -n skysat-server --port 80,3000 --priority 1010
```

Azure for Students only allows a few regions. A new subscription may first need
`az provider register -n Microsoft.Network --wait` (and the same for `Microsoft.Compute`).

On the VM: 2 GB of swap, Docker, a clone of this repo in `/opt/skysat`, and `/opt/skysat/.env` (mode 600, never in git):

```
MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD   the team's DagsHub
POSTGRES_PASSWORD, GRAFANA_ADMIN_PASSWORD                                 random, made for this server
MODEL_RELOAD_SECONDS=60                                                   not 10: this runs all day against DagsHub
API_IMAGE=ghcr.io/deensza/skysat-mlops-api:latest                         nothing is built on this small VM
COMPOSE_FILE=docker-compose.yml:docker-compose.server.yml
COMPOSE_PROJECT_NAME=skysat-mlops
```

Then `docker compose pull && docker compose up -d --no-build`.

**Reaching the VM.** SSH needs the key `~/.ssh/skysat-server`, and some networks block it (the university Wi-Fi does:
the handshake is answered by a Palo Alto firewall). `az vm run-command invoke` works from anywhere, over HTTPS.

## CD: deploy on every push to main

The `deploy` job in [ci.yml](../.github/workflows/ci.yml) runs on a self-hosted runner on this VM: it pulls the image
CI just built, restarts the stack and runs the smoke test. Only the repository owner can switch it on:

1. Settings → Actions → General → *Fork pull request workflows from outside collaborators* → **Require approval for
   all external contributors**. The repo is public: never approve runs from strangers, because their code would run
   on this server, which holds the DagsHub token.
2. Settings → Actions → Runners → **New self-hosted runner**, Linux x64: copy the token from the page (valid 1 hour).
3. On the VM, as user `skysat`, install the runner with the label `skysat`, and start it as a service.
4. Settings → Secrets and variables → Actions → Variables: `DEPLOY_TO_RUNNER` = `true`.

## Measured (2026-11-05)

| | 2 API copies | 1 API copy (in use) |
|---|---|---|
| Restart until `/predict` answers | about 6.5 min (heavy swapping) | about 2 min |
| Typical response from a PC in Turkey | 180 ms | 192 ms |
| Slowest response | 15.9 s (1 of 30) | 19 s (1 of 60) |

A full VM reboot came back by itself in 218 s, smoke test passed. 300 simulated passengers over the internet: 0 failures, accuracy 96.67%, 23 ms median inside the API. The occasional
stall is the price of 1 GB: on a machine with enough RAM the same requests do not have to wait for the disk.

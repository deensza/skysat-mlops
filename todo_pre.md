# Before the demo (19 November)

Run from the project folder, on this PC: the logins and the server's address are in `~/.config/skysat/`.
None of these print a password, so they are safe on screen. Do not `cat` the login files in front of anyone.

## The day before: 18 November

- [ ] **Is everything running?** Expect `VM running`, then the 5 services and a mostly empty log.

```bash
az vm get-instance-view -g skysat-server -n skysat-server --query "instanceView.statuses[1].displayStatus" -o tsv
```
```bash
az vm run-command invoke -g skysat-server -n skysat-server --command-id RunShellScript --scripts 'cd /opt/skysat; docker compose ps --format "{{.Service}} {{.Status}}"; echo restarts:; tail -5 /var/log/skysat-watchdog.log' --query "value[0].message" -o tsv
```

- [ ] **Does the API answer through the login?** Expect version 45, `"satisfied"` and a probability.

```bash
curl -s -u "team:$(cat ~/.config/skysat/login-team)" "https://$(cat ~/.config/skysat/server-address)/model" | cut -c1-120
```
```bash
bash .github/scripts/smoke-test.sh "https://team:$(cat ~/.config/skysat/login-team)@$(cat ~/.config/skysat/server-address)" | tail -2
```

- [ ] **Fill the dashboard with fresh data.** 2,000 normal passengers, about 15 minutes. `systemd-inhibit` stops this
  PC going to sleep during the run. Expect `2000 ok, 0 failed` (a few failures are fine). Use a new SEED each time, so
  they are new passengers.

```bash
systemd-inhibit --what=sleep:idle make simulate-server N=2000 SEED=3
```

- [ ] **Look at it the way the tutor will.** Print the links, open them, log in as `team`, and press Execute on
  `POST /predict`. Look at the accuracy on the dashboard.

```bash
echo "https://$(cat ~/.config/skysat/server-address)/docs"
```
```bash
echo "https://$(cat ~/.config/skysat/server-address)/grafana/d/skysat"
```

- [ ] **Is there still credit?** Azure portal → Cost Management. If the credit ends, the server stops.

## On the morning of 19 November

- [ ] Run the two "is everything running" checks and the API check above.
- [ ] Do not change, deploy or restart anything today. A restart takes the API away for about 2 minutes.

## If something is wrong

```bash
az vm start -g skysat-server -n skysat-server
```
Starts the VM if it is off. The stack comes back by itself in about 4 minutes.

```bash
az vm restart -g skysat-server -n skysat-server
```
The first thing to try if it answers badly. Also about 4 minutes.

## Optional: show the storm on the server's dashboard

Sends storm passengers, so the tutor would see accuracy drop. It does not retrain anything. It stays in the server's
history and lowers the overall accuracy number, so only do it on purpose.

```bash
systemd-inhibit --what=sleep:idle make simulate-server MODE=drift N=500 SEED=4
```

## Never on the server's data

- `make retrain SHARED=yes`: replaces v45, the model the server is serving.
- The k6 load test against the server: 1 GB of RAM cannot take it.
- A push to `main` that changes `training/` or `skysat/`: it starts CT, which can replace v45.

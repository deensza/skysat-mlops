#!/bin/bash
# Starts the drift check from the server (docs/server.md), run by cron every 10 minutes: one call to the GitHub REST API
# that dispatches drift.yml on main, the same as `gh workflow run drift.yml`.
# Why: GitHub's own hourly schedule for drift.yml rarely fires (2 runs in 22 hourly slots, measured), and this VM has a clock.
# Cron line: */10 * * * * root /usr/local/bin/skysat-drift-trigger >> /var/log/skysat-drift-trigger.log 2>&1
# The token is read from the file DRIFT_TRIGGER_TOKEN_FILE (default /opt/skysat/github-dispatch-token, mode 600, one
# line). It is a classic personal access token with the scopes `repo` and `workflow`, created by a person who can run
# workflows in the repository (a fine-grained token cannot reach a repository where the person is only a collaborator).
# The token never appears in the output, the log or the process list: curl reads the header from stdin (`--config -`).
# One line per call: "<UTC time> dispatch drift.yml: HTTP <code> (ok)" for 204, "(FAILED)" for anything else
# (401, 403, 404: the token or its scopes; 422: the workflow is disabled or the ref is wrong). Exit 0 only on 204.
# A call every 10 minutes is safe: drift.yml has its own concurrency group and train.yml has the cooldown.
# DRIFT_TRIGGER_TOKEN_FILE and DRIFT_TRIGGER_URL replace the token path and the URL below, for the test.
token_file=${DRIFT_TRIGGER_TOKEN_FILE:-/opt/skysat/github-dispatch-token}
url=${DRIFT_TRIGGER_URL:-https://api.github.com/repos/deensza/skysat-mlops/actions/workflows/drift.yml/dispatches}

token=$(tr -d '\r\n' < "$token_file" 2>/dev/null)
if [ -z "$token" ]; then
  echo "$(date -u +%FT%TZ) dispatch drift.yml: no token in $token_file (see docs/server.md) (FAILED)"
  exit 1
fi

code=$(printf 'header = "Authorization: Bearer %s"\n' "$token" | curl --config - -sS -o /dev/null -w '%{http_code}' \
  --connect-timeout 10 --max-time 20 -X POST \
  -H 'Accept: application/vnd.github+json' -H 'X-GitHub-Api-Version: 2022-11-28' \
  -d '{"ref":"main"}' "$url" 2>/dev/null)

if [ "$code" = 204 ]; then
  echo "$(date -u +%FT%TZ) dispatch drift.yml: HTTP $code (ok)"
  exit 0
fi
echo "$(date -u +%FT%TZ) dispatch drift.yml: HTTP ${code:-000} (FAILED)"
exit 1

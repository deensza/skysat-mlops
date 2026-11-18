#!/usr/bin/env bash
# Writes one comment on the open issue "Training runs", so a Train run can be followed from GitHub while it goes on.
# Usage: bash .github/scripts/train-comment.sh "<text>"
#        DRY_RUN=1 only says what it would post (it still lists the issues, and creates and comments nothing).
# Called by llm/runpod_pod.py (GPU rented, MLflow run started, pod gone) and by the last step of
# .github/workflows/train.yml (how the run ended). Each comment ends with a link to the run's Actions page.
# Needs gh with a token that may write issues (in a workflow: GH_TOKEN, GH_REPO and `issues: write`), and
# GITHUB_SERVER_URL, GITHUB_REPOSITORY and GITHUB_RUN_ID for the link (without a run id there is no link).
# The issue is found by its exact title in the list of open issues, not by a search: GitHub's search lags behind a
# creation, and two comments a minute apart would make two issues. The number is kept in a file for the next call
# of the same run. The caller ignores a failure: a comment that is not posted must never fail or slow a training.
set -euo pipefail
text="${1:?the text of the comment}"
title="Training runs"
cache="${TRAIN_ISSUE_FILE:-${RUNNER_TEMP:-/tmp}/skysat-training-runs-issue}"

number=""
if [ -s "$cache" ]; then
  number="$(tr -cd '0-9' < "$cache")"
fi
if [ -z "$number" ]; then
  number="$(timeout 30 gh issue list --state open --limit 100 --json number,title \
    --jq ".[] | select(.title == \"$title\") | .number" | head -n 1)"
fi
if [ -z "$number" ]; then
  body="The Train workflow writes one comment per event here: a GPU was rented, the MLflow run started, the pod is gone, how the run ended. Each comment links its run. Close this issue to start a fresh one."
  if [ "${DRY_RUN:-}" = 1 ]; then
    echo "Would create the issue \"$title\""
    number="NEW"
  else
    url="$(timeout 30 gh issue create --title "$title" --body "$body")"
    number="${url##*/}"
  fi
fi
[ "$number" = NEW ] || printf '%s\n' "$number" > "$cache"

comment="$text${GITHUB_RUN_ID:+ ([run $GITHUB_RUN_ID](${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-}/actions/runs/$GITHUB_RUN_ID))}"
if [ "${DRY_RUN:-}" = 1 ]; then
  echo "Would comment on issue #$number with: $comment"
else
  timeout 30 gh issue comment "$number" --body "$comment"
fi

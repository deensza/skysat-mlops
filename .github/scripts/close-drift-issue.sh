#!/usr/bin/env bash
# The loop's last step: the drift check (drift.yml) opens the issue "Drift detected: ..." and the Train workflow
# (train.yml) calls this when its run promoted a new champion, so the issue is closed by the run that answered it.
# Before 18 Nov 2026 nothing closed it: the issue of that morning stayed open after the new version was serving.
# Usage: bash .github/scripts/close-drift-issue.sh <promoted version> [<address of the run's page>]
#        DRY_RUN=1 only says what it would close.
# Needs gh with a token that may write issues (in a workflow: GH_TOKEN, GH_REPO and `issues: write`).
# A training that registered nothing, or a version that did not beat the champion, does not call it: the issue stays.
set -euo pipefail
version="${1:?the promoted model version}"
run="${2:-}"
title="Drift detected: the serving model fell below the gate"  # the same title as in .github/workflows/drift.yml

numbers="$(gh issue list --state open --search "\"$title\" in:title" --json number,title \
  --jq ".[] | select(.title == \"$title\") | .number")"
if [ -z "$numbers" ]; then
  echo "No open drift issue: nothing to close"
  exit 0
fi

comment="A retraining answered this alarm: version $version was trained on the new reviews, passed the gate, beat the champion on the same held-back reviews and is now the champion; the server loads it within a minute${run:+ ([the run]($run))}. If the new version drifts too, the drift check opens a new issue."
for number in $numbers; do
  if [ "${DRY_RUN:-}" = 1 ]; then
    echo "Would close issue #$number with: $comment"
  else
    gh issue close "$number" --comment "$comment"
    echo "Closed issue #$number"
  fi
done

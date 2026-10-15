---
name: verify-pipeline
description: Prove that a pipeline stage actually works end to end before calling it done — run the real command, capture raw output, check Docker/API/DB/MLflow state. Use before reporting any step complete, after changing Docker, API, training, or monitoring code, or when asked "does it work?".
---

# Verifying a pipeline stage

"It should work" is not a result. Run it and show the output.

## Rules

1. **Run the real path.** Use the same command a user or CI would run (`docker compose up`, `pytest`, `k6 run`, `curl` against the running API), not a shortcut that skips the container, the network, or the config.
2. **Show raw output.** Paste the relevant stdout/exit code, trimmed to the lines that matter. A summary without the output does not count.
3. **Check the effect, not just the exit code.** A command succeeding is not the same as the thing being true. Confirm the outcome directly:
   - Container up → `docker compose ps` shows healthy/running, logs have no tracebacks.
   - API → `curl` a request and show status code and JSON body; also one invalid payload to show it is rejected.
   - Model logged → the MLflow run exists, with the params, metrics, and artifact you expect.
   - Predictions logged → `SELECT count(*)` before and after sending N requests; the difference is N.
   - Load test → p95 latency, error rate, and requests/sec from the k6 summary, at the stated load.
   - Drift → alert/flag fires on drifted traffic and stays quiet on normal traffic.
4. **Test the failure case once.** Stop a dependency or send bad input and confirm the system fails in the way you expect.
5. **Re-run from clean when it matters.** For anything the README promises ("docker compose up gives X"), verify from a clean state (`docker compose down -v` first) so leftover state cannot hide a bug.
6. **Report honestly.** If a check failed or was skipped, say which and why. If a number did not reproduce on a second run, say so with both numbers.

## Output format

End with a short block:

```
Verified:   <what was run, with the command>
Result:     <key output lines>
Not checked: <anything skipped and why>
```

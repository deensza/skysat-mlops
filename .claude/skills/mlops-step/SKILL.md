---
name: mlops-step
description: Plan and implement one numbered step of the SkySat MLOps roadmap from README.md (data prep, training, API, load test, monitoring, drift/retraining, CI/CD, docs). Use when the user says "start step N", "do the next step", or asks to build a pipeline stage.
---

# Implementing a roadmap step

The roadmap lives in the `## Status` checklist of `README.md`. One step = one reviewable unit of work = one commit.

## Procedure

1. **Read first.** Read the README checklist, the folder(s) the step belongs to, and `docker-compose.yml`. Note what earlier steps already produced — reuse it, do not rebuild it.
2. **Write the contract before the code.** In a few lines, state:
   - Inputs and outputs of this step (files, tables, HTTP endpoints, MLflow entities).
   - Definition of done as a command the user can run and an expected result.
   - What is explicitly *out of scope* (later steps).
   Confirm with the user if the contract involves a choice they have not made (e.g. which model, which metric, which port).
3. **Build the smallest thing that meets the contract.** Follow the `working-principles` skill. Stay inside the step's folder; touch shared files (`docker-compose.yml`, `README.md`) only to add what this step needs.
4. **Verify by running it.** Execute the definition-of-done command and paste the real output. For anything that runs in Docker, run it in Docker, not just locally.
5. **Close the step.** Tick the README checklist item, add a short "how to run" note if the step introduced a new command, and propose a commit message `Step N: <short title>`. Do not commit or push unless asked.
6. **Guide.** Follow `project-guide`: when presenting the contract in step 2, flag any risk or gap you see in the plan; after closing, give the recommended next move, a "Possible improvements" block (1–3 optional items, not implemented), and a "Say it in the demo" line.

## Project conventions

- Folder ownership: `data/` dataset only (never committed), `simulator/` traffic sender, `training/` train + retrain + MLflow logging, `api/` FastAPI service, `monitoring/` Grafana + DB setup, `loadtest/` k6, `tests/` pytest, `.github/workflows/` CI/CD.
- Services are added to `docker-compose.yml` incrementally; each service must start with `docker compose up <service>` on its own where possible.
- Configuration (URLs, ports, credentials) comes from environment variables with a `.env.example` entry. Real `.env` is gitignored.
- Code and comments in English; user-facing console messages may stay in whatever language the step already uses.

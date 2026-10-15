---
name: competition-compliance
description: Guardrails from the Kaggle Playground S6E10 rules — competition data must not be redistributed or committed, code must not be shared privately outside the team, attribution (CC BY 4.0), open-source licence limits. Use whenever touching data/, .gitignore, .dockerignore, Dockerfiles, CI workflows, MLflow artifacts or datasets, README attribution, repo visibility or collaborators, external data, or new dependencies.
---

# Competition compliance

This repo is the working code for https://www.kaggle.com/competitions/playground-series-s6e10 (Predicting Airline Satisfaction, sponsor Google LLC, hosted by Kaggle). The rules are a binding agreement for both developers.

Source: the competition **Rules** tab, read 2026-10-11. Rules can change; if something here matters for a decision, re-read the tab. This is a practical checklist, not legal advice.

## What the rules say (relevant parts)

| Topic | Rule | Section |
|---|---|---|
| Data licence | CC BY 4.0; any use allowed, including academic research and education. **Attribution required.** | 1.7, 2.4.a |
| Data redistribution | Do not transmit, duplicate, publish, redistribute, or make the Competition Data available to anyone not participating in the competition. Use reasonable measures to keep non-participants from getting it. | 2.4.b |
| Private sharing | No private sharing of code or data outside the Team. Code may be shared if it is made available to **all** participants on the Kaggle forums/notebooks. | 3.5.d, 3.6.a |
| Public sharing | Allowed, but the stated channel is Kaggle discussion/notebooks for this competition; sharing there licenses the code under an OSI-approved licence that permits commercial use. | 3.6.b |
| Open source in the model | Only OSI-approved licences that do not limit commercial use. | 3.6.c |
| External data | Allowed if publicly available to all participants at no cost, or "reasonably accessible". | 2.6 |
| Teams | Max 3 people, each with their own single Kaggle account; one team per person. | 2.1, 3.5 |
| Submissions | 10 per day; 2 final submissions. No hand labelling or human prediction of test data. | 2.2, 3.4.b |

## Rules for work in this repo

**Data stays out of anything that leaves the team's machines.**
- Never commit competition files (`train.csv`, `test.csv`, `sample_submission.csv`, derived splits/parquet). `.gitignore` already excludes `data/*` — keep it that way and do not weaken it.
- Never `COPY` data into a Docker image or push an image containing data to a registry. Mount `data/` as a volume. Keep a `.dockerignore` that excludes `data/`.
- CI (GitHub Actions) must not download or store the competition data. Tests use a tiny synthetic fixture generated in code, not rows copied from the real files.
- MLflow: do not log the raw dataset or a sample of it as an artifact if the tracking server/artifact store is reachable by anyone outside the team. Log row counts, schema, and a hash instead.
- Screenshots, slides, and the README may show aggregate statistics and charts; avoid pasting large blocks of raw rows.

**Code and repository visibility.**
- Keep the GitHub repo **private** and limited to the two team members while the competition is running. Making it public, or adding anyone else as a collaborator, shares Competition Code outside the team and outside the sanctioned Kaggle channel. If the course instructor needs access, treat that as a decision for the team to check against the rules — flag it, don't decide it.
- If the team wants to publish code, do it as a Kaggle notebook/discussion post for this competition, under an OSI-approved licence, rather than a public repo.
- Both developers must be registered as one Kaggle team (each with their own account). Submissions come from that team only.

**Attribution and licences.**
- Credit the data in the README and any report/slides: competition name, host/sponsor, link, and "CC BY 4.0". Also credit the original dataset if it is used.
- Dependencies that end up in the model/serving code must be OSI-approved and allow commercial use (scikit-learn, MLflow, FastAPI, XGBoost/LightGBM, etc. are fine). Check before adding anything unusual.
- Before adding **external data**, confirm it is public, free, and available to every participant, and check its own licence. Note its source in the README.

## When to speak up

Raise it, in one line, when a task would: add data or data-derived files to git, a Docker image, CI, or a shared artifact store; change repo visibility or collaborators; add external data or an unusual dependency; or publish code/results outside Kaggle. Say what rule is involved and recommend the compliant alternative; then let the team decide.

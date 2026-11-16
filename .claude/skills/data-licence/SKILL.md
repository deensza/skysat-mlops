---
name: data-licence
description: Guardrails for the 128k airline reviews dataset (Kaggle, DbCL-1.0) — never commit it, keep it out of Docker images, CI, MLflow and the Grafana table, credit the source, check licences of dependencies. Use whenever touching data/, .gitignore, .dockerignore, Dockerfiles, CI workflows, MLflow artifacts or datasets, README attribution, the simulator, or new dependencies.
---

# Data licence

Data: [Airline Reviews (128k)](https://www.kaggle.com/datasets/joelljungstrom/128k-airline-reviews) by joelljungstrom
on Kaggle, licence DbCL-1.0. The reviews were scraped from airlinequality.com by the dataset's author. It lives on the
developer's PC in `data/reviews/` (see `data/README.md` and the README section "Data and licence"). This is a practical
checklist, not legal advice; if a decision depends on it, re-read the licence on the dataset page.

## Rules for work in this repo

The repo is public, so everything committed is visible to the world.

- Never commit dataset files or anything derived from them (splits, parquet, samples of review text). `data/*` is in
  `.gitignore`; keep it that way and do not weaken it.
- Never `COPY` data into a Docker image or push an image containing data to a registry. Keep `data/` in `.dockerignore`.
- CI (GitHub Actions) must not download or store the dataset. Tests use made-up reviews generated in code.
- MLflow (and DagsHub): never log dataset rows, review text or prompts as params or artifacts. Log counts, shares and
  metrics only.
- The table Grafana reads (`llm_scores`) holds numbers only, no review text.
- On the server the reviews exist only in the private queue. Only `simulator/replay.py` sends review text anywhere, to the
  intake URL you give it.
- Screenshots, slides and the README may show aggregate statistics; do not paste review text from the dataset.
- A backup copy stays private (owner-only Drive folder), never a public link. Keep the source and licence lines with any
  copy you pass on.

## Attribution and licences

- Credit the data in the README and any report or slides: dataset name, author, link and "DbCL-1.0".
- Dependencies and models in the serving code must allow the use we make of them (Qwen2.5-1.5B-Instruct is Apache-2.0,
  transformers and peft are Apache-2.0, the repo code is MIT). Check before adding anything unusual.
- Before adding other data, check its own licence and note its source in the README.

## When to speak up

Raise it, in one line, when a task would add data or data-derived files to git, a Docker image, CI, MLflow or the Grafana
table, or add other data or an unusual dependency. Name the rule and recommend the compliant alternative.

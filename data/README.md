# data/

Dataset files are **not** committed to Git (licence and size). `data/*` is in `.gitignore` and `.dockerignore`, so the
data cannot reach Git or a Docker image by accident. It is not on DagsHub either: no dataset row, review text or prompt
is ever logged to MLflow, and the prediction log in PostgreSQL holds numbers only.

**Source:** [Airline Reviews (128k)](https://www.kaggle.com/datasets/joelljungstrom/128k-airline-reviews) by joelljungstrom
on Kaggle, licence DbCL-1.0. The reviews were scraped from airlinequality.com by the dataset's author.

```bash
kaggle datasets download joelljungstrom/128k-airline-reviews -p data/reviews --unzip   # needs a free Kaggle account/token
```

```
data/
└── reviews/
    └── AirlineReviews.csv
```

`llm/reviews.py` cleans it (review present, score 1-10, parseable flight date, recommended yes/no) and splits it by
date and airline (see [docs/llm.md](../docs/llm.md)). Only `simulator/replay.py` sends review text anywhere, and only
to the API URL you give it.

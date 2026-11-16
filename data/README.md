# data/

Dataset files are **not** committed to Git (licence and size). `data/*` is in `.gitignore` and `.dockerignore`, so the
data cannot reach Git or a Docker image by accident. It is not on DagsHub either: no dataset row, review text or prompt
is ever logged to MLflow, and the prediction log in PostgreSQL holds numbers only.

**Source:** [Airline Reviews (128k)](https://www.kaggle.com/datasets/joelljungstrom/128k-airline-reviews) by joelljungstrom
on Kaggle, licence DbCL-1.0. The reviews were scraped from airlinequality.com by the dataset's author.

```bash
kaggle datasets download joelljungstrom/128k-airline-reviews -p data/reviews --unzip   # needs a free Kaggle account/token
```

**Backup, in case the Kaggle page goes away:** a private Google Drive folder `skysat-mlops-data` (owner only; ask the
owner to share it with you, never by public link). It holds the CSV gzipped and split into 7 parts, because uploads
from the tools we use are limited to 10 MB. Download all 7 `AirlineReviews.csv.gz.partNN` files into one folder, then:

```bash
cat AirlineReviews.csv.gz.part0* > AirlineReviews.csv.gz
sha256sum AirlineReviews.csv.gz   # b0630c9b9a29aef9241db89085fb424e939a3bfac64e8854a32aaa635f0bd65e
mkdir -p data/reviews && gunzip -c AirlineReviews.csv.gz > data/reviews/AirlineReviews.csv
sha256sum data/reviews/AirlineReviews.csv   # 857c214fbc8b56cf9f3f8ef847316018489c63acc7830dc1ceb18e1a216ebc17
```

Uploaded on 2026-11-16 from the copy downloaded from Kaggle on 2026-11-12 (123,235,669 bytes; 42,654,661 gzipped).
Keep the source and licence lines above with any copy you pass on.

```
data/
└── reviews/
    └── AirlineReviews.csv
```

`llm/reviews.py` cleans it (review present, score 1-10, parseable flight date, recommended yes/no) and splits it by
date and airline (see [docs/llm.md](../docs/llm.md)). Only `simulator/replay.py` sends review text anywhere: to the
`POST /reviews` URL you give it (the intake), where the reviews wait in the server's private queue.

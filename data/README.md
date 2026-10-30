# data/

Dataset files are **not** committed to Git (size and license).

They are not on DagsHub either: the competition data may only go to participants and the original dataset's licence does not allow redistribution. `make data-card` logs a data card instead (experiment `skysat-data`): per file the source, licence, rows, schema, satisfied share and sha256, so a downloaded copy can be checked with `sha256sum` against the one the models were trained on.

1. Open https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction (a free Kaggle account is needed).
2. Click **Download**, unzip the archive.
3. Copy `train.csv` and `test.csv` into this folder:

```
data/
├── train.csv   (103,904 passengers)
└── test.csv    (25,976 passengers)
```

`python -m training.prepare_data` then writes the clean files to `data/processed/`.

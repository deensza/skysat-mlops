# data/

Dataset files are **not** committed to Git (size and license).

1. Open https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction (a free Kaggle account is needed).
2. Click **Download**, unzip the archive.
3. Copy `train.csv` and `test.csv` into this folder:

```
data/
├── train.csv   (103,904 passengers)
└── test.csv    (25,976 passengers)
```

`python -m training.prepare_data` then writes the clean files to `data/processed/`.

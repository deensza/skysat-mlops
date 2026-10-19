"""Exploratory analysis: which features drive `satisfaction`, and how.

Reads the Kaggle Playground S6E10 competition train.csv (from data/playground-series-s6e10.zip
or an unzipped train.csv) and writes two PNG charts to artifacts/eda/ (gitignored):

    1_feature_weight.png   permutation importance, Spearman correlation, mutual information
    2_relationships.png    % satisfied per rating value / category / age / distance / delay

    python -m training.analysis.eda
    python -m training.analysis.eda --train data/train.csv --out artifacts/eda

Only aggregates are plotted; the competition data itself is never copied or committed.
The column names are the competition's raw ones (no "Inflight service", boolean `satisfaction`),
so this does not use skysat.schema, which describes the original dataset.
"""

import argparse
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "playground-series-s6e10.zip"
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "eda"

SEED = 42
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e3e2dd"


def load_train(path: Path) -> pd.DataFrame:
    """Read train.csv from a .zip (no extraction needed) or a plain csv."""
    if not path.exists():
        raise SystemExit(f"ERROR: {path} not found. Download the competition data (see data/README.md).")
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z, z.open("train.csv") as f:
            return pd.read_csv(f)
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description="Plot which features drive satisfaction")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN,
                        help="competition zip or train.csv (default: data/playground-series-s6e10.zip)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder (default: artifacts/eda)")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    df = load_train(args.train)
    needed = {"id", "satisfaction", "Class", "Type of Travel", "Arrival Delay in Minutes"}
    if not needed <= set(df.columns):
        raise SystemExit(
            f"ERROR: {args.train.name} lacks columns {sorted(needed - set(df.columns))}. "
            "This script expects the competition train.csv, not the original dataset."
        )
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e3e2dd"

    y = df["satisfaction"].astype(int)
    X = df.drop(columns=["id", "satisfaction"])
    for c in ["Gender", "Customer Type", "Type of Travel", "Class"]:
        X[c] = X[c].astype("category")
    ratings = [c for c in X.columns if c not in
               ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes",
                "Gender", "Customer Type", "Type of Travel", "Class"]]
    base = y.mean()
    print(f"rows={len(df):,} satisfied={base:.3f} features={X.shape[1]}")
    print("ratings value counts (0 = not rated):")
    print((X[ratings] == 0).mean().round(3).sort_values(ascending=False).head(5).to_string())

    # --- model-based weight: permutation importance on held-out data -----------------
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    gb = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=SEED).fit(Xtr, ytr)
    auc = roc_auc_score(yva, gb.predict_proba(Xva)[:, 1])
    print(f"default HistGB validation AUC = {auc:.4f}")
    sub = Xva.sample(40000, random_state=0)
    pi = permutation_importance(gb, sub, yva.loc[sub.index], scoring="roc_auc", n_repeats=3, random_state=0, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X.columns).sort_values()
    print("permutation importance (AUC drop):\n", imp.sort_values(ascending=False).round(4).to_string())

    # --- simple, model-free signal: spearman corr + mutual information ---------------
    enc = X.copy()
    for c in enc.select_dtypes("category"):
        enc[c] = enc[c].cat.codes
    print("NaN per column:", enc.isna().sum()[lambda v: v > 0].to_dict())
    enc["Arrival Delay in Minutes"] = enc["Arrival Delay in Minutes"].fillna(enc["Departure Delay in Minutes"])
    spearman = enc.corrwith(y, method="spearman")
    esub = enc.sample(100000, random_state=0)
    mi = pd.Series(mutual_info_classif(esub, y.loc[esub.index],
                                       discrete_features=[c in ratings or X[c].dtype.name == "category" for c in X.columns],
                                       random_state=0), index=X.columns)
    print("spearman:\n", spearman.round(3).sort_values().to_string())

    # --- Figure 1: weight overview ---------------------------------------------------
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, ax = plt.subplots(1, 3, figsize=(18, 7), gridspec_kw={"width_ratios": [1.1, 1, 1]})

    ax[0].barh(imp.index, imp.values, color=BLUE, height=0.65)
    ax[0].set_title("Permutation importance\nAUC lost when the feature is shuffled", loc="left", fontsize=11)
    for i, v in enumerate(imp.values):
        ax[0].text(v + imp.max() * 0.01, i, f"{v:.3f}", va="center", fontsize=8, color=MUTED)

    s = spearman.reindex(imp.index)
    cols = [BLUE if v >= 0 else ORANGE for v in s.values]
    ax[1].barh(s.index, s.values, color=cols, height=0.65)
    ax[1].axvline(0, color=MUTED, lw=0.8)
    ax[1].set_title("Spearman correlation with satisfied\nblue = more of it, more satisfied; orange = opposite", loc="left", fontsize=11)
    ax[1].set_yticklabels([])

    m = mi.reindex(imp.index)
    ax[2].barh(m.index, m.values, color=BLUE, height=0.65)
    ax[2].set_title("Mutual information with satisfied\n(captures non-linear / categorical)", loc="left", fontsize=11)
    ax[2].set_yticklabels([])
    for a in ax:
        a.grid(axis="x", color=GRID, lw=0.7); a.set_axisbelow(True)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.suptitle(f"What drives satisfaction  |  {len(df):,} passengers, {base:.0%} satisfied  |  default gradient-boosting AUC {auc:.3f}",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "1_feature_weight.png", dpi=140)
    plt.close(fig)

    # --- Figure 2: relationship detail -----------------------------------------------
    order = list(imp.sort_values(ascending=False).index)
    rate_by_rating = pd.DataFrame({c: y.groupby(X[c]).mean() for c in [r for r in order if r in ratings]}).T
    div = LinearSegmentedColormap.from_list("div", ["#eb6834", "#efeeea", "#2a78d6"])
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1])

    axh = fig.add_subplot(gs[0, :2])
    im = axh.imshow(rate_by_rating.values, aspect="auto", cmap=div, norm=TwoSlopeNorm(vcenter=base, vmin=0, vmax=1))
    axh.set_xticks(range(rate_by_rating.shape[1])); axh.set_xticklabels(rate_by_rating.columns)
    axh.set_yticks(range(len(rate_by_rating))); axh.set_yticklabels(rate_by_rating.index, fontsize=9)
    axh.set_xlabel("rating given (0 = not rated)")
    for i in range(rate_by_rating.shape[0]):
        for j in range(rate_by_rating.shape[1]):
            v = rate_by_rating.values[i, j]
            if not np.isnan(v):
                axh.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color=INK)
    axh.set_title(f"% satisfied at each rating value (centre = overall {base:.0%}). Rows sorted by importance", loc="left", fontsize=11)
    fig.colorbar(im, ax=axh, fraction=0.03, pad=0.01)

    axc = fig.add_subplot(gs[0, 2:])
    cats = ["Class", "Type of Travel", "Customer Type", "Gender"]
    labels, vals, ns = [], [], []
    for c in cats:
        for k, g in y.groupby(X[c], observed=True):
            labels.append(f"{c}: {k}"); vals.append(g.mean()); ns.append(len(g))
    idx = np.argsort(vals)
    axc.barh([labels[i] for i in idx], [vals[i] for i in idx], color=BLUE, height=0.6)
    axc.axvline(base, color=ORANGE, lw=1.5)
    axc.text(base, len(labels) - 0.4, f" overall {base:.0%}", color=ORANGE, fontsize=9, va="bottom")
    for r, i in enumerate(idx):
        axc.text(vals[i] + 0.01, r, f"{vals[i]:.0%}  (n={ns[i]:,})", va="center", fontsize=8, color=MUTED)
    axc.set_xlim(0, 1.0)
    axc.set_title("% satisfied by category", loc="left", fontsize=11)

    def binned(ax, series, bins, title, xlabel):
        b = pd.cut(series, bins, include_lowest=True)
        g = y.groupby(b, observed=True).agg(["mean", "size"])
        ax.plot(range(len(g)), g["mean"], marker="o", color=BLUE, lw=2)
        ax.axhline(base, color=ORANGE, lw=1, ls="--")
        lab = [("<=%d" % max(i.right, 0)) if k == 0 else (">%d" % i.left) if k == len(g) - 1 else ("%d-%d" % (i.left, i.right)) for k, i in enumerate(g.index)]
        ax.set_xticks(range(len(g))); ax.set_xticklabels(lab, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(title, loc="left", fontsize=11); ax.set_xlabel(xlabel)

    binned(fig.add_subplot(gs[1, 0]), X["Age"], [0, 18, 25, 35, 45, 55, 65, 120], "% satisfied by age", "age (years)")
    binned(fig.add_subplot(gs[1, 1]), X["Flight Distance"], [0, 300, 600, 1000, 1500, 2500, 4000, 10000], "% satisfied by flight distance", "km")
    binned(fig.add_subplot(gs[1, 2]), X["Departure Delay in Minutes"], [-1, 0, 5, 15, 30, 60, 120, 5000], "% satisfied by departure delay", "minutes")
    binned(fig.add_subplot(gs[1, 3]), X["Arrival Delay in Minutes"], [-1, 0, 5, 15, 30, 60, 120, 5000], "% satisfied by arrival delay", "minutes")

    for a in fig.axes:
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.suptitle("How satisfaction relates to each feature (orange dashed / line = overall satisfied rate)", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.savefig(out / "2_relationships.png", dpi=140)
    print(f"Charts written to {out}")


if __name__ == "__main__":
    main()

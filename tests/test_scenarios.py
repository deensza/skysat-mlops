"""Simulated drift (simulator/scenarios.py) on a synthetic cleaned-reviews frame. No model, no GPU, no real reviews."""

import numpy as np
import pandas as pd
import pytest

from simulator.scenarios import SCENARIOS, apply_scenario

TEXT_SCENARIOS = ["typos", "short", "html"]


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(1)
    n = 300
    score = rng.integers(1, 11, n)
    return pd.DataFrame({
        "Review": [" ".join(f"word{w}" for w in rng.integers(0, 50, rng.integers(5, 120))) for _ in range(n)],
        "Title": [f"Title {i}" for i in range(n)],
        "score": score,
        "recommended": np.where(score >= 6, "yes", "no"),
        "complaints": [["food"]] * n,
    })


def run(df, name, **kw):
    if name in ("typos", "mixed"):  # mixed draws typos for some rows
        pytest.importorskip("nlpaug")
    return apply_scenario(df, name, **kw)


def test_scenarios_list():
    assert set(SCENARIOS) == {"none", "typos", "short", "html", "harsh", "mixed"}


def test_unknown_name(df):
    with pytest.raises(ValueError):
        apply_scenario(df, "nope")


def test_none_is_a_copy(df):
    out = apply_scenario(df, "none")
    pd.testing.assert_frame_equal(out, df)
    assert out is not df


@pytest.mark.parametrize("name", SCENARIOS)
def test_input_not_mutated_and_deterministic(df, name):
    before = df.copy(deep=True)
    a = run(df, name, seed=3)
    b = run(df, name, seed=3)
    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == len(df) and list(a.columns) == list(df.columns)


@pytest.mark.parametrize("name", TEXT_SCENARIOS)
def test_text_scenarios_keep_labels_and_title(df, name):
    out = run(df, name)
    for col in ("score", "recommended", "Title"):
        assert out[col].tolist() == df[col].tolist()


def test_short_keeps_25_words(df):
    out = apply_scenario(df, "short")
    assert out["Review"].str.split().str.len().max() <= 25
    for new, old in zip(out["Review"], df["Review"]):
        assert new.split() == old.split()[:25]


def test_html_inserts_tags_and_keeps_words(df):
    out = apply_scenario(df, "html", strength=0.5)
    assert out["Review"].str.contains(r"<br>|<div>|</p>|&nbsp;|<span class=\"x\">").mean() > 0.9
    assert apply_scenario(df, "html", strength=0.0)["Review"].tolist() == df["Review"].tolist()
    for new, old in zip(out["Review"], df["Review"]):
        kept = [w for w in new.split(" ") if w.startswith("word")]
        assert kept == old.split()


def test_html_seed_changes_result(df):
    assert apply_scenario(df, "html", seed=0)["Review"].tolist() != apply_scenario(df, "html", seed=1)["Review"].tolist()


def test_harsh(df):
    out = apply_scenario(df, "harsh")
    assert (out["score"] <= df["score"]).all() and (out["score"] >= 1).all()
    assert (out["score"] < df["score"]).sum() > 0
    expected_rec = np.where((df["recommended"] == "yes") & (out["score"] >= 6), "yes", "no")
    assert (out["recommended"] == expected_rec).all()  # only ever flips yes -> no
    assert out["Review"].tolist() == df["Review"].tolist()
    assert out["score"].tolist() == [max(1, int(s) - 2) for s in df["score"]]
    assert apply_scenario(df, "harsh", strength=4)["score"].mean() < out["score"].mean()
    assert out["score"].nunique() > 3  # labels keep a spread: a constant answer must not look perfect


def test_mixed_rows_follow_one_scenario(df):
    pytest.importorskip("nlpaug")
    out = apply_scenario(df, "mixed", seed=5)
    assert len(out) == len(df)
    same_text = out["Review"] == df["Review"]
    same_label = (out["score"] == df["score"]) & (out["recommended"] == df["recommended"])
    assert (same_text & same_label).sum() > 0          # the "none" rows (and unlucky no-ops)
    assert (~same_text).sum() > 0 and (~same_label).sum() > 0
    assert not (~same_text & ~same_label).any()        # text and labels never both change in one row
    assert (out["score"] <= df["score"]).all()


def test_mixed_with_duplicate_index(df):
    pytest.importorskip("nlpaug")
    dup = pd.concat([df.iloc[:20], df.iloc[:20]])
    assert len(apply_scenario(dup, "mixed")) == 40


def test_typos_changes_text_only(df):
    out = run(df, "typos", strength=0.3)
    assert (out["Review"] != df["Review"]).mean() > 0.9
    assert out["score"].tolist() == df["score"].tolist()

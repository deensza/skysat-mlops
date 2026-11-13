"""The complaint box's data and gate (llm/reviews.py, llm/gate.py, simulator/replay.py). No model, no GPU, no real
reviews: a synthetic fixture with the same columns as the Kaggle file."""

import numpy as np
import pandas as pd
import pytest

from llm import gate
from llm.reviews import (COMPLAINT_NAMES, SUB_RATINGS, build_examples, clean_reviews, complaints, format_completion,
                         format_prompt, get_split, parse_output, split_reviews)
from simulator.replay import pick_reviews, request_body

MONTHS = ["January", "March", "June", "September", "December"]


def make_raw(n=3000, seed=0) -> pd.DataFrame:
    """Made-up reviews: 25 airlines, flown 2014-2022, sub-ratings 0 (= not rated) to 5."""
    rng = np.random.default_rng(seed)
    score = rng.integers(1, 11, n)
    raw = pd.DataFrame({
        "AirlineName": [f"Airline {i:02d}" for i in rng.integers(0, 25, n)],
        "CabinType": rng.choice(["Economy Class", "Business Class"], n),
        "TravelType": rng.choice(["Solo Leisure", "Business", "Couple Leisure"], n),
        "DateFlown": [f"{rng.choice(MONTHS)} {y}" for y in rng.integers(2014, 2023, n)],
        "OverallScore": score.astype(float),
        "Recommended": np.where(score >= 6, "yes", "no"),
        "Title": [f"Title number {i}" for i in range(n)],
        "Review": [" ".join(f"word{w}" for w in rng.integers(0, 50, rng.integers(5, 400))) + f" REVIEWID{i}"
                   for i in range(n)],
    })
    for col in SUB_RATINGS:
        raw[col] = rng.integers(0, 6, n)
    # dirty rows the cleaning must drop
    raw.loc[0, "Review"] = None
    raw.loc[1, "OverallScore"] = np.nan
    raw.loc[2, "DateFlown"] = np.nan
    raw.loc[3, "Recommended"] = "maybe"
    raw.loc[4, "OverallScore"] = 11
    return raw


@pytest.fixture(scope="module")
def reviews():
    return clean_reviews(make_raw())


@pytest.fixture(scope="module")
def parts(reviews):
    return split_reviews(reviews, seed=0)


def test_cleaning_drops_incomplete_rows(reviews):
    assert len(reviews) == 3000 - 5
    assert reviews["score"].between(1, 10).all() and reviews["recommended"].isin(["yes", "no"]).all()
    assert reviews["date"].notna().all() and (reviews["Review"] != "").all()


def test_parse_output_exact_and_tolerant():
    assert parse_output("Score: 7/10\nRecommended: yes\nComplaint: food") == {
        "score": 7, "recommended": "yes", "complaint": "food"}
    assert parse_output("score: 10 / 10\nrecommended: No.\nComplaint: Ground Service.") == {
        "score": 10, "recommended": "no", "complaint": "ground service"}
    assert parse_output("Score: 4 out of 10\nRecommended: yes\nComplaint: NONE")["score"] == 4
    assert parse_output("Score: 4/10\nRecommended: yes\nComplaint: ground-service")["complaint"] == "ground service"


def test_parse_output_malformed_fields_are_none():
    assert parse_output("") == {"score": None, "recommended": None, "complaint": None}
    assert parse_output("I loved it!") == {"score": None, "recommended": None, "complaint": None}
    p = parse_output("Score: 14/10\nRecommended: maybe\nComplaint: the pilot")
    assert p == {"score": None, "recommended": None, "complaint": None}
    assert parse_output("Score: 0/10")["score"] is None
    p = parse_output("Score: 6/10\nRecommended: yes")  # the third line is missing: only that field is None
    assert p == {"score": 6, "recommended": "yes", "complaint": None}


def test_completion_round_trips():
    text = format_completion(7, "yes", "food")
    assert text == "Score: 7/10\nRecommended: yes\nComplaint: food"
    assert parse_output(text) == {"score": 7, "recommended": "yes", "complaint": "food"}


def row_with(**ratings) -> dict:
    return dict.fromkeys(SUB_RATINGS, 0) | {SUB_RATINGS_BY_NAME[k]: v for k, v in ratings.items()}


SUB_RATINGS_BY_NAME = {name: col for col, name in SUB_RATINGS.items()}


def test_complaints_lowest_rating_with_ties():
    assert complaints(row_with(food=1, wifi=1, seat=4)) == ["food", "wifi"]
    assert complaints(row_with(food=2, seat=1, value=3)) == ["seat"]
    assert complaints(row_with(food=2, seat=2)) == ["seat", "food"]  # in the order of COMPLAINT_NAMES
    assert COMPLAINT_NAMES == ["seat", "service", "food", "entertainment", "ground service", "wifi", "value"]


def test_complaints_none_when_nothing_is_rated_that_low():
    assert complaints(row_with(food=3, seat=5)) == ["none"]
    assert complaints(row_with()) == ["none"]  # nothing rated: 0 is "not rated", not a bad rating
    assert complaints({col: np.nan for col in SUB_RATINGS} | {"FoodRating": 4}) == ["none"]
    assert complaints(row_with(food=0, seat=3)) == ["none"]


def test_split_held_out_airlines_never_in_train_or_validation(reviews, parts):
    held = set(parts["held_out_airlines"])
    assert len(held) == round(0.2 * reviews["AirlineName"].nunique())
    assert not held & set(parts["train"]["AirlineName"])
    assert not held & set(parts["validation"]["AirlineName"])
    assert held & set(parts["live"]["AirlineName"])  # live has all airlines


def test_split_is_a_time_split(reviews, parts):
    assert parts["train"]["year"].max() <= 2018 and parts["validation"]["year"].max() <= 2018
    assert parts["live"]["year"].min() >= 2019
    assert parts["live"]["date"].is_monotonic_increasing
    assert len(parts["live"]) == (reviews["year"] >= 2019).sum()


def test_validation_is_10_percent_and_disjoint_from_train(parts):
    train_ids = set(parts["train"]["Title"])  # titles are unique in the fixture
    val_ids = set(parts["validation"]["Title"])
    assert not train_ids & val_ids
    share = len(val_ids) / (len(val_ids) + len(train_ids))
    assert abs(share - 0.10) < 0.005


def test_split_is_reproducible_and_seed_matters(reviews, parts):
    again = split_reviews(reviews, seed=0)
    assert again["held_out_airlines"] == parts["held_out_airlines"]
    assert again["train"]["Title"].tolist() == parts["train"]["Title"].tolist()
    assert split_reviews(reviews, seed=1)["held_out_airlines"] != parts["held_out_airlines"]


def test_prompt_has_no_date_and_no_label(reviews):
    for _, row in reviews.head(200).iterrows():
        prompt = format_prompt(row)
        assert str(row["year"]) not in prompt.replace(row["Title"], "").replace(row["Review"], "")
        for forbidden in ["DateFlown", "Score", "Recommended", "Complaint", "OverallScore"]:
            assert forbidden not in prompt
        assert row["AirlineName"] in prompt and row["CabinType"] in prompt and row["TravelType"] in prompt
    row = reviews.iloc[0]
    assert row["date"].strftime("%B") not in prompt and row["date"].strftime("%Y-") not in prompt


def test_prompt_cuts_the_review_to_250_words(reviews):
    long = reviews[reviews["Review"].str.split().str.len() > 300].iloc[0]
    prompt = format_prompt(long)
    review = prompt.split("Review: ", 1)[1]
    assert len(review.replace(" ...", "").split()) == 250 and review.endswith("...")
    short = reviews[reviews["Review"].str.split().str.len() < 100].iloc[0]
    assert format_prompt(short).endswith(short["Review"])


def test_build_examples_labels_match_the_row(reviews):
    ex = build_examples(reviews, 200, seed=3)
    assert len(ex) == 200 and ex["prompt"].is_unique
    for _, row in ex.iterrows():
        p = parse_output(row["completion"])
        assert p["score"] == row["score"] and p["recommended"] == row["recommended"]
        assert p["complaint"] in row["complaints"]
    again = build_examples(reviews, 200, seed=3)
    assert again["completion"].tolist() == ex["completion"].tolist()


def test_get_split_train_and_live_year(reviews):
    train, val = get_split(reviews, "train")
    assert train["year"].max() <= 2018 and len(val)
    live_train, live_val = get_split(reviews, "live-2020")
    assert set(live_train["year"]) == {2020} and set(live_val["year"]) == {2020}
    assert not set(live_train["Title"]) & set(live_val["Title"])
    assert abs(len(live_val) / (len(live_val) + len(live_train)) - 0.10) < 0.01
    with pytest.raises(ValueError):
        get_split(reviews, "test")


def perfect_texts(rows: pd.DataFrame) -> list[str]:
    return [format_completion(r["score"], r["recommended"], r["complaints"][-1]) for _, r in rows.iterrows()]


def test_gate_on_perfect_outputs(reviews):
    rows = build_examples(reviews, 300, seed=1)
    m = gate.score(rows, perfect_texts(rows))
    assert m["score_mae"] == 0 and m["score_within1"] == 1 and m["recommended_accuracy"] == 1
    assert m["complaint_accuracy"] == 1 and m["combined"] == 1 and gate.passes(m)


def test_gate_complaint_right_if_in_the_tied_list():
    rows = pd.DataFrame({"score": [5, 5], "recommended": ["yes", "yes"], "complaints": [["food", "wifi"], ["none"]]})
    m = gate.score(rows, ["Score: 5/10\nRecommended: yes\nComplaint: wifi"] * 2)
    assert m["complaint_accuracy"] == 0.5


def test_gate_always_yes_fails(reviews):
    rows = build_examples(reviews, 500, seed=1)
    m = gate.score(rows, ["Score: 5/10\nRecommended: yes\nComplaint: none"] * len(rows))
    assert abs(m["recommended_accuracy"] - (rows["recommended"] == "yes").mean()) < 1e-9
    assert m["recommended_accuracy"] < gate.RECOMMENDED_MIN and not gate.passes(m)
    assert m["combined"] < 0.9


def test_gate_malformed_outputs_are_wrong(reviews):
    rows = build_examples(reviews, 100, seed=1)
    m = gate.score(rows, ["I think it was fine."] * len(rows))
    assert m["score_mae"] == 9 and m["score_within1"] == 0
    assert m["recommended_accuracy"] == 0 and m["complaint_accuracy"] == 0
    assert m["combined"] == 0 and not gate.passes(m)
    half = gate.score(rows.head(2), [perfect_texts(rows)[0], "???"])
    assert half["score_mae"] == 4.5 and half["recommended_accuracy"] == 0.5


def test_gate_limits_separate_the_measured_runs():
    """The limits must pass v1 on normal reviews and v2 after retraining, and fail v1 under the harsh survey
    (numbers from the runs recorded in llm/gate.py)."""
    def m(mae, rec, comp):
        return {"score_mae": mae, "recommended_accuracy": rec, "complaint_accuracy": comp}
    assert gate.passes(m(0.91, 0.945, 0.68))       # v1 on its 2018 validation
    assert not gate.passes(m(1.18, 0.895, 0.77))   # v1 on harsh 2019 (the drift)
    assert gate.passes(m(0.41, 0.955, 0.745))      # v2 retrained on harsh 2019
    assert not gate.passes(m(1.69, 0.788, 0.545))  # constant answers on harsh 2019


def test_replay_picks_live_reviews_in_date_order(parts):
    live = parts["live"]
    batch = pick_reviews(live, pd.Timestamp("2019-01-01"), months=3, per_month=10)
    assert batch["date"].is_monotonic_increasing and len(batch) <= 30
    assert batch["date"].min() >= pd.Timestamp("2019-01-01") and batch["date"].max() < pd.Timestamp("2019-04-01")
    body = request_body(batch.iloc[0])
    assert set(body) == {"airline", "cabin", "travel_type", "title", "review", "true_score", "true_recommended"}
    assert isinstance(body["true_score"], int) and body["true_recommended"] in ("yes", "no")


def test_promote_moves_the_champion_alias(tmp_path, monkeypatch, capsys):
    """llm.promote: the release step for a version trained with --no-promote (a throwaway sqlite registry)."""
    mlflow = pytest.importorskip("mlflow")
    from llm import promote
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    client = mlflow.MlflowClient()
    client.create_registered_model(promote.MODEL_NAME)
    for _ in range(2):
        client.create_model_version(promote.MODEL_NAME, source=str(tmp_path))
    client.set_registered_model_alias(promote.MODEL_NAME, promote.ALIAS, "1")

    monkeypatch.setattr("sys.argv", ["promote", "2"])
    promote.main()
    assert str(client.get_model_version_by_alias(promote.MODEL_NAME, promote.ALIAS).version) == "2"
    assert "version 1 -> version 2" in capsys.readouterr().out

    monkeypatch.setattr("sys.argv", ["promote", "3"])  # not registered: refuse, alias unchanged
    with pytest.raises(Exception):
        promote.main()
    assert str(client.get_model_version_by_alias(promote.MODEL_NAME, promote.ALIAS).version) == "2"

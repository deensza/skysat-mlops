"""The complaint-box API (llm/serve.py) without torch, transformers, peft, MLflow or a model download: the three heavy
steps (load the base model, attach an adapter, generate) are replaced by fakes."""

import json
import logging
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from llm import serve
from llm.reviews import SYSTEM_PROMPT

ANSWER = "Score: 7/10\nRecommended: yes\nComplaint: food"
REVIEW = "The crew were lovely but the lasagne at row 12 was cold. Flight BA284 to Lisbon."


def review(**changes) -> dict:
    """A made-up review."""
    return {"airline": "Imaginary Air", "cabin": "Economy Class", "travel_type": "Solo Leisure",
            "title": "Cold lasagne", "review": REVIEW} | changes


def write_adapter(folder, base=serve.BASE_MODEL, prompt_version=serve.PROMPT_VERSION):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / serve.INFO_FILE).write_text(json.dumps({"base_model": base, "prompt_version": prompt_version,
                                                      "split": "train"}))
    return str(folder)


class FakeTok:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return json.dumps(messages)


@pytest.fixture
def fake(monkeypatch):
    """A fresh ScoreModel whose heavy steps are fakes; records what they were called with."""
    calls = SimpleNamespace(base=0, attach=[], prompts=[], answer=ANSWER)

    def load_base(dtype):
        calls.base += 1
        return FakeTok(), "base-model"

    def attach_adapter(model, folder, name, old):
        calls.attach.append((model, name, old))
        return "peft-model"

    def generate(tok, model, prompt):
        calls.prompts.append(prompt)
        return calls.answer, 17

    monkeypatch.setattr(serve, "load_base", load_base)
    monkeypatch.setattr(serve, "attach_adapter", attach_adapter)
    monkeypatch.setattr(serve, "generate", generate)
    monkeypatch.setattr(serve, "serving", serve.ScoreModel())
    monkeypatch.setattr(serve, "db", None)
    return calls


def test_health_is_503_before_a_model_and_200_after(fake, tmp_path):
    with pytest.raises(HTTPException) as err:
        serve.health()
    assert err.value.status_code == 503
    with pytest.raises(HTTPException) as err:
        serve.score(serve.ScoreRequest(**review()))
    assert err.value.status_code == 503

    serve.serving.install(write_adapter(tmp_path / "adapter"), "local", "test")
    assert serve.health() == {"status": "ok", "model_version": "local"}
    info = serve.model_info()
    assert info["base_model"] == serve.BASE_MODEL and info["prompt_version"] == serve.PROMPT_VERSION
    assert info["version"] == "local" and info["loaded_at"]


def test_score_a_review(fake, tmp_path):
    serve.serving.install(write_adapter(tmp_path / "adapter"), "3", "test")
    out = serve.score(serve.ScoreRequest(**review()))

    assert set(out) == {"score", "recommended", "complaint", "model_version", "latency_ms"}
    assert (out["score"], out["recommended"], out["complaint"]) == (7, "yes", "food")
    assert out["model_version"] == "3" and isinstance(out["latency_ms"], int)
    # the prompt is llm/train_lora.py's chat() around llm/reviews.py's format_prompt()
    system, user = json.loads(fake.prompts[0])
    assert system["content"] == SYSTEM_PROMPT
    assert "Airline: Imaginary Air" in user["content"] and REVIEW in user["content"]


def test_malformed_model_output_gives_none_fields(fake, tmp_path):
    fake.answer = "I think this passenger was unhappy."
    serve.serving.install(write_adapter(tmp_path / "adapter"), "3", "test")
    out = serve.score(serve.ScoreRequest(**review()))
    assert (out["score"], out["recommended"], out["complaint"]) == (None, None, None)


def test_request_validation():
    assert serve.ScoreRequest(review="Fine.").true_score is None  # everything but the review is optional
    for bad in [{"review": ""}, {"review": "x", "true_score": 11}, {"review": "x", "true_recommended": "maybe"},
                {"review": "x", "passenger_id": 1}]:
        with pytest.raises(ValueError):
            serve.ScoreRequest(**bad)


def test_adapter_for_another_base_model_is_refused(fake, tmp_path, monkeypatch, caplog):
    wrong = write_adapter(tmp_path / "wrong", base="Qwen/Qwen2.5-7B-Instruct")
    with pytest.raises(serve.IncompatibleAdapter):
        serve.serving.install(wrong, "local", "test")
    assert serve.serving.current is None and fake.base == 0  # nothing was loaded

    # LLM_ADAPTER_DIR pointing at it: an error in the log, still no model, /health stays 503
    monkeypatch.setattr(serve, "ADAPTER_DIR", wrong)
    with caplog.at_level(logging.ERROR, logger="uvicorn.error"):
        serve.serving.refresh()
    assert "Refused" in caplog.text and serve.serving.current is None

    # and a model that is already serving stays
    serve.serving.install(write_adapter(tmp_path / "good"), "1", "test")
    with pytest.raises(serve.IncompatibleAdapter):
        serve.serving.install(write_adapter(tmp_path / "old-prompt", prompt_version=2), "2", "test")
    assert serve.serving.current["version"] == "1"


def test_new_champion_is_a_new_adapter_on_the_same_base(fake, tmp_path, monkeypatch):
    champion = SimpleNamespace(version="1")
    monkeypatch.setattr(serve, "ADAPTER_DIR", None)
    monkeypatch.setattr(serve, "mlflow_client", lambda: SimpleNamespace(
        get_model_version_by_alias=lambda name, alias: champion))
    monkeypatch.setattr(serve, "champion_adapter", lambda client, base: (
        champion.version, write_adapter(tmp_path / f"download-{champion.version}")))

    serve.serving.refresh()
    serve.serving.refresh()  # same version: nothing happens
    champion.version = "2"
    serve.serving.refresh()

    assert fake.base == 1  # the base model is loaded once
    assert fake.attach == [("base-model", "v_1", None), ("peft-model", "v_2", "v_1")]
    assert serve.health()["model_version"] == "2"
    assert not (tmp_path / "download-2").exists()  # the downloaded adapter is deleted once in memory


class FakePool:
    def __init__(self, fail=False):
        self.fail, self.executed = fail, []

    def connection(self, timeout=None):
        if self.fail:
            raise ConnectionError("database is down")
        pool = self

        class Conn:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, params=None):
                pool.executed.append((sql, params))

        return Conn()


def test_db_never_receives_the_review_the_title_or_the_airline(fake, tmp_path, monkeypatch):
    pool = FakePool()
    monkeypatch.setattr(serve, "db", pool)
    monkeypatch.setattr(serve, "_table_ready", False)
    serve.serving.install(write_adapter(tmp_path / "adapter"), "3", "test")
    out = serve.score(serve.ScoreRequest(**review(true_score=8, true_recommended="yes")))

    sqls = [sql for sql, _ in pool.executed]
    assert sqls[0].startswith("CREATE TABLE IF NOT EXISTS llm_scores") and sqls[1].startswith("INSERT INTO llm_scores")
    for column in ["created_at", "model_version", "latency_ms", "score", "recommended", "complaint", "true_score",
                   "true_recommended"]:
        assert column in sqls[0]
    for column in ["review", "title", "airline", "cabin", "travel", "prompt"]:
        assert column not in sqls[0].lower()

    params = pool.executed[1][1]
    assert params == {"model_version": "3", "latency_ms": out["latency_ms"], "score": 7, "recommended": "yes",
                      "complaint": "food", "true_score": 8, "true_recommended": "yes"}
    logged = repr(pool.executed)
    for text in [REVIEW, "lasagne", "Cold", "Imaginary", "Lisbon", "BA284"]:
        assert text not in logged


def test_db_gets_nulls_when_the_truth_is_not_sent(fake, tmp_path, monkeypatch):
    pool = FakePool()
    monkeypatch.setattr(serve, "db", pool)
    monkeypatch.setattr(serve, "_table_ready", False)
    serve.serving.install(write_adapter(tmp_path / "adapter"), "3", "test")
    serve.score(serve.ScoreRequest(**review()))
    params = pool.executed[1][1]
    assert params["true_score"] is None and params["true_recommended"] is None


def test_db_failure_does_not_fail_the_request(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "db", FakePool(fail=True))
    serve.serving.install(write_adapter(tmp_path / "adapter"), "3", "test")
    assert serve.score(serve.ScoreRequest(**review()))["score"] == 7


class FakeConn:
    """Answers the three aggregate queries of recent_scores() in order."""

    def __init__(self, summary, complaints, versions):
        self.queries, self._answers = [], [[summary], complaints, versions]

    def execute(self, sql, params=None):
        self.queries.append((sql, params))
        answer = self._answers[len(self.queries) - 1]
        return SimpleNamespace(fetchone=lambda: answer[0], fetchall=lambda: answer)


def test_recent_scores_are_aggregates_only():
    conn = FakeConn((200, 6.2345, 180, 0.7777, 190, 0.9473), [("food", 100), ("none", 80), ("malformed", 20)],
                    [("1", 120), ("2", 80)])
    out = serve.recent_scores(conn, 24)
    assert out == {"hours": 24, "count": 200, "mean_score": 6.23, "labelled_score": 180, "mae": 0.7777,
                   "labelled_recommended": 190, "recommended_accuracy": 0.9473,
                   "complaint_share": {"food": 0.5, "none": 0.4, "malformed": 0.1},
                   "model_versions": {"1": 120, "2": 80}}
    sql = " ".join(q.lower() for q, _ in conn.queries)
    for column in ["review", "title", "airline", "cabin", "prompt"]:
        assert column not in sql
    assert conn.queries[0][1] == {"hours": 24, "max_error": 9}


def test_recent_scores_with_no_traffic():
    out = serve.recent_scores(FakeConn((0, None, 0, None, 0, None), [], []), 1)
    assert out["count"] == 0 and out["mae"] is None and out["complaint_share"] == {}


def test_scores_recent_needs_the_database(monkeypatch):
    monkeypatch.setattr(serve, "db", None)
    with pytest.raises(HTTPException) as err:
        serve.scores_recent(hours=24)
    assert err.value.status_code == 503


def test_scores_recent_reads_through_the_pool(monkeypatch):
    conn = FakeConn((3, 7.0, 3, 1.0, 3, 1.0), [("food", 3)], [("1", 3)])

    class Pool:
        def connection(self, timeout=None):
            class Ctx:
                def __enter__(self_):
                    return conn

                def __exit__(self_, *exc):
                    return False
            return Ctx()

    monkeypatch.setattr(serve, "db", Pool())
    monkeypatch.setattr(serve, "_table_ready", True)
    assert serve.scores_recent(hours=6)["count"] == 3

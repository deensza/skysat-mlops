"""llm/drift_check.py: DRIFT only on enough labelled requests, with the gate's limits."""

import json

from llm import drift_check, gate


def summary(mae=0.8, rec=0.95, n_score=100, n_rec=100):
    return {"hours": 24, "count": 120, "mean_score": 6.1, "mae": mae, "labelled_score": n_score,
            "recommended_accuracy": rec, "labelled_recommended": n_rec, "complaint_share": {"food": 0.6, "none": 0.4}}


def test_healthy_traffic_is_no_drift():
    drift, rows = drift_check.decide(summary())
    assert drift is False and all(r[3].startswith("ok") for r in rows)


def test_mae_above_the_gate_is_drift():
    assert drift_check.decide(summary(mae=gate.MAE_MAX + 0.18))[0] is True


def test_recommended_below_the_gate_is_drift():
    assert drift_check.decide(summary(rec=gate.RECOMMENDED_MIN - 0.005))[0] is True


def test_exactly_at_the_limits_is_not_drift():
    assert drift_check.decide(summary(mae=gate.MAE_MAX, rec=gate.RECOMMENDED_MIN))[0] is False


def test_too_few_labelled_requests_is_not_drift():
    drift, rows = drift_check.decide(summary(mae=3.0, rec=0.2, n_score=49, n_rec=10))
    assert drift is False and "not enough" in rows[0][3] and "not enough" in rows[1][3]


def test_no_labels_at_all():
    drift, _ = drift_check.decide({"count": 5, "mae": None, "recommended_accuracy": None})
    assert drift is False


def test_markdown_table():
    text = drift_check.markdown(summary(mae=1.18, rec=0.895), True, drift_check.decide(summary(mae=1.18, rec=0.895))[1])
    assert "DRIFT" in text and "| score MAE | 1.180 | <= 1.0 |" in text and "| food | 60.0% |" in text


def test_main_writes_github_output_and_exits_zero(monkeypatch, tmp_path, capsys):
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(drift_check, "fetch", lambda url, hours: summary(mae=1.18, rec=0.895))
    monkeypatch.setattr("sys.argv", ["drift_check", "--url", "http://x"])
    drift_check.main()
    assert out.read_text() == "drift=true\n"
    assert "DRIFT" in capsys.readouterr().out


def test_unreachable_server_is_no_drift(monkeypatch, tmp_path):
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr("sys.argv", ["drift_check", "--url", "http://127.0.0.1:9"])
    drift_check.main()
    assert out.read_text() == "drift=false\n"


def test_fetch_sends_the_login(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"count": 1}).encode()

    def fake_urlopen(req, timeout):
        seen["url"], seen["auth"] = req.full_url, req.get_header("Authorization")
        return Resp()

    monkeypatch.setenv("API_USER", "github")
    monkeypatch.setenv("API_PASSWORD", "pw")
    monkeypatch.setattr(drift_check.urllib.request, "urlopen", fake_urlopen)
    assert drift_check.fetch("http://x/", 24) == {"count": 1}
    assert seen["url"] == "http://x/scores/recent?hours=24" and seen["auth"].startswith("Basic ")

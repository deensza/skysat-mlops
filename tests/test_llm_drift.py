"""llm/drift_check.py: DRIFT only on enough labelled requests, beyond the gate's limits by more than the noise."""

import json

import pytest

from llm import drift_check, gate


def summary(mae=0.8, rec=0.95, n_score=100, n_rec=100, mae_sd=1.2):
    out = {"hours": 24, "count": 120, "mean_score": 6.1, "mae": mae, "labelled_score": n_score,
           "recommended_accuracy": rec, "labelled_recommended": n_rec, "complaint_share": {"food": 0.6, "none": 0.4}}
    if mae_sd is not None:
        out["mae_sd"] = mae_sd
    return out


def test_healthy_traffic_is_no_drift():
    # mae 0.88 (se 0.12): 0.76 <= 1.0. accuracy 0.92 (se 0.027): 0.947 >= 0.90.
    drift, rows = drift_check.decide(summary(mae=0.88, rec=0.92))
    assert drift is False and all(r[3].startswith("ok") for r in rows)


def test_noise_that_the_old_rule_flagged_is_no_drift():
    # accuracy 0.89, n 100: se = sqrt(0.89 * 0.11 / 100) = 0.0313, 0.89 + 0.0313 = 0.921 >= 0.90.
    # mae 1.05, sd 1.2, n 100: se = 0.12, 1.05 - 0.12 = 0.93 <= 1.0.
    drift, rows = drift_check.decide(summary(mae=1.05, rec=0.89))
    assert drift is False and all(r[3].startswith("ok") for r in rows)


def test_mae_clearly_above_the_gate_is_drift():
    # mae 1.18, sd 1.2, n 150: se = 1.2 / sqrt(150) = 0.098, 1.18 - 0.098 = 1.082 > 1.0.
    drift, rows = drift_check.decide(summary(mae=1.18, n_score=150))
    assert drift is True and rows[0][3].startswith("DRIFT") and rows[1][3].startswith("ok")


def test_recommended_clearly_below_the_gate_is_drift():
    # accuracy 0.80, n 100: se = 0.04, 0.80 + 0.04 = 0.84 < 0.90.
    drift, rows = drift_check.decide(summary(rec=0.80))
    assert drift is True and rows[1][3].startswith("DRIFT") and rows[0][3].startswith("ok")


def test_exactly_at_the_limits_is_not_drift():
    # at the limit the noise only makes the margin bigger, so no drift.
    assert drift_check.decide(summary(mae=gate.MAE_MAX, rec=gate.RECOMMENDED_MIN))[0] is False


def test_too_few_labelled_requests_is_not_drift():
    # 99 < 100 for both measures, however bad the values.
    drift, rows = drift_check.decide(summary(mae=3.0, rec=0.2, n_score=99, n_rec=99))
    assert drift is False and "not enough" in rows[0][3] and "not enough" in rows[1][3]


def test_no_labels_at_all():
    drift, _ = drift_check.decide({"count": 5, "mae": None, "recommended_accuracy": None})
    assert drift is False


def test_summary_without_mae_sd_compares_plainly():
    # an older server: se 0 for the MAE, so 1.18 > 1.0 is DRIFT, and the live cell has no ±.
    drift, rows = drift_check.decide(summary(mae=1.18, n_score=150, mae_sd=None))
    assert drift is True and rows[0][1] == "1.180"
    assert drift_check.decide(summary(mae=1.05, mae_sd=None))[0] is True


def test_markdown_table():
    s = summary(mae=1.18, rec=0.895, n_score=150)
    text = drift_check.markdown(s, True, drift_check.decide(s)[1])
    assert "DRIFT" in text and "| score MAE | 1.180 ± 0.098 | <= 1.0 |" in text and "| food | 60.0% |" in text


def test_main_writes_github_output_and_exits_zero(monkeypatch, tmp_path, capsys):
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(drift_check, "fetch", lambda url, hours, version=None: summary(mae=1.18, n_score=150))
    monkeypatch.setattr(drift_check, "serving_version", lambda url: "2")
    monkeypatch.setattr("sys.argv", ["drift_check", "--url", "http://x"])
    drift_check.main()  # returns normally: exit code 0
    assert out.read_text() == "drift=true\nversion=2\n"
    assert "DRIFT" in capsys.readouterr().out


def test_unreachable_server_fails_with_exit_code_2(monkeypatch, tmp_path, capsys):
    out, step = tmp_path / "out", tmp_path / "step"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(step))
    monkeypatch.setattr("sys.argv", ["drift_check", "--url", "http://127.0.0.1:9"])
    with pytest.raises(SystemExit) as err:
        drift_check.main()
    assert err.value.code == 2
    assert capsys.readouterr().out.startswith("## Drift check: FAILED")
    assert step.read_text().startswith("## Drift check: FAILED")
    assert not out.exists() or "drift=" not in out.read_text()


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


def fake_server(monkeypatch, answers):
    """Fake urlopen: answers maps a path to a dict (or an exception to raise); returns the list of URLs asked."""
    seen = []

    class Resp:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(self.body).encode()

    def fake_urlopen(req, timeout):
        seen.append(req.full_url)
        body = answers[req.full_url.split("http://x")[1].split("?")[0]]
        if isinstance(body, Exception):
            raise body
        return Resp(body)

    monkeypatch.setattr(drift_check.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    return seen


def test_main_judges_the_serving_version(monkeypatch, tmp_path, capsys):
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    s = {**summary(), "version": "2", "model_versions": {"1": 300, "2": 57}}
    seen = fake_server(monkeypatch, {"/model": {"version": 2}, "/scores/recent": s})
    monkeypatch.setattr("sys.argv", ["drift_check", "--url", "http://x"])
    drift_check.main()
    assert seen == ["http://x/model", "http://x/scores/recent?hours=24&version=2"]
    text = capsys.readouterr().out
    assert "Last 24 h, model version 2 only: 120 requests" in text
    assert "Versions in the window: 1 (300 requests), 2 (57 requests)" in text
    assert "too old" not in text
    assert out.read_text() == "drift=false\nversion=2\n"


def test_all_versions_does_not_ask_for_the_model(monkeypatch, tmp_path, capsys):
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    seen = fake_server(monkeypatch, {"/scores/recent": {**summary(), "version": None}})
    monkeypatch.setattr("sys.argv", ["drift_check", "--url", "http://x", "--all-versions"])
    drift_check.main()
    assert seen == ["http://x/scores/recent?hours=24"]
    text = capsys.readouterr().out
    assert "Last 24 h: 120 requests" in text and "too old" not in text
    assert out.read_text() == "drift=false\nversion=all\n"


def test_failing_model_endpoint_fails_with_exit_code_2(monkeypatch, tmp_path, capsys):
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    seen = fake_server(monkeypatch, {"/model": OSError("boom")})
    monkeypatch.setattr("sys.argv", ["drift_check", "--url", "http://x"])
    with pytest.raises(SystemExit) as err:
        drift_check.main()
    assert err.value.code == 2
    assert capsys.readouterr().out.startswith("## Drift check: FAILED") and seen == ["http://x/model"]
    assert not out.exists()


def test_old_server_without_version_filter_is_reported(monkeypatch, capsys):
    fake_server(monkeypatch, {"/model": {"version": "2"}, "/scores/recent": summary()})
    monkeypatch.setattr("sys.argv", ["drift_check", "--url", "http://x"])
    drift_check.main()
    text = capsys.readouterr().out
    assert "The server is too old to filter by version: all versions are counted." in text
    assert "Last 24 h: 120 requests" in text


def test_fetch_quotes_the_version(monkeypatch):
    seen = fake_server(monkeypatch, {"/scores/recent": {"count": 1}})
    drift_check.fetch("http://x", 6, version="a b")
    assert seen == ["http://x/scores/recent?hours=6&version=a%20b"]

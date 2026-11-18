"""The trainer can cut the training short (--max-train-seconds), and the MLflow run starts before the training so DagsHub
shows it while it runs and gets the loss every 25 steps.

CI has no torch, peft or transformers, so the real train() and run() are run against fakes:
  - sys.modules gets a fake `torch` and `peft` (train() and run() import them inside the function): tensors are plain
    lists, the optimizer, scheduler and scaler do nothing, the "model" returns loss 1.0, 2.0, 3.0 ... per step;
  - the tokenizer gives one id per character, so the prefix check in encode() is real;
  - time.time() is a counter that advances 1.0 s per call, so the cut happens at an exact step;
  - run() uses a real MLflow on a throwaway sqlite file; load_model, train, generate and gate.score are stubbed
    (gate.passes is the real one: the metrics are chosen to pass or fail its limits).
What stays unproven here (it needs a GPU): that the real torch/peft calls accept what the fakes accept."""

import subprocess
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import pytest

from llm import gate, train_lora
from llm.trainer import parse_args

ROOT = Path(__file__).resolve().parent.parent


def test_max_train_seconds_is_a_float_and_off_by_default():
    assert parse_args(["--queue-url", "x", "--max-train-seconds", "20"]).max_train_seconds == 20.0
    assert parse_args(["--queue-url", "x"]).max_train_seconds is None


def test_help_mentions_max_train_seconds():
    done = subprocess.run([sys.executable, "-m", "llm.trainer", "--help"], capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert done.returncode == 0, done.stderr[-400:]
    assert "--max-train-seconds" in done.stdout


# ---------- fakes ----------

class FakeLoss:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def backward(self):
        pass


class FakeParam:
    requires_grad = True


class FakeModel:
    device = types.SimpleNamespace(type="cpu")

    def __init__(self):
        self.calls = 0
        self.params = [FakeParam()]

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        pass

    def enable_input_require_grads(self):
        pass

    def parameters(self):
        return self.params

    def train(self):
        pass

    def eval(self):
        pass

    def print_trainable_parameters(self):
        pass

    def __call__(self, input_ids=None, attention_mask=None, labels=None):
        self.calls += 1
        return types.SimpleNamespace(loss=FakeLoss(float(self.calls)))  # the loss of step n is n

    def save_pretrained(self, path):
        (Path(path) / "adapter_model.bin").write_bytes(b"fake adapter")


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "|".join(m["content"] for m in messages) + "|"  # with the answer last: the prompt text is a prefix

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[ord(c) for c in text])


class FakeScaler:
    def __init__(self, enabled=False):
        pass

    def scale(self, loss):
        return loss

    def unscale_(self, opt):
        pass

    def step(self, opt):
        pass

    def update(self):
        pass


class FakeOpt:
    def __init__(self, params, lr=None):
        pass

    def zero_grad(self, set_to_none=True):
        pass


class FakeSched:
    def __init__(self, opt, fn):
        pass

    def step(self):
        pass


def fake_torch():
    return types.SimpleNamespace(
        float16="float16", float32="float32", Tensor=type("Tensor", (), {}),  # MLflow checks isinstance(x, torch.Tensor)
        optim=types.SimpleNamespace(AdamW=FakeOpt, lr_scheduler=types.SimpleNamespace(LambdaLR=FakeSched)),
        amp=types.SimpleNamespace(GradScaler=FakeScaler),
        tensor=lambda data, device=None: data,
        autocast=lambda *a, **k: nullcontext(),
        nn=types.SimpleNamespace(utils=types.SimpleNamespace(clip_grad_norm_=lambda params, norm: 0.0)),
        cuda=types.SimpleNamespace(is_available=lambda: False),
        manual_seed=lambda seed: None,
    )


@pytest.fixture
def fakes(monkeypatch):
    """fake torch + peft in sys.modules, and a clock that advances 1.0 s per time.time() call (starting at 0)."""
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(LoraConfig=lambda **kw: kw, get_peft_model=lambda m, c: m))
    ticks = iter(range(10_000))
    monkeypatch.setattr(train_lora, "time", types.SimpleNamespace(time=lambda: float(next(ticks))))  # only this module: MLflow keeps the real clock


def rows(n):
    return pd.DataFrame({"prompt": [f"review {i}" for i in range(n)], "completion": [f"answer {i}" for i in range(n)]})


def args_for(*extra):
    return parse_args(["--queue-url", "x", "--epochs", "2", "--batch", "2", *extra])


# ---------- train() ----------

def test_train_stops_early_when_max_train_seconds_has_passed(fakes, capsys):
    # 8 rows, batch 2, 2 epochs = 8 steps. Clock: started = 0, then 1 s per step: step 3 is at 3 s (not > 3), step 4 at 4 s (> 3).
    _, losses, info = train_lora.train(FakeModel(), FakeTokenizer(), rows(8), args_for("--max-train-seconds", "3"), "float32")
    assert info == {"steps_done": 4, "steps_planned": 8, "cut": True}
    assert losses == [1.0, 2.0, 3.0, 4.0]
    assert "Training cut after 4 of 8 steps" in capsys.readouterr().out


def test_train_without_a_limit_does_all_steps(fakes, capsys):
    _, losses, info = train_lora.train(FakeModel(), FakeTokenizer(), rows(8), args_for(), "float32")
    assert info == {"steps_done": 8, "steps_planned": 8, "cut": False}
    assert len(losses) == 8
    assert "Training cut" not in capsys.readouterr().out


def test_train_plans_epochs_times_batches_rounded_up(fakes):
    _, losses, info = train_lora.train(FakeModel(), FakeTokenizer(), rows(7), args_for(), "float32")  # ceil(7/2) = 4 per epoch
    assert info["steps_planned"] == info["steps_done"] == len(losses) == 8


def test_train_calls_log_loss_every_25_steps_and_at_the_last_step(fakes):
    calls = []
    _, _, info = train_lora.train(FakeModel(), FakeTokenizer(), rows(30), args_for(), "float32",
                                  log_loss=lambda step, mean: calls.append((step, mean)))
    assert info["steps_done"] == 30  # 15 batches x 2 epochs
    assert calls == [(25, sum(range(1, 26)) / 25), (30, sum(range(6, 31)) / 25)]  # the mean of the last 25 losses


def test_train_without_log_loss_still_prints_the_progress(fakes, capsys):
    train_lora.train(FakeModel(), FakeTokenizer(), rows(30), args_for(), "float32")
    assert "step 25/30" in capsys.readouterr().out


# ---------- run() with a real MLflow on sqlite ----------

@pytest.fixture
def mlf(tmp_path, monkeypatch, fakes):
    mlflow = pytest.importorskip("mlflow")
    before = mlflow.get_tracking_uri()
    monkeypatch.chdir(tmp_path)  # the sqlite store puts artifacts in ./mlruns
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("GIT_COMMIT", "abc123")
    mlflow.set_tracking_uri(uri)
    yield mlflow
    mlflow.set_tracking_uri(before)


PASSING = {"score_mae": 0.5, "score_within1": 0.9, "recommended_accuracy": 0.95, "complaint_accuracy": 0.7, "n": 10}
FAILING = PASSING | {"score_mae": 2.0}


def stub_run(monkeypatch, metrics, train_fn):
    """Everything heavy in run() is replaced; the real MLflow, gate.passes and the registry steps are not."""
    monkeypatch.setattr(train_lora, "load_model", lambda base, device: (FakeTokenizer(), FakeModel(), "float32"))
    monkeypatch.setattr(train_lora, "generate", lambda model, tok, prompts: ["x"] * len(prompts))
    monkeypatch.setattr(train_lora.gate, "score", lambda rows_, texts: dict(metrics, combined=gate.combined(metrics)))
    monkeypatch.setattr(train_lora, "train", train_fn)


def run_it(tmp_path, *extra):
    df = rows(4).assign(score=5, recommended="yes")
    return train_lora.run(df.iloc[:3], df.iloc[3:], args_for("--out", str(tmp_path / "adapter"), *extra))


def done_train(model, log_loss=None):
    if log_loss is not None:
        log_loss(25, 0.5)
        log_loss(30, 0.4)
    return model, [0.5] * 30, {"steps_done": 30, "steps_planned": 30, "cut": False}


def simple_train(model, tok, examples, args, dtype, log_loss=None):
    return done_train(model, log_loss)


def last_run(mlflow):
    exp = mlflow.MlflowClient().get_experiment_by_name(train_lora.EXPERIMENT)
    runs = mlflow.search_runs([exp.experiment_id], output_format="list")
    assert len(runs) == 1
    return runs[0]


def test_run_is_open_with_tag_and_params_before_the_training_and_the_loss_is_logged(mlf, tmp_path, monkeypatch):
    seen = {}

    def inspecting_train(model, tok, examples, args, dtype, log_loss=None):
        got = mlf.MlflowClient().get_run(mlf.active_run().info.run_id)
        seen.update(status=got.info.status, tags=dict(got.data.tags), params=dict(got.data.params))
        return done_train(model, log_loss)

    stub_run(monkeypatch, FAILING, inspecting_train)
    run_it(tmp_path, "--max-train-seconds", "60")
    assert seen["status"] == "RUNNING"
    assert seen["tags"]["git_commit"] == "abc123"
    for key in ("n_train", "n_gate", "prompt_version", "max_train_seconds", "epochs"):
        assert key in seen["params"], key
    assert (seen["params"]["n_train"], seen["params"]["n_gate"], seen["params"]["max_train_seconds"]) == ("3", "1", "60.0")
    history = mlf.MlflowClient().get_metric_history(last_run(mlf).info.run_id, "train_loss")
    assert [(m.step, m.value) for m in sorted(history, key=lambda m: m.step)] == [(25, 0.5), (30, 0.4)]


def test_run_with_a_failing_gate_registers_nothing_and_finishes_the_run(mlf, tmp_path, monkeypatch):
    stub_run(monkeypatch, FAILING, simple_train)
    result = run_it(tmp_path)
    assert result["registered"] is False and result["version"] is None and result["promoted"] is False
    got = last_run(mlf)
    assert got.info.status == "FINISHED"
    assert got.data.tags["gate"] == "failed" and got.data.tags["registered"] == "false"
    assert got.data.tags["train_cut"] == "false"
    assert got.data.params["train_steps_done"] == "30" and got.data.params["train_steps_planned"] == "30"
    assert mlf.MlflowClient().search_model_versions(f"name='{train_lora.MODEL_NAME}'") == []


def test_run_with_a_passing_gate_registers_and_promotes_the_first_version(mlf, tmp_path, monkeypatch):
    stub_run(monkeypatch, PASSING, simple_train)
    result = run_it(tmp_path)
    assert result["registered"] is True and str(result["version"]) == "1" and result["promoted"] is True
    got = last_run(mlf)
    assert got.data.tags["gate"] == "passed" and got.data.tags["registered"] == "true"
    champion = mlf.MlflowClient().get_model_version_by_alias(train_lora.MODEL_NAME, train_lora.ALIAS)
    assert str(champion.version) == "1"


def test_run_with_no_promote_registers_but_leaves_the_alias_alone(mlf, tmp_path, monkeypatch):
    stub_run(monkeypatch, PASSING, simple_train)
    result = run_it(tmp_path, "--no-promote")
    assert result["registered"] is True and str(result["version"]) == "1" and result["promoted"] is False
    with pytest.raises(Exception):
        mlf.MlflowClient().get_model_version_by_alias(train_lora.MODEL_NAME, train_lora.ALIAS)


def test_a_loss_that_cannot_be_logged_does_not_stop_run(mlf, tmp_path, monkeypatch, capsys):
    real, calls = mlf.log_metric, []

    def flaky(*a, **k):  # fails for the first call only
        calls.append(a)
        if len(calls) == 1:
            raise RuntimeError("network down")
        return real(*a, **k)

    def train_with_a_failing_metric(model, tok, examples, args, dtype, log_loss=None):
        monkeypatch.setattr("mlflow.log_metric", flaky)
        return done_train(model, log_loss)

    stub_run(monkeypatch, FAILING, train_with_a_failing_metric)
    run_it(tmp_path)  # must not raise
    assert "train_loss at step 25 not logged (RuntimeError)" in capsys.readouterr().out
    monkeypatch.setattr("mlflow.log_metric", real)
    history = mlf.MlflowClient().get_metric_history(last_run(mlf).info.run_id, "train_loss")
    assert [m.step for m in history] == [30]


def test_run_with_no_mlflow_creates_no_run_and_passes_no_log_loss(mlf, tmp_path, monkeypatch):
    seen = []

    def train_without_mlflow(model, tok, examples, args, dtype, log_loss=None):
        seen.append(log_loss)
        return done_train(model, log_loss)

    stub_run(monkeypatch, PASSING, train_without_mlflow)
    result = run_it(tmp_path, "--no-mlflow")
    assert result["registered"] is False and result["version"] is None
    assert seen == [None]
    assert mlf.MlflowClient().get_experiment_by_name(train_lora.EXPERIMENT) is None

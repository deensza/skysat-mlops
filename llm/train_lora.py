"""LoRA-tune a small LLM to read an airline review (score /10, recommended, main complaint), gate it, log it to MLflow
and register it. The training loop and the registry steps; the trainer worker (llm/trainer.py) feeds it batches from
the review queue, on a RunPod GPU (llm/runpod_pod.py) or a local NVIDIA card.

Only the LoRA adapter (a few tens of MB) is trained and stored: the 1.5B base model is never changed, so a new
model version is a small file the server downloads next to the base model it already has.

MLflow gets parameters, metrics and the adapter. It never gets reviews, prompts or generated text: the review dataset
(DbCL-1.0) is not ours to redistribute and the shared MLflow is public.
"""

import contextlib
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd

from llm import gate
from llm.reviews import SYSTEM_PROMPT

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # Apache-2.0
EXPERIMENT = "skysat-llm"
MODEL_NAME = "skysat-reviews"
ALIAS = "champion"
PROMPT_VERSION = 3  # bump when llm/reviews.py changes the prompt or the completion format: older adapters were tuned on the old one
INFO_FILE = "skysat_llm.json"
DEFAULT_OUT = Path("artifacts/llm/adapter")
MAX_NEW_TOKENS = 24  # the answer is three short lines


def git_commit() -> str:
    if os.environ.get("GIT_COMMIT"):  # lets a build without a .git folder still say which commit it is
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def chat(tok, prompt: str, answer: str | None = None) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    if answer is None:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok.apply_chat_template(messages + [{"role": "assistant", "content": answer}], tokenize=False)


def encode(tok, prompts: list[str], answers: list[str]) -> list[tuple[list[int], list[int]]]:
    """(input_ids, labels) per example; the loss is only on the answer, prompt tokens are labelled -100."""
    out = []
    for prompt, answer in zip(prompts, answers, strict=True):
        p = tok(chat(tok, prompt), add_special_tokens=False).input_ids
        full = tok(chat(tok, prompt, answer), add_special_tokens=False).input_ids
        assert full[:len(p)] == p, "chat template: the prompt is not a prefix of prompt + answer"
        out.append((full, [-100] * len(p) + full[len(p):]))
    return out


def load_model(base: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()  # a T4 has no bf16: fp16 + loss scaling there
    dtype = torch.bfloat16 if bf16 else (torch.float16 if device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
    return tok, model, dtype


def train(model, tok, examples: pd.DataFrame, args, dtype, log_loss=None):
    """LoRA-tune model on the examples; returns (the peft model, per-step losses, info about the steps done).
    log_loss(step, mean loss of the last 25 steps) is called every 25 steps, so a network call is about once per 25 steps.
    With args.max_train_seconds the loops stop once that time has passed (info["cut"])."""
    import torch
    from peft import LoraConfig, get_peft_model
    # recompute activations in the backward pass: fits 8 GB cards (the cheapest RunPod GPU) at ~30% more time
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    data = encode(tok, examples["prompt"].tolist(), examples["completion"].tolist())
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    steps = args.epochs * math.ceil(len(data) / args.batch)
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warmup, max(0.0, (steps - s) / max(1, steps - warmup))))
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)
    pad = tok.pad_token_id
    rng = random.Random(args.seed)
    limit = getattr(args, "max_train_seconds", None)
    started = time.time()
    losses, step, cut = [], 0, False
    model.train()
    for _ in range(args.epochs):
        order = list(range(len(data)))
        rng.shuffle(order)
        for i in range(0, len(order), args.batch):
            batch = [data[j] for j in order[i:i + args.batch]]
            width = max(len(ids) for ids, _ in batch)
            ids = torch.tensor([x + [pad] * (width - len(x)) for x, _ in batch], device=model.device)
            labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in batch], device=model.device)
            mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in batch], device=model.device)
            with torch.autocast(model.device.type, dtype=dtype, enabled=model.device.type == "cuda"):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 25 == 0 or step == steps:
                mean_loss = sum(losses[-25:]) / len(losses[-25:])
                print(f"step {step}/{steps}  loss {mean_loss:.4f}", flush=True)
                if log_loss is not None:
                    log_loss(step, mean_loss)
            elapsed = time.time() - started
            if limit is not None and elapsed > limit:
                print(f"Training cut after {step} of {steps} steps ({elapsed:.0f} s > --max-train-seconds {limit:g})")
                cut = True
                break
        if cut:
            break
    model.eval()
    return model, losses, {"steps_done": step, "steps_planned": steps, "cut": cut}


def generate(model, tok, prompts: list[str], batch: int = 16, max_new_tokens: int = MAX_NEW_TOKENS) -> list[str]:
    import torch
    tok.padding_side = "left"
    texts = []
    for i in range(0, len(prompts), batch):
        enc = tok([chat(tok, p) for p in prompts[i:i + batch]], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        texts += tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    tok.padding_side = "right"
    return [t.strip() for t in texts]


def champion_adapter(client, base: str) -> tuple[str, str] | None:
    """(version, local folder) of the registered champion adapter, or None if there is none (or it is for
    another base model or prompt version, so it cannot be compared fairly)."""
    import mlflow
    from mlflow.exceptions import MlflowException
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except MlflowException:
        return None
    download, usable = tempfile.mkdtemp(), False
    try:
        folder = mlflow.artifacts.download_artifacts(artifact_uri=version.source, dst_path=download)
        info = json.loads((Path(folder) / INFO_FILE).read_text())
        usable = info["base_model"] == base and info["prompt_version"] == PROMPT_VERSION
    finally:
        if not usable:  # the API asks again every MODEL_RELOAD_SECONDS: a download nobody uses must not pile up
            shutil.rmtree(download, ignore_errors=True)
    if not usable:
        print(f"Champion version {version.version} was tuned for {info}; not comparable, treating as no champion")
        return None
    return str(version.version), folder  # a str everywhere (a local sqlite registry gives an int)


def score_adapter(model, tok, folder: str, name: str, rows: pd.DataFrame) -> dict:
    """Load another adapter next to the one being trained (same base in memory), score it, switch back."""
    model.load_adapter(folder, adapter_name=name)
    model.set_adapter(name)
    metrics = gate.score(rows, generate(model, tok, rows["prompt"].tolist()))
    model.set_adapter("default")
    model.delete_adapter(name)
    return metrics


def run(train_rows: pd.DataFrame, eval_rows: pd.DataFrame, args, params: dict | None = None) -> dict:
    """Train on train_rows, gate on eval_rows, save the adapter to args.out; unless args.no_mlflow, log the run,
    register the version if it passes the gate, and make it the champion if it beats the current one on the same
    reviews (and not args.no_promote). Both frames need `prompt`, `completion`, `score`, `recommended`.
    Returns {"metrics", "registered", "version", "promoted", "champion_combined"}."""
    import torch
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU found; this will be very slow")
    if args.no_mlflow:
        run_ctx = contextlib.nullcontext()
    else:
        import mlflow
        mlflow.set_experiment(EXPERIMENT)
        run_ctx = mlflow.start_run()  # before the model download and the training, so the run is visible from the first minute
    with run_ctx as run_:
        if not args.no_mlflow:
            mlflow.set_tag("git_commit", git_commit())
            mlflow.log_params({k: str(v) for k, v in vars(args).items() if k not in ("no_mlflow", "queue_url")}
                              | {"prompt_version": PROMPT_VERSION, "n_train": len(train_rows), "n_gate": len(eval_rows),
                                 "device": torch.cuda.get_device_name() if device == "cuda" else "cpu"}
                              | {k: str(v) for k, v in (params or {}).items()})
        tok, model, dtype = load_model(args.base, device)

        t0 = time.time()
        def log_loss(step, loss):  # a metric that cannot be sent is one line in the log, never the end of the training
            try:
                mlflow.log_metric("train_loss", loss, step=step)
            except Exception as err:
                print(f"MLflow: train_loss at step {step} not logged ({type(err).__name__})", flush=True)
        if args.no_mlflow:
            log_loss = None
        model, losses, info_train = train(model, tok, train_rows, args, dtype, log_loss)
        train_seconds = time.time() - t0
        t0 = time.time()
        metrics = gate.score(eval_rows, generate(model, tok, eval_rows["prompt"].tolist()))
        print(f"New adapter (trained on {len(train_rows)} reviews in {train_seconds:.0f} s, gate {time.time() - t0:.0f} s):")
        print(gate.report(metrics))

        args.out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.out)
        info = {"base_model": args.base, "prompt_version": PROMPT_VERSION, "split": (params or {}).get("batch_id", "local")}
        (args.out / INFO_FILE).write_text(json.dumps(info, indent=2))
        print(f"Adapter saved to {args.out} ({sum(f.stat().st_size for f in args.out.iterdir()) / 1e6:.0f} MB)")
        result = {"metrics": metrics, "registered": False, "version": None, "promoted": False, "champion_combined": None}
        if args.no_mlflow:
            return result

        client = mlflow.MlflowClient()
        print(f"MLflow: {mlflow.get_tracking_uri()}")
        mlflow.log_params({"train_cut": str(info_train["cut"]).lower(), "train_steps_done": info_train["steps_done"],
                           "train_steps_planned": info_train["steps_planned"]})
        mlflow.set_tag("train_cut", str(info_train["cut"]).lower())
        champion = champion_adapter(client, args.base)
        if champion is not None:
            champion_metrics = score_adapter(model, tok, champion[1], "champion", eval_rows)
            result["champion_combined"] = champion_metrics["combined"]
            print(f"Champion (version {champion[0]}) on the same reviews:\n{gate.report(champion_metrics)}")
        champion_score = result["champion_combined"]

        mlflow.log_metrics({k: v for k, v in metrics.items() if not (isinstance(v, float) and math.isnan(v))}
                           | {"train_loss_last50": sum(losses[-50:]) / len(losses[-50:]), "train_seconds": train_seconds})
        if champion_score is not None:
            mlflow.log_metric("champion_combined", champion_score)
        mlflow.log_artifacts(str(args.out), "adapter")

        mlflow.set_tag("gate", "passed" if gate.passes(metrics) else "failed")
        if not gate.passes(metrics):
            mlflow.set_tag("registered", "false")
            print("Not registered: the gate failed (limits are in llm/gate.py)")
            return result
        try:
            client.create_registered_model(MODEL_NAME)
        except Exception:
            pass  # already exists
        version = client.create_model_version(MODEL_NAME, source=f"{run_.info.artifact_uri}/adapter",
                                              run_id=run_.info.run_id)
        result.update(registered=True, version=version.version)
        mlflow.set_tag("registered", "true")
        print(f"Registered {MODEL_NAME} version {version.version}")
        beats = champion_score is None or metrics["combined"] > champion_score
        if args.no_promote:
            print(f"{ALIAS} not changed (--no-promote); version {version.version} "
                  f"{'beats' if beats else 'does not beat'} it on the same reviews. "
                  f"Promote it with: make llm-promote VERSION={version.version}")
        elif beats:
            try:
                client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
            except Exception as err:  # the version exists: do not fail the batch and train the same reviews twice
                print(f"Version {version.version} is registered, but the {ALIAS} alias could not be moved "
                      f"({type(err).__name__}: {err}). Promote it by hand: make llm-promote VERSION={version.version}")
            else:
                result["promoted"] = True
                print(f"Version {version.version} is the new {ALIAS} (combined {metrics['combined']:.3f}, "
                      f"previous {'none' if champion_score is None else f'{champion_score:.3f}'})")
        else:
            print(f"{ALIAS} stays: its combined {champion_score:.3f} >= {metrics['combined']:.3f}")
    return result

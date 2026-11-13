"""LoRA-tune a small LLM to read an airline review (score, recommended, main complaint), gate it, log it to MLflow and
register it.

    python -m llm.train_lora --split train                      # reviews flown <= 2018, gate on the validation reviews
    python -m llm.train_lora --split live-2020                  # retrain on 2020's live reviews, gate on 10% of that year
    python -m llm.train_lora --eval-only --split live-2020      # how does the current champion do on 2020?

Runs on a GPU (Colab T4, or a local NVIDIA card); see notebooks/train_lora.ipynb.
Only the LoRA adapter (a few tens of MB) is trained and stored: the 1.5B base model is never changed, so a new
model version is a small file the server downloads next to the base model it already has.

MLflow gets parameters, metrics and the adapter. It never gets reviews, prompts or generated text: the review
dataset (DbCL-1.0) is not ours to redistribute and the shared MLflow is public.
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd

from llm import gate
from llm.reviews import (REVIEWS_FILE, SYSTEM_PROMPT, build_examples, format_completion, format_prompt, get_split,
                         load_reviews)
from simulator.scenarios import SCENARIOS, apply_scenario

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


def train(model, tok, examples: pd.DataFrame, args, dtype):
    """LoRA-tune model on the examples; returns (the peft model, per-step losses)."""
    import torch
    from peft import LoraConfig, get_peft_model
    # recompute activations in the backward pass: fits 8 GB cards (and a Colab T4) at ~30% more time
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
    losses, step = [], 0
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
                print(f"step {step}/{steps}  loss {sum(losses[-25:]) / len(losses[-25:]):.4f}", flush=True)
    model.eval()
    return model, losses


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
    folder = mlflow.artifacts.download_artifacts(artifact_uri=version.source, dst_path=tempfile.mkdtemp())
    info = json.loads((Path(folder) / INFO_FILE).read_text())
    if info["base_model"] != base or info["prompt_version"] != PROMPT_VERSION:
        print(f"Champion version {version.version} was tuned for {info}; not comparable, treating as no champion")
        return None
    return version.version, folder


def score_adapter(model, tok, folder: str, name: str, rows: pd.DataFrame) -> dict:
    """Load another adapter next to the one being trained (same base in memory), score it, switch back."""
    model.load_adapter(folder, adapter_name=name)
    model.set_adapter(name)
    metrics = gate.score(rows, generate(model, tok, rows["prompt"].tolist()))
    model.set_adapter("default")
    model.delete_adapter(name)
    return metrics


def _drift(rows: pd.DataFrame, args, offset: int = 0) -> pd.DataFrame:
    """Apply --scenario to sampled reviews, then rebuild prompt and completion from the changed text and labels."""
    if args.scenario == "none":
        return rows
    drifted = apply_scenario(rows, args.scenario, args.strength, seed=args.seed + offset)
    return drifted.assign(
        prompt=[format_prompt(r) for r in drifted.to_dict("records")],
        completion=[format_completion(r["score"], r["recommended"], r["completion"].rsplit("Complaint: ", 1)[1])
                    for r in drifted.to_dict("records")])


def split_arg(value: str) -> str:
    if value != "train" and not re.fullmatch(r"live-\d{4}", value):
        raise argparse.ArgumentTypeError("must be 'train' or 'live-YYYY', e.g. live-2020")
    return value


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", default="train", type=split_arg,
                   help="train (flown <= 2018) or live-YYYY (e.g. live-2020: retrain on that year's live reviews)")
    p.add_argument("--data", type=Path, default=REVIEWS_FILE, help="the airline reviews CSV (never committed)")
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--scenario", default="none", choices=SCENARIOS,
                   help="simulated drift applied to the training and gate reviews (simulator/scenarios.py)")
    p.add_argument("--strength", type=float, default=None, help="scenario strength; default depends on the scenario")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where the adapter is saved")
    p.add_argument("--no-mlflow", action="store_true", help="train and gate only: log and register nothing")
    p.add_argument("--no-promote", action="store_true",
                   help="register the new version if it passes the gate, but leave the champion alias where it is "
                        "(promote it later: make llm-promote VERSION=N)")
    p.add_argument("--eval-only", action="store_true", help="no training: score an adapter on --split")
    p.add_argument("--adapter", default="champion", help="with --eval-only: 'champion' or a local adapter folder")
    return p.parse_args()


def main():
    import torch
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU found; this will be very slow (Colab: Runtime > Change runtime type > T4 GPU)")

    df = load_reviews(args.data)
    train_df, gate_df = get_split(df, args.split, seed=args.seed)
    if args.eval_only and args.split != "train":
        gate_df = pd.concat([train_df, gate_df])  # scoring, not training: the whole year
    train_rows = build_examples(train_df, args.n_train, seed=args.seed)
    eval_rows = build_examples(gate_df, args.n_eval, seed=args.seed + 1000)
    # simulated drift on the sampled rows, before the prompts are built (so the text and the labels both change)
    train_rows = _drift(train_rows, args)
    eval_rows = _drift(eval_rows, args, offset=1)
    print(f"split {args.split}: {len(train_rows)} training reviews, {len(eval_rows)} gate reviews")

    tok, model, dtype = load_model(args.base, device)
    client = None
    if not args.no_mlflow or (args.eval_only and args.adapter == "champion"):
        import mlflow
        client = mlflow.MlflowClient()
        print(f"MLflow: {mlflow.get_tracking_uri()}")

    if args.eval_only:
        from peft import PeftModel
        if args.adapter == "champion":
            champion = champion_adapter(client, args.base)
            if champion is None:
                raise SystemExit(f"No comparable '{ALIAS}' for {MODEL_NAME} in the registry")
            label, folder = f"{MODEL_NAME} version {champion[0]}", champion[1]
        else:
            label, folder = args.adapter, args.adapter
        model = PeftModel.from_pretrained(model, folder).eval()
        t0 = time.time()
        metrics = gate.score(eval_rows, generate(model, tok, eval_rows["prompt"].tolist()))
        print(f"{label} on {args.split} ({time.time() - t0:.0f} s):\n{gate.report(metrics)}")
        return

    t0 = time.time()
    model, losses = train(model, tok, train_rows, args, dtype)
    train_seconds = time.time() - t0
    t0 = time.time()
    metrics = gate.score(eval_rows, generate(model, tok, eval_rows["prompt"].tolist()))
    print(f"New adapter on {args.split} (trained {train_seconds:.0f} s, gate {time.time() - t0:.0f} s):")
    print(gate.report(metrics))

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    info = {"base_model": args.base, "prompt_version": PROMPT_VERSION, "split": args.split}
    (args.out / INFO_FILE).write_text(json.dumps(info, indent=2))
    print(f"Adapter saved to {args.out} ({sum(f.stat().st_size for f in args.out.iterdir()) / 1e6:.0f} MB)")
    if args.no_mlflow:
        return

    import mlflow
    champion = champion_adapter(client, args.base)
    champion_score = None
    if champion is not None:
        champion_metrics = score_adapter(model, tok, champion[1], "champion", eval_rows)
        champion_score = champion_metrics["combined"]
        print(f"Champion (version {champion[0]}) on the same reviews:\n{gate.report(champion_metrics)}")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.set_tag("git_commit", git_commit())
        mlflow.log_params({k: str(v) for k, v in vars(args).items() if k not in ("eval_only", "adapter", "no_mlflow", "data")}
                          | {"prompt_version": PROMPT_VERSION, "device": torch.cuda.get_device_name() if device == "cuda" else "cpu"})
        mlflow.log_metrics({k: v for k, v in metrics.items() if not (isinstance(v, float) and math.isnan(v))}
                           | {"train_loss_last50": sum(losses[-50:]) / len(losses[-50:]), "train_seconds": train_seconds})
        if champion_score is not None:
            mlflow.log_metric("champion_combined", champion_score)
        mlflow.log_artifacts(str(args.out), "adapter")

        if not gate.passes(metrics):
            print("Not registered: the gate failed (limits are in llm/gate.py)")
            return
        try:
            client.create_registered_model(MODEL_NAME)
        except Exception:
            pass  # already exists
        version = client.create_model_version(MODEL_NAME, source=f"{run.info.artifact_uri}/adapter",
                                              run_id=run.info.run_id)
        print(f"Registered {MODEL_NAME} version {version.version}")
        beats = champion_score is None or metrics["combined"] > champion_score
        if args.no_promote:
            print(f"{ALIAS} not changed (--no-promote); version {version.version} "
                  f"{'beats' if beats else 'does not beat'} it on the same reviews. "
                  f"Promote it with: make llm-promote VERSION={version.version}")
        elif beats:
            client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
            print(f"Version {version.version} is the new {ALIAS} (combined {metrics['combined']:.3f}, "
                  f"previous {'none' if champion_score is None else f'{champion_score:.3f}'})")
        else:
            print(f"{ALIAS} stays: its combined {champion_score:.3f} >= {metrics['combined']:.3f}")


if __name__ == "__main__":
    main()

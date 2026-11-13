"""Simulated drift for the complaint box: change a cleaned reviews frame (llm.reviews.clean_reviews) the way a real
change in the world would, so the model's drift check and retraining can be tried on drift that is harder than the real
2019-2022 one.

    apply_scenario(df, "typos", strength=0.3, seed=0)

    none    unchanged
    typos   keyboard-neighbour typos then swapped letters in the Review (nlpaug char augmenters, no model download)
    short   a mobile channel: only the first 25 words of the Review are kept
    html    HTML tags and &nbsp; pasted between the words of the Review (a broken web form)
    harsh   concept drift: a new survey rates the same text harsher; score drops by 2, recommended only at >= 6 (was yes at 6-7)
    mixed   each row independently one of typos / short / html / harsh / none

`strength` is the share of words touched (typos, html) or the points the score drops (harsh, default 2); None means the default.
The input frame is never changed: a copy is returned. Same df, name, strength and seed give the same result.
"""

import random

import numpy as np
import pandas as pd

SCENARIOS = ["none", "typos", "short", "html", "harsh", "mixed"]
_MIXED_PARTS = ["typos", "short", "html", "harsh", "none"]
SHORT_WORDS = 25
HARSH_RECOMMEND_MIN = 6
HTML_TAGS = ["<br>", "<div>", "</p>", "&nbsp;", '<span class="x">']


def _typos(df: pd.DataFrame, strength, seed: int) -> pd.DataFrame:
    import nlpaug.augmenter.char as nac
    p = 0.3 if strength is None else strength
    random.seed(seed)
    np.random.seed(seed)
    keyboard = nac.KeyboardAug(aug_word_p=p)
    swap = nac.RandomCharAug(action="swap", aug_word_p=p)
    out = []
    for text in df["Review"].tolist():
        if not isinstance(text, str) or not text.strip():
            out.append(text)
            continue
        text = keyboard.augment(text)
        text = text[0] if isinstance(text, list) else text
        text = swap.augment(text)
        out.append(text[0] if isinstance(text, list) else text)
    df["Review"] = out
    return df


def _short(df: pd.DataFrame, strength, seed: int) -> pd.DataFrame:
    df["Review"] = [" ".join(t.split()[:SHORT_WORDS]) if isinstance(t, str) else t for t in df["Review"]]
    return df


def _html(df: pd.DataFrame, strength, seed: int) -> pd.DataFrame:
    p = 0.2 if strength is None else strength
    rng = random.Random(seed)
    out = []
    for text in df["Review"].tolist():
        if not isinstance(text, str):
            out.append(text)
            continue
        words = text.split()
        pieces = []
        for i, word in enumerate(words):
            pieces.append(word)
            if i < len(words) - 1 and rng.random() < p:
                pieces.append(rng.choice(HTML_TAGS))
        out.append(" ".join(pieces))
    df["Review"] = out
    return df


def _harsh(df: pd.DataFrame, strength, seed: int) -> pd.DataFrame:
    # A shift, not a multiplier: scores keep their spread (a multiplier squeezed them into 1-6, so "recommended"
    # became "no" for everyone and a constant answer looked like a perfect model).
    drop = 2 if strength is None else int(strength)
    df["score"] = np.maximum(df["score"].to_numpy().astype(int) - max(drop, 0), 1)
    # the new survey only recommends what it rates HARSH_RECOMMEND_MIN or more: the old 6-7 "yes" reviews flip to "no"
    df["recommended"] = np.where((df["recommended"] == "yes") & (df["score"] >= HARSH_RECOMMEND_MIN), "yes", "no")
    return df


def _mixed(df: pd.DataFrame, strength, seed: int) -> pd.DataFrame:
    choice = np.random.default_rng(seed).integers(0, len(_MIXED_PARTS), len(df))
    for k, name in enumerate(_MIXED_PARTS):
        pos = np.flatnonzero(choice == k)
        if len(pos):
            changed = apply_scenario(df.iloc[pos], name, strength, seed + k)
            for col in ("Review", "score", "recommended"):
                df.iloc[pos, df.columns.get_loc(col)] = changed[col].to_numpy()
    return df


_APPLY = {"typos": _typos, "short": _short, "html": _html, "harsh": _harsh}


def apply_scenario(df: pd.DataFrame, name: str, strength: float | None = None, seed: int = 0) -> pd.DataFrame:
    """A drifted copy of the cleaned reviews `df`; see the module docstring."""
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; choose from {SCENARIOS}")
    out = df.copy()
    if name == "none" or out.empty:
        return out
    if name == "mixed":
        return _mixed(out, strength, seed)
    return _APPLY[name](out, strength, seed)

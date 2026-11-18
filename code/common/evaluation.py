"""Splitting protocols, metrics and the shared cross-validation runner.

Every model (base paper, baselines, proposed) is evaluated through `run_cv`, so all
approaches see exactly the same train/validation/test partitions.

Protocols
---------
random  : StratifiedKFold over posts (what the KeepUp paper effectively did) - posts of
          the same event can fall on both sides of the split.
event   : StratifiedGroupKFold with groups = event_id - every event (and therefore its
          claim, evidence and sibling posts) is entirely in train OR test.  This is the
          primary protocol of this study.
Inside each training fold a further grouped 85/15 split provides the validation set used
for early stopping / threshold choice, so the test fold is never used for model selection.
"""
from __future__ import annotations
import json, time
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef, confusion_matrix)

from .paths import N_FOLDS, SEEDS, PREDS


def make_folds(df: pd.DataFrame, protocol: str, seed: int, n_folds: int = N_FOLDS):
    y = df["label"].values
    idx = np.arange(len(df))
    if protocol == "random":
        cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in cv.split(idx, y)]
    if protocol == "event":
        cv = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in cv.split(idx, y, groups=df["event_id"].values)]
    raise ValueError(protocol)


def inner_val_split(df: pd.DataFrame, train_idx: np.ndarray, protocol: str, seed: int,
                    val_frac: float = 0.15):
    """Split a training fold into fit/val respecting the same protocol."""
    sub = df.iloc[train_idx]
    if protocol == "event":
        gss = GroupShuffleSplit(1, test_size=val_frac, random_state=seed)
        fit, val = next(gss.split(sub, groups=sub["event_id"].values))
    else:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(sub))
        n_val = int(round(val_frac * len(sub)))
        val, fit = perm[:n_val], perm[n_val:]
    return train_idx[fit], train_idx[val]


def metrics(y_true, y_prob, thr: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
    }
    try:
        out["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        out["auc"] = np.nan
    return out


@dataclass
class FoldContext:
    """Everything a model's fit/predict function receives for one fold."""
    df: pd.DataFrame
    fit_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    protocol: str
    seed: int
    fold: int
    extra: dict = field(default_factory=dict)


ModelFn = Callable[[FoldContext], np.ndarray]   # returns P(label==1) for test_idx


def run_cv(name: str, model_fn: ModelFn, df: pd.DataFrame, protocol: str = "event",
           seeds=SEEDS, n_folds: int = N_FOLDS, verbose: bool = True, save: bool = True):
    """Run repeated k-fold CV; returns (per-fold metrics DataFrame, OOF predictions).
    If complete saved predictions for (name, protocol) exist they are re-used (resume)."""
    safe = name.replace(" ", "_").replace("/", "-").replace("+", "plus")
    f_oof, f_res = PREDS / f"{safe}__{protocol}.csv", PREDS / f"{safe}__{protocol}__folds.csv"
    if save and f_res.exists() and f_oof.exists():
        res = pd.read_csv(f_res)
        if len(res) == len(seeds) * n_folds:
            if verbose:
                print(f"  [{name}|{protocol}] loaded saved results (resume)", flush=True)
            return res, pd.read_csv(f_oof)
    rows, oof_rows = [], []
    for seed in seeds:
        folds = make_folds(df, protocol, seed, n_folds)
        for k, (tr, te) in enumerate(folds):
            fit_idx, val_idx = inner_val_split(df, tr, protocol, seed + k)
            ctx = FoldContext(df, fit_idx, val_idx, te, protocol, seed, k)
            t0 = time.perf_counter()
            prob = np.asarray(model_fn(ctx), dtype=float)
            elapsed = time.perf_counter() - t0
            m = metrics(df["label"].values[te], prob)
            m.update(model=name, protocol=protocol, seed=seed, fold=k, fit_predict_sec=elapsed)
            rows.append(m)
            oof_rows.append(pd.DataFrame({"model": name, "protocol": protocol, "seed": seed,
                                          "fold": k, "post_id": df["post_id"].values[te],
                                          "y": df["label"].values[te], "prob": prob}))
            if verbose:
                print(f"  [{name}|{protocol}|s{seed}|f{k}] f1={m['f1']:.3f} acc={m['accuracy']:.3f} "
                      f"auc={m['auc']:.3f} ({elapsed:.1f}s)", flush=True)
    res = pd.DataFrame(rows)
    oof = pd.concat(oof_rows, ignore_index=True)
    if save:
        oof.to_csv(f_oof, index=False)
        res.to_csv(f_res, index=False)
    return res, oof


def summarize(res: pd.DataFrame, cols=("accuracy", "precision", "recall", "f1", "auc", "mcc")):
    g = res.groupby(["model", "protocol"])
    out = g[list(cols)].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()

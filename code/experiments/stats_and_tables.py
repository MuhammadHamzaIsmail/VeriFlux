"""Statistical significance and the final comparison tables / figures.

* Paired comparison of VeriFlux vs every other model on the SAME test folds:
    - Wilcoxon signed-rank test over the per-fold macro-F1 (15 folds, or 5 for 1-repeat models)
    - McNemar's exact test on pooled out-of-fold predictions (seed 42)
    - 95% bootstrap CI (2000 resamples, grouped by event) of the macro-F1 difference
* Writes results/tables/main_comparison.csv and LaTeX-ready rows.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PREDS, TABLES, PROCESSED

PROPOSED = "VeriFlux"


def load_preds(protocol):
    out = {}
    for p in PREDS.glob(f"*__{protocol}.csv"):
        d = pd.read_csv(p)
        out[d["model"].iloc[0]] = d
    return out


def load_folds(protocol):
    out = {}
    for p in PREDS.glob(f"*__{protocol}__folds.csv"):
        d = pd.read_csv(p)
        out[d["model"].iloc[0]] = d
    return out


def mcnemar(y, a, b):
    ca, cb = a == y, b == y
    n01 = int(np.sum(ca & ~cb)); n10 = int(np.sum(~ca & cb))
    if n01 + n10 == 0:
        return 1.0, n01, n10
    return binomtest(n01, n01 + n10, 0.5).pvalue, n01, n10


def boot_diff(y, pa, pb, groups, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    ug = np.unique(groups); gi = {g: np.where(groups == g)[0] for g in ug}
    diffs = []
    for _ in range(n):
        idx = np.concatenate([gi[g] for g in rng.choice(ug, len(ug), replace=True)])
        diffs.append(f1_score(y[idx], pa[idx], average="macro") - f1_score(y[idx], pb[idx], average="macro"))
    return np.percentile(diffs, [2.5, 97.5])


def compare(protocol="event"):
    posts = pd.read_csv(PROCESSED / "posts.csv")
    ev = posts.set_index("post_id")["event_id"]
    preds, folds = load_preds(protocol), load_folds(protocol)
    if PROPOSED not in preds:
        return None
    P = preds[PROPOSED]; Fp = folds[PROPOSED]
    rows = []
    for name, d in preds.items():
        if name == PROPOSED:
            continue
        # fold-level Wilcoxon on matching (seed, fold)
        fo = folds[name][["seed", "fold", "f1"]].merge(Fp[["seed", "fold", "f1"]], on=["seed", "fold"],
                                                       suffixes=("_b", "_p"))
        try:
            w = wilcoxon(fo["f1_p"], fo["f1_b"]).pvalue if len(fo) >= 5 else np.nan
        except ValueError:
            w = 1.0
        a = P[P["seed"] == 42].set_index("post_id"); b = d[d["seed"] == 42].set_index("post_id")
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        y = a["y"].values; pa = (a["prob"] >= .5).astype(int).values; pb = (b["prob"] >= .5).astype(int).values
        pm, n01, n10 = mcnemar(y, pa, pb)
        lo, hi = boot_diff(y, pa, pb, ev.loc[common].values)
        rows.append(dict(baseline=name, protocol=protocol, n_folds=len(fo),
                         f1_proposed=fo["f1_p"].mean(), f1_baseline=fo["f1_b"].mean(),
                         delta_f1=fo["f1_p"].mean() - fo["f1_b"].mean(), wilcoxon_p=w,
                         mcnemar_p=pm, proposed_only_correct=n01, baseline_only_correct=n10,
                         ci95_low=lo, ci95_high=hi))
    return pd.DataFrame(rows).sort_values("delta_f1")


if __name__ == "__main__":
    allr = [r for r in (compare("event"), compare("random")) if r is not None]
    out = pd.concat(allr)
    out.to_csv(TABLES / "significance.csv", index=False)
    print(out.round(4).to_string())

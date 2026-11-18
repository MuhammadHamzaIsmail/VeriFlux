"""Run the KeepUp base-paper reproduction under both evaluation protocols.

usage: python code/base_keepup/run_keepup.py [--variants pooled,sequence] [--protocols event,random]
"""
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CACHE, TABLES, SEEDS, set_seed
from common.features import load_tables, keepup_numeric
from common.evaluation import run_cv, summarize
from base_keepup.keepup_model import train_predict

LOG_COLS = ["likes", "comments", "followers", "followings", "account_age",
            "st_agree", "st_disagree", "st_query", "st_comment", "title_len"]


def numeric_block(X: pd.DataFrame):
    X = X.copy()
    X[LOG_COLS] = np.log1p(X[LOG_COLS].clip(lower=0))
    return X.values.astype(np.float32)


def make_fn(E, N, y, mask=None):
    def fn(ctx):
        sc = StandardScaler().fit(N[ctx.fit_idx])
        Ns = sc.transform(N).astype(np.float32)
        order = np.argsort(ctx.test_idx)
        prob_sorted, _ = train_predict(E, Ns, y, ctx.fit_idx, ctx.val_idx, ctx.test_idx,
                                       mask=mask, seed=ctx.seed + ctx.fold)
        prob = np.empty(len(ctx.test_idx)); prob[order] = prob_sorted
        return prob
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="pooled,sequence")
    ap.add_argument("--protocols", default="event,random")
    ap.add_argument("--stance", default="xstancefusion")
    args = ap.parse_args()
    set_seed()
    posts, com = load_tables()
    y = posts["label"].values.astype(int)
    stance = np.load(CACHE / f"stance_oof_{args.stance}.npy")
    N = numeric_block(keepup_numeric(posts, com, stance))
    allres = []
    for variant in args.variants.split(","):
        if variant == "pooled":
            E, mask, seeds = np.load(CACHE / "post_doc_xlmr.npy"), None, SEEDS
        else:
            E = np.load(CACHE / "post_doc_tokens_fp16.npy", mmap_mode="r")
            mask = np.load(CACHE / "post_doc_mask.npy")
            seeds = SEEDS[:1]          # CPU budget: one repeat of 5-fold CV
        name = f"KeepUp ({'released code' if variant == 'pooled' else 'paper spec'})"
        for prot in args.protocols.split(","):
            res, _ = run_cv(name, make_fn(E, N, y, mask), posts, prot, seeds=seeds)
            allres.append(res)
    s = summarize(pd.concat(allres))
    print(s.round(4).to_string())
    out = TABLES / "keepup_results.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(old):
        old = old[~old.set_index(["model", "protocol"]).index.isin(s.set_index(["model", "protocol"]).index)]
    pd.concat([old, s]).to_csv(out, index=False)


if __name__ == "__main__":
    main()

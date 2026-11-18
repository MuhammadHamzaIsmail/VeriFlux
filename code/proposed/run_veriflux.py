"""Run VeriFlux (proposed) and its experiments.

experiments
  final     VeriFlux (RBF-SVM on [E, Ec, F, Fc]) under both protocols + its ablation
  main      VeriFlux-Lite (LightGBM + text prior) under event-held-out and random protocols (3x5-fold CV)
  ablation  VeriFlux-Lite: remove each component / feature family, single families, classifier swap
  budget    VeriFlux-Lite comment-budget early-detection proxy: only the first k comments per post
usage: python code/proposed/run_veriflux.py --exp final
       python code/proposed/run_veriflux.py --exp main,ablation,budget
"""
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CACHE, TABLES, set_seed
from common.features import load_tables, veriflux_features, GROUPS
from common.evaluation import run_cv, summarize
from proposed.veriflux import VeriFlux, post_texts

STANCE = "tfidf_lr"                                   # lightweight stance classifier
POST_LEVEL = ["post_claim", "stance", "user", "engagement", "title"]   # get event contrast


def contrast_idx(cols):
    post_cols = {c for g in POST_LEVEL for c in GROUPS[g]}
    return [i for i, c in enumerate(cols) if c in post_cols]


def save(res_list, fname):
    s = summarize(pd.concat(res_list))
    out = TABLES / fname
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(old):
        old = old[~old.set_index(["model", "protocol"]).index.isin(s.set_index(["model", "protocol"]).index)]
    s = pd.concat([old, s])
    s.to_csv(out, index=False)
    print(s.round(4).to_string())


def run(name, F, y, texts, groups, posts, prot, **kw):
    cols = list(F.columns)
    kw.setdefault("contrast_idx", contrast_idx(cols))
    return run_cv(name, VeriFlux(**kw).fold_fn(F.values.astype(float), y, texts, groups), posts, prot)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="main,ablation,budget")
    ap.add_argument("--protocols", default="event,random")
    args = ap.parse_args()
    set_seed()
    posts, com = load_tables()
    y = posts["label"].values.astype(int)
    groups = posts["event_id"].values
    stance = np.load(CACHE / f"stance_oof_{STANCE}.npy")
    F = veriflux_features(posts, com, stance)
    F.to_csv(CACHE / "veriflux_features.csv", index=False)
    texts = post_texts(posts, com)

    if "final" in args.exp:
        from proposed.veriflux import contrast_representation, veriflux_svm_fn, event_contrast
        E = np.load(CACHE / "post_doc_xlmr.npy")
        Fi = F.fillna(F.median()).values
        Z = contrast_representation(E, F.values, groups)
        res = [run_cv("VeriFlux", veriflux_svm_fn(Z, y), posts, p)[0] for p in args.protocols.split(",")]
        save(res, "veriflux_final.csv")
        variants = {
            "VeriFlux w/o event contrast [E, F]": np.hstack([E, Fi]),
            "VeriFlux w/o named features [E, Ec]": np.hstack([E, event_contrast(E, groups)]),
            "VeriFlux w/o text embedding [F, Fc]": np.hstack([Fi, event_contrast(Fi, groups)]),
            "VeriFlux embedding only [E]": E,
        }
        res = [run_cv(n, veriflux_svm_fn(v.astype(np.float32), y), posts, "event")[0] for n, v in variants.items()]
        save(res, "veriflux_final_ablation.csv")

    if "main" in args.exp:
        # transformer-free variant; named "VeriFlux-Lite" so it never overwrites the final VeriFlux results
        res = [run("VeriFlux-Lite", F, y, texts, groups, posts, p) for p in args.protocols.split(",")]
        save(res, "veriflux_results.csv")

    if "ablation" in args.exp:
        prot, res = "event", []
        res.append(run("w/o event contrast", F, y, texts, groups, posts, prot, contrast_idx=None))
        res.append(run("w/o text prior", F, y, texts, groups, posts, prot, use_text_prior=False))
        res.append(run("w/o contrast and text prior", F, y, texts, groups, posts, prot,
                       contrast_idx=None, use_text_prior=False))
        for g, cols in GROUPS.items():
            keep = [c for c in F.columns if c not in cols]
            res.append(run(f"w/o {g} features", F[keep], y, texts, groups, posts, prot))
        for g, cols in GROUPS.items():
            res.append(run(f"only {g} features", F[cols], y, texts, groups, posts, prot,
                           use_text_prior=False))
        res.append(run("only text prior", F[[]], y, texts, groups, posts, prot, contrast_idx=None))
        for clf, nm in [("xgb", "XGBoost"), ("rf", "Random Forest"), ("lr", "Logistic Regression"),
                        ("mlp", "MLP")]:
            res.append(run(f"classifier: {nm}", F, y, texts, groups, posts, prot, classifier=clf))
        save(res, "veriflux_ablation.csv")

    if "budget" in args.exp:
        res = []
        for k in [0, 3, 10, 25, 50]:
            Fk = veriflux_features(posts, com, stance, max_comments=k)
            tk = post_texts(posts, com, max_comments=k)
            res.append(run(f"VeriFlux-Lite @{k} comments", Fk, y, tk, groups, posts, "event"))
        save(res, "veriflux_budget.csv")


if __name__ == "__main__":
    main()

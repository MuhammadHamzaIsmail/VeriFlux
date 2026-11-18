"""Explainability, efficiency and language-stratified analysis for VeriFlux.

explain   : exact TreeSHAP (LightGBM pred_contrib) on held-out events, global ranking,
            deletion fidelity (mask top-k vs random-k features with train medians) and
            attribution stability (Spearman rank correlation across folds)
efficiency: CPU training time, per-post inference latency (batch 1, 1000 repetitions),
            model size and peak RSS for KeepUp, the strongest baselines and VeriFlux
language  : per-title-language macro-F1 from pooled out-of-fold predictions
"""
import sys, time, os, pickle, argparse, tracemalloc
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CACHE, TABLES, FIGURES, PREDS, PROCESSED, set_seed
from common.features import load_tables, veriflux_features, keepup_numeric
from common.evaluation import make_folds, inner_val_split, FoldContext
from proposed.veriflux import VeriFlux, post_texts, text_prior, select_params, GRID, BASE


def explain():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    posts, com = load_tables()
    y = posts["label"].values.astype(int); groups = posts["event_id"].values
    F = pd.read_csv(CACHE / "veriflux_features.csv")
    texts = post_texts(posts, com)
    names = list(F.columns) + ["text_prior"]
    shap_all, imp_folds, fid = [], [], []
    rng = np.random.RandomState(0)
    for k, (tr, te) in enumerate(make_folds(posts, "event", 42)):
        fit, val = inner_val_split(posts, tr, "event", 42 + k)
        ctx = FoldContext(posts, fit, val, te, "event", 42, k)
        fn = VeriFlux().fold_fn(F.values, y, texts, groups, return_model=True)
        fn(ctx)
        m, Xa = ctx.extra["model"], ctx.extra["X"]
        contrib = m.predict_proba(Xa[te], pred_contrib=True) if False else m.booster_.predict(Xa[te], pred_contrib=True)
        sv = contrib[:, :-1]
        shap_all.append(pd.DataFrame(sv, columns=names))
        imp_folds.append(np.abs(sv).mean(0))
        # deletion fidelity: replace top-k (per post) vs random-k features with train medians
        med = np.nanmedian(Xa[tr], 0)
        p0 = m.predict_proba(Xa[te])[:, 1]
        for kk in [1, 3, 5]:
            Xt, Xr = Xa[te].copy(), Xa[te].copy()
            for i in range(len(te)):
                top = np.argsort(-np.abs(sv[i]))[:kk]
                rnd = rng.choice(len(names), kk, replace=False)
                Xt[i, top] = med[top]; Xr[i, rnd] = med[rnd]
            fid.append(dict(fold=k, k=kk,
                            drop_top=np.mean(np.abs(p0 - m.predict_proba(Xt)[:, 1])),
                            drop_random=np.mean(np.abs(p0 - m.predict_proba(Xr)[:, 1]))))
    S = pd.concat(shap_all)
    glob = S.abs().mean().sort_values(ascending=False)
    glob.to_csv(TABLES / "shap_global.csv", header=["mean_abs_shap"])
    imp = np.array(imp_folds)
    rho = [spearmanr(imp[i], imp[j]).correlation for i in range(len(imp)) for j in range(i + 1, len(imp))]
    pd.DataFrame(fid).groupby("k").mean().drop(columns="fold").assign(
        stability_spearman_mean=np.mean(rho), stability_spearman_min=np.min(rho)
    ).to_csv(TABLES / "shap_fidelity_stability.csv")
    top = glob.head(15)[::-1]
    plt.figure(figsize=(5, 4.2))
    plt.barh(top.index, top.values, color="#3b6ea8")
    plt.xlabel("mean |SHAP value| (log-odds)"); plt.tight_layout()
    plt.savefig(FIGURES / "shap_global.pdf"); plt.savefig(FIGURES / "shap_global.png", dpi=200)
    print(glob.head(15)); print(pd.read_csv(TABLES / "shap_fidelity_stability.csv"))


def _rss_mb():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def efficiency():
    """Classifier-stage cost on one event-held-out fold (features precomputed for all)."""
    import torch
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from base_keepup.keepup_model import train_predict, KeepUpNet
    from base_keepup.run_keepup import numeric_block
    from sklearn.preprocessing import StandardScaler
    set_seed()
    posts, com = load_tables(); y = posts["label"].values.astype(int)
    tr, te = make_folds(posts, "event", 42)[0]
    fit, val = inner_val_split(posts, tr, "event", 42)
    rows = []

    def latency(predict_one, x_one, reps=1000):
        predict_one(x_one)
        t0 = time.perf_counter()
        for _ in range(reps):
            predict_one(x_one)
        return (time.perf_counter() - t0) / reps * 1000

    # KeepUp (released-code variant: pooled embedding + 14 numeric)
    stance = np.load(CACHE / "stance_oof_xstancefusion.npy")
    N = numeric_block(keepup_numeric(posts, com, stance))
    N = StandardScaler().fit(N[fit]).transform(N).astype(np.float32)
    E = np.load(CACHE / "post_doc_xlmr.npy")
    t0 = time.perf_counter(); _, ep = train_predict(E, N, y, fit, val, te); t_train = time.perf_counter() - t0
    net = KeepUpNet(); net.eval()
    e1 = torch.from_numpy(E[te[:1]]).unsqueeze(1); n1 = torch.from_numpy(N[te[:1]])
    with torch.inference_mode():
        lat = latency(lambda _: net(e1, n1), None)
    params = sum(p.numel() for p in net.parameters())
    rows.append(dict(model="KeepUp classifier", train_sec=t_train, latency_ms=lat,
                     params_or_nodes=params, size_kb=params * 4 / 1024))
    # VeriFlux classifier
    F = pd.read_csv(CACHE / "veriflux_features.csv").values
    trn = np.concatenate([fit, val])
    t0 = time.perf_counter()
    p = select_params(F, y, fit, val, 42)
    m = LGBMClassifier(**p, **BASE, random_state=42).fit(F[trn], y[trn])
    t_train = time.perf_counter() - t0
    x1 = F[te[:1]]
    lat = latency(lambda _: m.predict_proba(x1), None)
    size = len(pickle.dumps(m)) / 1024
    rows.append(dict(model="VeriFlux classifier (LightGBM)", train_sec=t_train, latency_ms=lat,
                     params_or_nodes=int(m.booster_.num_trees()), size_kb=size))
    # XGBoost baseline on 782-d KeepUp input
    X = np.hstack([E, N])
    t0 = time.perf_counter()
    xg = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.5, n_jobs=8, eval_metric="logloss").fit(X[trn], y[trn])
    t_train = time.perf_counter() - t0
    lat = latency(lambda _: xg.predict_proba(X[te[:1]]), None)
    rows.append(dict(model="XGBoost baseline (782-d)", train_sec=t_train, latency_ms=lat,
                     params_or_nodes=300, size_kb=len(pickle.dumps(xg)) / 1024))
    pd.DataFrame(rows).to_csv(TABLES / "efficiency_classifier.csv", index=False)
    print(pd.DataFrame(rows))


def language():
    posts = pd.read_csv(PROCESSED / "posts.csv")
    lang = posts.set_index("post_id")["title_lang"].replace({"hindi": "other"})
    rows = []
    for p in PREDS.glob("*__event.csv"):
        d = pd.read_csv(p); name = d["model"].iloc[0]
        d["lang"] = d["post_id"].map(lang)
        for lg, g in d.groupby("lang"):
            pr = (g["prob"] >= .5).astype(int)
            rows.append(dict(model=name, lang=lg, n=len(g) / d["seed"].nunique(),
                             f1=f1_score(g["y"], pr, average="macro"), acc=accuracy_score(g["y"], pr)))
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "language_results.csv", index=False)
    print(out.pivot(index="model", columns="lang", values="f1").round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", default="explain,efficiency,language")
    a = ap.parse_args()
    if "explain" in a.parts: explain()
    if "efficiency" in a.parts: efficiency()
    if "language" in a.parts: language()

"""End-to-end CPU inference cost per post: KeepUp vs VeriFlux (run on an otherwise idle CPU).

Each pipeline runs in its own subprocess so peak memory (RSS) is measured separately.
Claim-level work (retrieval + claim/evidence NLI) is shared by all posts of an event, so it is
reported separately and amortised by the mean number of posts per event in the totals.
usage: python code/experiments/efficiency_e2e.py            (orchestrates both)
"""
import sys, time, json, subprocess, threading, os
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from common.paths import (CACHE, TABLES, PROCESSED, EVIDENCE_DIR, XLMR, BGE_M3, NORGLM, XNLI_MINI,
                          SBERT_MINI, RERANKER, set_seed)
from common.text import clean

N_POSTS, N_CLAIMS = 30, 10


def sample():
    posts = pd.read_csv(PROCESSED / "posts.csv", keep_default_na=False)
    com = pd.read_csv(PROCESSED / "comments.csv", keep_default_na=False)
    rng = np.random.RandomState(0)
    pid = rng.choice(posts.post_id.values, N_POSTS, replace=False)
    sp = posts[posts.post_id.isin(pid)]
    sc = com[com.post_id.isin(pid)]
    claims = pd.read_csv(PROCESSED / "claims.csv").sample(N_CLAIMS, random_state=0)
    return posts, com, sp, sc, claims


def timed(fn):
    t0 = time.perf_counter(); out = fn(); return out, time.perf_counter() - t0


def keepup():
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    torch.set_num_threads(8)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
    from base_keepup.keepup_model import KeepUpNet
    _, com, sp, sc, claims = sample()
    res = {}
    with torch.inference_mode():
        # ---- claim level: BGE-M3 query + dense search over cached doc vectors + rerank 20 + NorGLM
        E = np.load(CACHE / "evidence_bge_m3.npy")
        ids = sorted(p.stem for p in EVIDENCE_DIR.rglob("*.txt"))
        tokb = AutoTokenizer.from_pretrained(BGE_M3); bge = AutoModel.from_pretrained(BGE_M3).eval()
        tokr = AutoTokenizer.from_pretrained(RERANKER); rr = AutoModelForSequenceClassification.from_pretrained(RERANKER).eval()
        tokn = AutoTokenizer.from_pretrained(NORGLM)
        if tokn.pad_token is None:
            tokn.add_special_tokens({"pad_token": "[PAD]"})
        nor = AutoModelForSequenceClassification.from_pretrained(NORGLM); nor.resize_token_embeddings(len(tokn)); nor.eval()
        t0 = time.perf_counter()
        for c in claims["claim"].map(clean):
            q = bge(**tokb([c], return_tensors="pt", truncation=True, max_length=128)).last_hidden_state[:, 0]
            q = torch.nn.functional.normalize(q, dim=-1).numpy()
            top = np.argsort(-(E @ q[0]))[:20]
            docs = [clean((EVIDENCE_DIR / "evidence" / f"{ids[j]}.txt").read_text(encoding="utf-8", errors="ignore"))
                    if (EVIDENCE_DIR / "evidence" / f"{ids[j]}.txt").exists() else "" for j in top]
            s = rr(**tokr([c] * 20, docs, return_tensors="pt", padding=True, truncation=True, max_length=512)).logits
            best = docs[int(s.squeeze(-1).argmax())]
            nor(**tokn([c], [best], return_tensors="pt", truncation=True, max_length=512))
        res["claim_ms"] = (time.perf_counter() - t0) / len(claims) * 1000
        # ---- post level: XLM-R doc embedding + XStanceFusion over every comment + classifier
        tok = AutoTokenizer.from_pretrained(XLMR); enc = AutoModel.from_pretrained(XLMR).eval()
        import importlib
        stance_mod = importlib.import_module("05_stance")
        head = stance_mod.build_head("xstancefusion").eval()
        net = KeepUpNet().eval()
        t0 = time.perf_counter()
        for pid, title in zip(sp.post_id, sp.title):
            cm = [clean(t) or "." for t in sc[sc.post_id == pid]["text"]]
            doc = clean(title) + " " + " ".join(cm)
            e = enc(**tok([doc], return_tensors="pt", truncation=True, max_length=512)).last_hidden_state
            e = e.mean(1)
            for b in range(0, len(cm), 64):
                x = tok(cm[b:b + 64], return_tensors="pt", padding="max_length", truncation=True, max_length=48)
                h = enc(**x).last_hidden_state
                head(h, x["attention_mask"])
            net(e.unsqueeze(1), torch.zeros(1, 14))
        res["post_ms"] = (time.perf_counter() - t0) / len(sp) * 1000
        res["params_M"] = (sum(p.numel() for p in enc.parameters()) * 1 + sum(p.numel() for p in bge.parameters())
                           + sum(p.numel() for p in rr.parameters()) + sum(p.numel() for p in nor.parameters())
                           + sum(p.numel() for p in head.parameters()) + sum(p.numel() for p in net.parameters())) / 1e6
    return res


def veriflux():
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    torch.set_num_threads(8)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
    from common.features import DEBUNK_RE
    from proposed.veriflux import _tfidf_lr, GRID, BASE
    from lightgbm import LGBMClassifier
    import importlib, pickle
    retr = importlib.import_module("02_retrieval")
    stance_mod = importlib.import_module("05_stance")
    posts, com, sp, sc, claims = sample()
    res = {}
    # training cost of the light components (stance TF-IDF LR and text prior), excluded from latency
    tr_com = com[~com.post_id.isin(sp.post_id)].sample(20000, random_state=0)
    (vec_s, clf_s) = (None, None)
    from sklearn.pipeline import make_union
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    t0 = time.perf_counter()
    vec_s = make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=200000, sublinear_tf=True),
                       TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True))
    Xs = vec_s.fit_transform(tr_com.text.map(clean))
    clf_s = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced").fit(Xs, tr_com.stance)
    tr_posts = posts[~posts.post_id.isin(sp.post_id)]
    joined = com.groupby("post_id")["text"].apply(lambda s: " ".join(clean(x) for x in s)[:30000])
    txt = (tr_posts.title.map(clean) + " ||| " + tr_posts.post_id.map(joined).fillna("")).values
    vec_t, lr_t = _tfidf_lr(); lr_t.fit(vec_t.fit_transform(txt), tr_posts.label)
    F = pd.read_csv(CACHE / "veriflux_features.csv").values
    gbm = LGBMClassifier(**GRID[5], **BASE).fit(np.column_stack([F, F, np.zeros(len(F))])[:900], posts.label.values[:900])
    res["train_light_s"] = time.perf_counter() - t0
    res["size_MB_light"] = (len(pickle.dumps((vec_s, clf_s))) + len(pickle.dumps((vec_t, lr_t))) + len(pickle.dumps(gbm))) / 2**20
    tokn = AutoTokenizer.from_pretrained(XNLI_MINI); nli = AutoModelForSequenceClassification.from_pretrained(XNLI_MINI).eval()
    toks = AutoTokenizer.from_pretrained(SBERT_MINI); sb = AutoModel.from_pretrained(SBERT_MINI).eval()
    ids, docs = retr.load_evidence()
    from sklearn.feature_extraction.text import CountVectorizer
    with torch.inference_mode():
        # ---- claim level: BM25 (index built offline) + MiniLM NLI over top-3
        t0 = time.perf_counter()
        S = retr.bm25_scores(claims["claim"].map(clean).tolist(), docs)
        t_bm25_total = time.perf_counter() - t0      # includes building the index once
        t0 = time.perf_counter()
        for c, s in zip(claims["claim"].map(clean), S):
            top = np.argsort(-s)[:3]
            nli(**tokn([docs[j] for j in top], [c] * 3, return_tensors="pt", padding=True, truncation=True, max_length=512))
        res["claim_ms"] = (time.perf_counter() - t0) / len(claims) * 1000
        res["bm25_index_build_s"] = t_bm25_total
        # ---- post level
        cl = posts.set_index("event_id")
        claim_of = pd.read_csv(PROCESSED / "claims.csv").set_index("event_id")["claim"]
        t0 = time.perf_counter()
        for pid, title, ev in zip(sp.post_id, sp.title, sp.event_id):
            c = clean(str(claim_of.get(ev, ""))); t = clean(title)[:1000] or "."
            nli(**tokn([c], [t], return_tensors="pt", truncation=True, max_length=320))
            e = sb(**toks([c, t], return_tensors="pt", padding=True, truncation=True, max_length=256)).last_hidden_state
            cm = sc[sc.post_id == pid]["text"].map(clean).tolist()
            if cm:
                clf_s.predict_proba(vec_s.transform(cm))
                sum(bool(DEBUNK_RE.search(x)) for x in cm)
            lr_t.predict_proba(vec_t.transform([t + " ||| " + " ".join(cm)[:30000]]))
            gbm.predict_proba(np.zeros((1, gbm.n_features_in_)))
        res["post_ms"] = (time.perf_counter() - t0) / len(sp) * 1000
        res["params_M"] = (sum(p.numel() for p in nli.parameters()) + sum(p.numel() for p in sb.parameters())) / 1e6
        # extra cost of the full VeriFlux: XLM-R document embedding (512 tokens) + RBF-SVM
        from sklearn.svm import SVC
        tokx = AutoTokenizer.from_pretrained(XLMR); enc = AutoModel.from_pretrained(XLMR).eval()
        Z = np.random.RandomState(0).randn(900, 1612).astype(np.float32)
        svm = SVC(probability=True).fit(Z, posts.label.values[:900])
        t0 = time.perf_counter()
        for pid, title in zip(sp.post_id, sp.title):
            cm = sc[sc.post_id == pid]["text"].map(clean).tolist()
            e = enc(**tokx([clean(title) + " " + " ".join(cm)], return_tensors="pt", truncation=True,
                           max_length=512)).last_hidden_state.mean(1)
            svm.predict_proba(Z[:1])
        res["xlmr_svm_ms"] = (time.perf_counter() - t0) / len(sp) * 1000
        res["params_M_full"] = res["params_M"] + sum(p.numel() for p in enc.parameters()) / 1e6
    return res


def child(which):
    import psutil
    peak = [0]; stop = [False]
    def mon():
        p = psutil.Process(os.getpid())
        while not stop[0]:
            peak[0] = max(peak[0], p.memory_info().rss); time.sleep(0.05)
    th = threading.Thread(target=mon, daemon=True); th.start()
    r = keepup() if which == "keepup" else veriflux()
    stop[0] = True; r["peak_rss_MB"] = peak[0] / 2**20
    print("RESULT" + json.dumps(r))


def main():
    posts = pd.read_csv(PROCESSED / "posts.csv")
    ppe = posts.groupby("event_id").size().mean()
    rows = []
    for which in ["veriflux", "keepup"]:
        out = subprocess.run([sys.executable, "-X", "utf8", "-W", "ignore", __file__, which],
                             capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
        line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")]
        if not line:
            print(out.stdout[-2000:], out.stderr[-3000:]); continue
        r = json.loads(line[0][6:]); r["pipeline"] = which
        r["total_ms_per_post"] = r["post_ms"] + r["claim_ms"] / ppe
        rows.append(r); print(which, r)
    d = pd.DataFrame(rows)
    d.to_csv(TABLES / "efficiency_e2e.csv", index=False)
    a = d.set_index("pipeline")
    comp = []
    for p, nm in [("keepup", "KeepUp"), ("veriflux", "VeriFlux")]:
        if p in a.index:
            comp += [dict(component=f"{nm}: claim-level (per claim)", ms_per_post=a.loc[p, "claim_ms"], train_sec=np.nan, size_mb=np.nan),
                     dict(component=f"{nm}: post-level", ms_per_post=a.loc[p, "post_ms"], train_sec=np.nan, size_mb=np.nan),
                     dict(component=f"{nm} (total)", ms_per_post=a.loc[p, "total_ms_per_post"], train_sec=np.nan,
                          size_mb=a.loc[p, "params_M"] * 4)]
    if "veriflux" in a.index:
        v = a.loc["veriflux"]
        comp += [dict(component="VeriFlux: XLM-R doc embedding + SVM", ms_per_post=v["xlmr_svm_ms"], train_sec=np.nan, size_mb=np.nan),
                 dict(component="VeriFlux (total)", ms_per_post=v["total_ms_per_post"] + v["xlmr_svm_ms"],
                      train_sec=np.nan, size_mb=v["params_M_full"] * 4)]
        for c in comp:
            if c["component"] == "VeriFlux (total)" and c["ms_per_post"] == v["total_ms_per_post"]:
                c["component"] = "VeriFlux-Lite (total)"
            elif c["component"].startswith("VeriFlux:") and "XLM-R" not in c["component"]:
                c["component"] = c["component"].replace("VeriFlux:", "VeriFlux-Lite / VeriFlux:")
    pd.DataFrame(comp).to_csv(TABLES / "efficiency.csv", index=False)
    print(pd.DataFrame(comp))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
        child(sys.argv[1])
    else:
        main()

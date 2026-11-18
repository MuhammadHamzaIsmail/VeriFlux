"""Step 2 - knowledge extraction: retrieve evidence documents for every claim.

Reproduces KeepUp Sec. 4.1.1 (dense retrieval + cross-encoder re-ranking) and compares it
with a lexical BM25 baseline.  Metric: MRR@10 against the gold evidence id (Eq. 4).

Retrievers
  bm25          Okapi BM25 (k1=1.5, b=0.75) over word unigrams          [lightweight]
  bge_m3        BAAI/bge-m3 dense embeddings, cosine similarity (FAISS inner product)
  bge_m3+rerank top-20 dense candidates re-ranked by a multilingual cross-encoder
Outputs: results/cache/retrieval_<name>.csv  (claim_id, ranked doc ids, scores)
         results/tables/retrieval_mrr.csv
"""
import sys, re, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import PROCESSED, EVIDENCE_DIR, CACHE, TABLES, BGE_M3 as DENSE_MODEL, RERANKER
from common.text import clean

DOC_MAX_LEN = 256


def load_evidence():
    docs = {}
    for p in sorted(EVIDENCE_DIR.rglob("*.txt")):
        docs[p.stem] = clean(p.read_text(encoding="utf-8", errors="ignore"))
    ids = list(docs)
    return ids, [docs[i] for i in ids]


def mrr_at_k(ranked, gold, k=10):
    for r, d in enumerate(ranked[:k], 1):
        if d == gold:
            return 1.0 / r
    return 0.0


def bm25_scores(queries, docs, k1=1.5, b=0.75):
    vec = CountVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=True)
    D = vec.fit_transform(docs).tocsc().astype(np.float32)
    Q = vec.transform(queries)
    N = D.shape[0]
    df = np.asarray((D > 0).sum(0)).ravel()
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
    dl = np.asarray(D.sum(1)).ravel()
    avg = dl.mean()
    D = D.tocsr()
    rows, cols = D.nonzero()
    tf = np.asarray(D[rows, cols]).ravel()
    denom = tf + k1 * (1 - b + b * dl[rows] / avg)
    W = sparse.csr_matrix(((tf * (k1 + 1) / denom) * idf[cols], (rows, cols)), shape=D.shape)
    Qb = (Q > 0).astype(np.float32)
    return np.asarray((Qb @ W.T).todense())


def dense_encode(texts, model_name, max_len, batch=16, desc="", ckpt=None):
    """[CLS]-pooled, L2-normalised embeddings. With ckpt, progress is saved every 25 batches
    and an interrupted run resumes where it stopped."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    start = 0
    if ckpt is not None and Path(ckpt).exists():
        z = np.load(ckpt); out, start = z["out"], int(z["done"])
        print(f"  resuming {desc} at {start}/{len(texts)}", flush=True)
    t0 = time.time()
    with torch.inference_mode():
        for i in range(start, len(texts), batch):
            idx = order[i:i + batch]
            enc = tok([texts[j] or "." for j in idx], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt")
            cls = model(**enc).last_hidden_state[:, 0]            # BGE uses [CLS] pooling
            out[idx] = torch.nn.functional.normalize(cls, dim=-1).numpy()
            if (i // batch) % 25 == 0:
                print(f"  {desc} {i+len(idx)}/{len(texts)} {(time.time()-t0)/60:.1f}min", flush=True)
                if ckpt is not None:
                    np.savez(ckpt, out=out, done=i + len(idx))
    return out


def rerank(claims, cand_ids, cand_texts):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(RERANKER)
    model = AutoModelForSequenceClassification.from_pretrained(RERANKER).eval()
    new_ids, new_scores = [], []
    with torch.inference_mode():
        for c, ids, txts in zip(claims, cand_ids, cand_texts):
            enc = tok([c] * len(txts), txts, padding=True, truncation=True, max_length=512,
                      return_tensors="pt")
            s = model(**enc).logits.squeeze(-1).numpy()
            o = np.argsort(-s)
            new_ids.append([ids[j] for j in o]); new_scores.append(s[o].tolist())
    return new_ids, new_scores


def save_run(name, claims_df, ranked, scores):
    out = pd.DataFrame({"claim_id": claims_df["claim_id"], "gold": claims_df["gold_evidence"],
                        "ranked": [" ".join(r[:20]) for r in ranked],
                        "scores": [" ".join(f"{x:.5f}" for x in s[:20]) for s in scores]})
    out["mrr10"] = [mrr_at_k(r, g) for r, g in zip(ranked, claims_df["gold_evidence"])]
    out.to_csv(CACHE / f"retrieval_{name}.csv", index=False)
    print(f"{name}: MRR@10 = {out['mrr10'].mean():.4f}")
    return out["mrr10"].mean(), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="bm25,bge_m3,bge_m3_rerank")
    args = ap.parse_args()
    claims = pd.read_csv(PROCESSED / "claims.csv")
    claims["claim_clean"] = claims["claim"].map(clean)
    ids, docs = load_evidence()
    print(len(ids), "evidence docs;", claims.shape[0], "claims")
    rows = []
    t0 = time.time()
    S = bm25_scores(claims["claim_clean"].tolist(), docs)
    ranked = [[ids[j] for j in np.argsort(-s)[:20]] for s in S]
    sc = [np.sort(s)[::-1][:20] for s in S]
    m, _ = save_run("bm25", claims, ranked, sc)
    rows.append(dict(retriever="BM25", mrr10=m, minutes=(time.time() - t0) / 60))
    if "bm25_rerank" in args.methods:
        doc_map = dict(zip(ids, docs))
        r_ids, r_sc = rerank(claims["claim_clean"].tolist(), ranked,
                             [[doc_map[d] for d in r] for r in ranked])
        m, _ = save_run("bm25_rerank", claims, r_ids, r_sc)
        rows.append(dict(retriever="BM25 + cross-encoder", mrr10=m, minutes=(time.time() - t0) / 60))

    if "bge_m3" in args.methods.replace("bm25_rerank", ""):
        emb_path = CACHE / "evidence_bge_m3.npy"
        t0 = time.time()
        if emb_path.exists():
            E = np.load(emb_path)
        else:
            E = dense_encode(docs, DENSE_MODEL, DOC_MAX_LEN, desc="evidence", ckpt=CACHE / "evidence_bge_m3_ckpt.npz")
            np.save(emb_path, E)
        enc_min = (time.time() - t0) / 60
        Qe = dense_encode(claims["claim_clean"].tolist(), DENSE_MODEL, 128, desc="claims")
        try:
            import faiss
            index = faiss.IndexFlatIP(E.shape[1]); index.add(E)
            sc, I = index.search(Qe, 20)
        except ImportError:
            S = Qe @ E.T
            I = np.argsort(-S, 1)[:, :20]; sc = np.take_along_axis(S, I, 1)
        ranked = [[ids[j] for j in row] for row in I]
        m, _ = save_run("bge_m3", claims, ranked, sc)
        rows.append(dict(retriever="BGE-M3 (dense)", mrr10=m, minutes=enc_min))
        if "rerank" in args.methods:
            t0 = time.time()
            doc_map = dict(zip(ids, docs))
            r_ids, r_sc = rerank(claims["claim_clean"].tolist(), ranked,
                                 [[doc_map[d] for d in r] for r in ranked])
            m, _ = save_run("bge_m3_rerank", claims, r_ids, r_sc)
            rows.append(dict(retriever="BGE-M3 + cross-encoder", mrr10=m,
                             minutes=enc_min + (time.time() - t0) / 60))
    # the authors' released retrieval output, for reference
    ref = pd.read_csv(CACHE.parents[1] / "data" / "keepup_repo" / "retrieval_results.csv") \
        if (CACHE.parents[1] / "data" / "keepup_repo" / "retrieval_results.csv").exists() else None
    if ref is not None:
        rows.append(dict(retriever="KeepUp released output", mrr10=ref["MRR@10"].mean(), minutes=np.nan))
    out = TABLES / "retrieval_mrr.csv"
    new = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out); new = pd.concat([old[~old["retriever"].isin(new["retriever"])], new])
    new.to_csv(out, index=False)
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()

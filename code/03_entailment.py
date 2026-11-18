"""Step 3 - news-content module: natural language inference features.

(a) KeepUp reproduction (Sec. 4.2.1, released notebook): NorGLM/Entailment on
    (claim, top-1 retrieved evidence); feature = P(entailment)+P(neutral) as in the
    authors' code ("positive_prob").
(b) VeriFlux claim-evidence features: multilingual XNLI model
    (multilingual-MiniLMv2-L6-mnli-xnli), premise = evidence, hypothesis = claim, keeping all
    three probabilities for the top-1 document and the max over the top-3 documents.
(c) VeriFlux post-claim agreement features: same XNLI model with premise = claim,
    hypothesis = post title, plus multilingual sentence-embedding cosine similarity.
    Motivation: KeepUp's annotation guideline (Table 3) labels a post by combining the
    claim's veracity with the post's stance toward the claim.
Outputs: results/cache/nli_claims.csv, results/cache/nli_posts.csv
"""
import sys, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import PROCESSED, CACHE, EVIDENCE_DIR, NORGLM, XNLI_MINI as XNLI, SBERT_MINI as SBERT
from common.text import clean



def nli_probs(model_name, premises, hypotheses, max_len=512, batch=8, bert_pad_fix=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    if bert_pad_fix and tok.pad_token is None:      # replicates the authors' notebook
        tok.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tok))
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    out = np.zeros((len(premises), len(id2label)), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(premises), batch):
            enc = tok(premises[i:i + batch], hypotheses[i:i + batch], padding=True,
                      truncation="longest_first", max_length=max_len, return_tensors="pt")
            out[i:i + batch] = torch.softmax(model(**enc).logits, -1).numpy()
            if (i // batch) % 20 == 0:
                print(f"  {model_name.split('/')[-1]} {i+batch}/{len(premises)} "
                      f"{(time.time()-t0)/60:.1f}min", flush=True)
    return out, id2label


def label_cols(probs, id2label, prefix):
    cols = {}
    for k, name in id2label.items():
        short = {"entailment": "E", "neutral": "N", "contradiction": "C"}.get(name, name)
        cols[f"{prefix}_{short}"] = probs[:, k]
    return pd.DataFrame(cols)


def sbert_cos(a, b):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(SBERT)
    model = AutoModel.from_pretrained(SBERT).eval()

    def enc(texts):
        outs = []
        with torch.inference_mode():
            for i in range(0, len(texts), 32):
                e = tok(texts[i:i + 32], padding=True, truncation=True, max_length=256,
                        return_tensors="pt")
                h = model(**e).last_hidden_state
                m = e["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1)
                outs.append(torch.nn.functional.normalize(v, dim=-1).numpy())
        return np.vstack(outs)
    return (enc(a) * enc(b)).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", default="bge_m3_rerank")
    ap.add_argument("--parts", default="norglm,xnli_claim,xnli_post,sbert")
    args = ap.parse_args()
    claims = pd.read_csv(PROCESSED / "claims.csv")
    posts = pd.read_csv(PROCESSED / "posts.csv", keep_default_na=False)
    ret = pd.read_csv(CACHE / f"retrieval_{args.retrieval}.csv").set_index("claim_id")
    docs = {p.stem: clean(p.read_text(encoding="utf-8", errors="ignore"))
            for p in EVIDENCE_DIR.rglob("*.txt")}
    claims["claim_clean"] = claims["claim"].map(clean)
    ranked = claims["claim_id"].map(lambda c: ret.loc[c, "ranked"].split())
    top1 = [docs.get(r[0], "") for r in ranked]

    out_c = CACHE / "nli_claims.csv"
    C = pd.read_csv(out_c) if out_c.exists() else claims[["claim_id", "event_id"]].copy()
    if "norglm" in args.parts:
        p, lab = nli_probs(NORGLM, claims["claim_clean"].tolist(), top1, bert_pad_fix=True)
        # NorGLM's config names its classes label_0/1/2; the authors' notebook maps
        # {0: entailment, 1: neutral, 2: contradiction} and uses P(0)+P(1) as "positive_prob".
        C["keepup_pos_prob"] = p[:, 0] + p[:, 1]
        print("NorGLM labels:", lab, "| positive_prob mean:", float(C["keepup_pos_prob"].mean()))
    if "xnli_claim" in args.parts:
        prem, hyp, owner = [], [], []
        for i, r in enumerate(ranked):
            for d_id in r[:3]:
                prem.append(docs.get(d_id, "")); hyp.append(claims["claim_clean"].iloc[i]); owner.append(i)
        p, lab = nli_probs(XNLI, prem, hyp, max_len=512, batch=4)
        d = label_cols(p, lab, "ev")
        d["owner"] = owner
        d["rank"] = d.groupby("owner").cumcount()
        t1 = d[d["rank"] == 0].set_index("owner")
        agg = d.groupby("owner").agg(ev_maxE=("ev_E", "max"), ev_maxC=("ev_C", "max"))
        for c in ["ev_E", "ev_N", "ev_C"]:
            C[c + "_top1"] = t1[c].reindex(range(len(claims))).values
        C["ev_maxE_top3"] = agg["ev_maxE"].values
        C["ev_maxC_top3"] = agg["ev_maxC"].values
    C.to_csv(out_c, index=False)

    out_p = CACHE / "nli_posts.csv"
    P = pd.read_csv(out_p) if out_p.exists() else posts[["post_id", "event_id"]].copy()
    cl_map = claims.set_index("event_id")["claim_clean"]
    post_claim = posts["event_id"].map(cl_map).fillna("").tolist()
    titles = [clean(t)[:1000] or "." for t in posts["title"].astype(str)]
    if "xnli_post" in args.parts:
        p, lab = nli_probs(XNLI, post_claim, titles, max_len=320, batch=8)
        d = label_cols(p, lab, "pc")
        for c in d.columns:
            P[c] = d[c].values
    if "sbert" in args.parts:
        P["pc_cos"] = sbert_cos(post_claim, titles)
    P.to_csv(out_p, index=False)
    print(C.describe().T.round(3)); print(P.describe().T.round(3))


if __name__ == "__main__":
    main()

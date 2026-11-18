"""Step 5 - comment stance detection (agree / disagree / query / comment).

Cross-fitting over EVENTS (5 folds): every comment is predicted by a model that never saw
any comment of its own event.  These out-of-fold predictions are the stance features used
downstream, so no post's stance features are produced by a model trained on that post.

Models
  tfidf_lr       char(2-5)+word(1-2) TF-IDF + logistic regression    (VeriFlux stance)
  xlmr_lr        frozen XLM-R mean-pooled vector + logistic regression
  bilstm         frozen XLM-R tokens -> BiLSTM -> mean pool
  bigru          frozen XLM-R tokens -> BiGRU  -> mean pool
  xstancefusion  frozen XLM-R tokens -> BiLSTM -> BiGRU -> attention  (KeepUp, Sec. 4.3.1)

usage: python 05_stance.py [--models tfidf_lr,xlmr_lr,...] [--epochs 3]
"""
import sys, time, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_union
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import PROCESSED, CACHE, TABLES, SEED, set_seed
from common.text import clean

STANCES = ["agree", "disagree", "query", "comment"]


def comment_folds(com, posts, n=5, seed=SEED):
    ev = com["post_id"].map(posts.set_index("post_id")["event_id"])
    post_lab = com["post_id"].map(posts.set_index("post_id")["label"])
    cv = StratifiedGroupKFold(n, shuffle=True, random_state=seed)
    fold = np.full(len(com), -1)
    for k, (_, te) in enumerate(cv.split(np.zeros(len(com)), post_lab.values, groups=ev.values)):
        fold[te] = k
    return fold


# ---------------------------------------------------------------- classical models
def fit_tfidf_lr(texts_tr, y_tr, texts_te):
    vec = make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=200000,
                        sublinear_tf=True),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=100000,
                        sublinear_tf=True))
    Xtr = vec.fit_transform(texts_tr)
    Xte = vec.transform(texts_te)
    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, y_tr)
    return clf.predict_proba(Xte)


def fit_xlmr_lr(P_tr, y_tr, P_te):
    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
    clf.fit(P_tr, y_tr)
    return clf.predict_proba(P_te)


# ---------------------------------------------------------------- neural heads
def build_head(kind, d_in=768, hid=64, n_cls=4):
    import torch, torch.nn as nn

    class Attn(nn.Module):
        def __init__(self, d):
            super().__init__(); self.w = nn.Linear(d, 1)

        def forward(self, h, m):
            s = self.w(h).squeeze(-1).masked_fill(m == 0, -1e4)
            a = torch.softmax(s, 1).unsqueeze(-1)
            return (a * h).sum(1)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            if kind in ("bilstm", "xstancefusion"):
                self.lstm = nn.LSTM(d_in, hid, batch_first=True, bidirectional=True)
            if kind == "bigru":
                self.gru = nn.GRU(d_in, hid, batch_first=True, bidirectional=True)
            if kind == "xstancefusion":
                self.gru = nn.GRU(2 * hid, hid, batch_first=True, bidirectional=True)
                self.attn = Attn(2 * hid)
            self.drop = nn.Dropout(0.3)
            self.out = nn.Linear(2 * hid, n_cls)

        def forward(self, x, m):
            if self.kind == "bilstm":
                h, _ = self.lstm(x)
            elif self.kind == "bigru":
                h, _ = self.gru(x)
            else:
                h, _ = self.lstm(x)
                h, _ = self.gru(h)
            if self.kind == "xstancefusion":
                z = self.attn(h, m)
            else:
                mf = m.unsqueeze(-1).float()
                z = (h * mf).sum(1) / mf.sum(1).clamp(min=1)
            return self.out(self.drop(z))
    return Head()


def fit_neural(kind, tokens, mask, y, tr_idx, te_idx, epochs=3, bs=256, seed=SEED):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    model = build_head(kind)
    # class-balanced loss (stance classes are imbalanced)
    counts = np.bincount(y[tr_idx], minlength=4).astype(float)
    w = torch.tensor(counts.sum() / (4 * np.maximum(counts, 1)), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr_idx)
        t0, tot = time.time(), 0.0
        for b in range(0, len(perm), bs):
            ib = np.sort(perm[b:b + bs])
            xb = torch.from_numpy(tokens[ib].astype(np.float32))
            mb = torch.from_numpy(mask[ib].astype(np.int64))
            yb = torch.from_numpy(y[ib])
            opt.zero_grad()
            loss = lossf(model(xb, mb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(ib)
        print(f"    {kind} epoch {ep+1}/{epochs} loss={tot/len(perm):.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    out = np.zeros((len(te_idx), 4), dtype=np.float32)
    with torch.inference_mode():
        for b in range(0, len(te_idx), 1024):
            ib = te_idx[b:b + 1024]
            xb = torch.from_numpy(tokens[ib].astype(np.float32))
            mb = torch.from_numpy(mask[ib].astype(np.int64))
            out[b:b + len(ib)] = torch.softmax(model(xb, mb), -1).numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tfidf_lr,xlmr_lr,bilstm,bigru,xstancefusion")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    set_seed()
    posts = pd.read_csv(PROCESSED / "posts.csv")
    com = pd.read_csv(PROCESSED / "comments.csv", keep_default_na=False)
    com = com[com["stance"].isin(STANCES)].reset_index(drop=True) if False else com
    valid = com["stance"].isin(STANCES).values
    y = com["stance"].map({s: i for i, s in enumerate(STANCES)}).fillna(-1).astype(int).values
    fold = comment_folds(com, posts)
    texts = [clean(t) for t in com["text"].astype(str)]
    np.save(CACHE / "stance_fold.npy", fold)

    need_tokens = any(m in args.models for m in ("bilstm", "bigru", "xstancefusion"))
    pooled = np.load(CACHE / "comment_pooled.npy") if "xlmr_lr" in args.models else None
    tokens = np.load(CACHE / "comment_tokens_fp16.npy", mmap_mode="r") if need_tokens else None
    mask = np.load(CACHE / "comment_mask.npy") if need_tokens else None

    summary = []
    for name in args.models.split(","):
        t_start = time.time()
        oof = np.zeros((len(com), 4), dtype=np.float32)
        for k in range(5):
            tr = np.where((fold != k) & valid)[0]
            te = np.where(fold == k)[0]
            print(f"[{name}] fold {k}: train {len(tr)} test {len(te)}", flush=True)
            if name == "tfidf_lr":
                oof[te] = fit_tfidf_lr([texts[i] for i in tr], y[tr], [texts[i] for i in te])
            elif name == "xlmr_lr":
                oof[te] = fit_xlmr_lr(pooled[tr], y[tr], pooled[te])
            else:
                oof[te] = fit_neural(name, tokens, mask, y, tr, te, epochs=args.epochs)
        np.save(CACHE / f"stance_oof_{name}.npy", oof)
        pred = oof.argmax(1)
        p, r, f, _ = precision_recall_fscore_support(y[valid], pred[valid], average="macro", zero_division=0)
        row = dict(model=name, precision=p, recall=r, f1=f,
                   accuracy=accuracy_score(y[valid], pred[valid]),
                   minutes=(time.time() - t_start) / 60)
        print(row, flush=True)
        summary.append(row)
    out = TABLES / "stance_results.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    new = pd.DataFrame(summary)
    if len(old):
        old = old[~old["model"].isin(new["model"])]
    pd.concat([old, new]).to_csv(out, index=False)


if __name__ == "__main__":
    main()

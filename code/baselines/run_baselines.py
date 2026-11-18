"""Baseline approaches compared in the KeepUp paper (Table 8), re-implemented on identical folds.

Input for every baseline = KeepUp's "all extracted features":
    mean-pooled XLM-R(title + comments) [768]  +  the 14 structured KeepUp features.
Sequence baselines (LSTM, Bi-RNN, Bi-GRU, CNN+LSTM) read the token sequence of
title+comments (XLM-R sub-word ids, trainable 128-d embeddings) and concatenate the 14
structured features before the output layer.

ML      : Logistic Regression, SVM (RBF), Random Forest, Gradient Boosting, KNN, XGBoost
DL      : LSTM, Bi-RNN, Bi-GRU
Hybrid  : CNN+LSTM, XLM-R+LightGBM ("RoBERTa+LightGBM"), Bi-LSTM+XLM-R (frozen token states),
          soft-voting Ensemble Learning (LR+RF+XGB)

usage: python code/baselines/run_baselines.py [--models ...] [--protocols event,random]
"""
import sys, argparse, copy
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CACHE, TABLES, SEEDS, set_seed
from common.features import load_tables, keepup_numeric
from common.evaluation import run_cv, summarize
from common.text import clean
from base_keepup.run_keepup import numeric_block

ML_MODELS = {
    "Logistic Regression": lambda s: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
    "SVM": lambda s: make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", probability=True, random_state=s)),
    "Random Forest": lambda s: RandomForestClassifier(500, n_jobs=-1, random_state=s),
    "Gradient Boosting": lambda s: GradientBoostingClassifier(random_state=s),
    "KNN": lambda s: make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
    "XGBoost": lambda s: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.5, n_jobs=8,
                                       random_state=s, eval_metric="logloss"),
    "XLM-R + LightGBM": lambda s: LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                                 num_leaves=15, colsample_bytree=0.5,
                                                 subsample=0.8, subsample_freq=1,
                                                 random_state=s, verbose=-1),
    "Ensemble Learning": lambda s: VotingClassifier([
        ("lr", make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))),
        ("rf", RandomForestClassifier(300, n_jobs=-1, random_state=s)),
        ("xgb", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=8,
                              random_state=s, eval_metric="logloss"))], voting="soft"),
}
SEQ_MODELS = ["LSTM", "Bi-RNN", "Bi-GRU", "CNN+LSTM", "Bi-LSTM + XLM-R"]


def ml_fn(builder, X, y):
    def fn(ctx):
        m = builder(ctx.seed + ctx.fold)
        tr = np.concatenate([ctx.fit_idx, ctx.val_idx])     # no early stopping -> use all train
        m.fit(X[tr], y[tr])
        return m.predict_proba(X[ctx.test_idx])[:, 1]
    return fn


# ------------------------------------------------------------ sequence models (PyTorch)
def seq_fn(kind, ids, N, y, tokstates=None, tokmask=None):
    import torch, torch.nn as nn

    class Net(nn.Module):
        def __init__(self, vocab, n_num):
            super().__init__()
            self.kind = kind
            d = 768 if kind == "Bi-LSTM + XLM-R" else 128
            if kind != "Bi-LSTM + XLM-R":
                self.emb = nn.Embedding(vocab, 128, padding_idx=1)
            if kind == "LSTM":
                self.rnn = nn.LSTM(d, 64, batch_first=True); out = 64
            elif kind == "Bi-RNN":
                self.rnn = nn.RNN(d, 64, batch_first=True, bidirectional=True); out = 128
            elif kind == "Bi-GRU":
                self.rnn = nn.GRU(d, 64, batch_first=True, bidirectional=True); out = 128
            elif kind == "CNN+LSTM":
                self.conv = nn.Conv1d(d, 64, 5, padding=2)
                self.rnn = nn.LSTM(64, 64, batch_first=True); out = 64
            else:
                self.rnn = nn.LSTM(d, 64, batch_first=True, bidirectional=True); out = 128
                self.att = nn.Linear(128, 1)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(out + n_num, 2)

        def forward(self, x, m, num):
            h = x if self.kind == "Bi-LSTM + XLM-R" else self.emb(x)
            if self.kind == "CNN+LSTM":
                h = torch.relu(self.conv(h.transpose(1, 2))).transpose(1, 2)
            o, _ = self.rnn(h)
            mf = m.unsqueeze(-1).float()
            if self.kind == "Bi-LSTM + XLM-R":
                s = self.att(o).squeeze(-1).masked_fill(m == 0, -1e4)
                z = (torch.softmax(s, 1).unsqueeze(-1) * o).sum(1)
            else:
                z = (o * mf).sum(1) / mf.sum(1).clamp(min=1)
            return self.fc(torch.cat([self.drop(z), num], 1))

    def fn(ctx):
        torch.manual_seed(ctx.seed + ctx.fold); torch.set_num_threads(8)
        rng = np.random.RandomState(ctx.seed + ctx.fold)
        sc = StandardScaler().fit(N[ctx.fit_idx]); Ns = sc.transform(N).astype(np.float32)
        net = Net(250002, N.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()

        def batch(ib):
            ib = np.sort(ib)
            if kind == "Bi-LSTM + XLM-R":
                x = torch.from_numpy(np.asarray(tokstates[ib], dtype=np.float32))
                m = torch.from_numpy(tokmask[ib].astype(np.int64))
            else:
                x = torch.from_numpy(ids[ib]); m = (x != 1).long()
            return ib, x, m, torch.from_numpy(Ns[ib])

        def predict(idx):
            net.eval(); out = np.zeros(len(idx)); pos = {v: i for i, v in enumerate(idx)}
            with torch.inference_mode():
                for b in range(0, len(idx), 64):
                    ib, x, m, n = batch(idx[b:b + 64])
                    p = torch.softmax(net(x, m, n), -1)[:, 1].numpy()
                    for j, v in enumerate(ib):
                        out[pos[v]] = p[j]
            return out

        best, state, wait = np.inf, None, 0
        for ep in range(20):
            net.train()
            perm = rng.permutation(ctx.fit_idx)
            for b in range(0, len(perm), 32):
                ib, x, m, n = batch(perm[b:b + 32])
                if len(ib) < 2:
                    continue
                opt.zero_grad()
                loss = lossf(net(x, m, n), torch.from_numpy(y[ib]).long())
                loss.backward(); opt.step()
            pv = np.clip(predict(ctx.val_idx), 1e-6, 1 - 1e-6); yv = y[ctx.val_idx]
            vl = -np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv))
            if vl < best - 1e-4:
                best, state, wait = vl, copy.deepcopy(net.state_dict()), 0
            else:
                wait += 1
                if wait >= 4:
                    break
        net.load_state_dict(state)
        return predict(ctx.test_idx)
    return fn


def token_ids(posts, com, max_len=256):
    from transformers import AutoTokenizer
    from common.paths import XLMR
    tok = AutoTokenizer.from_pretrained(XLMR)
    c = com.sort_values(["post_id", "order"])
    joined = c.groupby("post_id")["text"].apply(lambda s: " ".join(clean(x) for x in s))
    docs = [(clean(t) + " " + joined.get(p, "")).strip() or "." for p, t in zip(posts["post_id"], posts["title"])]
    return tok(docs, padding="max_length", truncation=True, max_length=max_len,
               return_tensors="np")["input_ids"].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(list(ML_MODELS) + SEQ_MODELS))
    ap.add_argument("--protocols", default="event,random")
    ap.add_argument("--stance", default="xstancefusion")
    args = ap.parse_args()
    set_seed()
    posts, com = load_tables()
    y = posts["label"].values.astype(int)
    stance = np.load(CACHE / f"stance_oof_{args.stance}.npy")
    N = numeric_block(keepup_numeric(posts, com, stance))
    E = np.load(CACHE / "post_doc_xlmr.npy")
    X = np.hstack([E, N]).astype(np.float32)
    ids = None
    allres = []
    for name in args.models.split(","):
        for prot in args.protocols.split(","):
            if name in ML_MODELS:
                fn, seeds = ml_fn(ML_MODELS[name], X, y), SEEDS
            elif name == "Bi-LSTM + XLM-R":
                T = np.load(CACHE / "post_doc_tokens_fp16.npy", mmap_mode="r")
                M = np.load(CACHE / "post_doc_mask.npy")
                fn, seeds = seq_fn(name, None, N, y, T, M), SEEDS[:1]
            else:
                if ids is None:
                    ids = token_ids(posts, com)
                fn, seeds = seq_fn(name, ids, N, y), SEEDS
            res, _ = run_cv(name, fn, posts, prot, seeds=seeds)
            allres.append(res)
    s = summarize(pd.concat(allres))
    print(s.round(4).to_string())
    out = TABLES / "baseline_results.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(old):
        old = old[~old.set_index(["model", "protocol"]).index.isin(s.set_index(["model", "protocol"]).index)]
    pd.concat([old, s]).to_csv(out, index=False)


if __name__ == "__main__":
    main()

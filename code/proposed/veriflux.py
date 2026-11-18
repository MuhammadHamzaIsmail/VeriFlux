"""VeriFlux - lightweight, explainable, socially-informed fake-news detector (proposed).

Pipeline per fold (all fitting uses training events only):
  1. compact named feature vector x (common/features.py, ~38 dims, six signal families)
  2. "text prior" t: char/word TF-IDF + logistic regression over title + comments,
     produced as an out-of-fold score inside the training fold (stacking, no leakage)
  3. gradient-boosted trees (LightGBM) on [x, t], hyper-parameters chosen on the inner
     validation events from a small grid, then refit on all training events.
"""
import itertools
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_union
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier

from common.text import clean

GRID = [dict(num_leaves=nl, learning_rate=lr, n_estimators=ne, min_child_samples=mc)
        for nl, lr, ne, mc in itertools.product([7, 15], [0.03, 0.08], [200, 400], [10, 25])]
BASE = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
            verbose=-1, n_jobs=8)


def post_texts(posts, com, max_comments=None):
    c = com.sort_values(["post_id", "order"])
    if max_comments is not None:
        c = c[c["order"] < max_comments]
    joined = c.groupby("post_id")["text"].apply(lambda s: " ".join(clean(x) for x in s)[:30000])
    return np.array([(clean(t) + " ||| " + joined.get(p, "")).strip()
                     for p, t in zip(posts["post_id"], posts["title"])], dtype=object)


def _tfidf_lr():
    vec = make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=150000,
                        sublinear_tf=True),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=80000,
                        sublinear_tf=True))
    return vec, LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced")


def text_prior(texts, y, groups, train_idx, test_idx, seed, protocol):
    """OOF text score for train_idx (inner 5-fold) and full-fit score for test_idx.
    Cached on disk per (texts, split): identical for every ablation configuration."""
    import hashlib
    from common.paths import CACHE
    h = hashlib.md5("|".join(map(str, texts)).encode("utf-8")).hexdigest()[:10]
    h2 = hashlib.md5(np.asarray(test_idx).tobytes() + np.asarray(train_idx).tobytes()).hexdigest()[:10]
    f = CACHE / "textprior" / f"{protocol}_{seed}_{h}_{h2}.npz"
    if f.exists():
        z = np.load(f); return z["oof"], z["test"]
    f.parent.mkdir(exist_ok=True)
    oof, te = _text_prior(texts, y, groups, train_idx, test_idx, seed, protocol)
    np.savez(f, oof=oof, test=te)
    return oof, te


def _text_prior(texts, y, groups, train_idx, test_idx, seed, protocol):
    oof = np.zeros(len(train_idx))
    inner = (GroupKFold(5).split(train_idx, groups=groups[train_idx]) if protocol == "event"
             else KFold(5, shuffle=True, random_state=seed).split(train_idx))
    for a, b in inner:
        vec, lr = _tfidf_lr()
        lr.fit(vec.fit_transform(texts[train_idx[a]]), y[train_idx[a]])
        oof[b] = lr.predict_proba(vec.transform(texts[train_idx[b]]))[:, 1]
    vec, lr = _tfidf_lr()
    lr.fit(vec.fit_transform(texts[train_idx]), y[train_idx])
    return oof, lr.predict_proba(vec.transform(texts[test_idx]))[:, 1]


def emb_prior(E, y, groups, train_idx, test_idx, seed, protocol):
    """Stacked dense-text prior (VeriFlux v2): RBF-SVM on a frozen multilingual document
    embedding; OOF scores inside the training fold, full-fit score for the test fold."""
    import hashlib
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from common.paths import CACHE
    h = hashlib.md5(np.ascontiguousarray(E[:5]).tobytes()).hexdigest()[:8]
    h2 = hashlib.md5(np.asarray(test_idx).tobytes() + np.asarray(train_idx).tobytes()).hexdigest()[:10]
    f = CACHE / "embprior" / f"{protocol}_{seed}_{h}_{h2}.npz"
    if f.exists():
        z = np.load(f); return z["oof"], z["test"]
    f.parent.mkdir(exist_ok=True)
    mk = lambda: make_pipeline(StandardScaler(), SVC(C=1.0, probability=True, random_state=seed))
    oof = np.zeros(len(train_idx))
    inner = (GroupKFold(5).split(train_idx, groups=groups[train_idx]) if protocol == "event"
             else KFold(5, shuffle=True, random_state=seed).split(train_idx))
    for a, b in inner:
        oof[b] = mk().fit(E[train_idx[a]], y[train_idx[a]]).predict_proba(E[train_idx[b]])[:, 1]
    te = mk().fit(E[train_idx], y[train_idx]).predict_proba(E[test_idx])[:, 1]
    np.savez(f, oof=oof, test=te)
    return oof, te


def select_params(X, y, fit_idx, val_idx, seed):
    best, best_p = np.inf, GRID[0]
    for p in GRID:
        m = LGBMClassifier(**p, **BASE, random_state=seed).fit(X[fit_idx], y[fit_idx])
        l = log_loss(y[val_idx], m.predict_proba(X[val_idx])[:, 1], labels=[0, 1])
        if l < best:
            best, best_p = l, p
    return best_p


def event_contrast(M: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Eq. (contrast): value minus the mean over all posts of the same event (labels unused)."""
    df = pd.DataFrame(M)
    return (df - df.groupby(groups).transform("mean")).values


class VeriFlux:
    def __init__(self, use_text_prior=True, classifier="lgbm", tune=True, contrast_idx=None,
                 emb=None, semantic=None):
        """contrast_idx: column indices of post-level features that get event-context contrast
        features (None = no contrast component).  emb: optional document embedding used as a
        stacked prior (explored, not in the final model).  semantic: [n, d] input of the
        semantic expert (final VeriFlux: [XLM-R doc embedding, event-centred embedding]);
        when given, the prediction is the validation-weighted fusion of both experts."""
        self.use_text_prior = use_text_prior
        self.classifier = classifier
        self.tune = tune
        self.contrast_idx = contrast_idx
        self.emb = emb
        self.semantic = semantic
        self.weights_ = []

    def fold_fn(self, X: np.ndarray, y: np.ndarray, texts, groups, return_model=False):
        def fn(ctx):
            tr = np.concatenate([ctx.fit_idx, ctx.val_idx])
            te = ctx.test_idx
            Xa = X.copy()
            priors = []
            if self.use_text_prior:
                tp_tr, tp_te = text_prior(texts, y, groups, tr, te, ctx.seed + ctx.fold, ctx.protocol)
                col = np.full(len(X), np.nan); col[tr] = tp_tr; col[te] = tp_te
                priors.append(col)
            if self.emb is not None:
                ep_tr, ep_te = emb_prior(self.emb, y, groups, tr, te, ctx.seed + ctx.fold, ctx.protocol)
                col = np.full(len(X), np.nan); col[tr] = ep_tr; col[te] = ep_te
                priors.append(col)
            if priors:
                Xa = np.column_stack([Xa] + priors)
            if self.contrast_idx is not None:
                C = X[:, self.contrast_idx]
                if priors:
                    C = np.column_stack([C] + priors)
                n_sib = pd.Series(groups).map(pd.Series(groups).value_counts()).values
                Xa = np.column_stack([Xa, event_contrast(C, groups), np.log1p(n_sib)])
            model = self.make(Xa, y, ctx)
            model.fit(Xa[tr], y[tr])
            prob = model.predict_proba(Xa[te])[:, 1]
            if return_model:
                ctx.extra["model"], ctx.extra["X"] = model, Xa
            if self.semantic is not None:
                prob = self._fuse(prob, Xa, y, ctx, model)
            return prob
        return fn

    # ------------------------------------------------------------ two-expert fusion (v2)
    W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

    def _fuse(self, p_feat_test, Xa, y, ctx, model):
        """p = w * p_semantic + (1-w) * p_features, w chosen by log-loss on validation events."""
        from sklearn.svm import SVC
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        S, s = self.semantic, ctx.seed + ctx.fold
        tr = np.concatenate([ctx.fit_idx, ctx.val_idx])
        mk = lambda: make_pipeline(StandardScaler(), SVC(C=1.0, probability=True, random_state=s))
        # validation-stage predictions of both experts (fit on fit_idx only)
        sv = mk().fit(S[ctx.fit_idx], y[ctx.fit_idx]).predict_proba(S[ctx.val_idx])[:, 1]
        fv = type(model)(**model.get_params()).fit(Xa[ctx.fit_idx], y[ctx.fit_idx]) \
            .predict_proba(Xa[ctx.val_idx])[:, 1]
        yv = y[ctx.val_idx]
        losses = [log_loss(yv, np.clip(w * sv + (1 - w) * fv, 1e-6, 1 - 1e-6), labels=[0, 1])
                  for w in self.W_GRID]
        w = self.W_GRID[int(np.argmin(losses))]
        ctx.extra["fusion_w"] = w
        p_sem = mk().fit(S[tr], y[tr]).predict_proba(S[ctx.test_idx])[:, 1]
        ctx.extra["p_sem"], ctx.extra["p_feat"] = p_sem, p_feat_test
        self.weights_.append(dict(seed=ctx.seed, fold=ctx.fold, protocol=ctx.protocol, w=w))
        return w * p_sem + (1 - w) * p_feat_test

    def make(self, Xa, y, ctx):
        s = ctx.seed + ctx.fold
        if self.classifier == "lgbm":
            p = select_params(Xa, y, ctx.fit_idx, ctx.val_idx, s) if self.tune else GRID[5]
            return LGBMClassifier(**p, **BASE, random_state=s)
        if self.classifier == "xgb":
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.7, n_jobs=8, random_state=s, eval_metric="logloss")
        if self.classifier == "rf":
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.pipeline import make_pipeline
            from sklearn.impute import SimpleImputer
            return make_pipeline(SimpleImputer(strategy="median"),
                                 RandomForestClassifier(500, min_samples_leaf=3, n_jobs=-1, random_state=s))
        if self.classifier == "lr":
            from sklearn.pipeline import make_pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 LogisticRegression(C=0.5, max_iter=5000))
        if self.classifier == "mlp":
            from sklearn.neural_network import MLPClassifier
            from sklearn.pipeline import make_pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 MLPClassifier((64, 32), alpha=1e-3, max_iter=600,
                                               early_stopping=True, random_state=s))
        raise ValueError(self.classifier)


# ============================================================================ final VeriFlux
def contrast_representation(E: np.ndarray, F: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Event-contrast representation: [E, E - event mean, F, F - event mean].
    Event means use all posts of the event (training and test) but never their labels,
    so the representation is available at inference time."""
    Fi = pd.DataFrame(F).fillna(pd.DataFrame(F).median()).values
    return np.hstack([E, event_contrast(E, groups), Fi, event_contrast(Fi, groups)]).astype(np.float32)


def veriflux_svm_fn(Z: np.ndarray, y: np.ndarray, C: float = 1.0):
    """VeriFlux: standardised event-contrast representation + RBF-SVM (Platt probabilities)."""
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def fn(ctx):
        tr = np.concatenate([ctx.fit_idx, ctx.val_idx])
        m = make_pipeline(StandardScaler(), SVC(C=C, kernel="rbf", probability=True,
                                                random_state=ctx.seed + ctx.fold))
        m.fit(Z[tr], y[tr])
        ctx.extra["model"] = m
        return m.predict_proba(Z[ctx.test_idx])[:, 1]
    return fn

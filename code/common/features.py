"""Feature construction for the base paper (KeepUp) and the proposed method (VeriFlux).

All features are computed from information available when a post is classified
(title, author profile, engagement counters, comments, claim text, evidence corpus).
Gold comment stance labels are NEVER used as features: stance comes from the
cross-fitted stance classifiers (05_stance.py).
"""
import re
import numpy as np
import pandas as pd

from .paths import PROCESSED, CACHE, REPO
from .text import clean

STANCES = ["agree", "disagree", "query", "comment"]
CLICKBAIT = ["you won't believe", "shocking", "what happened next", "can't miss", "top 10",
             "reasons why"]   # list used in the authors' user_features notebook

# Debunk / doubt cue lexicon (English, Roman Urdu, Urdu).  Hand-built, 30 entries.
DEBUNK = ["fake", "false", "lie", "lies", "liar", "rumor", "rumour", "propaganda", "hoax",
          "fact check", "not true", "misleading", "old video", "edited", "jhoot", "jhooti",
          "jhoota", "jhuti", "jhoti", "afwah", "ghalat", "galat", "jaali", "jali", "bakwas",
          "جھوٹ", "جعلی", "افواہ", "غلط", "پروپیگنڈا"]
DEBUNK_RE = re.compile("|".join(re.escape(w) for w in DEBUNK), re.IGNORECASE)


def load_tables():
    posts = pd.read_csv(PROCESSED / "posts.csv", keep_default_na=False, na_values=[""])
    posts["title"] = posts["title"].fillna("").astype(str)
    com = pd.read_csv(PROCESSED / "comments.csv", keep_default_na=False)
    return posts, com


def textblob_polarity(texts):
    from textblob import TextBlob
    return np.array([TextBlob(t).sentiment.polarity for t in texts])


def stance_matrix(com, stance_probs, max_comments=None, hard=True):
    """Aggregate comment-level stance predictions to post level.

    returns DataFrame indexed by post_id with counts and ratios per stance."""
    sel = np.ones(len(com), bool) if max_comments is None else (com["order"].values < max_comments)
    P = stance_probs[sel]
    if hard:
        P = np.eye(4)[P.argmax(1)]
    d = pd.DataFrame(P, columns=STANCES)
    d["post_id"] = com["post_id"].values[sel]
    cnt = d.groupby("post_id")[STANCES].sum()
    n = cnt.sum(1)
    ratio = cnt.div(n.clip(lower=1), axis=0)
    return cnt, ratio, n


# ------------------------------------------------------------------- KeepUp (base paper)
KEEPUP_NUMERIC = ["likes", "comments", "followers", "followings", "unverified", "account_age",
                  "st_agree", "st_disagree", "st_query", "st_comment", "entail_pos_prob",
                  "title_len", "title_sentiment", "clickbait"]


def keepup_numeric(posts, com, stance_probs, entail_source="ours"):
    """The 14 structured features of KeepUp Sec. 4.5 (Fig. 18 numeric branch)."""
    cnt, _, _ = stance_matrix(com, stance_probs)
    cnt = cnt.reindex(posts["post_id"]).fillna(0)
    X = pd.DataFrame(index=posts.index)
    X["likes"] = posts["likescount"].fillna(0)
    X["comments"] = posts["commentscount"].fillna(0)
    X["followers"] = posts["followers"].fillna(0)
    X["followings"] = posts["followings"].fillna(0)
    X["unverified"] = posts["unverified"].fillna(1)
    X["account_age"] = posts["account_age_days"].fillna(0)
    for s in STANCES:
        X["st_" + s] = cnt[s].values
    if entail_source == "ours":
        nli = pd.read_csv(CACHE / "nli_claims.csv").set_index("event_id")
        X["entail_pos_prob"] = posts["event_id"].map(nli["keepup_pos_prob"]).fillna(0).values
    else:   # authors' released file (claim-level)
        rel = pd.read_csv(REPO / "entailment_predictions_results.csv").set_index("claim-id")
        X["entail_pos_prob"] = posts["claim_id"].map(rel["positive_prob"]).fillna(0).values
    titles = posts["title"].astype(str)
    X["title_len"] = titles.map(lambda t: len(t.split()))
    X["title_sentiment"] = textblob_polarity(titles.tolist())
    X["clickbait"] = titles.map(lambda t: int(any(p in t.lower() for p in CLICKBAIT)))
    return X[KEEPUP_NUMERIC].astype(float)


# ------------------------------------------------------------------- VeriFlux (proposed)
GROUPS = {
    "evidence": ["ev_E_top1", "ev_N_top1", "ev_C_top1", "ev_maxE_top3", "ev_maxC_top3",
                 "ret_score"],
    "post_claim": ["pc_E", "pc_N", "pc_C", "pc_cos", "consistency"],
    "stance": ["r_agree", "r_disagree", "r_query", "r_comment", "log_n_comments",
               "no_comments", "debunk_ratio", "doubt_index"],
    "user": ["log_followers", "log_followings", "unverified", "log_account_age",
             "log_posts_count"],
    "engagement": ["log_likes", "log_commentscount", "log_shares", "log_views",
                   "platform_facebook", "platform_twitter", "platform_youtube",
                   "platform_instagram", "media_type"],
    "title": ["title_words", "title_chars", "title_urdu_script", "title_has_url",
              "title_exclaim"],
}
ALL_VERIFLUX = [f for g in GROUPS.values() for f in g]


def veriflux_features(posts, com, stance_probs, max_comments=None, retrieval="bm25"):
    """Compact, named, CPU-cheap feature vector (Sec. III of the paper)."""
    X = pd.DataFrame(index=posts.index)
    # --- evidence (claim-level NLI against retrieved evidence)
    nli = pd.read_csv(CACHE / "nli_claims.csv").set_index("event_id")
    for c in GROUPS["evidence"][:-1]:
        X[c] = posts["event_id"].map(nli[c]).values
    ret = pd.read_csv(CACHE / f"retrieval_{retrieval}.csv").set_index("claim_id")
    top_score = ret["scores"].map(lambda s: float(str(s).split()[0]))
    X["ret_score"] = posts["claim_id"].map(top_score).values
    # --- post-claim agreement + veracity-consistency interaction (guideline Table 3)
    pc = pd.read_csv(CACHE / "nli_posts.csv").set_index("post_id")
    for c in ["pc_E", "pc_N", "pc_C", "pc_cos"]:
        X[c] = posts["post_id"].map(pc[c]).values
    claim_support = X["ev_E_top1"] - X["ev_C_top1"]          # evidence says claim true (+) / false (-)
    post_agree = X["pc_E"] - X["pc_C"]                        # post repeats (+) / refutes (-) claim
    X["consistency"] = claim_support * post_agree
    # --- stance (predicted, optionally only the first k comments)
    _, ratio, n = stance_matrix(com, stance_probs, max_comments)
    ratio = ratio.reindex(posts["post_id"]).fillna(0)
    n = n.reindex(posts["post_id"]).fillna(0)
    for s in STANCES:
        X["r_" + s] = ratio[s].values
    X["log_n_comments"] = np.log1p(n.values)
    X["no_comments"] = (n.values == 0).astype(int)
    sel = com if max_comments is None else com[com["order"] < max_comments]
    deb = sel["text"].astype(str).str.contains(DEBUNK_RE).groupby(sel["post_id"]).mean()
    X["debunk_ratio"] = posts["post_id"].map(deb).fillna(0).values
    X["doubt_index"] = X["r_disagree"] + X["r_query"] - X["r_agree"]
    # --- user profile
    X["log_followers"] = np.log1p(posts["followers"].fillna(0))
    X["log_followings"] = np.log1p(posts["followings"].fillna(0))
    X["unverified"] = posts["unverified"].fillna(1)
    X["log_account_age"] = np.log1p(posts["account_age_days"])          # NaN kept (trees handle)
    X["log_posts_count"] = np.log1p(posts["posts_count"])
    # --- engagement (final counters as released; see limitations)
    X["log_likes"] = np.log1p(posts["likescount"])
    X["log_commentscount"] = np.log1p(posts["commentscount"])
    X["log_shares"] = np.log1p(posts["shares"])
    X["log_views"] = np.log1p(posts["views"])
    for p in ["facebook", "twitter", "youtube", "instagram"]:
        X["platform_" + p] = posts["platform"].astype(str).str.contains(p[:4]).astype(int)
    X["media_type"] = pd.to_numeric(posts["media_type"], errors="coerce")
    # --- title style
    t = posts["title"].astype(str)
    X["title_words"] = t.map(lambda s: len(s.split()))
    X["title_chars"] = t.str.len()
    X["title_urdu_script"] = (posts["title_lang"] == "urdu").astype(int)
    X["title_has_url"] = t.str.contains(r"http|www\.", regex=True).astype(int)
    X["title_exclaim"] = t.str.count("!")
    return X[ALL_VERIFLUX].astype(float)

"""Step 1 - load the raw KeepUp workbook, clean it and write tidy parquet/csv tables.

Outputs (data/processed/):
  posts.csv     one row per post: ids, event, claim, label, title, engagement, user profile
  comments.csv  one row per comment: post-id, order within post, text, gold stance label
  claims.csv    one row per claim/event
  evidence/     extracted evidence documents (from evidence.7z)
"""
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import XLSX, EVIDENCE_7Z, EVIDENCE_DIR, PROCESSED


def parse_count(v):
    """'25k' -> 25000, '1.2M' -> 1200000, '' -> NaN."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower().replace(",", "")
    if s in ("", "nan", "none", "-"):
        return np.nan
    mult = 1.0
    if s.endswith("k"):
        mult, s = 1e3, s[:-1]
    elif s.endswith("m"):
        mult, s = 1e6, s[:-1]
    elif s.endswith("b"):
        mult, s = 1e9, s[:-1]
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group()) * mult if m else np.nan


def parse_date(v):
    if pd.isna(v):
        return pd.NaT
    if isinstance(v, (pd.Timestamp,)) or hasattr(v, "year"):
        return pd.Timestamp(v)
    s = str(v).strip().replace("\\", "/").replace("-", "/").replace(".", "/")
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
# Frequent Roman-Urdu function words (heuristic language tagging)
ROMAN_URDU = set("""hai hain ka ki ke ko se mein main nahi nahin kya ye yeh wo woh aur bhi tha thi
thy hy ho hota hoti kar karo kr diya gaya gya ap aap hum tum unka inka sab koi kuch bohat bohot bht
acha achha allah ki jo jab tak par pe sirf phir agar lekin magar k h ha na ni kia kyun kyon wala
wali log logon apna apni humain mujhe mera meri tera yar yaar bhai jee ji g hen gaya gayi raha rahi
rahe sahi galat ghalat jhoot jhoota khuda inshallah mashallah zinda abad hy hai hain kuch nai nhi
bilkul shukriya jazakallah ameen bhi sb sab bohot zaroor kyu""".split())


def detect_lang(text: str) -> str:
    t = str(text)
    n_ar = len(ARABIC.findall(t))
    n_lat = len(LATIN.findall(t))
    n_dev = len(DEVANAGARI.findall(t))
    if n_ar == n_lat == n_dev == 0:
        return "other"
    if n_dev > max(n_ar, n_lat):
        return "hindi"
    if n_ar >= n_lat:
        return "urdu"          # Arabic script (Urdu dominates the dataset)
    words = re.findall(r"[a-z]+", t.lower())
    if not words:
        return "english"
    ru = sum(w in ROMAN_URDU for w in words) / len(words)
    return "roman_urdu" if ru >= 0.10 else "english"


def norm_platform(p):
    p = re.sub(r"[^a-z]", "", str(p).lower())
    if p.startswith(("fa", "fc")): return "facebook"
    if p.startswith("ins"): return "instagram"
    if p.startswith(("tw", "te", "x")): return "twitter"
    if p.startswith("you"): return "youtube"
    return "other"


def main():
    xl = pd.ExcelFile(XLSX)
    events = pd.read_excel(xl, "Events")
    posts = pd.read_excel(xl, "post features")
    comments = pd.read_excel(xl, "comments")
    users = pd.read_excel(xl, "user features")
    for df in (events, posts, comments, users):
        df.columns = [str(c).strip() for c in df.columns]
        df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], inplace=True)

    # ---------- claims / events ----------
    claims = events.dropna(subset=["Event-id"]).copy()
    claims = claims[["Event-id", "Event Name", "claim-id", "claim", "claim-category",
                     "claim-goldevidence", "label"]].drop_duplicates("Event-id")
    claims.columns = ["event_id", "event_name", "claim_id", "claim", "claim_category",
                      "gold_evidence", "claim_label"]

    # ---------- posts ----------
    posts = posts.dropna(subset=["post-id"]).copy()
    posts["post_id"] = posts["post-id"].astype(int)
    posts = posts.rename(columns={"Event-id": "event_id", "post-title": "title",
                                  "post-label": "label", "platform": "platform",
                                  "image(image 0, video 1, if no image video 2)": "media_type"})
    for c in ["likescount", "commentscount", "views", "shares", "reposts"]:
        posts[c] = posts[c].map(parse_count)
    posts["post_date"] = posts["timestamp"].map(parse_date)
    posts["title"] = posts["title"].fillna("").astype(str)
    posts["platform"] = posts["platform"].astype(str).str.strip().str.lower().map(norm_platform)
    posts["event_id"] = posts["event_id"].astype(str).str.strip()

    # ---------- users (one row per post author) ----------
    users = users.dropna(subset=["post-id"]).copy()
    users["post_id"] = users["post-id"].astype(int)
    users = users.rename(columns={"is user verified(0 verified, 1 unverified)": "unverified",
                                  "posts count": "posts_count", "joining date": "join_date"})
    users["followers"] = users["followers"].map(parse_count)
    users["followings"] = users["followings"].map(parse_count)
    users["posts_count"] = users["posts_count"].map(parse_count)
    users["join_date"] = users["join_date"].map(parse_date)
    users = users[["post_id", "username", "followers", "followings", "unverified",
                   "posts_count", "join_date"]].drop_duplicates("post_id")

    df = posts.merge(users, on="post_id", how="left")
    df = df.merge(claims, on="event_id", how="left")
    # account age measured relative to the post date (no look-ahead to "today")
    ref = df["post_date"].fillna(df["post_date"].median())
    df["account_age_days"] = (ref - df["join_date"]).dt.days.clip(lower=0)
    df["title_lang"] = df["title"].map(detect_lang)
    df["label"] = df["label"].astype(int)

    keep = ["post_id", "event_id", "claim_id", "event_name", "claim", "claim_category",
            "claim_label", "gold_evidence", "platform", "media_type", "title", "title_lang",
            "label", "likescount", "commentscount", "views", "shares", "reposts", "post_date",
            "username", "followers", "followings", "unverified", "posts_count", "join_date",
            "account_age_days"]
    df = df[keep].sort_values("post_id").reset_index(drop=True)

    # ---------- comments ----------
    comments = comments.dropna(subset=["post-id"]).copy()
    comments["post_id"] = comments["post-id"].astype(int)
    comments = comments.rename(columns={"commenttext": "text", "comment label": "stance",
                                        "likescount on comment": "comment_likes"})
    comments["text"] = comments["text"].fillna("").astype(str)
    comments["stance"] = comments["stance"].astype(str).str.strip().str.lower()
    comments["order"] = comments.groupby("post_id").cumcount()   # sheet order within a post
    comments["lang"] = comments["text"].map(detect_lang)
    comments = comments[["post_id", "order", "text", "stance", "lang", "comment_likes"]]

    df.to_csv(PROCESSED / "posts.csv", index=False)
    comments.to_csv(PROCESSED / "comments.csv", index=False)
    claims.to_csv(PROCESSED / "claims.csv", index=False)

    # ---------- evidence ----------
    if not any(EVIDENCE_DIR.iterdir()):
        import py7zr
        with py7zr.SevenZipFile(EVIDENCE_7Z, "r") as z:
            z.extractall(EVIDENCE_DIR)

    # ---------- summary ----------
    print("posts", df.shape, "labels", df["label"].value_counts().to_dict())
    print("events", df["event_id"].nunique(), "claims", claims.shape[0])
    print("comments", comments.shape, comments["stance"].value_counts().to_dict())
    print("title lang", df["title_lang"].value_counts().to_dict())
    print("comment lang", comments["lang"].value_counts().to_dict())
    print("platform", df["platform"].value_counts().to_dict())
    print(df.isna().mean().round(3).to_dict())


if __name__ == "__main__":
    main()

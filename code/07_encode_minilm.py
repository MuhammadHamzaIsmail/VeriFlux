"""Step 7 - lightweight dense text representation for VeriFlux v2.

Multilingual MiniLM-L12 sentence encoder (118M params, 384-d, ~3x cheaper than XLM-R-base):
  post_doc_minilm.npy    mean-pooled over "title + first comments" (max 256 tokens)
  post_title_minilm.npy  title only
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import PROCESSED, CACHE, SBERT_MINI
from common.text import clean, mean_pool_encode


def main():
    posts = pd.read_csv(PROCESSED / "posts.csv", keep_default_na=False)
    com = pd.read_csv(PROCESSED / "comments.csv", keep_default_na=False).sort_values(["post_id", "order"])
    joined = com.groupby("post_id")["text"].apply(lambda s: " ".join(clean(x) for x in s)[:4000])
    docs = [(clean(t) + " " + joined.get(p, "")).strip() or "." for p, t in zip(posts.post_id, posts.title)]
    titles = [clean(t) or "." for t in posts.title]
    t0 = time.time()
    np.save(CACHE / "post_doc_minilm.npy", mean_pool_encode(docs, SBERT_MINI, max_len=256, batch_size=16, desc="doc"))
    np.save(CACHE / "post_title_minilm.npy", mean_pool_encode(titles, SBERT_MINI, max_len=128, batch_size=32, desc="title"))
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()

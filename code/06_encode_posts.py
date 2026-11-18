"""Step 6 - post-level XLM-RoBERTa representations used by KeepUp and the baselines.

  post_doc_xlmr.npy        mean-pooled XLM-R-base over "title + all comments" (512 tokens)
                           -> input of KeepUp's fusion network as in the released code
  post_doc_tokens_fp16.npy token states [N, 512, 768] -> e_T in R^{n x 768} as in the
                           paper's Eq. 17 (CNN / BiGRU branches over the token sequence)
  post_doc_mask.npy        attention mask [N, 512]
  post_title_xlmr.npy      mean-pooled XLM-R over the title only (title-only ablation)
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import PROCESSED, CACHE, XLMR
from common.text import clean, mean_pool_encode

MAX_LEN = 512


def post_documents(posts, comments, max_comments=None):
    com = comments.sort_values(["post_id", "order"])
    if max_comments is not None:
        com = com[com["order"] < max_comments]
    joined = com.groupby("post_id")["text"].apply(lambda s: " ".join(clean(x) for x in s))
    return [(clean(t) + " " + joined.get(pid, "")).strip() or "."
            for pid, t in zip(posts["post_id"], posts["title"])]


def encode_with_tokens(texts):
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(XLMR)
    model = AutoModel.from_pretrained(XLMR).eval()
    n = len(texts)
    T = np.lib.format.open_memmap(CACHE / "post_doc_tokens_fp16.npy", mode="w+",
                                  dtype=np.float16, shape=(n, MAX_LEN, 768))
    M = np.zeros((n, MAX_LEN), dtype=np.uint8)
    P = np.zeros((n, 768), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, n, 4):
            enc = tok(texts[i:i + 4], padding="max_length", truncation=True, max_length=MAX_LEN,
                      return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"]
            mf = m.unsqueeze(-1).float()
            P[i:i + 4] = ((h * mf).sum(1) / mf.sum(1).clamp(min=1)).numpy()
            T[i:i + 4] = h.numpy().astype(np.float16)
            M[i:i + 4] = m.numpy().astype(np.uint8)
            if (i // 4) % 25 == 0:
                print(f"  posts {i+4}/{n} {(time.time()-t0)/60:.1f}min", flush=True)
    T.flush()
    np.save(CACHE / "post_doc_mask.npy", M)
    np.save(CACHE / "post_doc_xlmr.npy", P)


def main():
    posts = pd.read_csv(PROCESSED / "posts.csv", keep_default_na=False)
    com = pd.read_csv(PROCESSED / "comments.csv", keep_default_na=False)
    if not (CACHE / "post_title_xlmr.npy").exists():
        titles = [clean(t) or "." for t in posts["title"]]
        np.save(CACHE / "post_title_xlmr.npy", mean_pool_encode(titles, max_len=512, desc="titles"))
    if not (CACHE / "post_doc_xlmr.npy").exists():
        encode_with_tokens(post_documents(posts, com))
    print("done")


if __name__ == "__main__":
    main()

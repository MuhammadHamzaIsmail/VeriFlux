"""Step 4a - encode every comment once with a frozen XLM-RoBERTa-base encoder.

Stores (results/cache/):
  comment_tokens_fp16.npy  [N, L, 768] token states (memmap) for XStanceFusion-style heads
  comment_mask.npy         [N, L] attention mask
  comment_pooled.npy       [N, 768] mean-pooled vectors
CPU-only constraint: the encoder is frozen instead of fine-tuned (documented deviation).
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/, for common.*
from common.paths import PROCESSED, CACHE, XLMR
from common.text import clean

MAX_LEN = 48
BATCH = 64


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(8)
    com = pd.read_csv(PROCESSED / "comments.csv", keep_default_na=False)
    texts = [clean(t) or "." for t in com["text"].astype(str)]
    n = len(texts)
    tok = AutoTokenizer.from_pretrained(XLMR)
    model = AutoModel.from_pretrained(XLMR).eval()

    tok_path = CACHE / "comment_tokens_fp16.npy"
    tokens = np.lib.format.open_memmap(tok_path, mode="w+", dtype=np.float16, shape=(n, MAX_LEN, 768))
    mask = np.zeros((n, MAX_LEN), dtype=np.uint8)
    pooled = np.zeros((n, 768), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])
    t0 = time.time()
    with torch.inference_mode():
        for b in range(0, n, BATCH):
            idx = order[b:b + BATCH]
            enc = tok([texts[i] for i in idx], padding="max_length", truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"]
            mf = m.unsqueeze(-1).float()
            pooled[idx] = ((h * mf).sum(1) / mf.sum(1).clamp(min=1)).numpy()
            tokens[idx] = h.numpy().astype(np.float16)
            mask[idx] = m.numpy().astype(np.uint8)
            if (b // BATCH) % 50 == 0:
                el = time.time() - t0
                print(f"  {b + len(idx)}/{n}  {el/60:.1f} min elapsed", flush=True)
    tokens.flush()
    np.save(CACHE / "comment_mask.npy", mask)
    np.save(CACHE / "comment_pooled.npy", pooled)
    print("done in", round((time.time() - t0) / 60, 1), "min")


if __name__ == "__main__":
    main()

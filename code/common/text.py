"""Text cleaning and transformer helpers (CPU friendly)."""
import re
import numpy as np

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
HASHTAG = re.compile(r"#(\w+)")
SPACES = re.compile(r"\s+")


def clean(text: str) -> str:
    """KeepUp preprocessing: remove URLs, usernames, hashtag symbols and extra spaces."""
    t = str(text)
    t = URL.sub(" ", t)
    t = MENTION.sub(" ", t)
    t = HASHTAG.sub(r"\1", t)
    t = SPACES.sub(" ", t)
    return t.strip()


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def mean_pool_encode(texts, model_name=None, max_len=512, batch_size=8,
                     desc="encode"):
    """Mean-pooled last-hidden-state embeddings (KeepUp Eq. 15)."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    from .paths import XLMR
    model_name = model_name or XLMR
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    # sort by length for efficient padding, restore order afterwards
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    done = 0
    with torch.inference_mode():
        for chunk in batched(list(order), batch_size):
            enc = tok([texts[i] for i in chunk], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            emb = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out[chunk] = emb.numpy()
            done += len(chunk)
            if done % (batch_size * 25) < batch_size:
                print(f"  {desc}: {done}/{len(texts)}", flush=True)
    return out

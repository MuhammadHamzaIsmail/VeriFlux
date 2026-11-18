"""KeepUp fake-news classifier (Wasim et al., Array 2026, Sec. 4.5, Fig. 18, Table 6).

Three branches fused by concatenation:
  CNN branch    Conv1D(768->64,k3)+BN+ReLU -> Conv1D(64->32,k3)+BN+ReLU -> AdaptiveMaxPool
                -> Dropout                                               h_CNN in R^32
  BiGRU branch  BiGRU(768, hidden 64) -> attention pooling -> BN -> Dropout h_GRU in R^128
  Numeric       Linear(14->32) -> BN -> ReLU -> Dropout                  h_N   in R^32
  Classifier    Linear(192 -> 2) + softmax, cross-entropy, Adam lr 1e-3, batch 32,
                <=30 epochs, early stopping patience 5 on VALIDATION loss.

Two input variants are supported:
  "pooled"   e_T is the mean-pooled 768-d vector treated as a length-1 sequence
             (this is what the authors' released notebook does)
  "sequence" e_T is the 512x768 token-state matrix (what Eq. 17 of the paper describes)
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, h, mask=None):
        s = self.w(h).squeeze(-1)
        if mask is not None:
            s = s.masked_fill(mask == 0, -1e4)
        a = torch.softmax(s, 1).unsqueeze(-1)
        return (a * h).sum(1)


class KeepUpNet(nn.Module):
    def __init__(self, n_numeric=14, emb=768, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv1d(emb, 64, 3, padding=1); self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 32, 3, padding=1); self.bn2 = nn.BatchNorm1d(32)
        self.gru = nn.GRU(emb, 64, batch_first=True, bidirectional=True)
        self.attn = Attention(128); self.bn_g = nn.BatchNorm1d(128)
        self.fc_n = nn.Linear(n_numeric, 32); self.bn_n = nn.BatchNorm1d(32)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(32 + 128 + 32, 2)

    def forward(self, e, num, mask=None):
        # e: [B, T, 768]
        x = e.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(1) == 0, -1e4)
        h_cnn = self.drop(F.adaptive_max_pool1d(x, 1).squeeze(-1))
        g, _ = self.gru(e)
        h_gru = self.drop(self.bn_g(self.attn(g, mask)))
        h_n = self.drop(F.relu(self.bn_n(self.fc_n(num))))
        return self.cls(torch.cat([h_cnn, h_gru, h_n], 1))


def _batches(idx, bs, rng=None):
    idx = np.array(idx)
    if rng is not None:
        idx = rng.permutation(idx)
    for i in range(0, len(idx), bs):
        yield np.sort(idx[i:i + bs])


def train_predict(E, N, y, fit_idx, val_idx, test_idx, mask=None, seed=42, epochs=30,
                  patience=5, bs=32, lr=1e-3, n_threads=8, verbose=False):
    """E: [n,768] (pooled) or [n,T,768] memmap (sequence). N: standardized numeric [n,14]."""
    torch.manual_seed(seed)
    torch.set_num_threads(n_threads)
    rng = np.random.RandomState(seed)
    seq = E.ndim == 3
    model = KeepUpNet(n_numeric=N.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    def tensors(ib):
        e = torch.from_numpy(np.asarray(E[ib], dtype=np.float32))
        if not seq:
            e = e.unsqueeze(1)
        m = torch.from_numpy(np.asarray(mask[ib], dtype=np.int64)) if (seq and mask is not None) else None
        return e, torch.from_numpy(N[ib].astype(np.float32)), m

    def predict(idx):
        model.eval(); out = []
        with torch.inference_mode():
            for ib in _batches(idx, 64):
                e, n, m = tensors(ib)
                out.append(torch.softmax(model(e, n, m), -1)[:, 1].numpy())
        # _batches sorts within batch; idx passed sorted so order is preserved
        return np.concatenate(out)

    fit_idx, val_idx, test_idx = map(np.sort, (fit_idx, val_idx, test_idx))
    best, best_state, wait = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for ib in _batches(fit_idx, bs, rng):
            if len(ib) < 2:
                continue
            e, n, m = tensors(ib)
            opt.zero_grad()
            loss = lossf(model(e, n, m), torch.from_numpy(y[ib]).long())
            loss.backward(); opt.step()
        pv = np.clip(predict(val_idx), 1e-6, 1 - 1e-6)
        yv = y[val_idx]
        vloss = -np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv))
        if verbose:
            print(f"      epoch {ep+1} val_loss={vloss:.4f}")
        if vloss < best - 1e-4:
            best, best_state, wait = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return predict(test_idx), ep + 1

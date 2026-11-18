"""Sanity baseline exposing event leakage: predict the majority label of the post's event
among TRAINING posts (global training prior for unseen events).  No text, no features.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import TABLES, PROCESSED
from common.evaluation import run_cv, summarize


def main():
    posts = pd.read_csv(PROCESSED / "posts.csv")

    def fn(ctx):
        tr = np.concatenate([ctx.fit_idx, ctx.val_idx])
        maj = posts.iloc[tr].groupby("event_id")["label"].mean()
        prior = posts["label"].iloc[tr].mean()
        p = posts["event_id"].iloc[ctx.test_idx].map(maj).fillna(prior).values
        return np.clip(p, 0, 1)

    res = [run_cv("Event-majority lookup", fn, posts, prot, verbose=False)[0] for prot in ("event", "random")]
    s = summarize(pd.concat(res)); print(s.round(4).to_string())
    s.to_csv(TABLES / "leakage_baseline.csv", index=False)

    g = posts.groupby("event_id")["label"].agg(["mean", "size"])
    pur = np.maximum(g["mean"], 1 - g["mean"])
    pd.DataFrame([dict(events=len(g), posts_per_event_median=g["size"].median(),
                       posts_per_event_max=g["size"].max(),
                       label_purity_weighted=(pur * g["size"]).sum() / g["size"].sum(),
                       share_posts_in_pure_events=g.loc[pur == 1, "size"].sum() / len(posts))]
                 ).to_csv(TABLES / "event_purity.csv", index=False)


if __name__ == "__main__":
    main()

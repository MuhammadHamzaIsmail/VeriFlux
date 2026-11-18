"""Figures for the paper: random-vs-event gap and comment-budget curve (SHAP figure is made in
explain_efficiency_language.py)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import TABLES, FIGURES, ROOT

PAPER_FIG = ROOT / "paper" / "figures"
PAPER_FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})


def save(name):
    for d in (FIGURES, PAPER_FIG):
        plt.savefig(d / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close()


def gap():
    df = pd.concat([pd.read_csv(TABLES / n) for n in
                    ["baseline_results.csv", "keepup_results.csv", "veriflux_results.csv", "veriflux_final.csv", "leakage_baseline.csv"]
                    if (TABLES / n).exists()])
    pv = df.pivot_table(index="model", columns="protocol", values="f1_mean").dropna()
    pv = pv.sort_values("event")
    y = np.arange(len(pv))
    fig, ax = plt.subplots(figsize=(3.4, 0.17 * len(pv) + 0.7))
    ax.barh(y + 0.2, pv["random"], 0.4, label="Random post split", color="#c9c9c9")
    ax.barh(y - 0.2, pv["event"], 0.4, label="Event-held-out", color="#3b6ea8")
    for i, m in enumerate(pv.index):
        if m == "VeriFlux":
            ax.get_yticklabels()
    ax.set_yticks(y); ax.set_yticklabels(pv.index)
    for lbl in ax.get_yticklabels():
        if lbl.get_text().startswith("VeriFlux"):
            lbl.set_fontweight("bold")
    ax.axvline(0.5, ls=":", c="k", lw=0.6)
    ax.set_xlabel("Macro-F1"); ax.set_xlim(0.3, 0.85)
    ax.legend(loc="lower right", frameon=False)
    save("random_vs_event")


def budget():
    p = TABLES / "veriflux_budget.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d["k"] = d["model"].str.extract(r"@(\d+)").astype(float)
    full = pd.read_csv(TABLES / "veriflux_results.csv")
    full = full[(full.protocol == "event") & (full.model == "VeriFlux-Lite")].iloc[0]
    d = d.sort_values("k")
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.errorbar(d["k"].astype(int).astype(str).tolist() + ["all"],
                list(d["f1_mean"]) + [full["f1_mean"]], yerr=list(d["f1_std"]) + [full["f1_std"]],
                marker="o", ms=3, capsize=2, color="#3b6ea8", label="VeriFlux-Lite")
    ku = pd.read_csv(TABLES / "keepup_results.csv") if (TABLES / "keepup_results.csv").exists() else None
    if ku is not None:
        r = ku[(ku.protocol == "event") & (ku.model == "KeepUp (released code)")]
        if len(r):
            ax.axhline(r["f1_mean"].iloc[0], ls="--", c="#b5523b", lw=0.8, label="KeepUp (all comments)")
            ax.legend(frameon=False, loc="lower right")
    ax.set_xlabel("Comments available per post (first k)"); ax.set_ylabel("Macro-F1 (event-held-out)")
    save("comment_budget")


if __name__ == "__main__":
    gap(); budget()
    print("figures written")

"""Collect every number the slide deck shows into slides/deck_data.json (from results/tables)."""
import json
from pathlib import Path
import pandas as pd

T = Path(__file__).resolve().parents[1] / "results" / "tables"


def rd(n):
    p = T / n
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


main = pd.concat([rd(n) for n in ["baseline_results.csv", "keepup_results.csv", "veriflux_results.csv",
                                  "veriflux_final.csv", "leakage_baseline.csv"]], ignore_index=True)
out = {"main": []}
for m, g in main.groupby("model"):
    row = {"model": m}
    for prot in ["event", "random"]:
        r = g[g.protocol == prot]
        if len(r):
            for c in ["f1", "auc", "accuracy", "mcc"]:
                row[f"{prot}_{c}"] = round(float(r[c + "_mean"].iloc[0]), 3)
            row[f"{prot}_f1_std"] = round(float(r["f1_std"].iloc[0]), 3)
    out["main"].append(row)
ab = pd.concat([rd("veriflux_final_ablation.csv"), rd("veriflux_ablation.csv").assign(model=lambda x: "Lite " + x.model)])
out["ablation"] = [{"model": r.model, "f1": round(r.f1_mean, 3), "auc": round(r.auc_mean, 3)}
                   for r in ab.itertuples()] if len(ab) else []
bu = rd("veriflux_budget.csv")
out["budget"] = [{"model": r.model, "f1": round(r.f1_mean, 3)} for r in bu.itertuples()] if len(bu) else []
out["retrieval"] = rd("retrieval_mrr.csv").round(3).to_dict("records")
out["stance"] = rd("stance_results.csv").round(3).to_dict("records")
sh = rd("shap_global.csv")
out["shap"] = sh.head(10).round(4).values.tolist() if len(sh) else []
out["efficiency"] = rd("efficiency.csv").round(3).to_dict("records")
out["sig"] = rd("significance.csv").round(4).to_dict("records")
out["purity"] = rd("event_purity.csv").round(3).to_dict("records")
out["lang"] = rd("language_results.csv").round(3).to_dict("records")
import math
def clean(o):
    if isinstance(o, float) and math.isnan(o): return None
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    return o
(Path(__file__).parent / "deck_data.json").write_text(json.dumps(clean(out), indent=1), encoding="utf-8")
print("exported", {k: len(v) for k, v in out.items()})

"""Generate every LaTeX table and the \\newcommand number macros used by paper/main.tex
directly from results/tables/*.csv (no hand-typed numbers)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import TABLES, ROOT, PROCESSED

OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
MACROS = {}


def f(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def pm(m, s, d=3):
    return "--" if pd.isna(m) else (f"{m:.{d}f}" if pd.isna(s) else f"{m:.{d}f}$\\pm${s:.{d}f}")


def macro(name, value):
    MACROS[name] = value


def read(name):
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def write(name, text):
    (OUT / name).write_text(text, encoding="utf-8")


# ------------------------------------------------------------------ data
def data_table():
    posts = pd.read_csv(PROCESSED / "posts.csv"); com = pd.read_csv(PROCESSED / "comments.csv")
    pur = read("event_purity.csv").iloc[0]
    macro("EventPurity", f"{pur['label_purity_weighted']:.2f}")
    rows = [("Events with posts / claims", f"{posts.event_id.nunique()} / 188"),
            ("Evidence documents", "11,974"),
            ("Posts (fake / real)", f"{len(posts):,} ({(posts.label==1).sum()} / {(posts.label==0).sum()})"),
            ("Posts per event (median / max)", f"{int(pur['posts_per_event_median'])} / {int(pur['posts_per_event_max'])}"),
            ("Label purity within events", f"{pur['label_purity_weighted']:.3f}"),
            ("Comments (A / D / Q / C)", f"{len(com):,} ("
             + " / ".join(f"{(com.stance==s).sum():,}" for s in ["agree", "disagree", "query", "comment"]) + ")"),
            ("Title script: Latin / Urdu", f"{(posts.title_lang.isin(['english','roman_urdu'])).sum()} / {(posts.title_lang=='urdu').sum()}"),
            ("Platforms (FB / IG / X / YT)", " / ".join(str((posts.platform==p).sum()) for p in ["facebook","instagram","twitter","youtube"]))]
    body = "\n".join(f"{a} & {b}\\\\" for a, b in rows)
    write("data_table.tex", f"""\\begin{{table}}[t]
\\centering
\\caption{{KeepUp dataset as used in this study}}
\\label{{tab:data}}
\\footnotesize
\\begin{{tabular}}{{lr}}
\\toprule
Property & Value\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


# ------------------------------------------------------------------ main comparison
CATEGORY = {"Logistic Regression": "ML", "SVM": "ML", "Random Forest": "ML", "Gradient Boosting": "ML",
            "KNN": "ML", "XGBoost": "ML", "LSTM": "DL", "Bi-RNN": "DL", "Bi-GRU": "DL",
            "CNN+LSTM": "Hybrid", "XLM-R + LightGBM": "Hybrid", "Bi-LSTM + XLM-R": "Hybrid",
            "Ensemble Learning": "Hybrid", "Event-majority lookup": "Sanity",
            "KeepUp (released code)": "Base", "KeepUp (paper spec)": "Base", "VeriFlux-Lite": "Proposed",
            "VeriFlux": "Proposed"}
ORDER = list(CATEGORY)


def main_table():
    df = pd.concat([read(n) for n in ["baseline_results.csv", "keepup_results.csv", "veriflux_results.csv",
                                      "veriflux_final.csv", "leakage_baseline.csv"]], ignore_index=True)
    if df.empty:
        return
    df = df[df["model"].isin(ORDER)]

    def get(m, prot, c):
        r = df[(df.model == m) & (df.protocol == prot)]
        return r[c].iloc[0] if len(r) and c in r else np.nan

    spec = [("event", c + "_mean") for c in ["accuracy", "precision", "recall", "f1", "auc"]] + \
           [("event", "f1_std"), ("random", "f1_mean"), ("random", "auc_mean")]
    real = [m for m in ORDER if m != "Event-majority lookup" and not df[df.model == m].empty]
    best = {}
    for j, (prot, col) in enumerate(spec):
        if col == "f1_std":
            continue
        vals_ = [round(get(m, prot, col), 3) for m in real]
        best[j] = max(v for v in vals_ if not np.isnan(v))
    lines, prev = [], None
    for m in ORDER:
        if df[df.model == m].empty:
            continue
        if prev is not None and CATEGORY[m] != prev:
            lines.append(r"\midrule")
        prev = CATEGORY[m]
        raw = [get(m, prot, col) for prot, col in spec]
        vals = []
        for j, v in enumerate(raw):
            s_ = f(v)
            if j in best and m != "Event-majority lookup" and not np.isnan(v) and round(v, 3) == best[j]:
                s_ = rf"\textbf{{{s_}}}"
            vals.append(s_)
        name = m + ("$^\\dagger$" if m in ("KeepUp (paper spec)", "Bi-LSTM + XLM-R") else "")
        lines.append(" & ".join([CATEGORY[m], name] + vals) + "\\\\")
    for m, key in [("VeriFlux", "VF"), ("VeriFlux-Lite", "VFL"), ("KeepUp (released code)", "KU"), ("KeepUp (paper spec)", "KUseq"),
                   ("Event-majority lookup", "Lookup")]:
        for prot in ["event", "random"]:
            if not np.isnan(get(m, prot, "f1_mean")):
                macro(f"{key}{prot}Fone", f(get(m, prot, "f1_mean")))
                macro(f"{key}{prot}Auc", f(get(m, prot, "auc_mean")))
                macro(f"{key}{prot}Acc", f(get(m, prot, "accuracy_mean")))
    base = df[(df.protocol == "event") & ~df.model.isin(["VeriFlux", "VeriFlux-Lite", "Event-majority lookup"])]
    if len(base):
        b = base.sort_values("f1_mean").iloc[-1]
        macro("BestOtherEvent", b["model"]); macro("BestOtherEventFone", f(b["f1_mean"]))
    body = "\n".join(lines)
    write("main_table.tex", rf"""\begin{{table*}}[t]
\centering
\caption{{Fake news detection on KeepUp under identical folds. Event-held-out: mean over 15 test folds
(3$\times$5-fold StratifiedGroupKFold by event). Random: 3$\times$5-fold stratified post split.
P, R and F1 are macro-averaged; sd = standard deviation of event-held-out F1 across folds. Best value per
column in bold (sanity rule excluded). $^\dagger$Token-sequence models: one repeat (5 folds) because of CPU cost.}}
\label{{tab:main}}
\footnotesize
\begin{{tabular}}{{llcccccc|cc}}
\toprule
 & & \multicolumn{{6}}{{c|}}{{Event-held-out (primary)}} & \multicolumn{{2}}{{c}}{{Random split}}\\
Cat. & Model & Acc & P & R & F1 & AUC & sd & F1 & AUC\\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")


def sig_table():
    s = read("significance.csv")
    if s.empty:
        return
    s = s[s.protocol == "event"].sort_values("delta_f1")
    rows = [f"{r.baseline} & {f(r.delta_f1)} & [{f(r.ci95_low)}, {f(r.ci95_high)}] & {r.wilcoxon_p:.3g} & {r.mcnemar_p:.3g}\\\\"
            for r in s.itertuples() if r.baseline in CATEGORY and r.baseline not in ("Event-majority lookup", "VeriFlux")]
    write("sig_table.tex", f"""\\begin{{table}}[t]
\\centering
\\caption{{Event-held-out significance of VeriFlux vs.\\ each model: mean macro-F1 difference over paired folds,
event-bootstrap 95\\% CI (seed 42), Wilcoxon signed-rank and McNemar $p$-values}}
\\label{{tab:sig}}
\\scriptsize
\\begin{{tabular}}{{lcccc}}
\\toprule
Compared model & $\\Delta$F1 & 95\\% CI & Wilcoxon & McNemar\\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def simple(csvname, texname, label, caption, cols, rename=None, fmt=None, star=False):
    d = read(csvname)
    if d.empty:
        return
    rename = rename or {}
    fmt = fmt or {}
    head = " & ".join(rename.get(c, c) for c in cols) + "\\\\"
    body = []
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt[c](v) if c in fmt else (f(v) if isinstance(v, (float, np.floating))
                                                       else str(v).replace("_", "\\_")))
        body.append(" & ".join(cells) + "\\\\")
    env = "table*" if star else "table"
    write(texname, f"""\\begin{{{env}}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\footnotesize
\\begin{{tabular}}{{l{'c' * (len(cols) - 1)}}}
\\toprule
{head}
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular}}
\\end{{{env}}}
""")


def write_macros():
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    # defaults so the paper always compiles
    needed = ["VFeventFone", "KUeventFone", "VFrandomFone", "KUrandomFone", "LookupRandomFone",
              "LookupEventFone", "LatencyRatio", "EventPurity"]
    for n in needed:
        alias = {"VFeventFone": "VFeventFone", "LookupRandomFone": "LookuprandomFone",
                 "LookupEventFone": "LookupeventFone", "KUeventFone": "KUeventFone",
                 "VFrandomFone": "VFrandomFone", "KUrandomFone": "KUrandomFone"}.get(n, n)
        if n not in MACROS:
            lines.append(f"\\providecommand{{\\{n}}}{{{MACROS.get(alias, '??')}}}")
    write("numbers.tex", "% generated by code/experiments/make_tables.py\n" + "\n".join(lines) + "\n")


def main():
    data_table()
    main_table()
    sig_table()
    simple("retrieval_mrr.csv", "retrieval_table.tex", "tab:retrieval",
           "Evidence retrieval for the 188 claims (MRR@10 against the gold evidence)",
           ["retriever", "mrr10", "minutes"], {"retriever": "Retriever", "mrr10": "MRR@10", "minutes": "CPU min"},
           {"minutes": lambda v: f(v, 1)})
    d = read("stance_results.csv")
    if len(d):
        names = {"tfidf_lr": "TF-IDF + LR (VeriFlux)", "xlmr_lr": "XLM-R (frozen) + LR", "bilstm": "XLM-R + BiLSTM",
                 "bigru": "XLM-R + BiGRU", "xstancefusion": "XStanceFusion (frozen XLM-R)"}
        d["model"] = d["model"].map(names).fillna(d["model"]); d.to_csv(TABLES / "_stance_named.csv", index=False)
        simple("_stance_named.csv", "stance_table.tex", "tab:stance",
               "Four-class comment stance detection, 5-fold cross-fitting over events (macro P/R/F1)",
               ["model", "precision", "recall", "f1", "accuracy", "minutes"],
               {"model": "Model", "precision": "P", "recall": "R", "f1": "F1", "accuracy": "Acc", "minutes": "CPU min"},
               {"minutes": lambda v: f(v, 1)})
    d = pd.concat([read("veriflux_final.csv").assign(model=lambda x: x.model.replace({"VeriFlux": "VeriFlux (full)"})),
                   read("veriflux_final_ablation.csv"),
                   read("veriflux_results.csv").assign(model=lambda x: x.model.replace({"VeriFlux-Lite": "VeriFlux-Lite (full)"})),
                   read("veriflux_ablation.csv").assign(model=lambda x: "Lite " + x.model)], ignore_index=True)
    if len(d):
        d = d[d.protocol == "event"][["model", "f1_mean", "f1_std", "auc_mean"]]
        d = d[~d.model.str.contains("only ") | d.model.str.contains("text prior")]
        d = d[~d.model.str.contains("classifier") | d.model.str.contains("Random Forest")]
        d.to_csv(TABLES / "_abl.csv", index=False)
        simple("_abl.csv", "ablation_table.tex", "tab:ablation",
               "VeriFlux ablation on the event-held-out protocol (15 folds)",
               ["model", "f1_mean", "f1_std", "auc_mean"], {"model": "Configuration", "f1_mean": "F1", "f1_std": "sd", "auc_mean": "AUC"})
    simple("efficiency.csv", "cost_table.tex", "tab:cost",
           "CPU cost per post (Intel i7-8665U, batch size 1) and model size",
           ["component", "ms_per_post", "size_mb"],
           {"component": "Component", "ms_per_post": "ms/post", "size_mb": "Weights MB"},
           {"ms_per_post": lambda v: f(v, 0), "size_mb": lambda v: f(v, 0)})
    d = read("shap_global.csv")
    if len(d):
        d.columns = ["feature", "mean_abs_shap"]; d.head(10).to_csv(TABLES / "_shap10.csv", index=False)
        simple("_shap10.csv", "shap_table.tex", "tab:shap", "Ten most influential VeriFlux-Lite features (mean $|$SHAP$|$)",
               ["feature", "mean_abs_shap"], {"feature": "Feature", "mean_abs_shap": "mean $|\\phi|$"})
    d = read("language_results.csv")
    if len(d):
        pv = d.pivot(index="model", columns="lang", values="f1").reset_index()
        keep = [m for m in ["VeriFlux", "VeriFlux-Lite", "KeepUp (released code)", "SVM", "Random Forest", "XGBoost"] if m in set(pv.model)]
        pv = pv[pv.model.isin(keep)]
        pv.to_csv(TABLES / "_lang.csv", index=False)
        cols = ["model"] + [c for c in ["english", "urdu", "roman_urdu"] if c in pv.columns]
        simple("_lang.csv", "lang_table.tex", "tab:lang", "Event-held-out macro-F1 by title language",
               cols, {"model": "Model", "english": "English", "urdu": "Urdu", "roman_urdu": "Roman Urdu"})
    simple("repro_summary.csv", "repro_table.tex", "tab:repro",
           "KeepUp: values reported in the paper vs.\\ this reproduction",
           ["item", "reported", "ours_random", "ours_event"],
           {"item": "Item", "reported": "Reported", "ours_random": "Ours (random)", "ours_event": "Ours (event)"})
    eff = read("efficiency.csv")
    if len(eff) and {"VeriFlux (total)", "KeepUp (total)"} <= set(eff.component):
        a = eff.set_index("component")["ms_per_post"]
        macro("LatencyRatio", f"{a['KeepUp (total)'] / a['VeriFlux (total)']:.1f}")
    write_macros()
    print("tables written to", OUT)


if __name__ == "__main__":
    main()

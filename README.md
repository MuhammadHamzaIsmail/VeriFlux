# VeriFlux — Lightweight, Explainable, Socially-Informed Fake News Detection on KeepUp

Source code for the final research paper (repository: https://github.com/MuhammadHamzaIsmail/VeriFlux).

## The idea in one paragraph

The base paper, **KeepUp**, reports 0.78 F1 on a *random* split of posts. Posts about the same news story (event)
end up in both training and testing, so a model can score well by **memorising stories**: a rule that never reads
the post and only returns the event's majority label already reaches 0.768 F1. **VeriFlux** fixes this with
(1) **fair testing** — test stories are never seen in training — and (2) the **event-context contrast** — each post
is compared with the average of the other posts about the same story, so the classifier (an RBF-SVM) cannot
memorise the story. On unseen stories VeriFlux is the best of 18 models (0.628 F1), significantly above KeepUp
(0.543), and 10.6x faster.

## Project map

| Step | Folder | What it is | Key file(s) |
|---|---|---|---|
| 0 | `code/common/` | Shared code: paths, data loading, feature builders, **fair (event-grouped) split** and metrics | `evaluation.py` (`make_folds`, `run_cv`) |
| 1 | `code/pipeline/` | Builds the inputs once (data cleaning, evidence retrieval, entailment, stance, XLM-R encodings); outputs are cached in `results/cache/` | `01_` … `06_` scripts |
| 2 | `code/base_keepup/` | **A. Base paper**: re-implementation of KeepUp (CNN + BiGRU + numeric fusion), two variants | `keepup_model.py`, `run_keepup.py` |
| 3 | `code/baselines/` | **B. Baselines**: LR, SVM, RF, Gradient Boosting, KNN, XGBoost, LSTM, Bi-RNN, Bi-GRU, CNN+LSTM, XLM-R+LightGBM, Bi-LSTM+XLM-R, ensemble | `run_baselines.py` |
| 4 | `code/proposed/` | **C. Proposed: VeriFlux** (event contrast + RBF-SVM) and the explainable **VeriFlux-Lite** (LightGBM + TreeSHAP) | `veriflux.py` (`event_contrast`, `contrast_representation`, `veriflux_svm_fn`), `run_veriflux.py` |
| 5 | `code/experiments/` | Proof of the problem (lookup rule), significance tests, SHAP, cost, language analysis, paper tables and figures | `leakage_baseline.py`, `stats_and_tables.py`, `make_tables.py` |
| — | `results/` | All outputs: `tables/` (CSV), `figures/`, `predictions/` (every model, every fold), `logs/` | |
| — | `paper/` | IEEE LaTeX paper (tables and numbers generated from `results/`) | `main.tex` |
| — | `slides/` | 15-minute presentation | `VeriFlux_presentation.pptx` |

## Quick check (about 1 minute, uses the saved cache)

```
python code/experiments/leakage_baseline.py          # the problem: lookup rule 0.768 (random) vs 0.344 (unseen events)
python code/proposed/run_veriflux.py --exp final     # the solution: VeriFlux 0.628 F1 / 0.687 AUC on unseen events
python code/experiments/make_tables.py               # regenerates every paper table from results/tables/*.csv
```
Finished folds are re-used. To force a model to retrain, move its files out of `results/predictions/` first
(e.g. `VeriFlux__*.csv`).

## 1. Environment

* Python 3.14 (tested), CPU only. Hardware used: Intel Core i7-8665U (4 cores / 8 threads), 16 GB RAM, Windows 10.
* Install: `python -m venv .venv` then `.venv\Scripts\pip install -r requirements.txt`
  (PyTorch CPU wheel: `pip install torch --index-url https://download.pytorch.org/whl/cpu`).

## 2. Data

* KeepUp dataset (public): https://github.com/sehrishsafdar/KeepUp-A-unified-framework-and-Dataset
  Download the repository zip and extract it to `data/keepup_repo/`.
* Pretrained models: `bash code/pipeline/00_download_models.sh` (≈5.2 GB into `data/models/`):
  XLM-RoBERTa-base, BGE-M3, NorGLM/Entailment, multilingual MiniLMv2-L6-XNLI,
  paraphrase-multilingual-MiniLM-L12-v2, mMiniLMv2 cross-encoder (mMARCO).
* The dataset, models and `results/cache/` are not in this repository (size / licence); the pipeline re-creates them.

## 3. Full run order (from the project root; several hours on CPU)

```
# Step 1 - build inputs (run once)
python code/pipeline/01_prepare_data.py          # tidy tables, extract evidence
python code/pipeline/02_retrieval.py             # BM25, BGE-M3, BGE-M3+rerank  -> MRR@10
python code/pipeline/03_entailment.py            # NorGLM (KeepUp) + MiniLM-XNLI (VeriFlux) features
python code/pipeline/04_encode_comments.py       # frozen XLM-R comment states  (~1.5 h CPU)
python code/pipeline/05_stance.py                # 5 stance models, event cross-fitted
python code/pipeline/06_encode_posts.py          # XLM-R post representations   (~0.5 h CPU)
# Step 2 - base paper
python code/base_keepup/run_keepup.py
# Step 3 - baselines
python code/baselines/run_baselines.py
# Step 4 - proposed
python code/proposed/run_veriflux.py --exp final                   # VeriFlux + its ablation
python code/proposed/run_veriflux.py --exp main,ablation,budget    # VeriFlux-Lite + its ablation + comment budget (~2.5 h)
# Step 5 - experiments, tables and figures
python code/experiments/leakage_baseline.py      # event-majority lookup (proof of the memorisation problem)
python code/experiments/stats_and_tables.py      # Wilcoxon, McNemar, bootstrap CI
python code/experiments/explain_efficiency_language.py --parts explain,language
python code/experiments/efficiency_e2e.py        # end-to-end CPU latency (run on an idle machine)
python code/experiments/make_figures.py
python code/experiments/make_framework_figure.py # paper Fig. 1 (framework diagram)
python code/experiments/make_tables.py           # writes paper/tables/*.tex and paper/tables/numbers.tex
```
`code/pipeline/run_remaining.sh` / `run_remaining2.sh` chain these steps.
`code/pipeline/07_encode_minilm.py` is an exploratory step from the design iteration (not used by the final model).

Slides: `cd slides && python export_data.py && NODE_PATH=node_modules node build_deck.js`
(needs `npm install pptxgenjs@3.12.0` in `slides/` and the PPTX skill's `apply_theme.js` via `PPTX_SKILL`).

## 4. Evaluation protocol (identical for every model)

* **Event-held-out** (primary): `StratifiedGroupKFold(5)` grouped by `event_id`, repeated with seeds 42/43/44 → 15 test folds. No event (claim, evidence, sibling posts) appears in both train and test.
* **Random post split** (KeepUp-style, for comparison): `StratifiedKFold(5)`, same seeds.
* Model selection / early stopping uses an inner validation split of the *training* fold only; the test fold is never used for selection.
* Stance features come from classifiers cross-fitted over events: no post's stance features are produced by a model trained on that post's event.
* Metrics: macro precision / recall / F1, accuracy, ROC-AUC, MCC; Wilcoxon signed-rank over folds, McNemar, event-bootstrap 95% CI.
* Seeds fixed in `code/common/paths.py`.

## 5. Main results (macro-F1 / ROC-AUC)

| Model | Event-held-out (15 folds) | Random split (15 folds) |
|---|---|---|
| **VeriFlux** | **0.628 / 0.687** | 0.699 / 0.775 |
| **VeriFlux-Lite** | 0.593 / 0.649 | **0.791 / 0.878** |
| Best baselines (SVM / Random Forest) | 0.620 / 0.663 · 0.620 / 0.656 | 0.704 / 0.778 · 0.703 / 0.768 |
| KeepUp, released-code variant | 0.543 / 0.633 | 0.686 / 0.755 |
| KeepUp, paper-spec variant (5 folds) | 0.565 / 0.653 | 0.712 / 0.781 |
| Event-majority lookup (sanity) | 0.344 / 0.500 | 0.768 / 0.837 |

* VeriFlux vs KeepUp on unseen events: +0.085 F1, Wilcoxon p = 0.003. VeriFlux vs SVM / RF: +0.008, **not significant** (p = 0.80 / 0.60).
* End-to-end CPU cost per post: KeepUp 6,028 ms, VeriFlux 569 ms (10.6x), VeriFlux-Lite 204 ms (29.5x).
* Full tables: `results/tables/`; per-fold predictions: `results/predictions/`.

## 6. Documented deviations from the base paper

* CPU only: XLM-R is used as a **frozen** encoder (XStanceFusion head trained on frozen token states) instead of fine-tuning.
* The released code selects the best epoch on the test set; here early stopping uses validation events.
* Comments carry no timestamps, so time-window early detection is replaced by a **first-k-comments** budget.
* NorGLM labels are `label_0/1/2`; `positive_prob = P(label_0) + P(label_1)` exactly as in the authors' notebook.
* The paper does not name its re-ranker; ours (mMARCO MiniLM cross-encoder) lowers MRR, and both numbers are reported.
* The paper-spec KeepUp variant and Bi-LSTM + XLM-R use one repeat (5 folds) because of CPU cost.
* VeriFlux was refined after v1 (now VeriFlux-Lite) failed to beat the kernel baselines; the redesign was developed on the
  seed-42 folds, which are part of the 15 reported folds (disclosed in the paper).

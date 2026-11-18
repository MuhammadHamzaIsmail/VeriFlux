#!/usr/bin/env bash
# Runs every remaining experiment. Safe to re-run: finished steps are resumed/skipped.
cd "$(dirname "$0")/.."
PY="./.venv/Scripts/python.exe -X utf8 -W ignore"
L=results/logs
mkdir -p $L

chain_a () {
  $PY code/02_retrieval.py --methods bm25,bge_m3,bge_m3_rerank      > $L/A1_retrieval.log 2>&1
  $PY code/03_entailment.py --retrieval bge_m3_rerank --parts norglm > $L/A2_norglm.log 2>&1
  $PY code/base_keepup/run_keepup.py                                 > $L/A3_keepup.log 2>&1
  $PY code/baselines/run_baselines.py                                > $L/A4_baselines.log 2>&1
  echo done > $L/A.done
}
chain_b () {
  $PY code/proposed/run_veriflux.py --exp ablation,budget            > $L/B1_ablation.log 2>&1
  $PY code/05_stance.py --models bilstm,bigru --epochs 3             > $L/B2_stance.log 2>&1
  echo done > $L/B.done
}
rm -f $L/A.done $L/B.done $L/ALL.done
chain_a & chain_b & wait
$PY code/experiments/leakage_baseline.py                      > $L/C1_leakage.log 2>&1
$PY code/experiments/stats_and_tables.py                      > $L/C2_stats.log 2>&1
$PY code/experiments/explain_efficiency_language.py --parts explain,language > $L/C3_explain.log 2>&1
$PY code/experiments/efficiency_e2e.py                        > $L/C4_efficiency.log 2>&1
$PY code/experiments/make_figures.py                          > $L/C5_figures.log 2>&1
$PY code/experiments/make_tables.py                           > $L/C6_tables.log 2>&1
echo done > $L/ALL.done

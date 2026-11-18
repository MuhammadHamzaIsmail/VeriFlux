#!/usr/bin/env bash
cd "$(dirname "$0")/.."
PY="./.venv/Scripts/python.exe -X utf8 -W ignore"
L=results/logs
rm -f $L/ALL.done
$PY code/base_keepup/run_keepup.py                            > $L/A3_keepup.log 2>&1
$PY code/baselines/run_baselines.py                           > $L/A4_baselines.log 2>&1
$PY code/experiments/leakage_baseline.py                      > $L/C1_leakage.log 2>&1
$PY code/experiments/stats_and_tables.py                      > $L/C2_stats.log 2>&1
$PY code/experiments/explain_efficiency_language.py --parts explain,language > $L/C3_explain.log 2>&1
$PY code/experiments/efficiency_e2e.py                        > $L/C4_efficiency.log 2>&1
$PY code/experiments/make_figures.py                          > $L/C5_figures.log 2>&1
$PY code/experiments/make_tables.py                           > $L/C6_tables.log 2>&1
echo done > $L/ALL.done

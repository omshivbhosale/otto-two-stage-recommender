#!/usr/bin/env bash
# Full pipeline: raw OTTO file -> final test-week score. Each step can also be run on its own.
# Needs data/raw/otto-recsys-train.jsonl, Java 21 (Spark) and about 30 GB of free disk.
set -euo pipefail
PY=${PY:-.venv/bin/python}

$PY -m scripts.prepare_data                      # events, time split, labels, ranker week
$PY -m scripts.baselines                         # popular / history / bought-together scores

for week in rank test; do
  $PY -m scripts.train_gru --week "$week"        # GRU session model, only data before the week
  $PY -m scripts.gru_candidates --week "$week"   # GRU candidates and session/item vectors
  for version in v1 v2; do
    $PY -m scripts.build_features --week "$week" --version "$version"
  done
done

$PY -m scripts.train_rankers --version v1
$PY -m scripts.train_rankers --version v2

# choose between systems on held-out ranker-week sessions (two samples) ...
$PY -m scripts.evaluate_ranking --versions v1 v2 --seed 42
$PY -m scripts.evaluate_ranking --versions v1 v2 --seed 7
# ... then score the test week once
$PY -m scripts.evaluate_ranking --versions v1 v2 --final

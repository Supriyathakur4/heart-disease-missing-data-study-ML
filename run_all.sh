#!/usr/bin/env bash
# Reproduce everything: leakage tests -> data audit -> experiments -> tables & figures -> error analysis.
# About 20 minutes on one CPU core (the tuning inside every training fold is the slow part).
# All randomness is seeded (MASTER_SEED = 42 in experiment/common.py).
# The experiment step is resumable: finished folds are cached, so if it is interrupted just run
# `python run_experiments.py` again from the experiment/ folder.
set -e
cd "$(dirname "$0")/experiment"
python tests/test_leakage.py
python data_audit.py
python run_experiments.py --fresh
python analyze.py
python error_analysis.py
echo "Done. See ../results/ (tables) and ../figures/ (plots)."

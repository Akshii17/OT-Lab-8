#!/bin/bash
# Runs every track (about 3-4 minutes) and rebuilds report.pdf
set -e
mkdir -p results
for f in track_a_continuous track_b_discrete track_c_quantum_opt track_d_parallel track_e_routing track_f_compiler track_g_sched_sim track_h_dbms track_i_quantum_algos track_j_testsuite; do
  echo ">>> $f"; python3 $f.py > /dev/null
done
python3 build_report.py

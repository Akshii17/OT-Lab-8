# Assignment 8 - Optimize Anything (Optimization Techniques)

Everything is plain Python (numpy, scipy, scikit-learn, matplotlib). No quantum or other special libraries: the quantum simulator is our own (`qsim.py`).

## Run
```
pip install -r requirements.txt
./run_all.sh          # runs all tracks, writes results/*.txt and report.pdf
```
Single track: `python3 track_a_continuous.py` (and so on). All random seeds are fixed.

## Files
| Track | File | What it covers |
|---|---|---|
| A | track_a_continuous.py | GD, Momentum, NAG, AdaGrad, RMSProp, Adam, Newton, Gauss-Newton, SGD/mini-batch, backprop + gradient check, HVP, Fisher, convex vs non-convex (saddle, local minima), noisy objective |
| B | track_b_discrete.py (+ common.py) | greedy, DP, ILP, Branch-and-Bound, QUBO/Ising (Max-Cut, feature selection with penalty), simulated annealing baseline |
| C | track_c_quantum_opt.py (+ qsim.py) | QAOA, VQE, parameter-shift, SPSA, quantum natural gradient, finite shots, noise model, barren plateau |
| D | track_d_parallel.py | submodular greedy, speedup/efficiency, load balancing, pipeline, Amdahl, FedAvg, MGDA, GradNorm |
| E | track_e_routing.py | main case study: delivery routing (TSP), strategy table in report |
| F | track_f_compiler.py | mini-compiler, 4 passes, pass-ordering optimization |
| G | track_g_xv6_priority_scheduler.patch, track_g_sched_sim.py | xv6 priority scheduler patch + `schedbench` user program; Python simulation of the 3 workloads |
| H | track_h_dbms.py | SQLite index selection: greedy vs DP vs ILP vs default optimizer |
| I | track_i_quantum_algos.py | Deutsch-Jozsa, Bernstein-Vazirani, Grover, QFT, QPE, Shor, QML vs classical |
| J | track_j_testsuite.py | test-suite minimisation (set cover): greedy vs ILP |

## AI tools register
| Tool / model | Tracks | Date | Kind of work |
|---|---|---|---|
| Claude (Anthropic, Claude Sonnet 5.5, claude.ai chat) | A-J | 2026-11-07 | planning, code generation, running scripts, report writing |

Add every other AI tool you use (debugging, explanation, documentation) to this table.

## Chapter 6 checklist (what is covered where)
See the "Chapter 6 coverage" table in report.pdf. Items NOT fully covered, to be honest: Fisher/low-rank curvature is shown only as a diagonal Fisher; model parallelism is shown only as a 2-stage pipeline; quantum annealing is represented by simulated annealing; no real quantum hardware was used; xv6 patch was not built or booted here.

## Things you must do before submitting
1. Read every file and be able to explain and change it live (viva).
2. Build the xv6 patch (`git apply track_g_xv6_priority_scheduler.patch` in xv6-riscv), boot in QEMU, run `schedbench 1/2/3` before and after, run `usertests`, and add the numbers.
3. Complete TOOL_COMPARISON.md with a real second tool, add your own chat exports under prompts/, and extend FAILURE_LOG.md with your own findings.
4. Push to a public GitHub repo with [AI:claude] commit prefixes.
5. Track J uses synthetic data; the assignment asks for a 1000+ line codebase or an open-source project. Swap in real coverage data if your instructor requires it.

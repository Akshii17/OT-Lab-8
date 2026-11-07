"""Builds report.pdf from results/*.txt (run after the track scripts)."""
import textwrap
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors

ss = getSampleStyleSheet(); B = ss["BodyText"]; H1 = ss["Heading1"]; H2 = ss["Heading2"]
mono = ParagraphStyle("mono", fontName="Courier", fontSize=6.6, leading=7.9)
def P(t): return Paragraph(t, B)
def res(name):
    txt = open(f"results/{name}.txt").read()
    return Preformatted("\n".join(w for l in txt.split("\n") for w in (textwrap.wrap(l, 118, subsequent_indent="    ") or [""])), mono)
def table(rows, widths=None):
    t = Table([[Paragraph(str(c), ParagraphStyle("c", fontSize=7.5, leading=9)) for c in r] for r in rows], colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP")])); return t

S = []
S += [Paragraph("Assignment 8 - Optimize Anything", ss["Title"]), P("Optimization Techniques - Technical report. All numbers below are copied from the output of the scripts in this repository (folder <i>results/</i>); settings and seeds are fixed in the code. Code was written with AI assistance (Claude); see the AI tools register in the README."), Spacer(1, 8)]

S += [Paragraph("1. Main case study (Track E): delivery routing", H1),
 P("<b>Process:</b> one vehicle must visit 60 delivery stops in a 100 x 100 km area and return to the depot. <b>Baseline:</b> visiting stops in the order they arrive (random order, mean of 20). <b>Formulation:</b> decision variable = order of the stops (a permutation); objective = minimise total tour length in km; constraint = every stop visited exactly once. <b>Assumptions:</b> straight-line distances, one vehicle, no time windows, synthetic data (seed 0)."),
 table([["Variable type", "Problem size", "Exact or heuristic", "Hardware", "Parallelism", "Chosen methods"],
        ["Discrete (permutation)", "60 stops (60! tours)", "Exact only feasible for about 12-15 stops (Held-Karp DP); heuristics for 60", "1 CPU core", "Not needed (runs in ms to seconds)", "Nearest neighbour, 2-opt, simulated annealing; Held-Karp for checking"]],
       [3*cm, 2.5*cm, 4*cm, 1.8*cm, 2.6*cm, 3.2*cm]), Spacer(1, 6), res("track_e"),
 P("<b>Result:</b> nearest neighbour is about 77% shorter than the baseline, 2-opt about 80%, and simulated annealing is similar to 2-opt but 100x slower and varies between seeds. On the 12-stop check, 2-opt is within 1% of the exact optimum. Limitations: no time windows or capacities; for larger real problems use ILP/OR-tools or parallel heuristics."), PageBreak()]

S += [Paragraph("2. Track A - Continuous optimization", H1), res("track_a"), Image("results/track_a_convergence.png", 9*cm, 5.8*cm),
 P("<b>Reading:</b> on the convex bowl every method converges and Newton needs one step. On the non-convex function the result depends on the start: Newton can land on the saddle or a local minimum, and the first-order methods end in the minimum on their side. A decaying step size removes most of the noise in the noisy-gradient test. Backprop was checked against finite differences."), PageBreak()]
S += [Paragraph("3. Track B - Discrete optimization", H1), res("track_b"),
 P("<b>Reading:</b> greedy is fast but 4% worse than optimal here; DP, ILP and Branch-and-Bound all give the optimum, and B&amp;B explores very few nodes. The QUBO and Ising energies match the direct Max-Cut value. For the constraint 'choose exactly k features' the penalty term works (k = 4 selected). Penalty weight must be large enough or constraints are violated."), Spacer(1, 6)]
S += [Paragraph("4. Track C - Quantum optimization (simulator, not hardware)", H1), res("track_c"),
 P("<b>Reading:</b> QAOA improves with depth p (0.83 to 0.97 approximation ratio). Few shots give noisy estimates (the error shrinks as 1/sqrt(shots)); a simple noise model lowers the result quickly with gate error; and gradient variance drops about 10x for every 2 extra qubits (barren plateau). All results are simulated with a noise-free state-vector simulator plus the simple noise formula shown; no real hardware was used."), PageBreak()]
S += [Paragraph("5. Track D - Submodular, parallel and distributed", H1), res("track_d"), Image("results/track_d_speedup.png", 8*cm, 5.3*cm),
 P("<b>Reading:</b> the machine used has <b>1 CPU core</b>, so CPU-bound tasks show no speedup (an honest negative result: more processes only add overhead). Tasks that wait (sleep) overlap well, as does the 2-stage pipeline, and dynamic scheduling fixes the imbalance of uneven tasks. Greedy coverage is submodular (0 violations found) and reached the optimum. FedAvg approaches centralized training as local steps and rounds increase, at the cost of communication. In the two-task test, equal weights and GradNorm gave balanced results while MGDA drove task 1 down but left task 2 (100x larger scale) high after 300 steps; this was not tuned, so no general claim is made."), PageBreak()]
S += [Paragraph("6. Track F - Compiler optimization", H1), res("track_f"),
 P("<b>Reading:</b> four passes (constant folding/propagation, strength reduction, common sub-expression elimination, dead-code elimination) reduced the estimated cycles from 41 to 13. Pass order matters: the best of 24 orders gives 13 cycles, the worst 28, and the greedy order gives 15. Output was identical on every tested input."), Spacer(1, 6)]
S += [Paragraph("7. Track G - xv6 operating system", H1),
 P("<b>Change:</b> priority scheduler (lower number = higher priority, ties by least recently run), new system call <i>setpriority(pid, prio)</i> and a benchmark program <i>schedbench</i> with 3 workloads. <b>Objective:</b> turnaround time of important jobs. The patch is in <i>track_g_xv6_priority_scheduler.patch</i>. <b>Status: the patch was NOT compiled, booted under QEMU or run through usertests in this environment (no RISC-V toolchain).</b> The table below is a Python <b>simulation</b> of the same three workloads, not an xv6 measurement. Real before/after numbers must be added after building the patch."), res("track_g"), PageBreak()]
S += [Paragraph("8. Track H - DBMS (SQLite)", H1), res("track_h"),
 P("<b>Formulation:</b> choose a subset of 8 candidate indexes to maximise measured workload time saved, subject to total index size at most 1500 KB (a knapsack). Greedy, DP and ILP were compared with the default (no index) plan. Query results were identical (hash check) and EXPLAIN QUERY PLAN shows two queries switching from a table scan to an index search. Benefits are measured one index at a time and assumed additive. Several indexes made the workload slower (the default planner chose them badly), which is why measured benefit, not intuition, is used. Timing noise exists: the three methods chose the same index but reported 1.4x to 2.2x."), Spacer(1, 6)]
S += [Paragraph("9. Track I - Quantum algorithms and QML", H1), res("track_i"),
 P("<b>Reading:</b> all circuits (Deutsch-Jozsa, Bernstein-Vazirani, Grover, QFT, QPE, Shor for N = 15) were written by hand on our own simulator and gave the correct answers. Grover needs about sqrt(N) queries versus N/2, which is a theoretical, query-count advantage; the simulator itself gets no wall-clock speedup. The small quantum classifiers were not better than classical models (honest negative result). Compare with Track D, where parallel speedups can be measured directly on real hardware when cores exist."), PageBreak()]
S += [Paragraph("10. Track J - Software engineering: test-suite minimisation", H1), res("track_j"),
 P("<b>Reading:</b> greedy set cover keeps all line coverage with 58% of the tests and 44% of the run time; the ILP is slightly better (48% of tests, 40% of run time) but slower to solve. Trade-off: removing tests can lose bug-detection power. Data are synthetic."), Spacer(1, 8)]

S += [Paragraph("11. Chapter 6 coverage checklist", H1), table([
 ["Chapter 6 item", "Where", "Status"],
 ["ML as optimization, convex vs non-convex, saddle/local minima", "Track A (A1, A2)", "Done"],
 ["GD, SGD, mini-batch, Momentum, NAG, AdaGrad, RMSProp, Adam", "Track A (A1, A4)", "Done"],
 ["Newton, Gauss-Newton, HVP, Fisher, noisy objective, backprop", "Track A (A2-A6)", "Done (Fisher: diagonal only)"],
 ["Greedy, ILP, Branch-and-Bound, DP, QUBO/Ising, penalty terms", "Track B", "Done"],
 ["QAOA, VQE, parameter-shift, SPSA, QNG, shots, noise, barren plateau", "Track C", "Done on simulator"],
 ["Deutsch-Jozsa, BV, Grover, QFT, QPE, Shor, VQC, quantum kernel", "Track I", "Done on simulator"],
 ["Quantum annealing", "Track I (I6)", "Simulated annealing stand-in only"],
 ["Submodular greedy and its conditions", "Track D (D1)", "Done"],
 ["Load balancing, work stealing, speedup, efficiency, Amdahl/Gustafson", "Track D (D2)", "Dynamic scheduling, pipeline, Amdahl shown; Gustafson only mentioned; 1-core machine"],
 ["Data / model / pipeline parallelism", "Track D (D2)", "Data and pipeline; model parallel = 2-stage pipeline"],
 ["FedAvg, MGDA, GradNorm", "Track D (D3, D4)", "Done (GradNorm simplified)"],
 ["Strategy selection", "Section 1 table", "Done"]], [7*cm, 4*cm, 5.5*cm]), Spacer(1, 8)]

S += [Paragraph("12. Final questions (short answers)", H1)]
for q, a in [
 ("1. Process / evidence", "Delivery routing; a random-order tour is about 3200 km vs about 660 km after optimization."),
 ("2. Formulation", "Section 1 and the formulation line at the top of each track's section."),
 ("3. Chapter 6 methods", "See the checklist; each was applied to a problem that fits its assumptions."),
 ("4. Optimized vs baseline", "Routing -80% distance; compiler 41 to 13 cycles; DB workload about 2x faster; test suite 56-60% less run time; priority scheduling halves turnaround of important jobs in simulation."),
 ("5. Trade-offs / failures", "Greedy loses quality; penalties need tuning; some indexes hurt; test minimisation may lose fault detection; quantum results are simulated and not faster; the CPU-bound parallel test showed no speedup on 1 core; the xv6 patch is untested."),
 ("6. Correctness / reproducibility", "Fixed seeds, brute-force checks (knapsack, Max-Cut, VQE exact energy), gradient check, identical program output, identical query results, one command (run_all.sh)."),
 ("7. If scale changes", "Larger routing: OR-tools / ILP / parallel heuristics; larger quantum: real hardware or tensor-network simulators; larger DB: more candidates and interaction-aware selection."),
 ("8. Compiler, xv6, DBMS", "Passes + pass ordering (output checked); priority scheduler (concept: scheduling and context switching; verification still to do on QEMU); index selection (results hashed and plans compared)."),
 ("9. Quantum vs classical vs parallel", "Quantum: provable query advantage (Grover) but simulation overhead, noise, barren plateaus and no QML win here; parallel: real speedup only with real cores or waiting tasks."),
 ("10. AI tools", "Claude (see README register); errors found are listed in FAILURE_LOG.md. A second-tool comparison still has to be filled in by the student.")]:
    S.append(P(f"<b>{q}:</b> {a}"))
SimpleDocTemplate("report.pdf", pagesize=A4, leftMargin=1.6*cm, rightMargin=1.6*cm, topMargin=1.5*cm, bottomMargin=1.5*cm).build(S)
print("report.pdf written")

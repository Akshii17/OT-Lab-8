"""Track J - software engineering: test-suite minimisation = set cover. Greedy vs ILP vs naive (keep all)."""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
r = np.random.default_rng(0)
T, L = 150, 500                       # 150 tests, 500 lines of code
cov = (r.random((T, L)) < 0.02)       # each test covers ~2% of lines
cost = r.uniform(0.1, 5.0, T)         # seconds per test
cov[r.integers(T, size=L), np.arange(L)] = True   # every line covered by at least one test
log(f"== J1. Synthetic suite: {T} tests, {L} lines, total runtime {cost.sum():.0f} s ==")
log("Formulation: min sum(cost_t * x_t)  s.t. for every line l: sum(x_t for tests covering l) >= 1, x_t in {0,1}")

def greedy():
    covered = np.zeros(L, bool); chosen = []
    while not covered.all():
        gain = (cov & ~covered).sum(1)/cost; t = int(np.argmax(gain)); chosen.append(t); covered |= cov[t]
    return chosen
def ilp():
    res = milp(cost, constraints=LinearConstraint(cov.T.astype(float), 1, np.inf), integrality=np.ones(T), bounds=Bounds(0, 1),
               options={"time_limit": 30})
    return [t for t in range(T) if res.x[t] > 0.5]
def check(sel): return cov[sel].any(0).all()
log(f"{'method':22s}{'tests':>7s}{'runtime(s)':>12s}{'coverage kept':>15s}{'solve time':>12s}")
log(f"{'naive: run all':22s}{T:7d}{cost.sum():12.1f}{'yes':>15s}{'-':>12s}")
for name, fn in [("greedy", greedy), ("ILP (exact)", ilp)]:
    t0 = time.perf_counter(); sel = fn(); dt = time.perf_counter()-t0
    log(f"{name:22s}{len(sel):7d}{cost[sel].sum():12.1f}{str(check(sel)):>15s}{dt:11.3f}s")
log("Cost of minimisation: coverage of lines is preserved, but fault detection is NOT guaranteed (a dropped test may catch a bug that line coverage cannot see).")
log("Limitation: data is synthetic; for the 1000+ line codebase requirement, replace `cov` with real coverage data (e.g. from pytest --cov) of a project.")
open("results/track_j.txt", "w").write("\n".join(out))

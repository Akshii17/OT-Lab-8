"""Track E - main case study: delivery-route optimization (TSP). Compare baseline, NN, 2-opt, SA, exact DP."""
import time, itertools
import numpy as np

out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
rng = np.random.default_rng(0)

def dist_matrix(P): return np.linalg.norm(P[:, None] - P[None], axis=2)
def length(D, tour): return sum(D[tour[i], tour[(i+1) % len(tour)]] for i in range(len(tour)))
def nearest_neighbour(D):
    tour = [0]; left = set(range(1, len(D)))
    while left: nxt = min(left, key=lambda j: D[tour[-1], j]); tour.append(nxt); left.remove(nxt)
    return tour
def two_opt(D, tour):
    tour = tour[:]; improved = True
    while improved:
        improved = False
        for i in range(1, len(tour)-1):
            for j in range(i+1, len(tour)):
                a, b, c, d = tour[i-1], tour[i], tour[j], tour[(j+1) % len(tour)]
                if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-12:
                    tour[i:j+1] = tour[i:j+1][::-1]; improved = True
    return tour
def sim_anneal(D, seed, iters=20000):
    r = np.random.default_rng(seed); tour = list(r.permutation(len(D))); cur = length(D, tour); best = cur
    for t in range(iters):
        T = 1.0*(1 - t/iters) + 1e-4; i, j = sorted(r.choice(len(D), 2, replace=False))
        new = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]; c2 = length(D, new)
        if c2 < cur or r.random() < np.exp((cur-c2)/T): tour, cur = new, c2; best = min(best, cur)
    return best
def held_karp(D):
    n = len(D); C = {(1 << 0, 0): 0.0}
    for mask in range(1, 1 << n, 2):
        for last in range(n):
            if (mask, last) not in C: continue
            for nx in range(1, n):
                if mask & (1 << nx): continue
                key = (mask | (1 << nx), nx); v = C[(mask, last)] + D[last, nx]
                if v < C.get(key, 1e18): C[key] = v
    full = (1 << n) - 1
    return min(C[(full, l)] + D[l, 0] for l in range(1, n))

log("== E1. Small instance (12 stops): exact optimum vs heuristics ==")
P = rng.random((12, 2))*100; D = dist_matrix(P)
t0 = time.perf_counter(); opt = held_karp(D); log(f"Held-Karp (exact DP): {opt:.2f}  ({time.perf_counter()-t0:.2f}s)")
log(f"nearest neighbour: {length(D, nearest_neighbour(D)):.2f}   NN+2-opt: {length(D, two_opt(D, nearest_neighbour(D))):.2f}")

log("\n== E2. Main instance: 60 delivery stops in a 100x100 km area (distance in km) ==")
P = rng.random((60, 2))*100; D = dist_matrix(P)
base = np.mean([length(D, list(np.random.default_rng(s).permutation(60))) for s in range(20)])
log(f"baseline (visit in given/random order, mean of 20): {base:.1f} km")
t0 = time.perf_counter(); nn = nearest_neighbour(D); tnn = time.perf_counter()-t0
log(f"nearest neighbour: {length(D, nn):.1f} km  ({tnn*1e3:.1f} ms)  improvement over baseline {100*(1-length(D, nn)/base):.1f}%")
t0 = time.perf_counter(); t2 = two_opt(D, nn); tt = time.perf_counter()-t0
log(f"NN + 2-opt:        {length(D, t2):.1f} km  ({tt*1e3:.1f} ms)  improvement over baseline {100*(1-length(D, t2)/base):.1f}%")
t0 = time.perf_counter(); sa = [sim_anneal(D, s) for s in range(5)]; tsa = (time.perf_counter()-t0)/5
log(f"simulated annealing (5 seeds): mean {np.mean(sa):.1f} +- {np.std(sa):.1f} km  ({tsa:.1f} s per run)")
assert sorted(t2) == list(range(60)), "tour must visit every stop exactly once"
log("correctness check passed: every stop visited exactly once")
open("results/track_e.txt", "w").write("\n".join(out))

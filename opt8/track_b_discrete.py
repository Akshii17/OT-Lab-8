"""Track B - discrete optimization: greedy, DP, ILP, Branch-and-Bound, QUBO/Ising (Max-Cut + feature selection)."""
import itertools, time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from common import *

out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

rng = np.random.default_rng(1)
n = 18
value = rng.integers(10, 100, n); weight = rng.integers(5, 40, n); cap = int(weight.sum()*0.4)
log(f"== B1. 0/1 knapsack (feature/project selection), n={n}, capacity={cap} ==")

def greedy():
    order = np.argsort(-value/weight); w = 0; v = 0
    for i in order:
        if w + weight[i] <= cap: w += weight[i]; v += value[i]
    return v
def dp():
    best = [0]*(cap+1)
    for i in range(n):
        for c in range(cap, weight[i]-1, -1): best[c] = max(best[c], best[c-weight[i]] + value[i])
    return best[cap]
def ilp():
    r = milp(-value.astype(float), constraints=LinearConstraint(weight[None, :].astype(float), -np.inf, cap),
             integrality=np.ones(n), bounds=Bounds(0, 1))
    return int(round(-r.fun))
nodes = [0]
def bnb():
    order = np.argsort(-value/weight); v = value[order]; w = weight[order]; best = [0]
    def bound(i, cw, cv):
        b = cv; room = cap - cw
        for j in range(i, n):
            if w[j] <= room: room -= w[j]; b += v[j]
            else: b += v[j]*room/w[j]; break
        return b
    def rec(i, cw, cv):
        nodes[0] += 1
        best[0] = max(best[0], cv)
        if i == n or bound(i, cw, cv) <= best[0]: return
        if cw + w[i] <= cap: rec(i+1, cw+w[i], cv+v[i])
        rec(i+1, cw, cv)
    rec(0, 0, 0); return best[0]
def brute():
    b = 0
    for mask in range(1 << n):
        bits = [(mask >> i) & 1 for i in range(n)]
        if sum(weight[i] for i in range(n) if bits[i]) <= cap:
            b = max(b, sum(value[i] for i in range(n) if bits[i]))
    return b
for name, fn in [("Greedy (ratio)", greedy), ("Dynamic programming", dp), ("ILP (scipy milp)", ilp), ("Branch-and-Bound", bnb), ("Brute force (check)", brute)]:
    t0 = time.perf_counter(); v = fn(); log(f"{name:22s} value={int(v)}  time={(time.perf_counter()-t0)*1e3:.2f} ms")
log("B&B nodes explored:", nodes[0], "of", 2**(n+1)-1, "possible")

log("\n== B2. Max-Cut as QUBO / Ising ==")
Q = maxcut_qubo(); e, x = brute_force_qubo(Q)
best_cut = max(cut_value(b) for b in itertools.product([0, 1], repeat=N_NODES))
log(f"QUBO optimum energy={e}  -> cut={-e}; direct brute-force cut={best_cut}  (match: {-e == best_cut})")

def qubo_to_ising(Q):
    n = len(Q); h = np.zeros(n); J = np.zeros((n, n)); c = 0.0
    for i in range(n):
        h[i] += Q[i, i]/2; c += Q[i, i]/2
        for j in range(i+1, n):
            J[i, j] = Q[i, j]/4; h[i] += Q[i, j]/4; h[j] += Q[i, j]/4; c += Q[i, j]/4
    return h, J, c
h, J, c = qubo_to_ising(Q)
bits = np.array(x); s = 1 - 2*bits     # x=(1-s)/2 convention is symmetric for Max-Cut; check energy with x=(1+s)/2:
s = 2*bits - 1
E_ising = c + h@s + sum(J[i, j]*s[i]*s[j] for i in range(N_NODES) for j in range(i+1, N_NODES))
log(f"Ising energy of the optimum = {E_ising:.1f} (should equal QUBO energy {e})")

log("\n== B3. Feature selection as QUBO with a penalty for 'choose exactly k' ==")
nf, k, P = 10, 4, 5.0
rel = rng.random(nf); red = rng.random((nf, nf)); red = (red + red.T)/2
Qf = np.zeros((nf, nf))
for i in range(nf):
    Qf[i, i] = -rel[i] + P*(1 - 2*k)           # penalty P*(sum x - k)^2 expanded (x^2 = x)
    for j in range(i+1, nf): Qf[i, j] = red[i, j] + 2*P
ef, xf = brute_force_qubo(Qf)
log(f"selected features {[i for i in range(nf) if xf[i]]}, count={sum(xf)} (constraint k={k} satisfied: {sum(xf)==k})")

log("\n== B4. Classical baseline for Track C: simulated annealing on the Max-Cut QUBO ==")
def sa(Q, seed, steps=2000):
    r = np.random.default_rng(seed); x = r.integers(0, 2, len(Q)); e = qubo_energy(Q, x); best = e
    for t in range(steps):
        T = 2.0*(1 - t/steps) + 1e-3; i = r.integers(len(Q)); x[i] ^= 1; e2 = qubo_energy(Q, x)
        if e2 < e or r.random() < np.exp((e-e2)/T): e = e2; best = min(best, e)
        else: x[i] ^= 1
    return best
res = [-sa(Q, s) for s in range(10)]
log(f"SA best cut over 10 seeds: mean={np.mean(res):.2f} min={min(res)} max={max(res)} (optimum {best_cut})")
open("results/track_b.txt", "w").write("\n".join(out))

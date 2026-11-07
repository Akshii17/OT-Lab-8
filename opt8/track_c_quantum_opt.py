"""Track C - quantum optimization on our own simulator: QAOA (Max-Cut), VQE, parameter-shift,
SPSA, quantum natural gradient, finite shots, noise, barren plateaus."""
import numpy as np
from scipy.optimize import minimize
from qsim import *
from common import *

out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
rng = np.random.default_rng(0)

# ---- QAOA for Max-Cut ----------------------------------------------------------
n = N_NODES
cuts = np.array([cut_value([(i >> (n-1-q)) & 1 for q in range(n)]) for i in range(2**n)], float)
OPT = cuts.max()
def qaoa_state(params):
    p = len(params)//2; s = np.ones(2**n, complex)/np.sqrt(2**n)
    for l in range(p):
        s = s*np.exp(-1j*params[l]*(-cuts))          # cost layer exp(-i*gamma*E), E=-cut
        for q in range(n): s = apply1(s, RX(2*params[p+l]), q, n)
    return s
def qaoa_exp(params): return float(np.sum(probs(qaoa_state(params))*cuts))   # expected cut (maximise)
log("== C1. QAOA on the Track B Max-Cut graph (exact simulation, not hardware) ==")
log(f"optimal cut = {OPT:.0f}, random-guess expected cut = {cuts.mean():.2f}")
for p in (1, 2, 3):
    best = max((minimize(lambda x: -qaoa_exp(x), rng.uniform(0, np.pi, 2*p), method="COBYLA", options={"maxiter": 300}) for _ in range(5)), key=lambda r: -r.fun)
    log(f"p={p}: expected cut={-best.fun:.3f}  approximation ratio={-best.fun/OPT:.3f}")
bestx = best.x
log("\n== C2. Finite shots and a simple noise model on the p=3 QAOA ==")
exact = qaoa_exp(bestx); pr = probs(qaoa_state(bestx))
for shots in (10, 100, 1000, 10000):
    est = [cuts[sample(qaoa_state(bestx), shots, rng)].mean() for _ in range(30)]
    log(f"shots={shots:6d}  estimate={np.mean(est):.3f} +- {np.std(est):.3f}  (exact {exact:.3f})")
n_gates = 3*(len(EDGES) + n)
for perr in (0.0, 0.001, 0.005, 0.02):
    lam = 1 - (1 - perr)**n_gates                      # depolarising-style: mix with uniform noise
    log(f"gate error {perr:.3f} -> expected cut {(1-lam)*exact + lam*cuts.mean():.3f}")

# ---- VQE ---------------------------------------------------------------------------
I2 = np.eye(2); Xm = np.array([[0, 1], [1, 0]]); Zm = np.diag([1., -1])
Ham = np.kron(Zm, Zm) + 0.5*np.kron(Xm, I2) + 0.5*np.kron(I2, Xm)
E0 = np.linalg.eigvalsh(Ham)[0]
def ansatz(th):
    s = zero(2)
    s = apply1(s, RY(th[0]), 0, 2); s = apply1(s, RY(th[1]), 1, 2); s = cnot(s, 0, 1, 2)
    s = apply1(s, RY(th[2]), 0, 2); s = apply1(s, RY(th[3]), 1, 2); return s
def energy(th): s = ansatz(th); return float(np.real(np.conj(s)@Ham@s))
def ps_grad(th):                                       # parameter-shift rule (exact for RY gates)
    g = np.zeros_like(th)
    for i in range(len(th)):
        e = np.zeros_like(th); e[i] = np.pi/2; g[i] = (energy(th+e) - energy(th-e))/2
    return g
def spsa_grad(th, k, r):
    c = 0.1/(k+1)**0.1; d = r.choice([-1, 1], len(th)); return (energy(th+c*d) - energy(th-c*d))/(2*c)*d
def qng_metric(th, eps=1e-4):
    d = []
    for i in range(len(th)):
        e = np.zeros_like(th); e[i] = eps; d.append((ansatz(th+e) - ansatz(th-e))/(2*eps))
    psi = ansatz(th); g = np.zeros((len(th), len(th)))
    for i in range(len(th)):
        for j in range(len(th)):
            g[i, j] = np.real(np.conj(d[i])@d[j]) - np.real(np.conj(d[i])@psi*np.conj(psi)@d[j])
    return g
log(f"\n== C3. VQE (2 qubits). exact ground energy = {E0:.4f} ==")
for name in ("parameter-shift GD", "SPSA", "quantum natural gradient"):
    finals = []; calls = 0
    for seed in range(5):
        r = np.random.default_rng(seed); th = r.uniform(0, 2*np.pi, 4)
        for k in range(100):
            if name == "parameter-shift GD": th = th - 0.2*ps_grad(th); calls = 8
            elif name == "SPSA": th = th - 0.3/(k+1)**0.2*spsa_grad(th, k, r); calls = 2
            else: th = th - 0.2*np.linalg.pinv(qng_metric(th) + 1e-3*np.eye(4))@ps_grad(th); calls = 8
        finals.append(energy(th))
    log(f"{name:26s} final energy {np.mean(finals):.4f} +- {np.std(finals):.4f}  ({calls} circuit evals/step)")

log("\n== C4. Barren plateau: variance of a parameter-shift gradient, global cost, random RY+CZ circuit ==")
def bp_var(nq, depth, samples=40):
    gs = []
    def cost(th):
        s = zero(nq)
        for l in range(depth):
            for q in range(nq): s = apply1(s, RY(th[l, q]), q, nq)
            for q in range(nq-1): s = apply_ctrl(s, Z, q, q+1, nq)
        return 1 - probs(s)[0]
    for _ in range(samples):
        th = rng.uniform(0, 2*np.pi, (depth, nq)); a = th.copy(); b = th.copy()
        a[0, 0] += np.pi/2; b[0, 0] -= np.pi/2; gs.append((cost(a) - cost(b))/2)
    return np.var(gs)
for nq in (2, 4, 6, 8, 10):
    log(f"qubits={nq:2d}  Var[grad]={bp_var(nq, nq):.2e}")
open("results/track_c.txt", "w").write("\n".join(out))

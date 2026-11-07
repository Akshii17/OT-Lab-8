"""Track I - quantum algorithms (own circuits on qsim.py), QML vs classical, Grover vs linear search."""
import time
from fractions import Fraction
from math import gcd
import numpy as np
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from qsim import *

out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
rng = np.random.default_rng(0)

def oracle(s, f, n):   # |x>|y> -> |x>|y xor f(x)>, ancilla = last qubit
    return permute(s, lambda i: (i & ~1) | ((i & 1) ^ f(i >> 1)))
def hall(s, qs, n):
    for q in qs: s = apply1(s, H, q, n)
    return s

log("== I1. Deutsch-Jozsa (3 input qubits) ==")
for name, f in [("constant", lambda x: 1), ("balanced (x0 xor x1)", lambda x: ((x >> 2) ^ (x >> 1)) & 1)]:
    n = 4; s = zero(n); s = apply1(s, X, 3, n); s = hall(s, range(4), n); s = oracle(s, f, n); s = hall(s, range(3), n)
    p0 = sum(probs(s)[i] for i in (0, 1))
    log(f"{name:22s} P(input=000) = {p0:.3f} -> {'constant' if p0 > .5 else 'balanced'}")
log("\n== I2. Bernstein-Vazirani, secret 1011 ==")
sec = 0b1011; n = 5; s = zero(n); s = apply1(s, X, 4, n); s = hall(s, range(5), n)
s = oracle(s, lambda x: bin(x & sec).count("1") % 2, n); s = hall(s, range(4), n)
log("measured input register:", format(int(np.argmax(probs(s)))>>1, "04b"))

log("\n== I3. Grover search vs classical linear search (oracle queries) ==")
def grover(n, target):
    s = hall(zero(n), range(n), n); N = 2**n; k = int(np.floor(np.pi/4*np.sqrt(N)))
    for _ in range(k):
        s[target] *= -1; m = s.mean(); s = 2*m - s
    return probs(s)[target], k
log(f"{'qubits':>6s}{'N':>8s}{'Grover queries':>16s}{'P(success)':>12s}{'classical avg':>15s}{'sim time':>10s}")
for nq in (4, 6, 8, 10, 12, 14):
    t0 = time.perf_counter(); p, k = grover(nq, 5); dt = time.perf_counter()-t0
    log(f"{nq:6d}{2**nq:8d}{k:16d}{p:12.4f}{2**nq/2:15.0f}{dt:9.4f}s")
log("Query count shows the quadratic advantage (theoretical). Wall-clock on a classical simulator is NOT faster: the simulator pays 2^n work per query.")

log("\n== I4. QFT check and quantum phase estimation ==")
nq = 3; F = np.array([[np.exp(2j*np.pi*j*k/8) for k in range(8)] for j in range(8)])/np.sqrt(8)
v = rng.normal(size=8) + 1j*rng.normal(size=8); v /= np.linalg.norm(v)
log("QFT circuit matches matrix:", np.allclose(qft(v.copy(), [0, 1, 2], 3), F@v))
t = 5; phi = 5/32; n = t+1; s = apply1(zero(n), X, t, n); s = hall(s, range(t), n)
for k in range(t): s = apply_ctrl(s, PH(2*np.pi*phi*2**(t-1-k)), k, t, n)
s = qft(s, list(range(t)), n, inverse=True)
m = int(np.argmax(probs(s))) >> 1
log(f"QPE: true phase {phi}, estimated {m}/{2**t} = {m/2**t}")

log("\n== I5. Shor's algorithm, N=15, a=7 (order finding with QPE, then classical post-processing) ==")
N, a, t = 15, 7, 8; n = t+4
s = zero(n); s = apply1(s, X, n-1, n)            # target register = 1
s = hall(s, range(t), n)
for k in range(t):
    mult = pow(a, 2**(t-1-k), N)
    def f(i, k=k, mult=mult):
        if (i >> (n-1-k)) & 1:
            y = i & 15
            if y < N: return (i & ~15) | ((y*mult) % N)
        return i
    s = permute(s, f)
s = qft(s, list(range(t)), n, inverse=True)
pc = probs(s).reshape(2**t, 16).sum(1)
top = np.argsort(-pc)[:4]
for m in top:
    fr = Fraction(int(m), 2**t).limit_denominator(N); r = fr.denominator
    if pow(a, r, N) == 1 and r % 2 == 0:
        log(f"measurement {m}/256 -> order r={r} -> factors {gcd(a**(r//2)-1, N)} x {gcd(a**(r//2)+1, N)}"); break

log("\n== I6. Quantum annealing: not simulated physically. Simulated annealing (Track B4) is used as the classical stand-in.")
log("Limit: real annealers need the QUBO embedded on a sparse hardware graph; dense problems need chains of physical qubits.")

log("\n== I7. QML vs classical on moons data (2 qubits, simulator) ==")
X_, y_ = make_moons(120, noise=0.15, random_state=1); X_ = (X_ - X_.min(0))/(X_.max(0) - X_.min(0))*np.pi
Xtr, ytr, Xte, yte = X_[:80], 2*y_[:80]-1, X_[80:], 2*y_[80:]-1
def circ(x, th):
    s = zero(2)
    for l in range(2):
        s = apply1(s, RY(x[0]), 0, 2); s = apply1(s, RY(x[1]), 1, 2)
        s = apply1(s, RY(th[4*l]), 0, 2); s = apply1(s, RY(th[4*l+1]), 1, 2); s = cnot(s, 0, 1, 2)
        s = apply1(s, RZ(th[4*l+2]), 0, 2); s = apply1(s, RY(th[4*l+3]), 1, 2)
    return s
def out_z(x, th): p = probs(circ(x, th)); return p[0] + p[1] - p[2] - p[3]
evals = [0]
def lossq(th, X, y): evals[0] += len(X); return np.mean([(out_z(x, th) - yy)**2 for x, yy in zip(X, y)])
th = rng.uniform(0, 2*np.pi, 8); t0 = time.perf_counter()
for ep in range(15):
    idx = rng.choice(80, 20, replace=False); g = np.zeros(8)
    for i in range(8):
        e = np.zeros(8); e[i] = np.pi/2; g[i] = (lossq(th+e, Xtr[idx], ytr[idx]) - lossq(th-e, Xtr[idx], ytr[idx]))/2
    th -= 0.5*g
tq = time.perf_counter()-t0
accq = np.mean([np.sign(out_z(x, th)) == yy for x, yy in zip(Xte, yte)])
def kern(A, B): return np.array([[abs(np.vdot(apply1(apply1(zero(2), RY(b[0]), 0, 2), RY(b[1]), 1, 2), apply1(apply1(zero(2), RY(a[0]), 0, 2), RY(a[1]), 1, 2)))**2 for b in B] for a in A])
t0 = time.perf_counter(); svm = SVC(kernel="precomputed").fit(kern(Xtr, Xtr), ytr); tk = time.perf_counter()-t0
acck = np.mean(svm.predict(kern(Xte, Xtr)) == yte)
t0 = time.perf_counter(); lr = LogisticRegression().fit(Xtr, ytr); tl = time.perf_counter()-t0
svc = SVC(kernel="rbf").fit(Xtr, ytr)
log(f"{'model':28s}{'test acc':>9s}{'params':>8s}{'train time':>12s}{'circuit evals':>15s}")
log(f"{'variational classifier (VQC)':28s}{accq:9.2f}{8:8d}{tq:11.2f}s{evals[0]:15d}")
log(f"{'quantum kernel + SVM':28s}{acck:9.2f}{'-':>8s}{tk:11.2f}s{'80*80 pairs':>15s}")
log(f"{'logistic regression':28s}{lr.score(Xte, yte):9.2f}{3:8d}{tl:11.4f}s{'0':>15s}")
log(f"{'RBF SVM':28s}{svc.score(Xte, yte):9.2f}{'-':>8s}{'<0.01':>11s}s{'0':>15s}")
log("Honest reading: on this small dataset the classical models are as good or better and far cheaper; the quantum models run on a classical simulator, so no quantum speedup is claimed.")
log("Hardware: no real quantum hardware was used; a noise model is in Track C (C2).")
open("results/track_i.txt", "w").write("\n".join(out))

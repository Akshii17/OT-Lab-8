"""Track A - continuous optimization (GD, momentum, NAG, AdaGrad, RMSProp, Adam, Newton,
Gauss-Newton, SGD/mini-batch, backprop, HVP, Fisher, noisy objective)."""
import time
import numpy as np
from sklearn.datasets import make_moons
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

# ---- objectives -------------------------------------------------------------
# convex (strongly): f = 0.5 (x^2 + 20 y^2)   non-convex: double well with a saddle at (0,0)
def f_cvx(p): return 0.5*(p[0]**2 + 20*p[1]**2)
def g_cvx(p): return np.array([p[0], 20*p[1]])
def H_cvx(p): return np.diag([1., 20.])
def f_ncv(p): return p[0]**4/4 - p[0]**2/2 + 0.2*p[0] + 0.5*p[1]**2   # local min x~0.8, global min x~-1.1
def g_ncv(p): return np.array([p[0]**3 - p[0] + 0.2, p[1]])
def H_ncv(p): return np.diag([3*p[0]**2 - 1, 1.])

def run(f, g, method, p0, iters=200, lr=0.05):
    p = np.array(p0, float); v = np.zeros(2); s = np.zeros(2); m = np.zeros(2); hist = [f(p)]
    t0 = time.perf_counter()
    for t in range(1, iters+1):
        if method == "GD": p -= lr*g(p)
        elif method == "Momentum": v = 0.9*v + g(p); p -= lr*v
        elif method == "NAG": v = 0.9*v + g(p - lr*0.9*v); p -= lr*v
        elif method == "AdaGrad": gr = g(p); s += gr**2; p -= 0.5*gr/(np.sqrt(s)+1e-8)
        elif method == "RMSProp": gr = g(p); s = 0.9*s + 0.1*gr**2; p -= 0.02*gr/(np.sqrt(s)+1e-8)
        elif method == "Adam":
            gr = g(p); m = 0.9*m + 0.1*gr; s = 0.999*s + 0.001*gr**2
            p -= 0.05*(m/(1-0.9**t))/(np.sqrt(s/(1-0.999**t))+1e-8)
        hist.append(f(p))
    return p, hist, time.perf_counter()-t0

def newton(f, g, Hf, p0, iters=20):
    p = np.array(p0, float)
    for _ in range(iters): p = p - np.linalg.solve(Hf(p), g(p))
    return p

log("== A1. Optimizers on convex and non-convex objectives (200 iters, lr=0.05) ==")
curves = {}
for name, (f, g, start) in {"convex": (f_cvx, g_cvx, [4., 1.]), "non-convex": (f_ncv, g_ncv, [1.5, 1.])}.items():
    log(f"-- {name} start={start}")
    for m in ["GD", "Momentum", "NAG", "AdaGrad", "RMSProp", "Adam"]:
        p, h, rt = run(f, g, m, start)
        log(f"{m:9s} final f={h[-1]:.5f} x={np.round(p,3)} time={rt*1e3:.2f} ms")
        if name == "convex": curves[m] = h
plt.figure(figsize=(5, 3.2))
for m, h in curves.items(): plt.semilogy(np.array(h)+1e-12, label=m)
plt.xlabel("iteration"); plt.ylabel("f (log)"); plt.title("Convex convergence"); plt.legend(fontsize=6); plt.tight_layout()
plt.savefig("results/track_a_convergence.png", dpi=130); plt.close()

log("\n== A2. Newton: convex converges in 1 step; non-convex converges to a saddle/min depending on start ==")
log("convex  from [4,1]  ->", np.round(newton(f_cvx, g_cvx, H_cvx, [4., 1.]), 4))
for s in ([1.5, 1.], [0.3, 1.], [0.05, 1.], [-1.5, 1.]):
    p = newton(f_ncv, g_ncv, H_ncv, s)
    log(f"non-cvx from {s} -> {np.round(p,3)} f={f_ncv(p):.4f} Hessian eig x-dir={3*p[0]**2-1:.3f} "
        f"({'min' if 3*p[0]**2-1>0 else 'max/saddle'})")

log("\n== A3. Gauss-Newton: fit y = a*exp(b*x) ==")
x = np.linspace(0, 1, 30); y = 2.0*np.exp(1.5*x) + rng.normal(0, 0.05, 30)
th = np.array([1., 1.])
for it in range(10):
    r = th[0]*np.exp(th[1]*x) - y
    J = np.stack([np.exp(th[1]*x), th[0]*x*np.exp(th[1]*x)], 1)
    th = th - np.linalg.solve(J.T@J, J.T@r)
log("Gauss-Newton estimate (a,b) =", np.round(th, 3), " true=(2.0,1.5)")

log("\n== A4. Backpropagation: 2-layer net on moons, plus gradient check ==")
X, Y = make_moons(400, noise=0.2, random_state=0); Y = Y.astype(float)
H_ = 16
def init(seed=0):
    r = np.random.default_rng(seed)
    return [r.normal(0, 0.5, (2, H_)), np.zeros(H_), r.normal(0, 0.5, H_), 0.0]
def fwd(P, X):
    z = np.tanh(X@P[0] + P[1]); return z, 1/(1+np.exp(-(z@P[2] + P[3])))
def loss(P, X, Y, lam=1e-3):
    _, p = fwd(P, X); e = 1e-9
    return -np.mean(Y*np.log(p+e) + (1-Y)*np.log(1-p+e)) + lam*(np.sum(P[0]**2) + np.sum(P[2]**2))
def backprop(P, X, Y, lam=1e-3):
    z, p = fwd(P, X); n = len(X)
    d_out = (p - Y)/n
    gW2 = z.T@d_out + 2*lam*P[2]; gb2 = d_out.sum()
    dz = np.outer(d_out, P[2])*(1-z**2)
    gW1 = X.T@dz + 2*lam*P[0]; gb1 = dz.sum(0)
    return [gW1, gb1, gW2, gb2]
P = init(); G = backprop(P, X, Y); eps = 1e-5; Pp = [q.copy() if isinstance(q, np.ndarray) else q for q in P]
Pp[0][0, 0] += eps; Pm = [q.copy() if isinstance(q, np.ndarray) else q for q in P]; Pm[0][0, 0] -= eps
log(f"gradient check dL/dW1[0,0]: backprop={G[0][0,0]:.6f}  finite-diff={(loss(Pp,X,Y)-loss(Pm,X,Y))/(2*eps):.6f}")

def train(mode, epochs=60, lr=0.1, bs=32, seed=0):
    P = init(); r = np.random.default_rng(seed); m = [np.zeros_like(q) for q in P]; s = [np.zeros_like(q) for q in P]; t = 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        idx = r.permutation(len(X))
        batches = [idx] if mode == "GD" else ([idx[i:i+1] for i in idx[:100]] if mode == "SGD" else [idx[i:i+bs] for i in range(0, len(X), bs)])
        for b in batches:
            G = backprop(P, X[b], Y[b]); t += 1
            for k in range(4):
                if mode == "Adam":
                    m[k] = 0.9*m[k] + 0.1*G[k]; s[k] = 0.999*s[k] + 0.001*G[k]**2
                    P[k] = P[k] - 0.02*(m[k]/(1-0.9**t))/(np.sqrt(s[k]/(1-0.999**t))+1e-8)
                else: P[k] = P[k] - lr*G[k]
    acc = np.mean((fwd(P, X)[1] > 0.5) == Y)
    return loss(P, X, Y), acc, time.perf_counter()-t0
for mode in ["GD", "SGD", "MiniBatch", "Adam"]:
    l, a, rt = train(mode); log(f"{mode:9s} loss={l:.4f} acc={a:.3f} time={rt:.2f}s")

log("\n== A5. Curvature tools: Hessian-vector product and diagonal Fisher ==")
p0 = np.array([0.7, 0.3]); v = np.array([1., 2.])
hvp = (g_cvx(p0 + 1e-5*v) - g_cvx(p0 - 1e-5*v))/2e-5
log("HVP (finite diff of gradients) =", np.round(hvp, 4), " exact H@v =", H_cvx(p0)@v)
z, p = fwd(init(), X); gs = (p - Y)[:, None]*z
log("diagonal Fisher (output layer, mean g^2) first 4:", np.round((gs**2).mean(0)[:4], 4))

log("\n== A6. Noisy objective: constant vs decaying step size ==")
for name, sched in [("constant lr=0.1", lambda t: 0.1), ("decay lr=0.5/(1+0.05t)", lambda t: 0.5/(1+0.05*t))]:
    finals = []
    for seed in range(10):
        r = np.random.default_rng(seed); p = np.array([4., 1.])
        for t in range(300): p = p - sched(t)*(g_cvx(p) + r.normal(0, 1.0, 2))
        finals.append(f_cvx(p))
    log(f"{name:24s} mean final f={np.mean(finals):.4f} +- {np.std(finals):.4f}")
open("results/track_a.txt", "w").write("\n".join(out))

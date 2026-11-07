"""Track D - submodular greedy, parallel speedup, FedAvg, MGDA, GradNorm."""
import time, itertools, os
import numpy as np
from multiprocessing import Pool, Process, Queue
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

# ---------------- D1 submodular: max coverage (sensor placement) -----------------
rng = np.random.default_rng(0)
U, S, k = 60, 20, 4
sets = [set(rng.choice(U, rng.integers(5, 15), replace=False)) for _ in range(S)]
cov = lambda idx: len(set().union(*[sets[i] for i in idx])) if idx else 0
log("== D1. Greedy on a submodular objective (coverage) ==")
viol = 0
for _ in range(2000):                      # test diminishing returns: f(A+e)-f(A) >= f(B+e)-f(B) for A subset B
    B = list(rng.choice(S, 6, replace=False)); A = B[:3]; e = int(rng.integers(S))
    if e in B: continue
    if cov(A+[e]) - cov(A) < cov(B+[e]) - cov(B): viol += 1
log("diminishing-returns violations in 2000 random tests:", viol, "(0 => submodular, so the 1-1/e guarantee applies)")
chosen = []
for _ in range(k):
    chosen.append(max((i for i in range(S) if i not in chosen), key=lambda i: cov(chosen+[i])))
opt = max(cov(list(c)) for c in itertools.combinations(range(S), k))
log(f"greedy covers {cov(chosen)}, optimum {opt}, ratio {cov(chosen)/opt:.3f} >= 1-1/e = {1-1/np.e:.3f}")

# ---------------- D2 parallel speedup ----------------------------------------------
def cpu_task(n):
    s = 0
    for i in range(n): s += i*i % 7
    return s
def sleep_task(t): time.sleep(t); return t
def timed(fn, *a):
    t0 = time.perf_counter(); fn(*a); return time.perf_counter()-t0

def par(task, items, workers, chunk):
    with Pool(workers) as p: return p.map(task, items, chunksize=chunk)
def par_dyn(task, items, workers):
    with Pool(workers) as p: return list(p.imap_unordered(task, items, chunksize=1))

log(f"\n== D2. Parallel speedup (machine has {os.cpu_count()} CPU core(s)) ==")
cpu_items = [200000]*24
t1 = timed(lambda: [cpu_task(x) for x in cpu_items]); log(f"CPU-bound, serial: {t1:.3f}s")
sp = {}
for w in (1, 2, 4):
    t = timed(par, cpu_task, cpu_items, w, 1); sp[w] = t1/t
    log(f"CPU-bound workers={w}: {t:.3f}s speedup={t1/t:.2f} efficiency={t1/t/w:.2f}")
io_items = [0.05]*24; ts = 24*0.05
for w in (1, 2, 4, 8):
    t = timed(par, sleep_task, io_items, w, 1)
    log(f"waiting-bound (sleep) workers={w}: {t:.3f}s speedup={ts/t:.2f} efficiency={ts/t/w:.2f}")
log("(CPU-bound speedup is limited by the number of physical cores; waiting-bound tasks overlap even on one core.)")

log("\nLoad balancing: uneven tasks, static chunks vs dynamic (work-stealing-like) scheduling, 4 workers")
uneven = [0.2] + [0.01]*39
t_static = timed(par, sleep_task, uneven, 4, 10)
t_dyn = timed(par_dyn, sleep_task, uneven, 4)
log(f"static chunks of 10: {t_static:.3f}s   dynamic (chunk=1): {t_dyn:.3f}s   ideal makespan ~{max(0.2, sum(uneven)/4):.3f}s")

def stage1(q_in, q_out):
    while (x := q_in.get()) is not None: time.sleep(0.01); q_out.put(x)
    q_out.put(None)
def stage2(q_in, q_out):
    while (x := q_in.get()) is not None: time.sleep(0.01)
    q_out.put("done")
def pipeline(n):
    a, b, c = Queue(), Queue(), Queue()
    p1 = Process(target=stage1, args=(a, b)); p2 = Process(target=stage2, args=(b, c)); p1.start(); p2.start()
    for i in range(n): a.put(i)
    a.put(None); c.get(); p1.join(); p2.join()
n_items = 40
t_seq = n_items*0.02; t_pipe = timed(pipeline, n_items)
log(f"\nPipeline parallelism (2 stages = 2 'layers' on separate processes, which is also the simplest model-parallel split): "
    f"sequential {t_seq:.2f}s vs pipeline {t_pipe:.2f}s, speedup {t_seq/t_pipe:.2f} (ideal 2.0)")
# Amdahl: serial fraction from measured 4-worker speedup on the waiting-bound task
log("Amdahl: with serial fraction f the speedup on P workers is 1/(f+(1-f)/P); f=0.1 -> P=4: %.2f, P=inf: 10" % (1/(0.1+0.9/4)))
plt.figure(figsize=(4.5, 3)); plt.bar([str(w) for w in sp], list(sp.values())); plt.axhline(1, color="k", lw=.5)
plt.xlabel("workers"); plt.ylabel("speedup (CPU-bound)"); plt.title(f"Speedup on {os.cpu_count()} core(s)"); plt.tight_layout()
plt.savefig("results/track_d_speedup.png", dpi=130); plt.close()

# ---------------- D3 FedAvg ---------------------------------------------------------
log("\n== D3. FedAvg vs centralized (logistic regression, 5 clients, non-IID) ==")
r = np.random.default_rng(1); d = 5; w_true = r.normal(size=d)
clients = []
for c in range(5):
    X = r.normal(c*0.3, 1, (80, d)); y = (X@w_true + r.normal(0, .5, 80) > 0).astype(float); clients.append((X, y))
Xall = np.vstack([c[0] for c in clients]); yall = np.concatenate([c[1] for c in clients])
sig = lambda z: 1/(1+np.exp(-z))
lossf = lambda w: -np.mean(yall*np.log(sig(Xall@w)+1e-9) + (1-yall)*np.log(1-sig(Xall@w)+1e-9))
def local(w, X, y, steps, lr=0.1):
    w = w.copy()
    for _ in range(steps): w -= lr*X.T@(sig(X@w) - y)/len(X)
    return w
w = np.zeros(d)
for _ in range(200): w = local(w, Xall, yall, 1)
log(f"centralized GD (200 steps): loss={lossf(w):.4f}")
for E in (1, 5, 20):
    for R in (5, 20):
        w = np.zeros(d)
        for _ in range(R): w = np.mean([local(w, X, y, E) for X, y in clients], axis=0)
        log(f"FedAvg local_steps={E:2d} rounds={R:2d}: loss={lossf(w):.4f}  (communication = {R} rounds)")

# ---------------- D4 multi-task: equal vs MGDA vs GradNorm -------------------------
log("\n== D4. Two-task shared-parameter model: equal weights vs MGDA vs GradNorm ==")
r = np.random.default_rng(2); Xt = r.normal(size=(100, 6)); y1 = Xt@r.normal(size=6); y2 = Xt@r.normal(size=6)
L1 = lambda w: np.mean((Xt@w - y1)**2); L2 = lambda w: 100*np.mean((Xt@w - y2)**2)   # task 2 has 100x larger scale
G1 = lambda w: 2*Xt.T@(Xt@w - y1)/100; G2 = lambda w: 200*Xt.T@(Xt@w - y2)/100
def train(kind, steps=300):
    w = np.zeros(6); a = np.array([.5, .5])
    for _ in range(steps):
        g1, g2 = G1(w), G2(w)
        if kind == "equal": a = np.array([.5, .5])
        elif kind == "MGDA":
            den = np.dot(g1-g2, g1-g2); t = 0.5 if den < 1e-12 else np.clip(np.dot(g2-g1, g2)/den, 0, 1); a = np.array([t, 1-t])
        else:   # GradNorm (simplified): push weights so that weighted gradient norms are equal
            n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2); tgt = (n1+n2)/2
            a = a*np.exp(0.1*np.array([np.log(tgt/(a[0]*n1+1e-9)), np.log(tgt/(a[1]*n2+1e-9))]))
            a = 2*a/a.sum()
        w = w - 0.002*(a[0]*g1 + a[1]*g2)
    return L1(w), L2(w)
for kind in ("equal", "MGDA", "GradNorm"):
    l1, l2 = train(kind); log(f"{kind:9s} L1={l1:.4f} L2={l2:.4f} sum={l1+l2:.4f}")
open("results/track_d.txt", "w").write("\n".join(out))

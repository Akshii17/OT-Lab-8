"""Track H - DBMS: index selection under a storage budget (SQLite, 100k-row synthetic table).
Formulation: choose a subset of candidate indexes maximising workload time saved subject to total size <= budget (knapsack)."""
import sqlite3, os, time, hashlib, itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

DB = "/tmp/track_h.db"
if os.path.exists(DB): os.remove(DB)
con = sqlite3.connect(DB); r = np.random.default_rng(0); N = 100_000
con.execute("CREATE TABLE orders(id INTEGER PRIMARY KEY, cust INT, product INT, region INT, status INT, day INT, amount REAL)")
rows = [(i, int(r.integers(5000)), int(r.integers(1000)), int(r.integers(10)), int(r.integers(4)), int(r.integers(365)), float(r.random()*500)) for i in range(N)]
con.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?)", rows); con.commit(); con.execute("ANALYZE")
log(f"== H1. Synthetic dataset: orders table, {N} rows (documented, seed 0) ==")

WORKLOAD = [   # (sql, executions per hour)
    ("SELECT COUNT(*), SUM(amount) FROM orders WHERE cust = 1234", 50),
    ("SELECT id, amount FROM orders WHERE product = 77 AND status = 1", 30),
    ("SELECT region, SUM(amount) FROM orders WHERE day BETWEEN 100 AND 110 GROUP BY region", 20),
    ("SELECT id FROM orders WHERE cust = 42 AND day > 300", 40),
    ("SELECT AVG(amount) FROM orders WHERE region = 3 AND status = 2 AND day < 30", 10),
]
CANDS = {"cust": "(cust)", "product_status": "(product, status)", "day": "(day)", "cust_day": "(cust, day)",
         "region_status_day": "(region, status, day)", "status": "(status)", "region": "(region)", "product": "(product)"}

def size_kb(): return (con.execute("PRAGMA page_count").fetchone()[0]-con.execute("PRAGMA freelist_count").fetchone()[0])*con.execute("PRAGMA page_size").fetchone()[0]/1024
def qtime(sql):
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter(); con.execute(sql).fetchall(); best = min(best, time.perf_counter()-t0)
    return best
def workload_time(): return sum(qtime(q)*f for q, f in WORKLOAD)
def results_hash(): return [hashlib.md5(repr(sorted(con.execute(q).fetchall())).encode()).hexdigest() for q, _ in WORKLOAD]
def plan(sql): return " | ".join(row[3] for row in con.execute("EXPLAIN QUERY PLAN " + sql))

base_t = workload_time(); base_hash = results_hash(); base_kb = size_kb()
log(f"baseline workload time (weighted by frequency) = {base_t*1e3:.1f} ms, db size = {base_kb:.0f} KB")
log("baseline plans:"); [log("   ", plan(q)) for q, _ in WORKLOAD]

log("\n== H2. Measure benefit and size of each candidate index ==")
benefit, size = {}, {}
for name, cols in CANDS.items():
    before = size_kb(); con.execute(f"CREATE INDEX ix_{name} ON orders{cols}"); con.execute("ANALYZE")
    size[name] = size_kb()-before; benefit[name] = base_t - workload_time()
    con.execute(f"DROP INDEX ix_{name}")
    log(f"{name:18s} size={size[name]:7.0f} KB  time saved={benefit[name]*1e3:7.2f} ms")
con.execute("ANALYZE"); names = list(CANDS); budget = 1500
log(f"\nstorage budget = {budget} KB")

def greedy():
    chosen, used = [], 0
    for nme in sorted(names, key=lambda x: -benefit[x]/max(size[x], 1)):
        if benefit[nme] > 0 and used + size[nme] <= budget: chosen.append(nme); used += size[nme]
    return chosen
def dp():
    w = [int(np.ceil(size[x])) for x in names]; v = [benefit[x] for x in names]
    best = [[0.0]*(budget+1) for _ in range(len(names)+1)]
    for i in range(1, len(names)+1):
        for c in range(budget+1):
            best[i][c] = best[i-1][c]
            if w[i-1] <= c: best[i][c] = max(best[i][c], best[i-1][c-w[i-1]] + v[i-1])
    c = budget; chosen = []
    for i in range(len(names), 0, -1):
        if best[i][c] != best[i-1][c]: chosen.append(names[i-1]); c -= w[i-1]
    return chosen
def ilp():
    res = milp(-np.array([benefit[x] for x in names]), constraints=LinearConstraint(np.array([[size[x] for x in names]]), -np.inf, budget),
               integrality=np.ones(len(names)), bounds=Bounds(0, 1))
    return [names[i] for i in range(len(names)) if res.x[i] > 0.5]

log("\n== H3. Greedy vs DP vs ILP vs default (no index) ==")
for label, fn in [("greedy (benefit/size)", greedy), ("dynamic programming", dp), ("ILP (milp)", ilp)]:
    sel = fn()
    for x in sel: con.execute(f"CREATE INDEX ix_{x} ON orders{CANDS[x]}")
    con.execute("ANALYZE"); t = workload_time(); kb = size_kb()-base_kb
    same = results_hash() == base_hash
    log(f"{label:22s} chose {sel}\n{'':22s} workload {t*1e3:.2f} ms (baseline {base_t*1e3:.1f} ms, speedup {base_t/t:.1f}x), extra storage {kb:.0f} KB, results identical: {same}")
    if label.startswith("dynamic"): log("  plans with DP choice:"); [log("     ", plan(q)) for q, _ in WORKLOAD]
    for x in sel: con.execute(f"DROP INDEX ix_{x}")
log("(Benefits are measured one index at a time and assumed additive; the combined measurement above checks that assumption.)")
open("results/track_h.txt", "w").write("\n".join(out))

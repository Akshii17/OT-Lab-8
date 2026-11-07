"""Track F - mini compiler: lexer, parser, 3-address IR, 4 optimization passes, register-VM codegen.
Optimization decision: pass ordering (greedy vs exhaustive search)."""
import re, time, itertools

out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

SRC = """
a = 4 * 8;
b = a + 0;
c = x * 2;
d = (x + y) * (x + y);
e = (x + y) * 3;
unused = d * 100 + 7;
f = c + e;
g = 2 * 5 + 1;
h = f - g;
print h;
print d;
print b * 1;
"""
INPUTS = {"x": 7, "y": 5}

# ---------------- lexer + parser (recursive descent) ----------------
def lex(src): return re.findall(r"\d+|[A-Za-z_]\w*|[=+\-*/();]", src)
class Parser:
    def __init__(s, toks): s.t = toks; s.i = 0
    def peek(s): return s.t[s.i] if s.i < len(s.t) else None
    def eat(s, x=None):
        v = s.t[s.i]; s.i += 1
        assert x is None or v == x, f"expected {x}, got {v}"; return v
    def program(s):
        stmts = []
        while s.peek():
            if s.peek() == "print": s.eat(); stmts.append(("print", s.expr())); s.eat(";")
            else: n = s.eat(); s.eat("="); stmts.append(("assign", n, s.expr())); s.eat(";")
        return stmts
    def expr(s):
        v = s.term()
        while s.peek() in ("+", "-"): op = s.eat(); v = (op, v, s.term())
        return v
    def term(s):
        v = s.atom()
        while s.peek() in ("*", "/"): op = s.eat(); v = (op, v, s.atom())
        return v
    def atom(s):
        t = s.eat()
        if t == "(": v = s.expr(); s.eat(")"); return v
        return int(t) if t.isdigit() else t

# ---------------- IR generation: (op, dst, a, b) ----------------
OPN = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
def gen_ir(stmts):
    ir = []; cnt = [0]
    def tmp(): cnt[0] += 1; return f"t{cnt[0]}"
    def ex(e):
        if not isinstance(e, tuple): return e
        a, b = ex(e[1]), ex(e[2]); d = tmp(); ir.append((OPN[e[0]], d, a, b)); return d
    for st in stmts:
        if st[0] == "print": ir.append(("print", None, ex(st[1]), None))
        else: r = ex(st[2]); ir.append(("mov", st[1], r, None))
    return ir

EV = {"add": lambda a, b: a+b, "sub": lambda a, b: a-b, "mul": lambda a, b: a*b, "div": lambda a, b: a//b}
isnum = lambda v: isinstance(v, int)

# ---------------- optimization passes ----------------
def const_prop_fold(ir):
    env = {}; res = []
    def val(v): return env.get(v, v) if not isnum(v) else v
    for op, d, a, b in ir:
        a = val(a) if a is not None else None; b = val(b) if b is not None else None
        if op in EV and isnum(a) and isnum(b) and not (op == "div" and b == 0): op, a, b = "mov", EV[op](a, b), None
        elif op == "add" and b == 0: op, b = "mov", None
        elif op == "add" and a == 0: op, a, b = "mov", b, None
        elif op == "mul" and (a == 0 or b == 0): op, a, b = "mov", 0, None
        elif op == "mul" and b == 1: op, b = "mov", None
        elif op == "mul" and a == 1: op, a, b = "mov", b, None
        if d is not None:
            env = {k: v for k, v in env.items() if k != d and v != d}
            if op == "mov": env[d] = a
        res.append((op, d, a, b))
    return res
def strength_reduce(ir):
    res = []
    for op, d, a, b in ir:
        if op == "mul":
            if isnum(a) and not isnum(b): a, b = b, a
            if isnum(b) and b > 0 and b & (b-1) == 0: res.append(("shl", d, a, b.bit_length()-1)); continue
        res.append((op, d, a, b))
    return res
def cse(ir):
    table = {}; res = []
    for op, d, a, b in ir:
        if op in ("add", "mul", "sub", "div", "shl"):
            key = (op, a, b) if op not in ("add", "mul") else (op,) + tuple(sorted((str(a), str(b))))
            if key in table: res.append(("mov", d, table[key], None))
            else: res.append((op, d, a, b))
            table = {k: v for k, v in table.items() if v != d and d not in k}
            if key not in table and res[-1][0] != "mov": table[key] = d
        else:
            res.append((op, d, a, b))
            if d is not None: table = {k: v for k, v in table.items() if v != d and d not in k}
    return res
def dce(ir):
    live = set(); res = []
    for op, d, a, b in reversed(ir):
        if op == "print" or d in live:
            if d is not None: live.discard(d)
            for v in (a, b):
                if isinstance(v, str): live.add(v)
            res.append((op, d, a, b))
    return res[::-1]
PASSES = {"const": const_prop_fold, "strength": strength_reduce, "cse": cse, "dce": dce}

COST = {"mov": 1, "add": 1, "sub": 1, "shl": 1, "mul": 3, "div": 8, "print": 1}
def run_vm(ir, inputs):
    regs = dict(inputs); outp = []
    g = lambda v: v if isnum(v) else regs[v]
    for op, d, a, b in ir:
        if op == "mov": regs[d] = g(a)
        elif op == "shl": regs[d] = g(a) << b
        elif op == "print": outp.append(g(a))
        else: regs[d] = EV[op](g(a), g(b))
    return outp
def cycles(ir): return sum(COST[i[0]] for i in ir)
def asm(ir): return "\n".join(f"{op} {d} {a} {b}" for op, d, a, b in ir)

t0 = time.perf_counter(); stmts = Parser(lex(SRC)).program(); base = gen_ir(stmts); t_compile_base = time.perf_counter()-t0
ref = run_vm(base, INPUTS)
def apply_order(ir, order):
    for p in order: ir = PASSES[p](ir)
    return ir
log("== F1. Unoptimized program ==")
log(f"IR instructions={len(base)}  est. cycles={cycles(base)}  output={ref}")

log("\n== F2. Each pass alone, then pass-ordering as an optimization problem ==")
for p in PASSES:
    o = apply_order(base, [p]); log(f"{p:9s} alone: instrs={len(o):2d} cycles={cycles(o):3d} output_ok={run_vm(o, INPUTS)==ref}")
results = {}
for order in itertools.permutations(PASSES):
    o = apply_order(base, order); assert run_vm(o, INPUTS) == ref; results[order] = (cycles(o), len(o))
best = min(results, key=lambda k: results[k]); worst = max(results, key=lambda k: results[k])
log(f"exhaustive search over 24 orders: best {best} -> cycles={results[best][0]} instrs={results[best][1]}; worst {worst} -> cycles={results[worst][0]}")
cur = base; greedy_order = []; left = list(PASSES)
while left:
    p = min(left, key=lambda q: cycles(PASSES[q](cur))); cur = PASSES[p](cur); greedy_order.append(p); left.remove(p)
log(f"greedy (apply whichever pass helps most next): {greedy_order} -> cycles={cycles(cur)}")

log("\n== F3. Before/after measurement ==")
t0 = time.perf_counter(); opt = apply_order(base, best); t_opt = time.perf_counter()-t0
def bench(ir): 
    t0 = time.perf_counter()
    for _ in range(20000): run_vm(ir, INPUTS)
    return time.perf_counter()-t0
tb, to = bench(base), bench(opt)
log(f"{'':14s}{'instrs':>8s}{'cycles':>8s}{'run time (20k runs)':>22s}")
log(f"{'unoptimized':14s}{len(base):8d}{cycles(base):8d}{tb:19.3f} s")
log(f"{'optimized':14s}{len(opt):8d}{cycles(opt):8d}{to:19.3f} s")
log(f"extra compile time for the passes: {t_opt*1e3:.2f} ms;  same output: {run_vm(opt, INPUTS) == ref}")
for xy in [{"x": 1, "y": 2}, {"x": -3, "y": 9}, {"x": 0, "y": 0}]:
    assert run_vm(base, xy) == run_vm(opt, xy)
log("correctness: optimized and unoptimized programs give identical output on 4 different inputs")
log("\noptimized IR:\n" + asm(opt))
open("results/track_f.txt", "w").write("\n".join(out))

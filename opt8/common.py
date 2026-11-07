"""Shared helpers: the Max-Cut graph used by Track B (QUBO) and Track C (QAOA)."""
import itertools
import numpy as np

N_NODES = 8
EDGES = [(0,1),(0,2),(1,3),(2,3),(2,4),(3,5),(4,5),(4,6),(5,7),(6,7),(0,7),(1,6)]

def cut_value(bits, edges=EDGES):
    return sum(1 for i, j in edges if bits[i] != bits[j])

def maxcut_qubo(n=N_NODES, edges=EDGES):
    """Minimise x^T Q x (upper-triangular Q) where energy = -cut(x)."""
    Q = np.zeros((n, n))
    for i, j in edges:
        Q[i, i] -= 1; Q[j, j] -= 1; Q[min(i, j), max(i, j)] += 2
    return Q

def qubo_energy(Q, x):
    x = np.asarray(x, float)
    return float(x @ Q @ x)

def brute_force_qubo(Q):
    n = len(Q); best = (1e18, None)
    for bits in itertools.product([0, 1], repeat=n):
        e = qubo_energy(Q, bits)
        if e < best[0]: best = (e, bits)
    return best

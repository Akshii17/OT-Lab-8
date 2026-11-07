"""Tiny state-vector quantum simulator (numpy only). Qubit 0 is the most significant bit."""
import numpy as np

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], complex)
Z = np.diag([1, -1]).astype(complex)
def RX(t): return np.array([[np.cos(t/2), -1j*np.sin(t/2)], [-1j*np.sin(t/2), np.cos(t/2)]])
def RY(t): return np.array([[np.cos(t/2), -np.sin(t/2)], [np.sin(t/2), np.cos(t/2)]], complex)
def RZ(t): return np.diag([np.exp(-1j*t/2), np.exp(1j*t/2)])
def PH(t): return np.diag([1, np.exp(1j*t)])

def zero(n):
    s = np.zeros(2**n, complex); s[0] = 1; return s

def apply1(s, U, q, n):
    s = s.reshape([2]*n)
    s = np.moveaxis(np.tensordot(U, s, axes=([1], [q])), 0, q)
    return s.reshape(-1)

def apply_ctrl(s, U, c, t, n):
    s = s.reshape([2]*n).copy()
    idx = [slice(None)]*n; idx[c] = 1
    sub = s[tuple(idx)]
    tt = t if t < c else t-1
    sub = np.moveaxis(np.tensordot(U, sub, axes=([1], [tt])), 0, tt)
    s[tuple(idx)] = sub
    return s.reshape(-1)

def cnot(s, c, t, n): return apply_ctrl(s, X, c, t, n)
def swap(s, a, b, n): return cnot(cnot(cnot(s, a, b, n), b, a, n), a, b, n)

def bit(i, q, n): return (i >> (n-1-q)) & 1

def permute(s, f):
    """Apply a basis-state permutation i -> f(i)."""
    out = np.zeros_like(s)
    for i in range(len(s)): out[f(i)] += s[i]
    return out

def probs(s): return np.abs(s)**2

def sample(s, shots, rng):
    p = probs(s); return rng.choice(len(p), size=shots, p=p/p.sum())

def qft(s, qs, n, inverse=False):
    m = len(qs); ops = []
    for j in range(m):
        ops.append(('h', qs[j]))
        for k in range(j+1, m): ops.append(('cp', qs[k], qs[j], np.pi/2**(k-j)))
    for j in range(m//2): ops.append(('sw', qs[j], qs[m-1-j]))
    if inverse: ops = ops[::-1]
    for op in ops:
        if op[0] == 'h': s = apply1(s, H, op[1], n)
        elif op[0] == 'cp': s = apply_ctrl(s, PH(-op[3] if inverse else op[3]), op[1], op[2], n)
        else: s = swap(s, op[1], op[2], n)
    return s

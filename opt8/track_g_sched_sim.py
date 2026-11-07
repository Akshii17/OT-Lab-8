"""Track G (support) - scheduler simulation: round robin (xv6 default) vs priority scheduling.
This is a Python SIMULATION of the same 3 workloads used by xv6's schedbench.c. It is NOT a measurement of xv6 under QEMU."""
import numpy as np
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

def simulate(jobs, policy, quantum=1):
    """jobs: list of dict(burst, prio, io) ; io=True means it sleeps 1 tick after every burst unit chunk of 10."""
    rem = [j["burst"] for j in jobs]; ready_at = [0]*len(jobs); done = [None]*len(jobs); last = [-1]*len(jobs); t = 0; since_io = [0]*len(jobs)
    while any(d is None for d in done):
        cand = [i for i in range(len(jobs)) if done[i] is None and ready_at[i] <= t]
        if not cand: t += 1; continue
        if policy == "round robin": i = min(cand, key=lambda i: last[i])
        else: i = min(cand, key=lambda i: (jobs[i]["prio"], last[i]))
        rem[i] -= quantum; since_io[i] += quantum; last[i] = t; t += quantum
        if rem[i] <= 0: done[i] = t
        elif jobs[i]["io"] and since_io[i] >= 10: since_io[i] = 0; ready_at[i] = t + 5
    return done

workloads = {
 "W1 CPU-bound, 2 high + 2 low priority": [dict(burst=200, prio=2, io=False)]*2 + [dict(burst=200, prio=18, io=False)]*2,
 "W2 twelve short equal jobs": [dict(burst=20, prio=10, io=False)]*12,
 "W3 CPU + sleepy mix": [dict(burst=150, prio=5, io=False), dict(burst=150, prio=5, io=True), dict(burst=150, prio=15, io=False), dict(burst=150, prio=15, io=True)],
}
log("== G1. Scheduler simulation (time units = ticks). Turnaround of every job, high-priority jobs listed first ==")
for name, jobs in workloads.items():
    log(name)
    for pol in ("round robin", "priority"):
        d = simulate(jobs, pol); first = np.mean([d[i] for i in range(len(jobs)) if jobs[i]["prio"] == min(j["prio"] for j in jobs)])
        log(f"   {pol:12s} turnarounds={d}  mean={np.mean(d):.1f}  mean of highest-priority jobs={first:.1f}  makespan={max(d)}")
log("Reading: priority scheduling cuts turnaround for the important jobs (W1, W3) at the cost of the low-priority ones; with equal priorities (W2) it behaves like round robin.")
open("results/track_g.txt", "w").write("\n".join(out))

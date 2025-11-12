#!/usr/bin/env python
# %%
import numpy as np

# Parameters (copied from your setup)
N   = 100
Nr0 = 13.3
lam = 1.3
l   = 0.9
abl = 20
ild = 1
dt  = 1e-4

# Recompute rates
r0   = Nr0 / N
r_db = (2*abl + ild)/2
l_db = (2*abl - ild)/2
pr   = 10**(r_db/20)
pl   = 10**(l_db/20)
den  = pr**(lam*l) + pl**(lam*l)
rr   = pr**lam / den
rl   = pl**lam / den
r_right = r0 * rr
r_left  = r0 * rl

mu       = N * (r_right - r_left)
sigma_sq = N * (r_right + r_left)
sigma    = np.sqrt(sigma_sq)

print(f"Parameters:")
print(f"r_right = {r_right:.6f}, r_left = {r_left:.6f}")
print(f"mu = {mu:.6f}, sigma_sq = {sigma_sq:.6f}, sigma = {sigma:.6f}")
print(f"dt = {dt}")

def ddm_theory(theta):
    fc = 1.0 / (1.0 + np.exp(-2 * mu * theta / sigma_sq))
    dt_mean = theta / mu * np.tanh(mu * theta / sigma_sq)
    return fc, dt_mean

def ddm_sim(theta, n_trials=20000, seed=0):
    rng = np.random.default_rng(seed)
    max_steps = int(20.0 / dt)
    choices = []
    rts = []
    for _ in range(n_trials):
        x = 0.0
        t = 0.0
        for _ in range(max_steps):
            x += mu * dt + sigma * np.sqrt(dt) * rng.normal()
            t += dt
            if x >= theta:
                choices.append(1); rts.append(t); break
            if x <= -theta:
                choices.append(-1); rts.append(t); break
        else:
            choices.append(0); rts.append(np.nan)
    rts = np.array(rts)
    choices = np.array(choices)
    mask = ~np.isnan(rts)
    fc_sim = np.mean(choices[mask] == 1)
    dt_sim = np.mean(rts[mask])
    frac_decided = np.mean(mask)
    return fc_sim, dt_sim, frac_decided

print("\nTheory vs. Simulation Comparison:")
for theta in [1, 4, 7, 10]:
    fc_th, dt_th = ddm_theory(theta)
    fc_sim, dt_sim, frac = ddm_sim(theta)
    print(f"θ={theta}:  Theory FC={fc_th:.3f}, DT={dt_th:.3f}  |  Sim FC={fc_sim:.3f}, DT={dt_sim:.3f}, decided={frac:.3f}")

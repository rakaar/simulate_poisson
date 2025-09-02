# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from sim_utils import poisson_spk_train

# Helper for parallel continuous DDM single trial (resample until first hit within [0, T])
def cont_ddm_first_hit(T, dt, theta_E, mu, sigma_sq):
    rng = np.random.default_rng()
    dB = dt ** 0.5
    while True:
        dv = 0.0
        t = 0.0
        while t < T:
            dv += (mu * dt) + ((sigma_sq ** 0.5) * rng.normal(0, dB))

            t += dt
            if dv >= theta_E:
                return t, True
            if dv <= -theta_E:
                return t, False
        # no hit by T -> resample a fresh path

# %% 
# Params
T = 5

rate_lambda = 2.3
Nr0 = 1000/65 # 15.38
theta_E = 2.8
l  = 0.9

N_sim = int(100e3)

ABL = 0
ILD = 0

SL_R = ABL + (ILD/2)
SL_L = ABL - (ILD/2)

chi = 17.37

# %%
# firing rate terms
common_rate_denom =  (10**(rate_lambda * l * SL_R / 20)) + (10**(rate_lambda * l * SL_L / 20))
right_rate_num = 10 ** ( rate_lambda * ( SL_R / 20))
left_rate_num = 10 ** ( rate_lambda *  ( SL_L / 20))
right_rate = Nr0 * (right_rate_num / common_rate_denom)
left_rate = Nr0 * (left_rate_num / common_rate_denom)
print(f'Nr0 = {Nr0:.4g} Hz')
print(f"Right rate: {right_rate:.4g} Hz")
print(f"Left rate: {left_rate:.4g} Hz")

# %%
# DDM terms
scaling_term = (10 **(rate_lambda * (1-l) * ABL / 20)) * Nr0
hyp_arg_term = rate_lambda * ILD / chi
hyp_arg_term_with_l = rate_lambda * l * ILD / chi


mu = scaling_term * np.sinh(hyp_arg_term) / np.cosh(hyp_arg_term_with_l)
# if mu is nan, set it to 0
if np.isnan(mu) and ABL == 0:
    mu = 0
sigma_sq = scaling_term * np.cosh(hyp_arg_term) / np.cosh(hyp_arg_term_with_l)
print(f"Nr0 = {Nr0:.4g} Hz")
print(f"mu: {mu:.4g}")
print(f"sigma_sq: {sigma_sq:.4g}")

# %%
# #### Discrete DDM ##
# Event-based Monte Carlo (N_sim trials)
first_pos_times = []
first_neg_times = []
for s in range(N_sim):
    # ensure a first-passage hit within [0, T]; resample if none
    while True:
        r_times = poisson_spk_train(right_rate, T)
        l_times = poisson_spk_train(left_rate, T)
        if r_times.size == 0 and l_times.size == 0:
            # no events at all; try again
            continue
        times = np.concatenate((r_times, l_times))
        labels = np.concatenate((
            np.ones_like(r_times, dtype=int),
            -np.ones_like(l_times, dtype=int),
        ))
        order = np.argsort(times)
        times = times[order]
        labels = labels[order]

        DV = 0
        pos_time = np.nan
        neg_time = np.nan
        hit = False
        for t, lab in zip(times, labels):
            DV += lab
            if DV >= theta_E:
                pos_time = t
                hit = True
                break
            if DV <= -theta_E:
                neg_time = t
                hit = True
                break
        if hit:
            if not np.isnan(pos_time):
                first_pos_times.append(pos_time)
            else:
                first_neg_times.append(neg_time)
            break  # proceed to next trial

# Plot crossing-time distributions like sim_parallel.py (lines 103–121)
pos_times = np.array(first_pos_times)
neg_times = np.array(first_neg_times)
frac_pos = pos_times.size / N_sim
frac_neg = neg_times.size / N_sim

# %%
# continuous DDM (N_sim trials; parallel)
dt = 1e-4
cont_pos_times = []
cont_neg_times = []
with ProcessPoolExecutor(max_workers=(os.cpu_count() or 1)) as ex:
    futures = [ex.submit(cont_ddm_first_hit, T, dt, theta_E, mu, sigma_sq) for _ in range(N_sim)]
    for fut in as_completed(futures):
        t, is_pos = fut.result()
        if is_pos:
            cont_pos_times.append(t)
        else:
            cont_neg_times.append(t)

cont_pos_times = np.array(cont_pos_times)
cont_neg_times = np.array(cont_neg_times)
cont_frac_pos = cont_pos_times.size / N_sim
cont_frac_neg = cont_neg_times.size / N_sim

    
# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
bins = np.arange(0, T, 0.01)
if pos_times.size > 0:
    pos_pdf, edges = np.histogram(pos_times, bins=bins, density=True)
    pos_pdf *= frac_pos  # area equals fraction of + crossings
    ax.stairs(pos_pdf, edges, fill=False, color='tab:red', lw=1.5, label=f'+θ_E (frac={frac_pos:.3f})')
    print(f'Event-based: +θ_E fraction = {frac_pos:.5f}')
if neg_times.size > 0:
    neg_pdf, edges = np.histogram(neg_times, bins=bins, density=True)
    neg_pdf *= -frac_neg  # negative to go below zero; area magnitude equals fraction of - crossings
    ax.stairs(neg_pdf, edges, fill=False, color='tab:blue', lw=1.5, label=f'-θ_E (frac={frac_neg:.3f})')
    print(f'Event-based: -θ_E fraction = {frac_neg:.5f}')
if cont_pos_times.size > 0:
    cont_pos_pdf, edges = np.histogram(cont_pos_times, bins=bins, density=True)
    cont_pos_pdf *= cont_frac_pos
    ax.stairs(cont_pos_pdf, edges, fill=False, color='tab:red', lw=1.0, ls='--', label=f'DDM +θ_E (frac={cont_frac_pos:.3f})')
    print(f'Cont DDM: +θ_E fraction = {cont_frac_pos:.5f}')
if cont_neg_times.size > 0:
    cont_neg_pdf, edges = np.histogram(cont_neg_times, bins=bins, density=True)
    cont_neg_pdf *= -cont_frac_neg
    ax.stairs(cont_neg_pdf, edges, fill=False, color='tab:blue', lw=1.0, ls='--', label=f'DDM -θ_E (frac={cont_frac_neg:.3f})')
    print(f'Cont DDM: -θ_E fraction = {cont_frac_neg:.5f}')
ax.axhline(0, color='k', lw=0.8)
ax.set_title('Crossing-time distributions')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Fraction density (+ above, - below)')
ax.legend(frameon=False)

fig.savefig('event_mc_crossing_dists.png', dpi=150, bbox_inches='tight')
fig.show()

# %%
# Crossing-time distributions normalized per bound (area=1), no fraction scaling
fig2, ax2 = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
bins = np.arange(0, T, 0.01)
if pos_times.size > 0:
    pos_pdf_n, edges = np.histogram(pos_times, bins=bins, density=True)
    ax2.stairs(pos_pdf_n, edges, fill=False, color='tab:red', lw=1.5, label='+θ_E (area=1)')
if neg_times.size > 0:
    neg_pdf_n, edges = np.histogram(neg_times, bins=bins, density=True)
    ax2.stairs(-neg_pdf_n, edges, fill=False, color='tab:blue', lw=1.5, label='-θ_E (area=1)')
if cont_pos_times.size > 0:
    cont_pos_pdf_n, edges = np.histogram(cont_pos_times, bins=bins, density=True)
    ax2.stairs(cont_pos_pdf_n, edges, fill=False, color='tab:red', lw=1.0, ls='--', label='DDM +θ_E (area=1)')
if cont_neg_times.size > 0:
    cont_neg_pdf_n, edges = np.histogram(cont_neg_times, bins=bins, density=True)
    ax2.stairs(-cont_neg_pdf_n, edges, fill=False, color='tab:blue', lw=1.0, ls='--', label='DDM -θ_E (area=1)')
ax2.axhline(0, color='k', lw=0.8)
ax2.set_title('Crossing-time distributions (normalized per bound)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Density (+ above, - below)')
ax2.legend(frameon=False)

fig2.savefig('event_mc_crossing_dists_area1.png', dpi=150, bbox_inches='tight')
fig2.show()

# Poisson DDM - spike trains - inefficient
# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x
from sim_utils import poisson_spk_train, gen_side_spk_trains

T = 2
dt = 1e-3

N = 5
Nr0 = 1000/65 # 15.38
theta_E = 2.8
N_sim = int(50e3)

R0 = Nr0 / N
# %%
# Poisson Neural activity
n_bins = int(round(T / dt))
time_bins = np.arange(n_bins + 1) * dt  # for plotting with initial 0
cum_diffs = np.zeros((N_sim, n_bins), dtype=int)
cross_times = np.full(N_sim, np.nan)
cross_times_pos = np.full(N_sim, np.nan)
cross_times_neg = np.full(N_sim, np.nan)
first_pos_times = []  # earliest +theta_E crossing per trial
first_neg_times = []  # earliest -theta_E crossing per trial
def simulate_one_trial(s, N, T, dt, R0, theta_E):
    while True:
        lmat, _ = gen_side_spk_trains(N, T, dt, R0)
        rmat, _ = gen_side_spk_trains(N, T, dt, R0)
        lc = lmat.sum(axis=0)
        rc = rmat.sum(axis=0)
        diff = rc - lc
        cum = np.cumsum(diff)
        pos_idxs = np.flatnonzero(cum >= theta_E)
        neg_idxs = np.flatnonzero(cum <= -theta_E)
        if pos_idxs.size == 0 and neg_idxs.size == 0:
            # no hit, redo this trial
            continue

        pos_time = pos_idxs[0] * dt if pos_idxs.size > 0 else np.nan
        neg_time = neg_idxs[0] * dt if neg_idxs.size > 0 else np.nan

        # earliest crossing (either bound)
        if pos_idxs.size > 0 and neg_idxs.size > 0:
            first_idx = min(pos_idxs[0], neg_idxs[0])
            first_is_pos = pos_idxs[0] < neg_idxs[0]
        elif pos_idxs.size > 0:
            first_idx = pos_idxs[0]
            first_is_pos = True
        else:
            first_idx = neg_idxs[0]
            first_is_pos = False
        first_time = first_idx * dt
        return s, cum, pos_time, neg_time, first_time, first_is_pos

# Parallelize trials
with ProcessPoolExecutor(max_workers=(os.cpu_count() or 1)) as ex:
    futures = [ex.submit(simulate_one_trial, s, N, T, dt, R0, theta_E) for s in range(N_sim)]
    for fut in tqdm(as_completed(futures), total=N_sim, desc='Simulating', unit='trial'):
        s, cum, pos_time, neg_time, first_time, first_is_pos = fut.result()
        # record per-trial cum and crossing times now that we have a hit
        cum_diffs[s] = cum
        cross_times[s] = first_time
        cross_times_pos[s] = pos_time
        cross_times_neg[s] = neg_time
        if first_is_pos:
            first_pos_times.append(first_time)
        else:
            first_neg_times.append(first_time)

# %%
# DDM

# %%
# 3 plots in a 1x3 layout
fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

# (1) All cum_diffs with thresholds
# Plot only a random sample of up to 50 trajectories to avoid clutter
plot_k = min(5, N_sim)
if N_sim > plot_k:
    _plot_idxs = np.random.default_rng().choice(N_sim, size=plot_k, replace=False)
else:
    _plot_idxs = np.arange(N_sim)
for s in _plot_idxs:
    axes[0].plot(time_bins, np.concatenate(([0], cum_diffs[s])), color='k')
axes[0].axhline(theta_E, color='r', ls='--')
axes[0].axhline(-theta_E, color='r', ls='--')
axes[0].axhline(0, color='gray', lw=0.6, ls=':')
axes[0].set_title('Sample trajectories')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Cum (R-L)')
axes[0].set_ylim(-5,5)

# (2) Crossing-time distributions (split, normalized to fractions of first-hits)
pos_times = np.array(first_pos_times)
neg_times = np.array(first_neg_times)
frac_pos = pos_times.size / N_sim
frac_neg = neg_times.size / N_sim
bins = np.arange(0, T, 0.01)
if pos_times.size > 0:
    pos_pdf, edges = np.histogram(pos_times, bins=bins, density=True)
    pos_pdf *= frac_pos  # area equals fraction of + crossings
    axes[1].stairs(pos_pdf, edges, fill=False, color='tab:red', lw=1.5, label='+θ_E')
if neg_times.size > 0:
    neg_pdf, edges = np.histogram(neg_times, bins=bins, density=True)
    neg_pdf *= -frac_neg  # negative to go below zero; area magnitude equals fraction of - crossings
    axes[1].stairs(neg_pdf, edges, fill=False, color='tab:blue', lw=1.5, label='-θ_E')
axes[1].axhline(0, color='k', lw=0.8)
axes[1].set_title('Crossing-time distributions')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Fraction density (+ above, - below)')
axes[1].legend(frameon=False)

# (3) Fraction of trials crossing +θ_E vs -θ_E
axes[2].bar(['+θ_E', '-θ_E'], [frac_pos, frac_neg], color=['tab:red', 'tab:blue'], alpha=0.8)
axes[2].set_ylim(0, 1)
axes[2].set_ylabel('Fraction of trials')
axes[2].set_title('Crossing fractions')
title_str = f"N neurons={N}, N_sim={N_sim}, Nr0={Nr0:.3g}, R0={R0:.3g}, dt={dt:g}, theta_E  = {theta_E:.3g}"
fig.suptitle(title_str)
fig.savefig('mc_summary.png', dpi=150, bbox_inches='tight')
fig.show()
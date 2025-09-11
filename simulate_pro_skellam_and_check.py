# %% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

N_TRIALS = int(50e3)
R_true=87
L_true=107
theta_true=7
# Globals populated in main() for use by worker processes
skellam_times = None
skellam_choices = None
# Stimulus times
rng = np.random.default_rng(0)
stim_times = rng.exponential(scale=0.4, size=N_TRIALS) + 0.2
def simulate_trial(mu1, mu2, theta, rng=None):
    """
    Simulate one Skellam trial dX = dN1 - dN2 with absorbing boundaries at +/- theta.
    Returns (first_passage_time, choice) where choice is +1 for +theta, -1 for -theta.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = 0
    time = 0.0
    total_rate = mu1 + mu2
    prob_up = mu1 / total_rate
    while abs(x) < theta:
        dt = rng.exponential(1.0 / total_rate)
        time += dt
        if rng.random() < prob_up:
            x += 1
        else:
            x -= 1
    choice = 1 if x >= theta else -1
    return time, choice

# Worker function for parallel simulation of trials
def _worker_chunk(start: int, end: int, seed: int):
    """
    Compute proactive vs reactive outcome for trials in [start, end).
    Returns (times_out, choices_out, proactive_count, reactive_count).
    """
    local_rng = np.random.default_rng(seed)
    size = end - start
    times_out = np.empty(size, dtype=float)
    choices_out = np.empty(size, dtype=int)
    proactive_count = 0
    reactive_count = 0
    for idx, i in enumerate(range(start, end)):
        # proactive trial
        pro_dv = 0.0
        t_pro = t_A_aff
        while True:
            pro_dv += V_A * dt + local_rng.normal(0.0, dB)
            t_pro += dt
            if pro_dv >= theta_A:
                break

        # reactive hit time
        t_skellam = skellam_times[i]
        t_skellam_with_delay = t_skellam - t_E_aff - stim_times[i]
        if t_skellam_with_delay < 0:
            t_skellam_with_delay = np.inf

        # case 1: reactive wins
        if t_skellam_with_delay < t_pro:
            times_out[idx] = t_skellam_with_delay
            choices_out[idx] = skellam_choices[i]
            reactive_count += 1
        # case 2: proactive wins
        elif t_pro < t_skellam_with_delay:
            times_out[idx] = t_pro
            # case 2a: reactive would have hit within go window
            if t_pro < t_skellam_with_delay < t_pro + del_go - t_E_aff:
            # TEMP: NOTE: IS it just del_go?
            # if t_pro < t_skellam_with_delay < t_pro + del_go:

                choices_out[idx] = skellam_choices[i]
            # case 2b: no reactive hit in the window, coin flip
            else:
                choices_out[idx] = 1 if local_rng.random() < 0.5 else -1
            proactive_count += 1
        else:
            raise ValueError(f'unknown case (skellam with delay = {t_skellam_with_delay}, t_pro = {t_pro})')

    return times_out, choices_out, proactive_count, reactive_count

V_A=1.1774684269283593
theta_A=1
t_A_aff=0.04163924630404911
dt = 1e-4
dB = 1e-2

t_E_aff = 0.073
del_go = 1.90

# %%
def cont_ddm_simulate(t_stim):
    AI = 0; DV = 0; t = t_A_aff; dB = dt**0.5
    mu = R_true - L_true
    sigma = np.sqrt(R_true + L_true)
    is_act = 0
    theta = theta_true

    while True:
        AI += V_A*dt + np.random.normal(0, dB)

        if t > t_stim + t_E_aff:
            DV += mu*dt + sigma*np.random.normal(0, dB)
        
        
        t += dt
        
        if DV >= theta:
            choice = +1; RT = t
            break
        elif DV <= -theta:
            choice = -1; RT = t
            break
        
        if AI >= theta_A:
            both_AI_hit_and_EA_hit = 0 # see if both AI and EA hit 
            is_act = 1
            AI_hit_time = t
            while t <= (AI_hit_time + del_go):
                if t > t_stim + t_E_aff: 
                    DV += mu*dt + sigma*np.random.normal(0, dB)
                    if DV >= theta:
                        DV = theta
                        both_AI_hit_and_EA_hit = 1
                        break
                    elif DV <= -theta:
                        DV = -theta
                        both_AI_hit_and_EA_hit = -1
                        break
                t += dt
            
            break
        
        
    if is_act == 1:
        RT = AI_hit_time
        if both_AI_hit_and_EA_hit != 0:
            choice = both_AI_hit_and_EA_hit
        else:
            randomly_choose_up = np.random.rand() >= 0.5
            if randomly_choose_up:
                choice = 1
            else:
                choice = -1
    
    return choice, RT, t_stim

# N_TRIALS cont_DDM sim

def cont_ddm_sim_worker(args):
    rng_seed, = args
    rng = np.random.default_rng(rng_seed)
    results = []
    for _ in range(chunk_size):
        # Set globals like t_stim, Z_E, etc. as needed
        t_stim = rng.choice(stim_times)
        result = cont_ddm_simulate(t_stim)
        results.append(result)
    return results

n_workers = os.cpu_count() or 2
chunk_size = max(1, (N_TRIALS + n_workers - 1) // n_workers)
cont_ddm_results = []

with ProcessPoolExecutor(max_workers=n_workers) as ex:
    futures = []
    for i in range(0, N_TRIALS, chunk_size):
        rng_seed = int(rng.integers(0, 2**63 - 1))
        futures.append(ex.submit(cont_ddm_sim_worker, (rng_seed,)))
    for fut in as_completed(futures):
        cont_ddm_results.extend(fut.result())

cont_ddm_results = np.array(cont_ddm_results[:N_TRIALS])
cont_choices = cont_ddm_results[:,0]
cont_rts = cont_ddm_results[:,1]
cont_stim_times = cont_ddm_results[:,2]

#%%
#  Generate simulated dataset (reactive component)
skellam_times = np.empty(N_TRIALS, dtype=float)
skellam_choices = np.empty(N_TRIALS, dtype=int)
for i in range(N_TRIALS):
    t, ch = simulate_trial(R_true, L_true, theta_true, rng=rng)
    skellam_times[i] = t
    skellam_choices[i] = ch



# Run trials in parallel
pro_and_skellam = np.zeros((N_TRIALS, 2))
proactive_win_count = 0
reactive_win_count = 0

n_workers = os.cpu_count() or 2
# Chunk the work to balance overhead and speed
chunk_size = max(1, (N_TRIALS + n_workers - 1) // n_workers)
jobs = []
with ProcessPoolExecutor(max_workers=n_workers) as ex:
    start_indices = list(range(0, N_TRIALS, chunk_size))
    for start in start_indices:
        end = min(start + chunk_size, N_TRIALS)
        seed = int(rng.integers(0, 2**63 - 1))
        jobs.append((start, end, ex.submit(_worker_chunk, start, end, seed)))

    for start, end, fut in jobs:
        times_out, choices_out, pro_c, re_c = fut.result()
        pro_and_skellam[start:end, 0] = times_out
        pro_and_skellam[start:end, 1] = choices_out
        proactive_win_count += pro_c
        reactive_win_count += re_c

print(f'pro win count = {proactive_win_count}')
print(f'reactive win count = {reactive_win_count}')
# %%
# t
# sample 1000 t_stim from stim_times
from vbmc_skellam_utils import up_or_down_hit_fn
N_theory = int(1e3)
sampled_stim_times = rng.choice(stim_times, size=N_theory, replace=True)
t_pts_wrt_stim = np.arange(-5, 5, 0.05)
up_and_down_density = np.zeros((N_theory, len(t_pts_wrt_stim)))
for idx, t_stim in enumerate(sampled_stim_times):
    t_pts_wrt_fix = t_pts_wrt_stim + t_stim
    up_and_down_density[idx, :] = \
        [ up_or_down_hit_fn(rt, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, R_true, L_true, theta_true, 1) \
        + up_or_down_hit_fn(rt, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, R_true, L_true, theta_true, -1) \
         for rt in t_pts_wrt_fix]

    
# P_joint_rt_choice = up_or_down_hit_fn(rt, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, choice)



# %%
bins = np.arange(-5, 5, 0.05)
plt.hist(pro_and_skellam[:, 0] - stim_times, bins=bins, density=True, histtype='step');
theory_mean = np.mean(up_and_down_density, axis=0)
plt.plot(t_pts_wrt_stim, theory_mean, 'r', alpha=0.5, lw=3, ls='--')
cont_rt_wrt_stim = cont_rts - cont_stim_times
plt.hist(cont_rt_wrt_stim, bins=bins, density=True, histtype='step', color='g');
plt.legend(['simulated', 'theory', 'cont'])
plt.show()
# %%
area_theory = np.trapezoid(theory_mean, t_pts_wrt_stim)
print(f'theory area = {area_theory}')

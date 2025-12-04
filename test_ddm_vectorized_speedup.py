# %%
"""
Benchmark: Original single-trial DDM vs Vectorized DDM
"""
import numpy as np
import time
from joblib import Parallel, delayed
import multiprocessing

# %%
# Parameters from heat_map_ddm_poisson_acc_mean.py
N = 10
rho = 1e-3
lam = 1.3
l = 0.9
Nr0_base = 13.3
abl = 20
ild = 4  # Pick one ILD for testing
dt_sim = 1e-4  # Time step
theta_ddm = 2
T_max = 2  # Max simulation time in seconds

# Calculate rates
r0 = Nr0_base / N
r_db = (2*abl + ild)/2
l_db = (2*abl - ild)/2
pr = (10 ** (r_db/20))
pl = (10 ** (l_db/20))
den = (pr ** (lam * l)) + (pl ** (lam * l))
rr = (pr ** lam) / den
rl = (pl ** lam) / den
r_right = r0 * rr
r_left = r0 * rl

print(f"r_right = {r_right:.4f}, r_left = {r_left:.4f}")
print(f"theta = {theta_ddm}, dt = {dt_sim}, T_max = {T_max}")

# %%
# ============ ORIGINAL: Single trial loop ============
def run_ddm_trial_single(trial_idx, N, r_right, r_left, theta, dt_sim=1e-4):
    """Original single-trial DDM simulation."""
    mu = N * (r_right - r_left)
    sigma = np.sqrt(N * (r_right + r_left))
    dB = np.sqrt(dt_sim)
    max_steps = int(T_max / dt_sim)
    
    position = 0
    for step in range(max_steps):
        position += mu * dt_sim + sigma * np.random.normal(0, dB)
        time_now = (step + 1) * dt_sim
        
        if position >= theta:
            return (time_now, 1)
        elif position <= -theta:
            return (time_now, -1)
    
    return (np.nan, 0)


def simulate_ddm_original(n_trials, use_parallel=True):
    """Original approach with joblib parallelization."""
    if use_parallel:
        results = Parallel(n_jobs=multiprocessing.cpu_count()-2)(
            delayed(run_ddm_trial_single)(i, N, r_right, r_left, theta_ddm, dt_sim)
            for i in range(n_trials)
        )
    else:
        results = [run_ddm_trial_single(i, N, r_right, r_left, theta_ddm, dt_sim) 
                   for i in range(n_trials)]
    
    results = np.array(results)
    return results[:, 0], results[:, 1]


# %%
# ============ VECTORIZED: All trials at once ============
def simulate_ddm_vectorized(n_trials):
    """Vectorized DDM - all trials updated simultaneously."""
    mu = N * (r_right - r_left)
    sigma = np.sqrt(N * (r_right + r_left))
    dB = np.sqrt(dt_sim)
    max_steps = int(T_max / dt_sim)
    
    # State arrays
    position = np.zeros(n_trials)
    rt = np.full(n_trials, np.nan)
    choice = np.zeros(n_trials)
    active = np.ones(n_trials, dtype=bool)
    
    for step in range(max_steps):
        n_active = active.sum()
        if n_active == 0:
            break
        
        # Update all active trials at once
        position[active] += mu * dt_sim + sigma * np.random.normal(0, dB, size=n_active)
        time_now = (step + 1) * dt_sim
        
        # Check thresholds
        hit_upper = active & (position >= theta_ddm)
        hit_lower = active & (position <= -theta_ddm)
        
        rt[hit_upper] = time_now
        choice[hit_upper] = 1
        rt[hit_lower] = time_now
        choice[hit_lower] = -1
        
        active[hit_upper | hit_lower] = False
    
    return rt, choice


# %%
# ============ BENCHMARK ============
n_trials_test = 1000  # Start with 1K for quick test

print(f"\n{'='*60}")
print(f"BENCHMARK: {n_trials_test} trials")
print(f"{'='*60}")

# Warm up
_ = simulate_ddm_vectorized(100)

# Test vectorized
t0 = time.time()
rt_vec, choice_vec = simulate_ddm_vectorized(n_trials_test)
t_vectorized = time.time() - t0
print(f"\nVectorized:     {t_vectorized:.3f}s")

# Test original (no parallel) - only if small
if n_trials_test <= 500:
    t0 = time.time()
    rt_orig_seq, choice_orig_seq = simulate_ddm_original(n_trials_test, use_parallel=False)
    t_original_seq = time.time() - t0
    print(f"Original (seq): {t_original_seq:.3f}s  | Speedup: {t_original_seq/t_vectorized:.1f}x")

# Test original with parallel
t0 = time.time()
rt_orig, choice_orig = simulate_ddm_original(n_trials_test, use_parallel=True)
t_original_par = time.time() - t0
print(f"Original (par): {t_original_par:.3f}s  | Speedup: {t_original_par/t_vectorized:.1f}x")

# %%
# Verify results are statistically similar
print(f"\n{'='*60}")
print("VERIFICATION (results should be similar)")
print(f"{'='*60}")
print(f"Vectorized  - Mean RT: {np.nanmean(rt_vec):.4f}, Accuracy: {np.nanmean(choice_vec == 1):.3f}")
print(f"Original    - Mean RT: {np.nanmean(rt_orig):.4f}, Accuracy: {np.nanmean(choice_orig == 1):.3f}")

# %%
# Test with 10K trials (the actual use case)
print(f"\n{'='*60}")
print(f"FULL SCALE TEST: 10,000 trials")
print(f"{'='*60}")

t0 = time.time()
rt_vec_10k, choice_vec_10k = simulate_ddm_vectorized(10_000)
t_vec_10k = time.time() - t0
print(f"Vectorized: {t_vec_10k:.3f}s")

t0 = time.time()
rt_orig_10k, choice_orig_10k = simulate_ddm_original(10_000, use_parallel=True)
t_orig_10k = time.time() - t0
print(f"Original (parallel): {t_orig_10k:.3f}s")

print(f"\nSpeedup: {t_orig_10k/t_vec_10k:.1f}x faster")
print(f"\nVectorized  - Mean RT: {np.nanmean(rt_vec_10k):.4f}, Accuracy: {np.nanmean(choice_vec_10k == 1):.3f}")
print(f"Original    - Mean RT: {np.nanmean(rt_orig_10k):.4f}, Accuracy: {np.nanmean(choice_orig_10k == 1):.3f}")

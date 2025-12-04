# %%
"""
Benchmark: Poisson simulation timing
"""
import numpy as np
import pandas as pd
import time
from joblib import Parallel, delayed
import multiprocessing
from corr_poisson_utils_subtractive import run_poisson_trial

# %%
# Parameters (same as analysis file)
N = 10
rho = 1e-3  # correlation
lam = 1.3
l = 0.9
Nr0_base = 13.3
abl = 20
ild = 4
theta_poisson = 3  # typical optimized value

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
print(f"theta = {theta_poisson}, rho (correlation) = {rho}")

# %%
# ============ TIME SINGLE TRIAL ============
print(f"\n{'='*60}")
print("SINGLE TRIAL TIMING")
print(f"{'='*60}")

n_single_trials = 100
times = []
for i in range(n_single_trials):
    t0 = time.time()
    _ = run_poisson_trial(N, rho, r_right, r_left, theta_poisson)
    times.append(time.time() - t0)

mean_time = np.mean(times) * 1000  # ms
std_time = np.std(times) * 1000
print(f"Single trial: {mean_time:.2f} ± {std_time:.2f} ms")
print(f"Estimated time for 10K trials (sequential): {mean_time * 10000 / 1000:.1f}s")

# %%
# ============ PARALLEL TIMING ============
print(f"\n{'='*60}")
print("PARALLEL TIMING (like your current code)")
print(f"{'='*60}")

def run_poisson_parallel(n_trials):
    results = Parallel(n_jobs=multiprocessing.cpu_count()-2)(
        delayed(run_poisson_trial)(N, rho, r_right, r_left, theta_poisson)
        for _ in range(n_trials)
    )
    return np.array(results)

# 1K trials
t0 = time.time()
results_1k = run_poisson_parallel(1000)
t_1k = time.time() - t0
print(f"1K trials (parallel):  {t_1k:.2f}s")

# 10K trials
t0 = time.time()
results_10k = run_poisson_parallel(10_000)
t_10k = time.time() - t0
print(f"10K trials (parallel): {t_10k:.2f}s")

# %%
# ============ IDENTIFY BOTTLENECK ============
print(f"\n{'='*60}")
print("PROFILING: Where is the time spent?")
print(f"{'='*60}")

# Profile one trial step by step
rng = np.random.default_rng(42)
T = 50

# Step 1: Generate spikes
t0 = time.time()
for _ in range(100):
    # Simplified spike generation (c=rho is small, so mostly independent)
    right_spikes = []
    left_spikes = []
    for i in range(N):
        n_right = rng.poisson(r_right * T)
        n_left = rng.poisson(r_left * T)
        right_spikes.append(np.sort(rng.random(n_right) * T))
        left_spikes.append(np.sort(rng.random(n_left) * T))
t_spikes = (time.time() - t0) / 100 * 1000
print(f"Spike generation: {t_spikes:.3f} ms")

# Step 2: Concatenate
t0 = time.time()
for _ in range(100):
    all_right = np.concatenate(right_spikes)
    all_left = np.concatenate(left_spikes)
    all_times = np.concatenate([all_right, all_left])
    all_evidence = np.concatenate([np.ones(len(all_right)), -np.ones(len(all_left))])
t_concat = (time.time() - t0) / 100 * 1000
print(f"Concatenation: {t_concat:.3f} ms")

# Step 3: Pandas DataFrame + groupby (THE BOTTLENECK!)
t0 = time.time()
for _ in range(100):
    events_df = pd.DataFrame({'time': all_times, 'evidence_jump': all_evidence})
    evidence_events = events_df.groupby('time')['evidence_jump'].sum().reset_index()
t_pandas = (time.time() - t0) / 100 * 1000
print(f"Pandas groupby: {t_pandas:.3f} ms  <-- BOTTLENECK!")

# Step 4: Cumsum + threshold check
t0 = time.time()
for _ in range(100):
    event_times = evidence_events['time'].values
    event_jumps = evidence_events['evidence_jump'].values
    dv = np.cumsum(event_jumps)
    pos_cross = np.where(dv >= theta_poisson)[0]
    neg_cross = np.where(dv <= -theta_poisson)[0]
t_cumsum = (time.time() - t0) / 100 * 1000
print(f"Cumsum + threshold: {t_cumsum:.3f} ms")

print(f"\nTotal per trial: {t_spikes + t_concat + t_pandas + t_cumsum:.2f} ms")

# %%
# ============ FASTER VERSION WITHOUT PANDAS ============
print(f"\n{'='*60}")
print("OPTIMIZED VERSION (no pandas)")
print(f"{'='*60}")

def run_poisson_trial_fast(N, c, r_right, r_left, theta, rng=None):
    """Faster Poisson trial without pandas overhead."""
    if rng is None:
        rng = np.random.default_rng()
    
    T = 50
    
    # Generate spikes (simplified for c ≈ 0)
    if c < 0.01:  # Essentially independent
        right_spikes = []
        left_spikes = []
        for i in range(N):
            n_r = rng.poisson(r_right * T)
            n_l = rng.poisson(r_left * T)
            right_spikes.append(rng.random(n_r) * T)
            left_spikes.append(rng.random(n_l) * T)
        all_right = np.concatenate(right_spikes) if right_spikes else np.array([])
        all_left = np.concatenate(left_spikes) if left_spikes else np.array([])
    else:
        # Use original correlated method
        from corr_poisson_utils_subtractive import generate_correlated_pool
        right_pool = generate_correlated_pool(N, c, r_right, T, rng)
        left_pool = generate_correlated_pool(N, c, r_left, T, rng)
        all_right = np.concatenate(list(right_pool.values()))
        all_left = np.concatenate(list(left_pool.values()))
    
    if len(all_right) == 0 and len(all_left) == 0:
        return (np.nan, 0)
    
    # Combine and sort (INSTEAD of pandas groupby)
    all_times = np.concatenate([all_right, all_left])
    all_evidence = np.concatenate([np.ones(len(all_right)), -np.ones(len(all_left))])
    
    # Sort by time
    sort_idx = np.argsort(all_times)
    sorted_times = all_times[sort_idx]
    sorted_evidence = all_evidence[sort_idx]
    
    # Cumsum and find first crossing
    dv = np.cumsum(sorted_evidence)
    
    pos_cross = np.where(dv >= theta)[0]
    neg_cross = np.where(dv <= -theta)[0]
    
    first_pos = pos_cross[0] if len(pos_cross) > 0 else np.inf
    first_neg = neg_cross[0] if len(neg_cross) > 0 else np.inf
    
    if first_pos < first_neg:
        return (sorted_times[first_pos], 1)
    elif first_neg < first_pos:
        return (sorted_times[first_neg], -1)
    else:
        return (np.nan, 0)


# Time single trial
times_fast = []
for i in range(100):
    t0 = time.time()
    _ = run_poisson_trial_fast(N, rho, r_right, r_left, theta_poisson)
    times_fast.append(time.time() - t0)

mean_time_fast = np.mean(times_fast) * 1000
print(f"Fast single trial: {mean_time_fast:.2f} ms (vs {mean_time:.2f} ms original)")
print(f"Speedup: {mean_time/mean_time_fast:.1f}x")

# %%
# 10K trials with fast version
def run_poisson_fast_parallel(n_trials):
    results = Parallel(n_jobs=multiprocessing.cpu_count()-2)(
        delayed(run_poisson_trial_fast)(N, rho, r_right, r_left, theta_poisson)
        for _ in range(n_trials)
    )
    return np.array(results)

t0 = time.time()
results_fast_10k = run_poisson_fast_parallel(10_000)
t_fast_10k = time.time() - t0
print(f"\n10K trials (fast parallel): {t_fast_10k:.2f}s (vs {t_10k:.2f}s original)")
print(f"Speedup: {t_10k/t_fast_10k:.1f}x")

# Verify results similar
rts_orig = results_10k[:, 0]
rts_fast = results_fast_10k[:, 0]
print(f"\nOriginal  - Mean RT: {np.nanmean(rts_orig):.4f}, Accuracy: {np.nanmean(results_10k[:, 1] == 1):.3f}")
print(f"Fast      - Mean RT: {np.nanmean(rts_fast):.4f}, Accuracy: {np.nanmean(results_fast_10k[:, 1] == 1):.3f}")

# Test - Simulate poisson spike trains
# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
np.random.seed(42)
param_type = 'norm' # vanilla


if param_type == 'norm':
    N = 5
    Nr0 = 1000/65
    theta_E = 2.8
elif param_type == 'vanilla':
    N = 10
    Nr0 = 1000/0.3
    theta_E = 38

# print the params
print(f'Param type: {param_type}')
print(f"N: {N}")
print(f"Nr0: {Nr0}")
print(f"theta_E: {theta_E}")
# %%
T = 1
dt = 1e-3

def poisson_spk_train(rate, T, dt):
    rng = np.random.default_rng()
    spk_times = []
    t = 0
    while t <= T:
        t += rng.exponential(1/rate)
        if t > T:
            break
        spk_times.append(t)
        
    # convert spk times to bin indices
    spk_times = np.array(spk_times)
    spk_bins = np.floor(spk_times/dt).astype(int)
    return spk_times

def gen_side_spk_trains(N, T, dt, rate_each):
    """Generate an N x (T/dt) 0/1 spike train matrix and list of spike time arrays."""
    n_bins = int(round(T / dt))
    times_list = [poisson_spk_train(rate_each, T, dt) for _ in range(N)]
    mat = np.zeros((N, n_bins), dtype=int)
    for i, times in enumerate(times_list):
        if times.size == 0:
            continue
        bins = np.floor(times / dt).astype(int)
        mat[i, bins] = 1
    return mat, times_list

rate_each = Nr0 / N
spk_times = [poisson_spk_train(rate_each, T, dt) for _ in range(N)]
# Check if ISI is exponential
plt.hist(np.diff(np.concatenate(spk_times)), bins=100)  
plt.show()
offsets = np.arange(N)
# plotting raster
plt.eventplot(spk_times, colors='k', lineoffsets=offsets, linelengths=0.8, orientation='horizontal');
plt.ylim(-0.5, N - 0.5)
plt.yticks(range(N), range(1, N+1))
plt.xlabel('Time (s)')
plt.ylabel('Train #')
plt.tight_layout()
print(f'Generated {N} spike trains at {rate_each:.4g} Hz each. Total spikes: {sum(len(st) for st in spk_times)}')

# %%
# Total number of spikes(all neurons combined) ~ Nr0, check
total = 0
for i in range(N):
    total += len(spk_times[i])
    print(f"Train {i+1}: {len(spk_times[i])} spikes")
print(f"Total spikes: {total}")
print(f'N r0: {Nr0:.4g} Hz')

# %%
# 1. Left and right spike trains
left_spk_train, left_times = gen_side_spk_trains(N, T, dt, rate_each)
right_spk_train, right_times = gen_side_spk_trains(N, T, dt, rate_each)

# Checks (single realization)
left_total = int(left_spk_train.sum())
right_total = int(right_spk_train.sum())
print(f"Left total spikes in 1 s: {left_total}")
print(f"Right total spikes in 1 s: {right_total}")

# Monte Carlo estimate over multiple runs
reps = 50
left_totals = []
right_totals = []
for _ in range(reps):
    lmat, _ = gen_side_spk_trains(N, T, dt, rate_each)
    rmat, _ = gen_side_spk_trains(N, T, dt, rate_each)
    left_totals.append(int(lmat.sum()))
    right_totals.append(int(rmat.sum()))

print(f"Avg left spikes over {reps} runs: {np.mean(left_totals):.2f} ± {np.std(left_totals):.2f}")
print(f"Avg right spikes over {reps} runs: {np.mean(right_totals):.2f} ± {np.std(right_totals):.2f}")

# Print R0 (using Nr0 as the parameter value)
print(f"R0: {Nr0:.4g} Hz")
# %%
# Per-dt totals and cumulative difference (Right - Left)
# Use the left/right spike trains generated above
left_counts = left_spk_train.sum(axis=0)   # shape: (int(T/dt),)
right_counts = right_spk_train.sum(axis=0) # shape: (int(T/dt),)
diff_counts = right_counts - left_counts   # per-dt difference
cum_diff = np.cumsum(diff_counts)          # cumulative sum over time
time = np.arange(cum_diff.size) * dt

plt.figure()
plt.plot(time, cum_diff, 'k-')
plt.axhline(0, color='gray', lw=0.8, ls='--')
plt.xlabel('Time (s)')
plt.ylabel('Cumulative (Right - Left)')
plt.axhline(theta_E, color='gray', lw=0.8, ls='--')
plt.title('Cumulative spike difference (Right - Left)')
plt.tight_layout()

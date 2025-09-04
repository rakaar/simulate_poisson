# spike timings and spike train generating funcs
import numpy as np

def poisson_spk_train(rate, T):
    rng = np.random.default_rng()
    # Vectorized homogeneous Poisson process: sample N~Poisson(rate*T)
    # and place N event times as the sorted order statistics of Uniform(0, T).
    n_spikes = rng.poisson(rate * T)
    if n_spikes == 0:
        return np.empty(0, dtype=float)
    spk_times = np.sort(rng.random(n_spikes) * T)
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


# %%

# generate correlated spike trains
# %%
import numpy as np

N = 100 # number of neurons
c = 0.8 # correlation coefficient
r = 30 # rate of each neuron /s
T = 3 # duration

# Generate correlateed spike train timings
rng = np.random.default_rng()
corr_spk_times_dict = {}
uncorr_spk_times_dict = {}

n_common_spikes = rng.poisson(c*r*T)
common_spk_timings = np.sort(rng.random(n_common_spikes) * T)

for i in range(N):
    n_spikes_for_corr = rng.poisson((1-c) * r * T)
    spk_times_for_corr = np.sort(rng.random(n_spikes_for_corr) * T)
    corr_spk_times_dict[i] = np.sort(np.concatenate([common_spk_timings, spk_times_for_corr]))
    uncorr_spk_times_dict[i] = np.sort(rng.random(rng.poisson(r * T)) * T)

# Test correlation
corr_value_corr_spk_trains = []
corr_value_uncorr_spk_trains = []

for i in range(N-1):
    for j in range(i+1, N):
        # corr



# generate correlated spike trains
# %%
import numpy as np
from correlated_spk_utils import calc_ccvf
import matplotlib.pyplot as plt

N = 3 # number of neurons
c = 0.4 # correlation coefficient
r = 30 # rate of each neuron /s
T = 500 # duration

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

bin_size = 1e-4
window_size = 0.1

# Calculate number of combinations (N choose 2)
import math
n_combinations = math.comb(N, 2)

# Create a single figure with n_combinations rows and 2 columns
fig, axes = plt.subplots(n_combinations, 2, figsize=(12, 4*n_combinations))

# If N=2, axes won't be a 2D array, so we need to handle that case
if n_combinations == 1:
    axes = [axes]

plot_idx = 0
for i in range(N-1):
    for j in range(i+1, N):
        # corr btn corr spike trains
        ccvf_y_corr, ccvf_x_corr, corr_1 = calc_ccvf(corr_spk_times_dict[i], corr_spk_times_dict[j], T, bin_size, window_size)
        # corr btn uncorr spike trains
        ccvf_y_uncorr, ccvf_x_uncorr, corr_2 = calc_ccvf(uncorr_spk_times_dict[i], uncorr_spk_times_dict[j], T, bin_size, window_size)
        
        # Plot on the appropriate subplot
        axes[plot_idx][0].plot(ccvf_x_corr, ccvf_y_corr)
        axes[plot_idx][0].set_title(f'Correlated pair N1,N2= {i} {j}, corr={corr_1:.2f}, ideal c={c}')
        axes[plot_idx][0].set_ylim(0, 5000)
        axes[plot_idx][0].set_ylabel('CCVF')
        
        axes[plot_idx][1].plot(ccvf_x_uncorr, ccvf_y_uncorr)
        axes[plot_idx][1].set_title(f'UNcorr pair N1,N2={i} {j}, corr={corr_2:.2f}')
        axes[plot_idx][1].set_ylabel('CCVF')
        
        plot_idx += 1

plt.tight_layout()
plt.show()
        
        
# %%

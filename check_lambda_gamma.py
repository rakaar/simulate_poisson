# Gamma from psychometric curve
# %% 
import numpy as np
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from sim_utils import poisson_spk_train

# %% 
# Params
T = 2

Nr0 = 1000/65 # 15.38
theta_E = 2.8

N_sim = int(100e3)

ABL_arr = [60]
ILD_arr = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
chi = 17.37
rate_lambda = 2.13
l = 0.9
 
output_filename = f'discrete_ddm_results_lambda_{rate_lambda}_l_{l}_ABL_{ABL_arr[0]}.pkl'

if os.path.exists(output_filename):
    print(f'Loading existing results from {output_filename}')
    with open(output_filename, 'rb') as f:
        discrete_ddm_results = pickle.load(f)
else:
    print(f'Running simulation for rate_lambda={rate_lambda}, l={l}')
    discrete_ddm_results = {}
    
    for ABL in ABL_arr:
        for ILD in ILD_arr:
            print(f'  Running simulation for ABL={ABL}, ILD={ILD}')
            stimulus_pair = (ABL, ILD)

            # Firing rate terms
            SL_R = ABL + (ILD / 2)
            SL_L = ABL - (ILD / 2)
            common_rate_denom = (10**(rate_lambda * l * SL_R / 20)) + (10**(rate_lambda * l * SL_L / 20))
            right_rate_num = 10**(rate_lambda * (SL_R / 20))
            left_rate_num = 10**(rate_lambda * (SL_L / 20))
            right_rate = Nr0 * (right_rate_num / common_rate_denom)
            left_rate = Nr0 * (left_rate_num / common_rate_denom)

            # #### Discrete DDM ##
            first_pos_times = []
            first_neg_times = []
            for s in range(N_sim):
                while True:
                    r_times = poisson_spk_train(right_rate, T)
                    l_times = poisson_spk_train(left_rate, T)
                    if r_times.size == 0 and l_times.size == 0:
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
                        break
            
            discrete_ddm_results[stimulus_pair] = {
                'pos_times': np.array(first_pos_times),
                'neg_times': np.array(first_neg_times)
            }

    # Save the results for the current lambda and l
    with open(output_filename, 'wb') as f:
        pickle.dump(discrete_ddm_results, f)
    print(f'Saved results to {output_filename}')

# %%
# Plot the psychometric curve
import matplotlib.pyplot as plt

# Load the results
with open(output_filename, 'rb') as f:
    results = pickle.load(f)

ild_arr_sorted = sorted(ILD_arr)
prob_upper_bound = []

for ild in ild_arr_sorted:
    # ABL is fixed at 40 in this script
    stimulus_pair = (ABL_arr[0], ild)
    
    res = results[stimulus_pair]
    pos_times = res['pos_times']
    neg_times = res['neg_times']
    
    total_trials = len(pos_times) + len(neg_times)
    if total_trials == 0:
        prob = np.nan
    else:
        prob = len(pos_times) / total_trials
    prob_upper_bound.append(prob)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(ild_arr_sorted, prob_upper_bound, 'o-', label=f'ABL={ABL_arr[0]}, λ={rate_lambda}, l={l}')
plt.xlabel('ILD (dB)')
plt.ylabel('P(choose right)')
plt.title('Psychometric Curve')
plt.grid(True)
plt.ylim([0, 1])
plt.legend()

plot_filename = f'psychometric_curve_lambda_{rate_lambda}_l_{l}.png'
plt.savefig(plot_filename)
print(f'Saved plot to {plot_filename}')
# %%
# Calculate gamma for each ILD
gamma_values = []
ild_for_gamma = []

# Add a small epsilon to prevent log(0) errors
epsilon = 1e-9

for ild, p in zip(ild_arr_sorted, prob_upper_bound):
    if not np.isnan(p) and p > epsilon and p < 1 - epsilon:
        gamma = -0.5 * np.log((1 - p) / p)
        gamma_values.append(gamma)
        ild_for_gamma.append(ild)

# Calculate theoretical gamma
ild_theory_range = np.linspace(min(ild_for_gamma), max(ild_for_gamma), 200)
theoretical_gamma = theta_E * np.tanh(rate_lambda * ild_theory_range / chi)

# Plotting gamma vs ILD
plt.figure(figsize=(8, 6))
plt.plot(ild_for_gamma, gamma_values, 'o', label='0.5 ln(p/(1-p))')
plt.plot(ild_theory_range, theoretical_gamma, '-', label='thetaE tanh(.)')
plt.xlabel('ILD (dB)')
plt.ylabel('Gamma (γ)')
plt.title('Gamma vs. ILD: Simulated vs. Theoretical')
# plt.grid(True)
plt.legend()

gamma_plot_filename = f'gamma_vs_ild_lambda_{rate_lambda}_l_{l}.png'
plt.savefig(gamma_plot_filename)
print(f'Saved gamma plot to {gamma_plot_filename}')

# %%

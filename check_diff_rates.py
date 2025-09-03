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

ABL_arr = [20, 40, 60]
ILD_arr = [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
chi = 17.37
rate_lambda = 2.13
l = 0.9
 
scaling_factors = [1, 2, 5, 10, 15]
output_filename = f'discrete_ddm_results_lambda_R_L_scaled.pkl'

if os.path.exists(output_filename):
    print(f'Loading existing results from {output_filename}')
    with open(output_filename, 'rb') as f:
        all_results = pickle.load(f)
else:
    print(f'No existing results file found. Starting fresh.')
    all_results = {}

for sf in scaling_factors:
    if sf in all_results:
        print(f'Skipping existing results for scaling_factor={sf}')
        continue

    print(f'Running simulation for scaling_factor={sf}, rate_lambda={rate_lambda}, l={l}')
    sf_results = {}
    
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

            # Scale the rates
            right_rate *= sf
            left_rate *= sf

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
            
            sf_results[stimulus_pair] = {
                'pos_times': np.array(first_pos_times),
                'neg_times': np.array(first_neg_times)
            }
    
    all_results[sf] = sf_results

# Save the results for all scaling factors
with open(output_filename, 'wb') as f:
    pickle.dump(all_results, f)
print(f'Saved all results to {output_filename}')
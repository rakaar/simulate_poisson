# Poisson DDM - efficient via spk times
# %%
import numpy as np
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from sim_utils import poisson_spk_train

# %% 
# Params
T = 2
param_type = 'vanilla'
if param_type == 'norm':
    rate_lambda = 2.3
    Nr0 = 1000/65 # 15.38
    theta_E = 2.8
    l  = 0.9
elif param_type == 'vanilla':
    rate_lambda = 0.09
    Nr0 = 1000/0.32
    theta_E = 38.4
    l  = 0

N_sim = int(25e3)

ABL_arr = [20, 60]
ILD_arr = [-16,-4, -2, 2, 4, 16]
chi = 17.37

discrete_ddm_results = {}
continuous_ddm_results = {}

for ABL in ABL_arr:
    for ILD in ILD_arr:
        print(f'Running simulation for ABL={ABL}, ILD={ILD}')
        stimulus_pair = (ABL, ILD)

        # Firing rate terms
        SL_R = ABL + (ILD / 2)
        SL_L = ABL - (ILD / 2)
        if param_type == 'norm':
            common_rate_denom = (10**(rate_lambda * l * SL_R / 20)) + (10**(rate_lambda * l * SL_L / 20))
        else:
            common_rate_denom = 1
        right_rate_num = 10**(rate_lambda * (SL_R / 20))
        left_rate_num = 10**(rate_lambda * (SL_L / 20))
        right_rate = Nr0 * (right_rate_num / common_rate_denom)
        left_rate = Nr0 * (left_rate_num / common_rate_denom)

        # DDM terms
        if param_type == 'norm':
            scaling_term = (10**(rate_lambda * (1 - l) * ABL / 20)) * Nr0
        else:
            scaling_term = (10**(rate_lambda * (1 - l) * ABL / 20)) * Nr0 * 2
        hyp_arg_term = rate_lambda * ILD / chi
        hyp_arg_term_with_l = rate_lambda * l * ILD / chi
        mu = scaling_term * np.sinh(hyp_arg_term) / np.cosh(hyp_arg_term_with_l)
        if np.isnan(mu) and ABL == 0:
            mu = 0
        sigma_sq = scaling_term * np.cosh(hyp_arg_term) / np.cosh(hyp_arg_term_with_l)

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

        # #### Continuous DDM ##
        dt = 1e-4
        n_steps = int(T / dt)
        rng = np.random.default_rng()

        noise = (sigma_sq**0.5) * rng.normal(0, dt**0.5, size=(N_sim, n_steps))
        increments = mu * dt + noise
        dv_paths = np.cumsum(increments, axis=1)

        pos_hit_indices = np.argmax(dv_paths >= theta_E, axis=1)
        neg_hit_indices = np.argmax(dv_paths <= -theta_E, axis=1)

        pos_crossed = dv_paths[np.arange(N_sim), pos_hit_indices] >= theta_E
        neg_crossed = dv_paths[np.arange(N_sim), neg_hit_indices] <= -theta_E

        pos_hit_indices[~pos_crossed] = n_steps
        neg_hit_indices[~neg_crossed] = n_steps

        first_hit_indices = np.minimum(pos_hit_indices, neg_hit_indices)

        valid_hits = first_hit_indices < n_steps
        valid_indices = np.where(valid_hits)[0]
        first_hit_indices = first_hit_indices[valid_hits]

        is_pos_hit = pos_hit_indices[valid_hits] == first_hit_indices

        cont_pos_times = (first_hit_indices[is_pos_hit] + 1) * dt
        cont_neg_times = (first_hit_indices[~is_pos_hit] + 1) * dt
        
        continuous_ddm_results[stimulus_pair] = {
            'pos_times': cont_pos_times,
            'neg_times': cont_neg_times
        }

# Save results to a pickle file
results_to_save = {
    'discrete_ddm': discrete_ddm_results,
    'continuous_ddm': continuous_ddm_results
}

with open(f'ddm_simulation_results_{param_type}.pkl', 'wb') as f:
    pickle.dump(results_to_save, f)

print(f"Simulations complete. Results saved to ddm_simulation_results_{param_type}.pkl")

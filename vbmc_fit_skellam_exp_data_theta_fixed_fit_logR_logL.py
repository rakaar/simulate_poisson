# %%
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from joblib import Parallel, delayed
from scipy.integrate import trapezoid as trapz
from pyvbmc import VBMC
from scipy.integrate import cumulative_trapezoid as cumtrapz
import pickle
import io
import sys
import contextlib
from collections import defaultdict
from vbmc_skellam_utils import cum_pro_and_reactive_trunc_fn, up_or_down_hit_fn, up_or_down_hit_truncated_proactive_V2_fn   

# %%
# collect animals
# --- Get Batch-Animal Pairs ---
# DESIRED_BATCHES = ['SD', 'LED34', 'LED6', 'LED8', 'LED7', 'LED34_even']
DESIRED_BATCHES = ['LED8']
batch_dir = '/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/batch_csvs'
batch_files = [f'batch_{batch_name}_valid_and_aborts.csv' for batch_name in DESIRED_BATCHES]

merged_data = pd.concat([
    pd.read_csv(os.path.join(batch_dir, fname)) for fname in batch_files if os.path.exists(os.path.join(batch_dir, fname))
], ignore_index=True)

merged_valid = merged_data[merged_data['success'].isin([1, -1])].copy()

# --- Print animal table ---
batch_animal_pairs = sorted(list(map(tuple, merged_valid[['batch_name', 'animal']].drop_duplicates().values)))

print(f"Found {len(batch_animal_pairs)} batch-animal pairs from {len(set(p[0] for p in batch_animal_pairs))} batches:")

if batch_animal_pairs:
    batch_to_animals = defaultdict(list)
    for batch, animal in batch_animal_pairs:
        # Ensure animal is a string and we don't add duplicates
        animal_str = str(animal)
        if animal_str == '112':
            batch_to_animals[batch].append(animal_str)

    # Determine column widths for formatting
    max_batch_len = max(len(b) for b in batch_to_animals.keys()) if batch_to_animals else 0
    animal_strings = {b: ', '.join(sorted(a)) for b, a in batch_to_animals.items()}
    max_animals_len = max(len(s) for s in animal_strings.values()) if animal_strings else 0

    # Header
    print(f"{'Batch':<{max_batch_len}}  {'Animals'}")
    print(f"{'=' * max_batch_len}  {'=' * max_animals_len}")

    # Rows
    for batch in sorted(animal_strings.keys()):
        animals_str = animal_strings[batch]
        print(f"{batch:<{max_batch_len}}  {animals_str}")

# %%
def get_params_from_animal_pkl_file(batch_name, animal_id):
    pkl_file = f'/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/results_{batch_name}_animal_{animal_id}.pkl'
    with open(pkl_file, 'rb') as f:
        fit_results_data = pickle.load(f)
    vbmc_aborts_param_keys_map = {
        'V_A_samples': 'V_A',
        'theta_A_samples': 'theta_A',
        't_A_aff_samp': 't_A_aff'
    }
    vbmc_vanilla_tied_param_keys_map = {
        'rate_lambda_samples': 'rate_lambda',
        'T_0_samples': 'T_0',
        'theta_E_samples': 'theta_E',
        'w_samples': 'w',
        't_E_aff_samples': 't_E_aff',
        'del_go_samples': 'del_go'
    }
    vbmc_norm_tied_param_keys_map = {
        **vbmc_vanilla_tied_param_keys_map,
        'rate_norm_l_samples': 'rate_norm_l'
    }
    abort_keyname = "vbmc_aborts_results"
    if MODEL_TYPE == 'vanilla':
        tied_keyname = "vbmc_vanilla_tied_results"
        tied_param_keys_map = vbmc_vanilla_tied_param_keys_map
        is_norm = False
    elif MODEL_TYPE == 'norm':
        tied_keyname = "vbmc_norm_tied_results"
        tied_param_keys_map = vbmc_norm_tied_param_keys_map
        is_norm = True
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
    abort_params = {}
    tied_params = {}
    rate_norm_l = 0
    if abort_keyname in fit_results_data:
        abort_samples = fit_results_data[abort_keyname]
        for param_samples_name, param_label in vbmc_aborts_param_keys_map.items():
            abort_params[param_label] = np.mean(abort_samples[param_samples_name])
    if tied_keyname in fit_results_data:
        tied_samples = fit_results_data[tied_keyname]
        for param_samples_name, param_label in tied_param_keys_map.items():
            tied_params[param_label] = np.mean(tied_samples[param_samples_name])
        if is_norm:
            rate_norm_l = tied_params.get('rate_norm_l', np.nan)
        else:
            rate_norm_l = 0
    else:
        print(f"Warning: {tied_keyname} not found in pickle for {batch_name}, {animal_id}")
    return abort_params, tied_params, rate_norm_l, is_norm

# %%
# VBMC funcs
def compute_loglike_trial(row, mu1, mu2, theta_E):
        # data
        c_A_trunc_time = 0.3
        if 'timed_fix' in row:
            rt = row['timed_fix']
        else:
            rt = row['TotalFixTime']
        
        t_stim = row['intended_fix']

        response_poke = row['response_poke']
        choice = 2*response_poke - 5

        
        trunc_factor_p_joint = cum_pro_and_reactive_trunc_fn(
                            t_stim + 1, c_A_trunc_time,
                            V_A, theta_A, t_A_aff,
                            t_stim, t_E_aff, mu1, mu2, theta_E) - \
                        cum_pro_and_reactive_trunc_fn(
                            t_stim, c_A_trunc_time,
                            V_A, theta_A, t_A_aff,
                            t_stim, t_E_aff, mu1, mu2, theta_E)

        
        P_joint_rt_choice = up_or_down_hit_truncated_proactive_V2_fn(np.array([rt]), V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, choice)[0]
        

        
        P_joint_rt_choice_trunc = max(P_joint_rt_choice / (trunc_factor_p_joint + 1e-10), 1e-10)
        
        wt_log_like = np.log(P_joint_rt_choice_trunc)

        return wt_log_like




"""
Fit (logR, logL) with theta fixed per condition (ABL, ILD) on real data.
Uniform priors over logR and logL using the same bounds as in the SIM-based script.
"""

# Bounds and plausible bounds for logR, logL
logR_bounds = [0.5, 4]
logL_bounds = [0.5, 4]

logR_plausible_bounds = [1, 3.8]
logL_plausible_bounds = [1, 3.8]

logR_width = logR_bounds[1] - logR_bounds[0]
logL_width = logL_bounds[1] - logL_bounds[0]

def vbmc_prior_fn(params):
    # Uniform proper log-prior over (logR, logL)
    return - (np.log(logR_width) + np.log(logL_width))

# %%
all_ABLs_cond = [20, 40, 60]
all_ILDs_cond = [1, -1, 2, -2, 4, -4, 8, -8, 16, -16]
vbmc_fit_saving_path = 'real_data_cond_wise_theta_fixed'
K_max = 10

for batch_name, animal_id in batch_animal_pairs:

    print('##########################################')
    print(f'Batch: {batch_name}, Animal: {animal_id}')
    print('##########################################')

    MODEL_TYPE = 'vanilla'
    abort_params, vanilla_tied_params, rate_norm_l, is_norm = get_params_from_animal_pkl_file(batch_name, animal_id)
    MODEL_TYPE = 'norm'
    abort_params, norm_tied_params, rate_norm_l, is_norm = get_params_from_animal_pkl_file(batch_name, animal_id)
    
    # take t_E_aff, del_go avg from both vanilla and norm tied params
    t_E_aff = (vanilla_tied_params['t_E_aff'] + norm_tied_params['t_E_aff']) / 2
    del_go = (vanilla_tied_params['del_go'] + norm_tied_params['del_go']) / 2
    
    print(f"Batch: {batch_name}, Animal: {animal_id}")
    print(f"t_E_aff: {t_E_aff}")
    print(f"t_E_aff: {t_E_aff}")
    print(f"del_go: {del_go}")
    print("\n")

    # abort params
    V_A = abort_params['V_A']
    theta_A = abort_params['theta_A']
    t_A_aff = abort_params['t_A_aff']

    # get the database from batch_csvs
    file_name = f'/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
    df = pd.read_csv(file_name)
    df_animal = df[df['animal'] == int(animal_id)]
    df_animal_success = df_animal[df_animal['success'].isin([1, -1])]
    df_animal_success_rt_filter = df_animal_success[(df_animal_success['RTwrtStim'] <= 1) & (df_animal_success['RTwrtStim'] > 0)]

    for cond_ABL in all_ABLs_cond:
        for cond_ILD in all_ILDs_cond:
            condition_dir = os.path.join(vbmc_fit_saving_path, f'ABL_{cond_ABL}_ILD_{cond_ILD}')
            # Determine exactly which theta files are missing, and skip the condition only if none are missing
            missing_thetas = None
            if os.path.isdir(condition_dir):
                missing_thetas = [theta for theta in range(1, 51)
                                  if not os.path.exists(os.path.join(condition_dir, f'ABL_{cond_ABL}_ILD_{cond_ILD}_theta{theta:02d}.pkl'))]
                if len(missing_thetas) == 0:
                    print(f"Condition ABL={cond_ABL}, ILD={cond_ILD} already complete (50/50 PKLs); skipping...")
                    continue
                else:
                    print(f"Condition ABL={cond_ABL}, ILD={cond_ILD} missing thetas: {missing_thetas}")
            # Ensure the directory exists and proceed to fill missing thetas (or all if new)
            os.makedirs(condition_dir, exist_ok=True)

            print('********************************')
            print(f'ABL: {cond_ABL}, ILD: {cond_ILD}')
            print('********************************')
            
            conditions = {'ABL': [cond_ABL], 'ILD': [cond_ILD]}
            df_animal_cond_filter = df_animal_success_rt_filter[
                (df_animal_success_rt_filter['ABL'].isin(conditions['ABL'])) & 
                (df_animal_success_rt_filter['ILD'].isin(conditions['ILD']))
            ]
            print(f'len(df_animal_cond_filter): {len(df_animal_cond_filter)}')

            if len(df_animal_cond_filter) == 0:
                continue

            # Bounds arrays for (logR, logL)
            lb = np.array([logR_bounds[0], logL_bounds[0]])
            ub = np.array([logR_bounds[1], logL_bounds[1]])
            plb = np.array([logR_plausible_bounds[0], logL_plausible_bounds[0]])
            pub = np.array([logR_plausible_bounds[1], logL_plausible_bounds[1]])

            # Loop over fixed theta values: only the missing ones (or all if new)
            target_thetas = missing_thetas if (missing_thetas is not None) else list(range(1, 51))
            for theta_fixed in target_thetas:
                out_pkl = os.path.join(condition_dir, f'ABL_{cond_ABL}_ILD_{cond_ILD}_theta{theta_fixed:02d}.pkl')
                if os.path.exists(out_pkl):
                    print(f'{out_pkl} already exists, skipping')
                    continue

                def vbmc_loglike_fn(params):
                    logR, logL = params
                    R = 10**logR
                    L = 10**logL
                    all_loglike = [
                        compute_loglike_trial(row, R, L, int(theta_fixed))
                        for _, row in df_animal_cond_filter.iterrows()
                    ]
                    return float(np.sum(all_loglike))

                def vbmc_joint_fn(params):
                    priors = vbmc_prior_fn(params)
                    loglike = vbmc_loglike_fn(params)
                    return priors + loglike

                # Initialize with random values within plausible bounds (seeded per theta)
                rng_init = np.random.default_rng(42 + theta_fixed)
                logR_0 = rng_init.uniform(logR_plausible_bounds[0], logR_plausible_bounds[1])
                logL_0 = rng_init.uniform(logL_plausible_bounds[0], logL_plausible_bounds[1])
                x_0 = np.array([logR_0, logL_0])

                # Run VBMC and capture console output to save alongside PKL
                log_stream = io.StringIO()
                class _Tee:
                    def __init__(self, *streams):
                        self.streams = streams
                    def write(self, s):
                        for st in self.streams:
                            st.write(s)
                        return len(s)
                    def flush(self):
                        for st in self.streams:
                            st.flush()

                _tee = _Tee(sys.stdout, log_stream)
                with contextlib.redirect_stdout(_tee):
                    vbmc = VBMC(vbmc_joint_fn, x_0, lb, ub, plb, pub, options={'display': 'on', 'max_fun_evals': 50 * (2 + 2)})
                    vp, results = vbmc.optimize()

                # Save VBMC result for this condition and theta
                os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
                vbmc.save(out_pkl, overwrite=True)

                # Also save a text log with the VBMC console output and basic context
                out_txt = os.path.splitext(out_pkl)[0] + '.txt'
                try:
                    with open(out_txt, 'w') as f:
                        f.write(f'Batch: {batch_name}, Animal: {animal_id}\n')
                        f.write(f'Condition: ABL={cond_ABL}, ILD={cond_ILD}, theta={theta_fixed:02d}\n')
                        f.write(f'Init x_0: {x_0.tolist()}\n')
                        f.write(f'Bounds: lb={lb.tolist()}, ub={ub.tolist()}, plb={plb.tolist()}, pub={pub.tolist()}\n')
                        f.write(f'Timestamp: {pd.Timestamp.now().isoformat()}\n')
                        f.write('\n=== VBMC results object ===\n')
                        f.write(repr(results) + '\n')
                        f.write('\n=== VBMC console output ===\n')
                        f.write(log_stream.getvalue())
                    print(f'Saved log: {out_txt}')
                except Exception as e:
                    print(f'Warning: failed to save log to {out_txt}: {e}')

                print(f'Saved: {out_pkl}')

# %%

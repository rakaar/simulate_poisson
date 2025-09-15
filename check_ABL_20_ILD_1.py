# %%
# check ABL 20 ILD 1 vbmc theta fixed fit
import os
import numpy as np
import matplotlib.pyplot as plt
from pyvbmc import VBMC
import pandas as pd
from vbmc_skellam_utils import cum_pro_and_reactive_trunc_fn, up_or_down_hit_truncated_proactive_V2_fn
def _load_animal_params(batch_name: str, animal_id: str):
    """
    Load proactive (V_A, theta_A, t_A_aff) and executive offsets (t_E_aff, del_go)
    from the per-animal results pickle, following the same conventions as the fit script.

    Returns dict with keys: V_A, theta_A, t_A_aff, t_E_aff, del_go
    """
    import pickle
    pkl_file = f"/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/results_{batch_name}_animal_{animal_id}.pkl"
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"Animal results pickle not found: {pkl_file}")
    with open(pkl_file, 'rb') as f:
        fit_results_data = pickle.load(f)

    # Maps as in fit script
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

    V_A = theta_A = t_A_aff = np.nan
    t_E_aff_vals = []
    del_go_vals = []

    abort_keyname = "vbmc_aborts_results"
    if abort_keyname in fit_results_data:
        abort_samples = fit_results_data[abort_keyname]
        V_A = float(np.mean(abort_samples.get('V_A_samples', np.nan)))
        theta_A = float(np.mean(abort_samples.get('theta_A_samples', np.nan)))
        t_A_aff = float(np.mean(abort_samples.get('t_A_aff_samp', np.nan)))
    else:
        print(f"[WARN] {abort_keyname} not found in {pkl_file}")

    # vanilla tied
    tied_keyname = "vbmc_vanilla_tied_results"
    if tied_keyname in fit_results_data:
        tied_samples = fit_results_data[tied_keyname]
        if 't_E_aff_samples' in tied_samples:
            t_E_aff_vals.append(float(np.mean(tied_samples['t_E_aff_samples'])))
        if 'del_go_samples' in tied_samples:
            del_go_vals.append(float(np.mean(tied_samples['del_go_samples'])))
    # norm tied
    tied_keyname = "vbmc_norm_tied_results"
    if tied_keyname in fit_results_data:
        tied_samples = fit_results_data[tied_keyname]
        if 't_E_aff_samples' in tied_samples:
            t_E_aff_vals.append(float(np.mean(tied_samples['t_E_aff_samples'])))
        if 'del_go_samples' in tied_samples:
            del_go_vals.append(float(np.mean(tied_samples['del_go_samples'])))

    # average t_E_aff and del_go
    t_E_aff = float(np.mean(t_E_aff_vals)) if t_E_aff_vals else np.nan
    del_go = float(np.mean(del_go_vals)) if del_go_vals else np.nan

    return {
        'V_A': V_A,
        'theta_A': theta_A,
        't_A_aff': t_A_aff,
        't_E_aff': t_E_aff,
        'del_go': del_go,
    }


def _map_choice(row):
    """Map row to choice +1 / -1 using response_poke if available; fallback to success."""
    if 'response_poke' in row:
        try:
            rp = int(row['response_poke'])
            if rp in (2, 3):
                return 2 * rp - 5  # 3->+1, 2->-1
            if rp in (-1, 1):
                return rp
        except Exception:
            pass
    if 'success' in row and row['success'] in (-1, 1):
        return int(row['success'])
    return None

# %%
batch_name = "LED8"
animal_id = "112"
rt_max = 1
params = _load_animal_params(batch_name, animal_id)

V_A = params['V_A']
theta_A = params['theta_A']
t_A_aff = params['t_A_aff']
t_E_aff = params['t_E_aff']
del_go = params['del_go']

# Load data CSV and prepare RTs relative to stim
csv_path = f"/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/batch_csvs/batch_{batch_name}_valid_and_aborts.csv"
if not os.path.exists(csv_path):
    print(f"[ERROR] Data CSV not found: {csv_path}")
df = pd.read_csv(csv_path)
df_animal = df[df['animal'] == int(animal_id)].copy()
df_animal = df_animal[df_animal['success'].isin([1, -1])].copy()

# Compute RT relative to stim: timed_fix - intended_fix (fallback TotalFixTime)
if 'timed_fix' in df_animal.columns:
    timed = df_animal['timed_fix']
else:
    timed = df_animal['TotalFixTime']
rt_rel = timed - df_animal['intended_fix']
df_animal['rt_rel'] = rt_rel
df_animal = df_animal[(df_animal['rt_rel'] > 0) & (df_animal['rt_rel'] <= rt_max)].copy()

# choice mapping
df_animal['choice'] = df_animal.apply(_map_choice, axis=1)
eps = 1e-12

ABL = 20
ILD = 1
df_animal_cond = df_animal[
    (df_animal['ABL'] == ABL) & (df_animal['ILD'] == ILD)
]
N_samples = int(500e3)
bins = np.arange(0,1,0.05)
t_pts_wrt_stim = np.arange(0,1,0.001)
N_theory = int(1e3)
c_A_trunc_time = 0.3
folder = "/home/ragha/code/simulate_poisson/real_data_cond_wise_theta_fixed/ABL_20_ILD_1"
fig, axes = plt.subplots(10, 5, figsize=(20, 30))
axes = axes.flatten()
for idx, theta_fixed in enumerate(range(1,51)):
    pkl_file = os.path.join(folder, f"ABL_{ABL}_ILD_{ILD}_theta{theta_fixed:02d}.pkl")
    vp = VBMC.load(pkl_file)
    samples_obj = vp.vp.sample(N_samples)
    samples = samples_obj[0] if isinstance(samples_obj, (tuple, list)) else samples_obj
    R = (10 ** samples[:, 0]).mean()
    L = (10 ** samples[:, 1]).mean()
    stats = getattr(vp.vp, 'stats', None)
    stable = bool(stats.get('stable', False))
    # if stable:
    #     print(f'{theta_fixed:02d} is stable')
    ax = axes[idx]
    up_theory_samples = np.zeros((N_theory, len(t_pts_wrt_stim)))
    down_theory_samples = np.zeros((N_theory, len(t_pts_wrt_stim)))
    t_stim_sampels = df_animal_cond['intended_fix'].sample(N_theory, replace=True)
    for j, t_stim in enumerate(t_stim_sampels):
        trunc_factor_p_joint = cum_pro_and_reactive_trunc_fn(
                                t_stim + 1, c_A_trunc_time,
                                V_A, theta_A, t_A_aff,
                                t_stim, t_E_aff, R, L, theta_fixed) - \
                            cum_pro_and_reactive_trunc_fn(
                                t_stim, c_A_trunc_time,
                                V_A, theta_A, t_A_aff,
                                t_stim, t_E_aff, R, L, theta_fixed)
    
        t_pts_wrt_fix = t_pts_wrt_stim + t_stim
        up_theory_samples[j, :] = \
            up_or_down_hit_truncated_proactive_V2_fn(t_pts_wrt_fix, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, R, L, theta_fixed, c_A_trunc_time, 1)
        down_theory_samples[j, :] = \
            up_or_down_hit_truncated_proactive_V2_fn(t_pts_wrt_fix, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, R, L, theta_fixed, c_A_trunc_time, -1)
            
        
        up_theory_samples[j, :] /= trunc_factor_p_joint
        down_theory_samples[j, :] /= trunc_factor_p_joint
    
    up_samples_mean = up_theory_samples.mean(axis=0)
    down_samples_mean = down_theory_samples.mean(axis=0)
    total_mean = up_samples_mean + down_samples_mean
    ax.hist(df_animal_cond['rt_rel'], bins=bins, density=True, histtype='step')
    ax.plot(t_pts_wrt_stim, total_mean, 'k', ls='--', lw=3)
    # ax.set_title(f'R = {R :.2f}, L = {L :.2f}, theta_fixed = {theta_fixed}, stable={stable}', color=('green' if stable else 'red'))
    elbo_val = float(stats.get('elbo', np.nan))
    color = 'green' if stable else 'red'
    ax.set_title(f'theta_fixed = {theta_fixed}, ELBO = {elbo_val:.2f}', color=color)
    # print(f'area under curve: {np.trapezoid(total_mean, t_pts_wrt_stim)}')
    

plt.tight_layout()
plt.show()
    

# %%
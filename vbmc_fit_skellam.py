# VBMC single animal
# %% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyvbmc import VBMC
import os
import pickle 
from vbmc_skellam_utils import fpt_density_skellam, fpt_cdf_skellam, fpt_choice_skellam
# %%
trunc_time = 0.085
batch_name = 'LED8'
animal_id = 112


df = pd.read_csv('outLED8.csv')
df_1 = df[
    (df['animal'] == animal_id) &
    (df['session_type'] == 1) &
    (df['LED_trial'].isin([0, np.nan])) &
    (df['repeat_trial'].isin([0, 2])) &
    (df['training_level'] == 16) &
    (df['success'].isin([1, -1]))
]

# %%
rt = df_1['timed_fix'] - df_1['intended_fix']
df_1 = df_1.copy()  # Create an explicit copy to avoid SettingWithCopyWarning
df_1.loc[:, 'abs_ILD'] = df_1['ILD'].abs()
df_1.loc[:, 'RTwrtStim'] = df_1['timed_fix'] - df_1['intended_fix']

# %%
# Plot CDF of RTwrtStim for each absolute ILD value
# Remove rows with NaN values in abs_ILD or RTwrtStim
# df_plot = df_1.dropna(subset=['abs_ILD', 'RTwrtStim'])

# # Get unique absolute ILD values
# unique_abs_ilds = sorted(df_plot['abs_ILD'].unique())

# # Create the CDF plot
# plt.figure(figsize=(10, 6))

# for ild in unique_abs_ilds:
#     # Filter data for this absolute ILD value
#     data = df_plot[df_plot['abs_ILD'] == ild]['RTwrtStim']
    
#     # Sort the data
#     sorted_data = np.sort(data)
    
#     # Calculate the cumulative probabilities
#     yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
#     # Plot the CDF
#     plt.plot(sorted_data, yvals, label=f'|ILD| = {ild}', marker='.', linestyle='-', alpha=0.7)

# plt.xlabel('RTwrtStim')
# plt.ylabel('Cumulative Probability')
# plt.title('CDF of RTwrtStim for Each Absolute ILD Value')
# plt.legend()
# plt.xlim(0, 0.12)
# plt.grid(True, alpha=0.3)
# plt.show()
# 100ms cut off then
# %%
# df_1 RTwrtStim only > 0.1 and < 1s
# NOTE - comment if u don't want to truncate
df_1 = df_1[(df_1['RTwrtStim'] > trunc_time) & (df_1['RTwrtStim'] < 1.0)]
 
# %%
# to find range of rates to decide bound vals
# vanilla rates
# r0 = 1000/0.32
# rate_lambda = 0.1
# ABL = 60; ILD = 16
# print(f'vanilla largest and smallest rates')
# sample_rate = r0 * (10 **(rate_lambda * (ABL/20))) * (10 **(rate_lambda * (ILD/40)))
# print(f'sample_rate: {sample_rate:.2f}')
# ABL = 20; ILD = 1
# sample_rate = r0 * (10 **(rate_lambda * (ABL/20))) * (10 **(rate_lambda * (ILD/40)))
# print(f'sample_rate: {sample_rate:.2f}')

# # norm rates
# r0 = 1000/65
# rate_lambda = 2.13
# l = 0.9

# print(f'norm largest and smallest rates')
# r_num = r0 * (10 ** (rate_lambda * 68/20))
# r_den = (10 ** (rate_lambda * 68 * l/20)) + (10 ** (rate_lambda * 52 * l/20))
# print(f'rate: {r_num/r_den:.2f}')

# r_num = r0 * (10 ** (rate_lambda * 21/20))
# r_den = (10 ** (rate_lambda * 19 * l/20)) + (10 ** (rate_lambda * 21 * l/20))
# print(f'rate: {r_num/r_den:.2f}')

# # %%
# print(np.log10(5),np.log10(7000) )
# # 0.5, 3.84

# %%
# get t_E_aff
pkl_file = f'/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/results_{batch_name}_animal_{animal_id}.pkl'
with open(pkl_file, 'rb') as f:
    results = pickle.load(f)

t_E_aff = results['vbmc_vanilla_tied_results']['t_E_aff_samples'].mean()
print(f"t_E_aff: {t_E_aff:.2f}")

# %%
# funcs
def loglike_fn(params):
    logR, logL, theta = params
    
    # param modifying
    R = 10**logR
    L = 10**logL
    theta += np.random.uniform(-0.5, 0.5)
    theta = round(theta)
    if theta < 1:
        theta = 1

    # Precompute factors that do not vary across trials
    cdf_trunc = 1 - fpt_cdf_skellam(trunc_time - t_E_aff, R, L, theta)
    # Probability of hitting + boundary (does not depend on the trial)
    p_pos = fpt_choice_skellam(R, L, theta, 1)
    # Vectorize across trials
    rts = df_1_stim['RTwrtStim'].to_numpy()
    choices = (2 * df_1_stim['response_poke'].to_numpy() - 5).astype(int)
    p_choice = np.where(choices == 1, p_pos, 1 - p_pos)
    # FPT density for all RTs in one vectorized call
    rho_t = fpt_density_skellam(rts, R, L, theta)
    # Truncation factor per trial (guard against zeros)
    trunc_factor = cdf_trunc * p_choice
    trunc_factor = np.where(trunc_factor == 0, 1e-80, trunc_factor)
    # Compute per-trial probability with cut-off handling and positivity clamp
    p = np.where(rts < trunc_time, 1e-50, (rho_t * p_choice) / trunc_factor)
    p = np.where(p <= 0, 1e-50, p)
    # NaN guard: if any, raise with details for the first offending trial
    logp = np.log(p)
    if np.any(np.isnan(logp)):
        bad_idx = np.where(np.isnan(logp))[0][0]
        bad_rt = float(rts[bad_idx])
        bad_choice = int(choices[bad_idx])
        bad_trunc = float(trunc_factor[bad_idx])
        raise ValueError(f'log(p) is nan for params: {params}, rt = {bad_rt}, choice = {bad_choice}, '
                         f'trunc_factor = {bad_trunc}, cdf_trunc = {cdf_trunc}, '
                         f'rho(t)={fpt_density_skellam(bad_rt, R, L, theta)} '
                         f'rho(+) = {fpt_choice_skellam(R, L, theta, bad_choice)}')
    # Sum log-probabilities across trials
    loglike = float(np.sum(logp))
    return loglike

def prior(params):
    logR, logL, theta = params
    
    # Proper normalized uniform log-density inside bounds
    logR_width = logR_bounds[1] - logR_bounds[0]
    logL_width = logL_bounds[1] - logL_bounds[0]
    theta_width = theta_bounds[1] - theta_bounds[0]
    return - (np.log(logR_width) + np.log(logL_width) + np.log(theta_width))


def joint_likelihood(params):
    return loglike_fn(params) + prior(params)

# %%
# Declare lb, ub, plaus_lb, plaus_ub
# %%
    
ABL_unique = df_1['ABL'].unique()
ILD_unique = df_1['ILD'].unique()
print(f'ABL_unique: {ABL_unique}')
print(f'ILD_unique: {ILD_unique}')

# Define bounds for logR and logL (using the existing bounds)
# For theta, we'll use a separate set of bounds
logR_bounds = [0.5, 4]
logL_bounds = [0.5, 4]
theta_bounds = [0.5, 60.5]

logR_plausible_bounds = [1, 3.8]
logL_plausible_bounds = [1, 3.8]
theta_plausible_bounds = [1, 45]

# Define the joint likelihood function for VBMC
# This function should take the parameters as a single array
# and return the joint log-likelihood


for ABL in ABL_unique:
    for ILD in ILD_unique:
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print(f'ABL = {ABL}, ILD = {ILD}')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        df_1_stim = df_1[(df_1['ABL'] == ABL) & (df_1['ILD'] == ILD)]
        if len(df_1_stim) == 0:
            raise ValueError(f'No data for ABL={ABL}, ILD={ILD}')
        
        print(f'Number of trials for ABL={ABL}, ILD={ILD}: {len(df_1_stim)}')
        continue

        
        # Define bounds for VBMC
        lb = np.array([logR_bounds[0], logL_bounds[0], theta_bounds[0]])
        ub = np.array([logR_bounds[1], logL_bounds[1], theta_bounds[1]])
        plb = np.array([logR_plausible_bounds[0], logL_plausible_bounds[0], theta_plausible_bounds[0]])
        pub = np.array([logR_plausible_bounds[1], logL_plausible_bounds[1], theta_plausible_bounds[1]])

        # Initialize with random values within plausible bounds
        np.random.seed(42)
        logR_0 = np.random.uniform(logR_plausible_bounds[0], logR_plausible_bounds[1])
        logL_0 = np.random.uniform(logL_plausible_bounds[0], logL_plausible_bounds[1])
        theta_0 = np.random.uniform(theta_plausible_bounds[0], theta_plausible_bounds[1])
        x_0 = np.array([logR_0, logL_0, theta_0])

        # Run VBMC
        print('Initializing VBMC...')
        try:
            vbmc = VBMC(joint_likelihood, x_0, lb, ub, plb, pub, options={'display': 'on', 'max_fun_evals': 200 * (2 + 3)})
            vp, results = vbmc.optimize()
            print('VBMC optimization completed successfully')
            os.makedirs('poisson_fit_pkls', exist_ok=True)
            vbmc.save(f'poisson_fit_pkls/vbmc_poisson_fit_{ABL}_{ILD}_{batch_name}_{animal_id}.pkl', overwrite=True)
        except Exception as e:
            raise ValueError(f'VBMC optimization failed: {e}')


# %%
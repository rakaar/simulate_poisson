# VBMC single animal
# %% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyvbmc import VBMC

df = pd.read_csv('outLED8.csv')
df_1 = df[
    (df['animal'] == 112) &
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

# Plot CDF of RTwrtStim for each absolute ILD value
# Remove rows with NaN values in abs_ILD or RTwrtStim
df_plot = df_1.dropna(subset=['abs_ILD', 'RTwrtStim'])

# Get unique absolute ILD values
unique_abs_ilds = sorted(df_plot['abs_ILD'].unique())

# Create the CDF plot
plt.figure(figsize=(10, 6))

for ild in unique_abs_ilds:
    # Filter data for this absolute ILD value
    data = df_plot[df_plot['abs_ILD'] == ild]['RTwrtStim']
    
    # Sort the data
    sorted_data = np.sort(data)
    
    # Calculate the cumulative probabilities
    yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Plot the CDF
    plt.plot(sorted_data, yvals, label=f'|ILD| = {ild}', marker='.', linestyle='-', alpha=0.7)

plt.xlabel('RTwrtStim')
plt.ylabel('Cumulative Probability')
plt.title('CDF of RTwrtStim for Each Absolute ILD Value')
plt.legend()
plt.xlim(0, 0.12)
plt.grid(True, alpha=0.3)
plt.show()
# 100ms cut off then
# %%
# df_1 RTwrtStim only > 0.1 and < 1s
df_1 = df_1[(df_1['RTwrtStim'] > 0.1) & (df_1['RTwrtStim'] < 1.0)]

# %%
# Filter ILDs and ABL
# ILDs = -16, -8, -4, -2, 2, 4, 8, 16
# ABLs = 20, 60
# filter
# df_1 = df_1[(df_1['ILD'].isin([-16, -8, -4, -2, 2, 4, 8, 16])) & (df_1['ABL'].isin([20, 60]))]


# %%
# vanilla rates
r0 = 1000/0.32
rate_lambda = 0.1
ABL = 60; ILD = 16
print(f'vanilla largest and smallest rates')
sample_rate = r0 * (10 **(rate_lambda * (ABL/20))) * (10 **(rate_lambda * (ILD/40)))
print(f'sample_rate: {sample_rate:.2f}')
ABL = 20; ILD = 1
sample_rate = r0 * (10 **(rate_lambda * (ABL/20))) * (10 **(rate_lambda * (ILD/40)))
print(f'sample_rate: {sample_rate:.2f}')

# norm rates
r0 = 1000/65
rate_lambda = 2.13
l = 0.9

print(f'norm largest and smallest rates')
r_num = r0 * (10 ** (rate_lambda * 68/20))
r_den = (10 ** (rate_lambda * 68 * l/20)) + (10 ** (rate_lambda * 52 * l/20))
print(f'rate: {r_num/r_den:.2f}')

r_num = r0 * (10 ** (rate_lambda * 21/20))
r_den = (10 ** (rate_lambda * 19 * l/20)) + (10 ** (rate_lambda * 21 * l/20))
print(f'rate: {r_num/r_den:.2f}')

# %%
print(np.log10(5),np.log10(7000) )
# 0.5, 3.84

# %%
# bounds
logR_bounds = [0.5, 4]
logR_plausible_bounds = [1, 3.8]

logL_bounds = [0.5, 4]
logL_plausible_bounds = [1, 3.8]

theta_bounds = [0.5, 60.5]
theta_plausible_bounds = [1, 45]

trunc_time = 0.1

# funcs
def loglike_fn(params):
    logR, logL, theta = params
    
    # param modifying
    R = 10**logR
    L = 10**logL
    theta += np.random.uniform(-0.5, 0.5)
    theta = round(theta)

    cdf_trunc = 1 - fpt_cdf_skellam(trunc_time, mu1, mu2, theta)    
    loglike = 0
    for row in df_1.iterrows():
        rt = row['RTwrtStim']
        t_stim = row['intended_fix']
        choice = 2 * row['response_poke'] - 5
        trunc_factor = cdf_trunc * fpt_choice_skellam(mu1, mu2, choice)
        if rt < trunc_time:
            p = 1e-50
        else:
            p = fpt_density_skellam(rt, R, L, theta) * fpt_choice_skellam(mu1, mu2, choice)
        p /= trunc_factor

        if p <= 0:
            p = 1e-50
        elif p >= 1:
            p = 1 - 1e-50
        loglike += np.log(p)
    return loglike

def prior(params):
    logR, logL, theta = params
    # Uniform priors within hard bounds; -inf outside
    if not (logR_bounds[0] <= logR <= logR_bounds[1]):
        return -np.inf
    if not (logL_bounds[0] <= logL <= logL_bounds[1]):
        return -np.inf
    if not (theta_bounds[0] <= theta <= theta_bounds[1]):
        return -np.inf

    # Proper normalized uniform log-density inside bounds
    logR_width = logR_bounds[1] - logR_bounds[0]
    logL_width = logL_bounds[1] - logL_bounds[0]
    theta_width = theta_bounds[1] - theta_bounds[0]
    return - (np.log(logR_width) + np.log(logL_width) + np.log(theta_width))


def joint_likelihood(params):
    return loglike_fn(params) + prior(params)
# %%
    

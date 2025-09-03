# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the simulation results
fname = 'discrete_ddm_results_lambda_R_L_scaled.pkl'
with open(fname, 'rb') as f:
    results = pickle.load(f)

# Get the unique scaling factors, ABL, and ILD values
scaling_factors = sorted(results.keys())
all_stim_pairs = list(results[scaling_factors[0]].keys())
ABL_arr = sorted(list(set(k[0] for k in all_stim_pairs)))
ILD_arr = sorted(list(set(k[1] for k in all_stim_pairs)))
print(f'ABL_arr: {ABL_arr}')
print(f'ILD_arr: {ILD_arr}')
# Create the plot
num_sf = len(scaling_factors)
fig, axes = plt.subplots(1, num_sf, figsize=(5 * num_sf, 4), sharex=True, sharey=True)
if num_sf == 1:
    axes = [axes] # Make it iterable if there's only one subplot

# Define plot styles for each ABL
styles = [
    {'color': 'k', 'marker': '.', 'linestyle': '-', 'label': f'ABL={ABL_arr[0]}'}, 
    {'color': 'r', 'marker': 'x', 'linestyle': '--', 'label': f'ABL={ABL_arr[1]}'}, 
    {'color': 'g', 'marker': '^', 'linestyle': ':', 'label': f'ABL={ABL_arr[2]}'} 
]

# Process and plot data for each scaling factor
for i, sf in enumerate(scaling_factors):
    ax = axes[i]
    sf_results = results[sf]

    for abl_idx, abl in enumerate(ABL_arr):
        prob_upper = []
        for ild in ILD_arr:
            stimulus_pair = (abl, ild)
            if stimulus_pair in sf_results:
                res = sf_results[stimulus_pair]
                n_pos = len(res['pos_times'])
                n_neg = len(res['neg_times'])
                total_trials = n_pos + n_neg
                if total_trials > 0:
                    prob_upper.append(n_pos / total_trials)
                else:
                    prob_upper.append(np.nan)
            else:
                prob_upper.append(np.nan)
        
        # Plot psychometric curve for the current ABL
        style = styles[abl_idx % len(styles)]
        ax.plot(ILD_arr, prob_upper, **style)

    ax.set_title(f'Scaling Factor = {sf}')
    ax.set_xlabel('ILD (dB)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(-0.05, 1.05)

# Set shared y-label and legend
axes[0].set_ylabel('Probability of Choosing Upper Bound')
axes[0].legend()

fig.suptitle('Psychometric Curves for Different Rate Scaling Factors', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure
output_fig = 'psychometric_curves_scaled.png'
plt.savefig(output_fig, dpi=150, bbox_inches='tight')

print(f"Analysis plot saved to {output_fig}")

# %% 
# Quantile-Quantile plots

# Define parameters
abs_ild_values = [1, 2, 4, 8, 16]
quantiles = np.arange(0.05, 0.96, 0.01)

# Create the plot grid
fig_qq, axes_qq = plt.subplots(len(scaling_factors), len(abs_ild_values), 
                               figsize=(15, 9), sharex=False, sharey=False)

# Process and plot data
for i, sf in enumerate(scaling_factors):
    sf_results = results[sf]
    row_max_val = 0
    row_quantile_data = {}

    # First pass: calculate all quantiles for the row and find the max value
    for j, abs_ild in enumerate(abs_ild_values):
        rt_data = {abl: [] for abl in ABL_arr}
        for abl in ABL_arr:
            for ild_sign in [-1, 1]:
                ild = abs_ild * ild_sign
                stimulus_pair = (abl, ild)
                if stimulus_pair in sf_results:
                    res = sf_results[stimulus_pair]
                    all_times = np.concatenate([res['pos_times'], res['neg_times']])
                    rt_data[abl].extend(all_times)

        quantile_values = {}
        for abl in ABL_arr:
            if rt_data[abl]:
                quantile_values[abl] = np.quantile(rt_data[abl], quantiles)
                row_max_val = np.nanmax([row_max_val, np.nanmax(quantile_values[abl])])
            else:
                quantile_values[abl] = np.full_like(quantiles, np.nan)
        row_quantile_data[abs_ild] = quantile_values

    # Second pass: plot with synchronized axes for the row
    for j, abs_ild in enumerate(abs_ild_values):
        ax = axes_qq[i, j]
        quantile_values = row_quantile_data[abs_ild]

        # Plot Q-Q plots
        if ABL_arr[2] in quantile_values: # ABL 60 on x-axis
            x_quantiles = quantile_values[ABL_arr[2]]
            if ABL_arr[0] in quantile_values: # ABL 20 on y-axis
                ax.plot(x_quantiles, quantile_values[ABL_arr[0]], 'k.', markersize=2, label=f'{ABL_arr[0]} vs {ABL_arr[2]}')
            if ABL_arr[1] in quantile_values: # ABL 40 on y-axis
                ax.plot(x_quantiles, quantile_values[ABL_arr[1]], 'b.', markersize=2, label=f'{ABL_arr[1]} vs {ABL_arr[2]}')

        # Set dynamic limits and add diagonal line
        if row_max_val > 0:
            lim = row_max_val * 1.05
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.plot([0, lim], [0, lim], 'r--', alpha=0.75, zorder=0)
            ax.set_aspect('equal', 'box')

        if i == 0:
            ax.set_title(f'ILD = +/-{abs_ild} dB')
        if j == 0:
            ax.set_ylabel(f'SF={sf}\nQuantiles')
        if i == len(scaling_factors) - 1:
            ax.set_xlabel(f'Quantiles ABL={ABL_arr[2]}')
        # if i == 0 and j == 0:
        #     ax.legend()
        
        

fig_qq.suptitle('Q-Q Plots of Reaction Times', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure
output_fig_qq = 'qq_plots_scaled.png'
plt.savefig(output_fig_qq, dpi=150, bbox_inches='tight')

print(f"Q-Q plot saved to {output_fig_qq}")

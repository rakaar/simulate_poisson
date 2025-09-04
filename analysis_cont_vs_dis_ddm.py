# Analysis - discrete DDM vs cont DDM 
# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the simulation results
fname = 'ddm_simulation_results_vanilla.pkl'
# fname = 'ddm_simulation_results.pkl'
with open(fname, 'rb') as f:
    results = pickle.load(f)

discrete_ddm_results = results['discrete_ddm']
continuous_ddm_results = results['continuous_ddm']

# Get the unique ABL and ILD values from the keys
all_keys = list(discrete_ddm_results.keys())
ABL_arr = sorted(list(set(k[0] for k in all_keys)))
ILD_arr = sorted(list(set(k[1] for k in all_keys)))

# Prepare data for plotting
prob_upper_discrete = {abl: [] for abl in ABL_arr}
prob_upper_continuous = {abl: [] for abl in ABL_arr}

for abl in ABL_arr:
    for ild in ILD_arr:
        stimulus_pair = (abl, ild)

        # Discrete DDM probabilities
        disc_res = discrete_ddm_results[stimulus_pair]
        n_pos_disc = len(disc_res['pos_times'])
        n_neg_disc = len(disc_res['neg_times'])
        total_trials_disc = n_pos_disc + n_neg_disc
        if total_trials_disc > 0:
            prob_upper_discrete[abl].append(n_pos_disc / total_trials_disc)
        else:
            prob_upper_discrete[abl].append(np.nan)

        # Continuous DDM probabilities
        cont_res = continuous_ddm_results[stimulus_pair]
        n_pos_cont = len(cont_res['pos_times'])
        n_neg_cont = len(cont_res['neg_times'])
        total_trials_cont = n_pos_cont + n_neg_cont
        if total_trials_cont > 0:
            prob_upper_continuous[abl].append(n_pos_cont / total_trials_cont)
        else:
            prob_upper_continuous[abl].append(np.nan)

# Prepare data for mean RT plot
mean_rt_discrete = {abl: [] for abl in ABL_arr}
sem_rt_discrete = {abl: [] for abl in ABL_arr}
mean_rt_continuous = {abl: [] for abl in ABL_arr}
sem_rt_continuous = {abl: [] for abl in ABL_arr}

for abl in ABL_arr:
    for ild in ILD_arr:
        stimulus_pair = (abl, ild)

        # Discrete DDM mean RT and SEM
        disc_res = discrete_ddm_results[stimulus_pair]
        all_times_disc = np.concatenate([disc_res['pos_times'], disc_res['neg_times']])
        if all_times_disc.size > 0:
            mean_rt_discrete[abl].append(np.mean(all_times_disc))
            sem_rt_discrete[abl].append(np.std(all_times_disc, ddof=1) / np.sqrt(all_times_disc.size))
        else:
            mean_rt_discrete[abl].append(np.nan)
            sem_rt_discrete[abl].append(np.nan)

        # Continuous DDM mean RT and SEM
        cont_res = continuous_ddm_results[stimulus_pair]
        all_times_cont = np.concatenate([cont_res['pos_times'], cont_res['neg_times']])
        if all_times_cont.size > 0:
            mean_rt_continuous[abl].append(np.mean(all_times_cont))
            sem_rt_continuous[abl].append(np.std(all_times_cont, ddof=1) / np.sqrt(all_times_cont.size))
        else:
            mean_rt_continuous[abl].append(np.nan)
            sem_rt_continuous[abl].append(np.nan)

# Create the 3x3 plot
fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharey='row', constrained_layout=True)

bins = np.arange(0, 1, 0.01)

for i, abl in enumerate(ABL_arr):
    # Row 1: Psychometric curves
    ax_psych = axes[0, i]
    ax_psych.plot(ILD_arr, prob_upper_discrete[abl], 'r-o', label='Discrete DDM')
    ax_psych.plot(ILD_arr, prob_upper_continuous[abl], 'k-o', label='Continuous DDM')
    ax_psych.set_title(f'ABL = {abl} dB')
    ax_psych.grid(True, linestyle='--', alpha=0.6)

    # Row 2: Mean Reaction Times
    ax_rt = axes[1, i]
    ax_rt.errorbar(ILD_arr, mean_rt_discrete[abl], yerr=sem_rt_discrete[abl], color='red', linestyle='-', marker='o', capsize=5)
    ax_rt.errorbar(ILD_arr, mean_rt_continuous[abl], yerr=sem_rt_continuous[abl], color='black', linestyle='-', marker='o', capsize=5)
    ax_rt.grid(True, linestyle='--', alpha=0.6)

    # Row 3: Reaction Time Distributions
    ax_dist = axes[2, i]
    pos_ild_times_disc = np.concatenate([np.concatenate([discrete_ddm_results[(abl, ild)]['pos_times'], discrete_ddm_results[(abl, ild)]['neg_times']]) for ild in ILD_arr if ild > 0])
    neg_ild_times_disc = np.concatenate([np.concatenate([discrete_ddm_results[(abl, ild)]['pos_times'], discrete_ddm_results[(abl, ild)]['neg_times']]) for ild in ILD_arr if ild < 0])
    pos_ild_times_cont = np.concatenate([np.concatenate([continuous_ddm_results[(abl, ild)]['pos_times'], continuous_ddm_results[(abl, ild)]['neg_times']]) for ild in ILD_arr if ild > 0])
    neg_ild_times_cont = np.concatenate([np.concatenate([continuous_ddm_results[(abl, ild)]['pos_times'], continuous_ddm_results[(abl, ild)]['neg_times']]) for ild in ILD_arr if ild < 0])

    counts_pos_disc, _ = np.histogram(pos_ild_times_disc, bins=bins, density=True)
    counts_neg_disc, _ = np.histogram(neg_ild_times_disc, bins=bins, density=True)
    ax_dist.step(bins[:-1], counts_pos_disc, where='post', color='red', label='Discrete (Pos ILD)', lw=2, alpha=0.4)
    ax_dist.step(bins[:-1], -counts_neg_disc, where='post', color='red', linestyle='--', lw=2, alpha=0.4)

    counts_pos_cont, _ = np.histogram(pos_ild_times_cont, bins=bins, density=True)
    counts_neg_cont, _ = np.histogram(neg_ild_times_cont, bins=bins, density=True)
    ax_dist.step(bins[:-1], counts_pos_cont, where='post', color='black', label='Continuous (Pos ILD)')
    ax_dist.step(bins[:-1], -counts_neg_cont, where='post', color='black', linestyle='--')
    
    ax_dist.axhline(0, color='gray', linewidth=0.8)
    ax_dist.set_xlabel('ILD (dB)')
    ax_dist.grid(True, linestyle='--', alpha=0.3)

# Set shared labels and legends
axes[0, 0].set_ylabel('Probability of Choosing Upper Bound')
axes[1, 0].set_ylabel('Mean Reaction Time (s)')
axes[2, 0].set_ylabel('Density (Positive/Negative ILD)')
axes[0, 0].legend()
axes[2, 0].legend()

fig.suptitle('Full DDM Analysis', fontsize=16)

# Save the figure
plt.savefig('full_analysis_plot.png', dpi=150, bbox_inches='tight')

print("Full analysis plot saved to full_analysis_plot.png")



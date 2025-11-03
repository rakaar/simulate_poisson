# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from pathlib import Path
from scipy.stats import ks_2samp
from joblib import Parallel, delayed
from tqdm import tqdm
from bads_utils import (
    lr_rates_from_ABL_ILD,
    simulate_single_ddm_trial,
    simulate_poisson_rts
)

# %%
# Find and load the most recent multi-stimulus BADS results (with choice)
pkl_pattern = 'bads_multistim_scaling_results_*.pkl'
pkl_files = glob.glob(pkl_pattern)

if not pkl_files:
    raise FileNotFoundError(f"No files matching '{pkl_pattern}' found in current directory")

# Get the most recent file
latest_pkl = max(pkl_files, key=lambda x: Path(x).stat().st_mtime)
print(f"Loading results from: {latest_pkl}")

# Define dummy obj_func to allow unpickling
def obj_func(x):
    """Dummy function for unpickling - not actually used."""
    pass

def obj_func_with_scaling(x, ddm_rts_decided_dict, ddm_data_dict, n_trials, seed):
    """Dummy function for unpickling - not actually used."""
    pass

with open(latest_pkl, 'rb') as f:
    results = pickle.load(f)

# Extract data
optimize_result = results['bads_result']
optimized_params = results['optimized_params']
ddm_params = results['ddm_params']
stimuli = results['stimuli']
ddm_simulation = results['ddm_simulation']

# Extract shared parameters (including rate_scaling_factor)
N_opt = optimized_params['N']
k_opt = optimized_params['k']
c_opt = optimized_params['c']
theta_opt = optimized_params['theta']
rate_scaling_factor_opt = optimized_params['rate_scaling_factor']
x_opt = optimized_params['x_opt']

print("\n" + "="*70)
print("LOADED MULTI-STIMULUS BADS OPTIMIZATION RESULTS (WITH CHOICE)")
print("="*70)
print(f"\nShared optimized parameters:")
print(f"  N:                   {N_opt}")
print(f"  k:                   {k_opt:.4f}")
print(f"  c:                   {c_opt:.6f} (= k/N)")
print(f"  theta:               {theta_opt:.4f}")
print(f"  rate_scaling_factor: {rate_scaling_factor_opt:.4f}")

print(f"\nPoisson rates for each stimulus:")
print(f"  (Poisson rate = DDM rate × rate_scaling_factor)")
poisson_rates_per_stimulus = optimized_params['poisson_rates_per_stimulus']
for stim in stimuli:
    stim_key = f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"
    ddm_right = poisson_rates_per_stimulus[stim_key]['ddm_right_rate']
    ddm_left = poisson_rates_per_stimulus[stim_key]['ddm_left_rate']
    poisson_right = poisson_rates_per_stimulus[stim_key]['poisson_right_rate']
    poisson_left = poisson_rates_per_stimulus[stim_key]['poisson_left_rate']
    print(f"\n  {stim_key}:")
    print(f"    DDM right rate:     {ddm_right:.6f}")
    print(f"    DDM left rate:      {ddm_left:.6f}")
    print(f"    Poisson right rate: {poisson_right:.6f}")
    print(f"    Poisson left rate:  {poisson_left:.6f}")

print(f"\nOptimization Statistics:")
print(f"  Objective: MSE of mean RT and P(right)")
print(f"  Total MSE (sum):      {optimize_result['fval']:.6f}")
print(f"  Average MSE per stim: {optimize_result['fval']/len(stimuli):.6f}")
print(f"  Function evaluations: {optimize_result['func_count']}")
print(f"  Success:              {optimize_result['success']}")

# %%
# Simulate DDM and Poisson for each stimulus condition
print("\n" + "="*70)
print("SIMULATING DDM AND POISSON FOR ALL STIMULI")
print("="*70)

# Simulation parameters
dt = 1e-4
dB = 1e-2
N_sim = int(100e3)  # 100K trials per condition
T = 20
n_steps = int(T/dt)

# Store results for each stimulus
simulation_results = {}

for stim in stimuli:
    ABL = stim['ABL']
    ILD = stim['ILD']
    stim_key = f"ABL_{ABL}_ILD_{ILD}"
    
    print(f"\n{'='*70}")
    print(f"STIMULUS: {stim_key}")
    print(f"{'='*70}")
    
    # Calculate DDM rates
    ddm_right_rate, ddm_left_rate = lr_rates_from_ABL_ILD(
        ABL, ILD, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
    )
    
    # Calculate Poisson rates using scaling factor
    poisson_right_rate = ddm_right_rate * rate_scaling_factor_opt
    poisson_left_rate = ddm_left_rate * rate_scaling_factor_opt
    
    # DDM simulation
    print(f"\n--- DDM Simulation ---")
    mu_ddm = ddm_right_rate - ddm_left_rate
    sigma_sq_ddm = ddm_right_rate + ddm_left_rate
    sigma_ddm = np.sqrt(sigma_sq_ddm)
    
    print(f"DDM rates: right={ddm_right_rate:.6f}, left={ddm_left_rate:.6f}")
    print(f"DDM params: mu={mu_ddm:.6f}, sigma={sigma_ddm:.6f}, theta={ddm_params['theta']:.4f}")
    print(f"Simulating {N_sim} DDM trials...")
    
    ddm_results = Parallel(n_jobs=-1)(
        delayed(simulate_single_ddm_trial)(j, mu_ddm, sigma_ddm, ddm_params['theta'], dt, dB, n_steps) 
        for j in tqdm(range(N_sim), desc=f'DDM {stim_key}', ncols=80)
    )
    ddm_data = np.array(ddm_results)
    ddm_decided_mask = ~np.isnan(ddm_data[:, 0])
    ddm_rts_decided = ddm_data[ddm_decided_mask, 0]
    ddm_choices_decided = ddm_data[ddm_decided_mask, 1]
    
    # DDM metrics
    mean_rt_ddm = np.mean(ddm_rts_decided)
    p_right_ddm = np.mean(ddm_choices_decided == 1)
    
    print(f"DDM: {len(ddm_rts_decided)}/{N_sim} decided")
    print(f"  Mean RT:   {mean_rt_ddm:.4f}s")
    print(f"  P(right):  {p_right_ddm:.4f}")
    
    # Poisson simulation
    print(f"\n--- Poisson Simulation ---")
    poisson_params = {
        'N_right': N_opt,
        'N_left': N_opt,
        'c': c_opt,
        'r_right': poisson_right_rate,
        'r_left': poisson_left_rate,
        'theta': theta_opt,
        'T': 20,
        'exponential_noise_scale': 0
    }
    
    print(f"Poisson rates: right={poisson_right_rate:.6f}, left={poisson_left_rate:.6f}")
    print(f"  (DDM rates × scaling_factor={rate_scaling_factor_opt:.4f})")
    print(f"Poisson params: N={N_opt}, c={c_opt:.6f}, theta={theta_opt:.4f}")
    print(f"Simulating {N_sim} Poisson trials...")
    
    poisson_results = simulate_poisson_rts(poisson_params, n_trials=N_sim, 
                                           seed=42, verbose=False)
    poisson_decided_mask = ~np.isnan(poisson_results[:, 0])
    poisson_rts_decided = poisson_results[poisson_decided_mask, 0]
    poisson_choices_decided = poisson_results[poisson_decided_mask, 1]
    
    # Poisson metrics
    mean_rt_poisson = np.mean(poisson_rts_decided)
    p_right_poisson = np.mean(poisson_choices_decided == 1)
    
    print(f"Poisson: {len(poisson_rts_decided)}/{N_sim} decided")
    print(f"  Mean RT:   {mean_rt_poisson:.4f}s")
    print(f"  P(right):  {p_right_poisson:.4f}")
    
    # Compute metrics
    ks_stat, ks_pval = ks_2samp(ddm_rts_decided, poisson_rts_decided)
    rt_mse = (mean_rt_ddm - mean_rt_poisson) ** 2
    choice_mse = (p_right_ddm - p_right_poisson) ** 2
    total_mse = rt_mse + choice_mse
    
    print(f"\n--- Comparison Metrics ---")
    print(f"KS-statistic:   {ks_stat:.6f}, p-value: {ks_pval:.6e}")
    print(f"RT MSE:         {rt_mse:.6f}")
    print(f"Choice MSE:     {choice_mse:.6f}")
    print(f"Total MSE:      {total_mse:.6f}")
    
    # Store results
    simulation_results[stim_key] = {
        'ABL': ABL,
        'ILD': ILD,
        'ddm_right_rate': ddm_right_rate,
        'ddm_left_rate': ddm_left_rate,
        'poisson_right_rate': poisson_right_rate,
        'poisson_left_rate': poisson_left_rate,
        'ddm_data': ddm_data,
        'ddm_rts_decided': ddm_rts_decided,
        'ddm_choices_decided': ddm_choices_decided,
        'mean_rt_ddm': mean_rt_ddm,
        'p_right_ddm': p_right_ddm,
        'poisson_data': poisson_results,
        'poisson_rts_decided': poisson_rts_decided,
        'poisson_choices_decided': poisson_choices_decided,
        'mean_rt_poisson': mean_rt_poisson,
        'p_right_poisson': p_right_poisson,
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
        'rt_mse': rt_mse,
        'choice_mse': choice_mse,
        'total_mse': total_mse
    }

print(f"\n{'='*70}")
print("ALL SIMULATIONS COMPLETE")
print(f"{'='*70}")

# Print summary table
print(f"\n{'='*70}")
print("SUMMARY TABLE")
print(f"{'='*70}")
print(f"\n{'Stimulus':<15} {'Mean RT DDM':<12} {'Mean RT Pois':<12} {'P(R) DDM':<10} {'P(R) Pois':<10} {'Total MSE':<10}")
print("-" * 70)
for stim in stimuli:
    stim_key = f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"
    data = simulation_results[stim_key]
    print(f"{stim_key:<15} {data['mean_rt_ddm']:<12.4f} {data['mean_rt_poisson']:<12.4f} "
          f"{data['p_right_ddm']:<10.4f} {data['p_right_poisson']:<10.4f} {data['total_mse']:<10.6f}")
print()

# %%
# Plot RTD comparison for all stimuli in a 2x4 grid
print("="*70)
print("PLOTTING RTD COMPARISONS (2x4 GRID)")
print("="*70)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

bins = np.arange(0, 2.01, 0.01)

for idx, stim in enumerate(stimuli):
    stim_key = f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"
    data = simulation_results[stim_key]
    ax = axes[idx]
    
    # Plot histograms
    ax.hist(data['ddm_rts_decided'], bins=bins, density=True, color='b', 
            alpha=0.6, label='DDM', histtype='step', linewidth=2)
    ax.hist(data['poisson_rts_decided'], bins=bins, density=True, color='r', 
            alpha=0.6, label='Poisson', histtype='step', linewidth=2)
    
    # Add step outlines
    ax.hist(data['ddm_rts_decided'], bins=bins, density=True, color='b', 
            histtype='step', linewidth=2)
    ax.hist(data['poisson_rts_decided'], bins=bins, density=True, color='r', 
            histtype='step', linewidth=2)
    
    # Formatting
    ax.set_xlabel('Reaction Time (s)', fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title(f'{stim_key}\nRT MSE={data["rt_mse"]:.6f}, P(R): {data["p_right_ddm"]:.3f} vs {data["p_right_poisson"]:.3f}', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

plt.suptitle(f'DDM vs Poisson RTD Comparison (Multi-Stimulus with Choice)\n'
             f'N={N_opt}, c={c_opt:.6f}, theta={theta_opt:.4f}, rate_scaling={rate_scaling_factor_opt:.4f}',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('rtd_comparison_multistim_choice.png', dpi=300, bbox_inches='tight')
print("✓ Saved: rtd_comparison_multistim_choice.png")
plt.show()

# %%
# Plot CDFs for all stimuli in a 2x4 grid
print("\n" + "="*70)
print("PLOTTING CDF COMPARISONS (2x4 GRID)")
print("="*70)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, stim in enumerate(stimuli):
    stim_key = f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"
    data = simulation_results[stim_key]
    ax = axes[idx]
    
    # Sort and compute CDFs
    ddm_sorted = np.sort(data['ddm_rts_decided'])
    poisson_sorted = np.sort(data['poisson_rts_decided'])
    ddm_cdf = np.arange(1, len(ddm_sorted) + 1) / len(ddm_sorted)
    poisson_cdf = np.arange(1, len(poisson_sorted) + 1) / len(poisson_sorted)
    
    # Plot CDFs
    ax.plot(ddm_sorted, ddm_cdf, 'b-', linewidth=2, label='DDM', alpha=0.7)
    ax.plot(poisson_sorted, poisson_cdf, 'r--', linewidth=2, label='Poisson', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Reaction Time (s)', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title(f'{stim_key}\nKS={data["ks_stat"]:.6f}', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.05)

plt.suptitle(f'DDM vs Poisson CDF Comparison (Multi-Stimulus with Choice)\n'
             f'N={N_opt}, c={c_opt:.6f}, theta={theta_opt:.4f}, rate_scaling={rate_scaling_factor_opt:.4f}',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('cdf_comparison_multistim_choice.png', dpi=300, bbox_inches='tight')
print("✓ Saved: cdf_comparison_multistim_choice.png")
plt.show()

# %%
# Plot choice probability comparison
print("\n" + "="*70)
print("PLOTTING CHOICE PROBABILITY COMPARISON")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar plot of P(right)
ax = axes[0]
x_pos = np.arange(len(stimuli))
width = 0.35

stim_labels = [f"ABL_{stim['ABL']}_ILD_{stim['ILD']}" for stim in stimuli]
p_right_ddm_vals = [simulation_results[label]['p_right_ddm'] for label in stim_labels]
p_right_poisson_vals = [simulation_results[label]['p_right_poisson'] for label in stim_labels]

bars1 = ax.bar(x_pos - width/2, p_right_ddm_vals, width, label='DDM', color='blue', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, p_right_poisson_vals, width, label='Poisson', color='red', alpha=0.7)

ax.set_xlabel('Stimulus', fontsize=12)
ax.set_ylabel('P(right)', fontsize=12)
ax.set_title('Choice Probability Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(stim_labels, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Scatter plot: DDM vs Poisson P(right)
ax = axes[1]
ax.scatter(p_right_ddm_vals, p_right_poisson_vals, s=100, alpha=0.7, c='purple', edgecolors='black')

# Add diagonal line (perfect agreement)
min_val = min(min(p_right_ddm_vals), min(p_right_poisson_vals))
max_val = max(max(p_right_ddm_vals), max(p_right_poisson_vals))
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Agreement')

# Label points
for i, label in enumerate(stim_labels):
    ax.annotate(label.replace('ABL_', '').replace('_ILD_', ','), 
                (p_right_ddm_vals[i], p_right_poisson_vals[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('DDM P(right)', fontsize=12)
ax.set_ylabel('Poisson P(right)', fontsize=12)
ax.set_title('DDM vs Poisson P(right)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.suptitle(f'Choice Probability Analysis\n'
             f'N={N_opt}, c={c_opt:.6f}, theta={theta_opt:.4f}, rate_scaling={rate_scaling_factor_opt:.4f}',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('choice_probability_comparison_multistim.png', dpi=300, bbox_inches='tight')
print("✓ Saved: choice_probability_comparison_multistim.png")
plt.show()

# %%
# Plot combined metrics: RT and Choice MSE
print("\n" + "="*70)
print("PLOTTING MSE BREAKDOWN")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

stim_labels = [f"ABL_{stim['ABL']}_ILD_{stim['ILD']}" for stim in stimuli]
rt_mse_vals = [simulation_results[label]['rt_mse'] for label in stim_labels]
choice_mse_vals = [simulation_results[label]['choice_mse'] for label in stim_labels]
total_mse_vals = [simulation_results[label]['total_mse'] for label in stim_labels]

# Stacked bar chart
ax = axes[0]
x_pos = np.arange(len(stimuli))
bars1 = ax.bar(x_pos, rt_mse_vals, label='RT MSE', color='skyblue', alpha=0.8)
bars2 = ax.bar(x_pos, choice_mse_vals, bottom=rt_mse_vals, label='Choice MSE', color='salmon', alpha=0.8)

ax.set_xlabel('Stimulus', fontsize=12)
ax.set_ylabel('MSE', fontsize=12)
ax.set_title('MSE Breakdown by Stimulus', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(stim_labels, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Individual MSE components
ax = axes[1]
width = 0.25
bars1 = ax.bar(x_pos - width, rt_mse_vals, width, label='RT MSE', color='skyblue', alpha=0.8)
bars2 = ax.bar(x_pos, choice_mse_vals, width, label='Choice MSE', color='salmon', alpha=0.8)
bars3 = ax.bar(x_pos + width, total_mse_vals, width, label='Total MSE', color='purple', alpha=0.8)

ax.set_xlabel('Stimulus', fontsize=12)
ax.set_ylabel('MSE', fontsize=12)
ax.set_title('Individual MSE Components', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(stim_labels, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'MSE Analysis Across Stimuli\n'
             f'Total MSE (sum): {sum(total_mse_vals):.6f}, Average: {np.mean(total_mse_vals):.6f}',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('mse_breakdown_multistim.png', dpi=300, bbox_inches='tight')
print("✓ Saved: mse_breakdown_multistim.png")
plt.show()

print("\n" + "="*70)
print("ALL PLOTS COMPLETE")
print("="*70)

# %%
# Psychometric curves: P(right) vs ILD for different ABL values
print("\n" + "="*70)
print("GENERATING PSYCHOMETRIC CURVES")
print("="*70)

# Define ABL and ILD ranges
ABL_values = [20, 40, 60]  # Test generalization to ABL=40 (not in training)
ILD_values = np.arange(-16, 18, 2)  # -16 to 16 in steps of 2
n_trials_psychometric = int(50e3)  # 50K trials per condition for psychometric

print(f"\nABL values: {ABL_values}")
print(f"ILD range: {ILD_values[0]} to {ILD_values[-1]} (step={ILD_values[1]-ILD_values[0]})")
print(f"Trials per condition: {n_trials_psychometric}")

# Store psychometric data
psychometric_data = {}

for ABL in ABL_values:
    print(f"\n{'='*70}")
    print(f"ABL = {ABL}")
    print(f"{'='*70}")
    
    p_right_ddm_list = []
    p_right_poisson_list = []
    
    for ILD in tqdm(ILD_values, desc=f'Psychometric ABL={ABL}', ncols=80):
        # Calculate DDM rates
        ddm_right_rate, ddm_left_rate = lr_rates_from_ABL_ILD(
            ABL, ILD, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
        )
        
        # Calculate Poisson rates using scaling factor
        poisson_right_rate = ddm_right_rate * rate_scaling_factor_opt
        poisson_left_rate = ddm_left_rate * rate_scaling_factor_opt
        
        # DDM simulation
        mu_ddm = ddm_right_rate - ddm_left_rate
        sigma_ddm = np.sqrt(ddm_right_rate + ddm_left_rate)
        
        ddm_results = Parallel(n_jobs=-1)(
            delayed(simulate_single_ddm_trial)(j, mu_ddm, sigma_ddm, ddm_params['theta'], dt, dB, n_steps) 
            for j in range(n_trials_psychometric)
        )
        ddm_data = np.array(ddm_results)
        ddm_decided_mask = ~np.isnan(ddm_data[:, 0])
        ddm_choices = ddm_data[ddm_decided_mask, 1]
        p_right_ddm = np.mean(ddm_choices == 1)
        
        # Poisson simulation
        poisson_params = {
            'N_right': N_opt,
            'N_left': N_opt,
            'c': c_opt,
            'r_right': poisson_right_rate,
            'r_left': poisson_left_rate,
            'theta': theta_opt,
            'T': 20,
            'exponential_noise_scale': 0
        }
        
        poisson_results = simulate_poisson_rts(poisson_params, n_trials=n_trials_psychometric, 
                                               seed=42, verbose=False)
        poisson_decided_mask = ~np.isnan(poisson_results[:, 0])
        poisson_choices = poisson_results[poisson_decided_mask, 1]
        p_right_poisson = np.mean(poisson_choices == 1)
        
        p_right_ddm_list.append(p_right_ddm)
        p_right_poisson_list.append(p_right_poisson)
    
    psychometric_data[ABL] = {
        'ILD_values': ILD_values,
        'p_right_ddm': np.array(p_right_ddm_list),
        'p_right_poisson': np.array(p_right_poisson_list)
    }
    
    print(f"  Completed ABL={ABL}: {len(ILD_values)} ILD conditions")

print(f"\n{'='*70}")
print("PSYCHOMETRIC DATA GENERATION COMPLETE")
print(f"{'='*70}\n")

# %%
# Plot psychometric curves
print("="*70)
print("PLOTTING PSYCHOMETRIC CURVES")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, ABL in enumerate(ABL_values):
    ax = axes[idx]
    data = psychometric_data[ABL]
    
    # Plot psychometric curves
    ax.plot(data['ILD_values'], data['p_right_ddm'], 'o-', color='blue', 
            linewidth=2, markersize=8, label='DDM', alpha=0.7)
    ax.plot(data['ILD_values'], data['p_right_poisson'], 's--', color='red', 
            linewidth=2, markersize=8, label='Poisson', alpha=0.7)
    
    # Add horizontal line at 0.5
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Add vertical line at ILD=0
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('ILD (dB)', fontsize=12)
    ax.set_ylabel('P(right)', fontsize=12)
    ax.set_title(f'ABL = {ABL} dB', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(ILD_values[0] - 1, ILD_values[-1] + 1)

plt.suptitle(f'Psychometric Curves: P(right) vs ILD\n'
             f'N={N_opt}, c={c_opt:.6f}, theta={theta_opt:.4f}, rate_scaling={rate_scaling_factor_opt:.4f}',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('psychometric_curves_multistim.png', dpi=300, bbox_inches='tight')
print("✓ Saved: psychometric_curves_multistim.png")
plt.show()

print("\n" + "="*70)
print("PSYCHOMETRIC ANALYSIS COMPLETE")
print("="*70)

# %%
# Mean RT curves: mean RT vs ILD for different ABL values
print("\n" + "="*70)
print("GENERATING MEAN RT CURVES")
print("="*70)

# Define ABL and ILD ranges for RT analysis
ABL_values_rt = [20, 40, 60]  # Test generalization to ABL=40 (not in training)
ILD_values_rt = np.array([1, 2, 4, 8, 16])
n_trials_rt = int(50e3)  # 50K trials per condition

print(f"\nABL values: {ABL_values_rt}")
print(f"ILD values: {ILD_values_rt}")
print(f"Trials per condition: {n_trials_rt}")

# Store mean RT data
mean_rt_data = {}

for ABL in ABL_values_rt:
    print(f"\n{'='*70}")
    print(f"ABL = {ABL}")
    print(f"{'='*70}")
    
    mean_rt_ddm_list = []
    mean_rt_poisson_list = []
    raw_ddm_rts_dict = {}
    raw_ddm_choices_dict = {}
    raw_poisson_rts_dict = {}
    raw_poisson_choices_dict = {}
    
    for ILD in tqdm(ILD_values_rt, desc=f'Mean RT ABL={ABL}', ncols=80):
        # Calculate DDM rates
        ddm_right_rate, ddm_left_rate = lr_rates_from_ABL_ILD(
            ABL, ILD, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
        )
        
        # Calculate Poisson rates using scaling factor
        poisson_right_rate = ddm_right_rate * rate_scaling_factor_opt
        poisson_left_rate = ddm_left_rate * rate_scaling_factor_opt
        
        # DDM simulation
        mu_ddm = ddm_right_rate - ddm_left_rate
        sigma_ddm = np.sqrt(ddm_right_rate + ddm_left_rate)
        
        ddm_results = Parallel(n_jobs=-1)(
            delayed(simulate_single_ddm_trial)(j, mu_ddm, sigma_ddm, ddm_params['theta'], dt, dB, n_steps) 
            for j in range(n_trials_rt)
        )
        ddm_data = np.array(ddm_results)
        ddm_decided_mask = ~np.isnan(ddm_data[:, 0])
        ddm_rts = ddm_data[ddm_decided_mask, 0]
        ddm_choices = ddm_data[ddm_decided_mask, 1]
        mean_rt_ddm = np.mean(ddm_rts)
        
        # Poisson simulation
        poisson_params = {
            'N_right': N_opt,
            'N_left': N_opt,
            'c': c_opt,
            'r_right': poisson_right_rate,
            'r_left': poisson_left_rate,
            'theta': theta_opt,
            'T': 20,
            'exponential_noise_scale': 0
        }
        
        poisson_results = simulate_poisson_rts(poisson_params, n_trials=n_trials_rt, 
                                               seed=42, verbose=False)
        poisson_decided_mask = ~np.isnan(poisson_results[:, 0])
        poisson_rts = poisson_results[poisson_decided_mask, 0]
        poisson_choices = poisson_results[poisson_decided_mask, 1]
        mean_rt_poisson = np.mean(poisson_rts)
        
        # Store mean values
        mean_rt_ddm_list.append(mean_rt_ddm)
        mean_rt_poisson_list.append(mean_rt_poisson)
        
        # Store raw data
        raw_ddm_rts_dict[ILD] = ddm_rts
        raw_ddm_choices_dict[ILD] = ddm_choices
        raw_poisson_rts_dict[ILD] = poisson_rts
        raw_poisson_choices_dict[ILD] = poisson_choices
    
    mean_rt_data[ABL] = {
        'ILD_values': ILD_values_rt,
        'mean_rt_ddm': np.array(mean_rt_ddm_list),
        'mean_rt_poisson': np.array(mean_rt_poisson_list),
        'raw_ddm_rts': raw_ddm_rts_dict,
        'raw_ddm_choices': raw_ddm_choices_dict,
        'raw_poisson_rts': raw_poisson_rts_dict,
        'raw_poisson_choices': raw_poisson_choices_dict
    }
    
    print(f"  Completed ABL={ABL}: {len(ILD_values_rt)} ILD conditions")

print(f"\n{'='*70}")
print("MEAN RT DATA GENERATION COMPLETE")
print(f"{'='*70}\n")

# %%
# Plot mean RT curves (all ABL values on same plot)
print("="*70)
print("PLOTTING MEAN RT CURVES")
print("="*70)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Define colors for different ABL values
colors = ['tab:blue', 'tab:orange', 'tab:green']  # One color per ABL value

for idx, ABL in enumerate(ABL_values_rt):
    data = mean_rt_data[ABL]
    color = colors[idx]
    
    # Plot mean RT curves for this ABL
    ax.plot(data['ILD_values'], data['mean_rt_ddm'], 'o-', color=color, 
            linewidth=2, markersize=8, label=f'DDM (ABL={ABL})', alpha=0.7)
    ax.plot(data['ILD_values'], data['mean_rt_poisson'], 's--', color=color, 
            linewidth=2, markersize=8, label=f'Poisson (ABL={ABL})', alpha=0.7)

# Formatting
ax.set_xlabel('ILD (dB)', fontsize=12)
ax.set_ylabel('Mean RT (s)', fontsize=12)
ax.set_title(f'Mean RT vs ILD for Different ABL Values\n'
             f'N={N_opt}, c={c_opt:.6f}, theta={theta_opt:.4f}, rate_scaling={rate_scaling_factor_opt:.4f}',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='best', ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(ILD_values_rt[0] - 0.5, ILD_values_rt[-1] + 0.5)

plt.tight_layout()
plt.savefig('mean_rt_curves_multistim.png', dpi=300, bbox_inches='tight')
print("✓ Saved: mean_rt_curves_multistim.png")
plt.show()

print("\n" + "="*70)
print("MEAN RT ANALYSIS COMPLETE")
print("="*70)

# %%
# Choice-conditioned RT distributions
# +1 choice RTD on positive y-axis (area = P(choice=+1))
# -1 choice RTD on negative y-axis (area = P(choice=-1))
print("\n" + "="*70)
print("PLOTTING CHOICE-CONDITIONED RT DISTRIBUTIONS")
print("="*70)

# Create 3x5 grid (3 ABL rows × 5 ILD columns)
fig, axes = plt.subplots(3, 5, figsize=(20, 12))

# RT bins for histograms
rt_bins = np.arange(0, 2.01, 0.02)
rt_bin_centers = (rt_bins[:-1] + rt_bins[1:]) / 2

for row_idx, ABL in enumerate(ABL_values_rt):
    data_abl = mean_rt_data[ABL]
    
    for col_idx, ILD in enumerate(ILD_values_rt):
        ax = axes[row_idx, col_idx]
        
        # Get raw data for this ABL-ILD combination
        ddm_rts = data_abl['raw_ddm_rts'][ILD]
        ddm_choices = data_abl['raw_ddm_choices'][ILD]
        poisson_rts = data_abl['raw_poisson_rts'][ILD]
        poisson_choices = data_abl['raw_poisson_choices'][ILD]
        
        # Separate by choice for DDM
        ddm_rts_plus1 = ddm_rts[ddm_choices == 1]
        ddm_rts_minus1 = ddm_rts[ddm_choices == -1]
        p_plus1_ddm = len(ddm_rts_plus1) / len(ddm_rts)
        p_minus1_ddm = len(ddm_rts_minus1) / len(ddm_rts)
        
        # Separate by choice for Poisson
        poisson_rts_plus1 = poisson_rts[poisson_choices == 1]
        poisson_rts_minus1 = poisson_rts[poisson_choices == -1]
        p_plus1_poisson = len(poisson_rts_plus1) / len(poisson_rts)
        p_minus1_poisson = len(poisson_rts_minus1) / len(poisson_rts)
        
        # Compute histograms for DDM
        if len(ddm_rts_plus1) > 0:
            ddm_hist_plus1, _ = np.histogram(ddm_rts_plus1, bins=rt_bins, density=True)
            ddm_hist_plus1_weighted = ddm_hist_plus1 * p_plus1_ddm  # Scale by choice probability
        else:
            ddm_hist_plus1_weighted = np.zeros(len(rt_bin_centers))
        
        if len(ddm_rts_minus1) > 0:
            ddm_hist_minus1, _ = np.histogram(ddm_rts_minus1, bins=rt_bins, density=True)
            ddm_hist_minus1_weighted = -ddm_hist_minus1 * p_minus1_ddm  # Negative y-axis
        else:
            ddm_hist_minus1_weighted = np.zeros(len(rt_bin_centers))
        
        # Compute histograms for Poisson
        if len(poisson_rts_plus1) > 0:
            poisson_hist_plus1, _ = np.histogram(poisson_rts_plus1, bins=rt_bins, density=True)
            poisson_hist_plus1_weighted = poisson_hist_plus1 * p_plus1_poisson
        else:
            poisson_hist_plus1_weighted = np.zeros(len(rt_bin_centers))
        
        if len(poisson_rts_minus1) > 0:
            poisson_hist_minus1, _ = np.histogram(poisson_rts_minus1, bins=rt_bins, density=True)
            poisson_hist_minus1_weighted = -poisson_hist_minus1 * p_minus1_poisson
        else:
            poisson_hist_minus1_weighted = np.zeros(len(rt_bin_centers))
        
        # Plot +1 choice (positive y-axis)
        ax.plot(rt_bin_centers, ddm_hist_plus1_weighted, '-', color='blue', 
                linewidth=1.5, alpha=0.7, label='DDM (+1)' if row_idx == 0 and col_idx == 0 else '')
        ax.plot(rt_bin_centers, poisson_hist_plus1_weighted, '--', color='red', 
                linewidth=1.5, alpha=0.7, label='Poisson (+1)' if row_idx == 0 and col_idx == 0 else '')
        
        # Plot -1 choice (negative y-axis)
        ax.plot(rt_bin_centers, ddm_hist_minus1_weighted, '-', color='blue', 
                linewidth=1.5, alpha=0.7, label='DDM (-1)' if row_idx == 0 and col_idx == 0 else '')
        ax.plot(rt_bin_centers, poisson_hist_minus1_weighted, '--', color='red', 
                linewidth=1.5, alpha=0.7, label='Poisson (-1)' if row_idx == 0 and col_idx == 0 else '')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Formatting
        ax.set_xlim(0, 1.0)
        ax.grid(True, alpha=0.3)
        
        # Labels only on edges
        if col_idx == 0:
            ax.set_ylabel(f'ABL={ABL}\nWeighted Density', fontsize=9)
        else:
            ax.set_yticklabels([])
        
        if row_idx == 2:  # Last row (0-indexed, so row 2 is the 3rd row)
            ax.set_xlabel('RT (s)', fontsize=9)
        else:
            ax.set_xticklabels([])
        
        # Title with ILD value
        if row_idx == 0:
            ax.set_title(f'ILD={ILD}', fontsize=10, fontweight='bold')
        
        # Add text showing P(+1) and P(-1)
        ax.text(0.98, 0.98, f'P(+1):\nD:{p_plus1_ddm:.2f}\nP:{p_plus1_poisson:.2f}', 
                transform=ax.transAxes, fontsize=7, verticalalignment='top', 
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Add legend to first subplot
axes[0, 0].legend(fontsize=8, loc='upper right')

plt.suptitle(f'Choice-Conditioned RT Distributions\n'
             f'Top: choice=+1 (right), Bottom: choice=-1 (left) | '
             f'Solid=DDM, Dashed=Poisson\n'
             f'N={N_opt}, c={c_opt:.6f}, theta={theta_opt:.4f}, rate_scaling={rate_scaling_factor_opt:.4f}',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('choice_conditioned_rtd_grid.png', dpi=300, bbox_inches='tight')
print("✓ Saved: choice_conditioned_rtd_grid.png")
plt.show()

print("\n" + "="*70)
print("CHOICE-CONDITIONED RTD ANALYSIS COMPLETE")
print("="*70)

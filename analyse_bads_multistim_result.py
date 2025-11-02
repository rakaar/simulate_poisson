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
# Find and load the most recent multi-stimulus BADS results
pkl_pattern = 'bads_multistim_results_*.pkl'
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

with open(latest_pkl, 'rb') as f:
    results = pickle.load(f)

# Extract data
optimize_result = results['bads_result']
optimized_params = results['optimized_params']
ddm_params = results['ddm_params']
stimuli = results['stimuli']
ddm_simulation = results['ddm_simulation']

# Extract shared parameters
N_opt = optimized_params['N']
k_opt = optimized_params['k']
c_opt = optimized_params['c']
theta_opt = optimized_params['theta']
x_opt = optimized_params['x_opt']

print("\n" + "="*70)
print("LOADED MULTI-STIMULUS BADS OPTIMIZATION RESULTS")
print("="*70)
print(f"\nShared optimized parameters:")
print(f"  N:     {N_opt}")
print(f"  k:     {k_opt:.4f}")
print(f"  c:     {c_opt:.6f} (= k/N)")
print(f"  theta: {theta_opt:.4f}")

print(f"\nOptimized rates for each stimulus:")
rates_per_stimulus = optimized_params['rates_per_stimulus']
for stim in stimuli:
    stim_key = f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"
    r1 = rates_per_stimulus[stim_key]['r1_right']
    r2 = rates_per_stimulus[stim_key]['r2_left']
    print(f"  {stim_key}:")
    print(f"    r1 (right): {r1:.6f}")
    print(f"    r2 (left):  {r2:.6f}")
    print(f"    r1 - r2:    {r1 - r2:.6f}")

print(f"\nOptimization Statistics:")
print(f"  Total KS-statistic (sum): {optimize_result['fval']:.6f}")
print(f"  Average KS per stimulus:  {optimize_result['fval']/len(stimuli):.6f}")
print(f"  Function evaluations:     {optimize_result['func_count']}")
print(f"  Success:                  {optimize_result['success']}")

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
    
    # Get rates for this stimulus
    r_right_poisson = rates_per_stimulus[stim_key]['r1_right']
    r_left_poisson = rates_per_stimulus[stim_key]['r2_left']
    
    # DDM simulation
    print(f"\n--- DDM Simulation ---")
    ddm_right_rate, ddm_left_rate = lr_rates_from_ABL_ILD(
        ABL, ILD, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
    )
    
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
    
    print(f"DDM: {len(ddm_rts_decided)}/{N_sim} decided, mean RT={np.mean(ddm_rts_decided):.4f}s")
    
    # Poisson simulation
    print(f"\n--- Poisson Simulation ---")
    poisson_params = {
        'N_right': N_opt,
        'N_left': N_opt,
        'c': c_opt,
        'r_right': r_right_poisson,
        'r_left': r_left_poisson,
        'theta': theta_opt,
        'T': 20,
        'exponential_noise_scale': 0
    }
    
    print(f"Poisson rates: right={r_right_poisson:.6f}, left={r_left_poisson:.6f}")
    print(f"Poisson params: N={N_opt}, c={c_opt:.6f}, theta={theta_opt:.4f}")
    print(f"Simulating {N_sim} Poisson trials...")
    
    poisson_results = simulate_poisson_rts(poisson_params, n_trials=N_sim, 
                                           seed=42, verbose=False)
    poisson_decided_mask = ~np.isnan(poisson_results[:, 0])
    poisson_rts_decided = poisson_results[poisson_decided_mask, 0]
    
    print(f"Poisson: {len(poisson_rts_decided)}/{N_sim} decided, mean RT={np.mean(poisson_rts_decided):.4f}s")
    
    # Compute KS statistic
    ks_stat, ks_pval = ks_2samp(ddm_rts_decided, poisson_rts_decided)
    print(f"\nKS-statistic: {ks_stat:.6f}, p-value: {ks_pval:.6e}")
    
    # Store results
    simulation_results[stim_key] = {
        'ABL': ABL,
        'ILD': ILD,
        'ddm_right_rate': ddm_right_rate,
        'ddm_left_rate': ddm_left_rate,
        'poisson_r_right': r_right_poisson,
        'poisson_r_left': r_left_poisson,
        'ddm_data': ddm_data,
        'ddm_rts_decided': ddm_rts_decided,
        'poisson_data': poisson_results,
        'poisson_rts_decided': poisson_rts_decided,
        'ks_stat': ks_stat,
        'ks_pval': ks_pval
    }

print(f"\n{'='*70}")
print("ALL SIMULATIONS COMPLETE")
print(f"{'='*70}\n")

# %%
# Plot RTD comparison for all stimuli in a 2x2 grid
print("="*70)
print("PLOTTING RTD COMPARISONS (2x2 GRID)")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
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
    ax.set_title(f'{stim_key}\nKS={data["ks_stat"]:.6f}', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

plt.suptitle(f'DDM vs Poisson RTD Comparison (Multi-Stimulus)\n'
             f'N={N_opt}, c={c_opt:.6f}, theta={theta_opt:.4f}',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('rtd_comparison_multistim.png', dpi=300, bbox_inches='tight')
print("✓ Saved: rtd_comparison_multistim.png")
plt.show()

# %%
# Plot CDFs for all stimuli in a 2x2 grid
print("\n" + "="*70)
print("PLOTTING CDF COMPARISONS (2x2 GRID)")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
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

plt.suptitle(f'DDM vs Poisson CDF Comparison (Multi-Stimulus)\n'
             f'N={N_opt}, c={c_opt:.6f}, theta={theta_opt:.4f}',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('cdf_comparison_multistim.png', dpi=300, bbox_inches='tight')
print("✓ Saved: cdf_comparison_multistim.png")
plt.show()

# %%
# Summary statistics table
print("\n" + "="*70)
print("SUMMARY STATISTICS TABLE")
print("="*70)

print("\n{:<15} {:<12} {:<12} {:<12} {:<12}".format(
    "Stimulus", "Mean RT (s)", "Std RT (s)", "N Decided", "KS-stat"
))
print("{:<15} {:<12} {:<12} {:<12} {:<12}".format(
    "", "DDM / Pois", "DDM / Pois", "DDM / Pois", ""
))
print("-" * 75)

for stim in stimuli:
    stim_key = f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"
    data = simulation_results[stim_key]
    
    ddm_mean = np.mean(data['ddm_rts_decided'])
    poisson_mean = np.mean(data['poisson_rts_decided'])
    ddm_std = np.std(data['ddm_rts_decided'])
    poisson_std = np.std(data['poisson_rts_decided'])
    ddm_n = len(data['ddm_rts_decided'])
    poisson_n = len(data['poisson_rts_decided'])
    
    print("{:<15} {:.4f}/{:.4f}  {:.4f}/{:.4f}  {:>5}/{:<5}  {:.6f}".format(
        stim_key, ddm_mean, poisson_mean, ddm_std, poisson_std, 
        ddm_n, poisson_n, data['ks_stat']
    ))

total_ks = sum(simulation_results[f"ABL_{s['ABL']}_ILD_{s['ILD']}"]['ks_stat'] for s in stimuli)
avg_ks = total_ks / len(stimuli)
print("-" * 75)
print(f"{'Total KS (sum):':<15} {total_ks:.6f}")
print(f"{'Average KS:':<15} {avg_ks:.6f}")

# %%
# Parameter comparison table
print("\n" + "="*70)
print("PARAMETER COMPARISON: DDM vs POISSON")
print("="*70)

print("\n{:<15} {:<15} {:<15} {:<15} {:<12}".format(
    "Stimulus", "Rate", "DDM", "Poisson", "Ratio (P/D)"
))
print("-" * 75)

for stim in stimuli:
    stim_key = f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"
    data = simulation_results[stim_key]
    
    # Right rate comparison
    ratio_right = data['poisson_r_right'] / (data['ddm_right_rate'] / N_opt)
    print("{:<15} {:<15} {:<15.6f} {:<15.6f} {:<12.4f}".format(
        stim_key, "right", data['ddm_right_rate']/N_opt, 
        data['poisson_r_right'], ratio_right
    ))
    
    # Left rate comparison
    ratio_left = data['poisson_r_left'] / (data['ddm_left_rate'] / N_opt)
    print("{:<15} {:<15} {:<15.6f} {:<15.6f} {:<12.4f}".format(
        "", "left", data['ddm_left_rate']/N_opt, 
        data['poisson_r_left'], ratio_left
    ))
    print()

print("\nShared Poisson Parameters:")
print(f"  N:     {N_opt}")
print(f"  k:     {k_opt:.4f}")
print(f"  c:     {c_opt:.6f}")
print(f"  theta: {theta_opt:.4f} (DDM theta: {ddm_params['theta']:.4f})")

# %%
# Plot mean RT vs ILD for each ABL
print("\n" + "="*70)
print("PLOTTING MEAN RT vs ILD")
print("="*70)

# Organize data by ABL
abl_values = sorted(set(s['ABL'] for s in stimuli))

fig, ax = plt.subplots(figsize=(12, 6))

for ABL in abl_values:
    # Get stimuli for this ABL
    stim_subset = [s for s in stimuli if s['ABL'] == ABL]
    ild_vals = sorted([s['ILD'] for s in stim_subset])
    
    ddm_means = []
    poisson_means = []
    
    for ILD in ild_vals:
        stim_key = f"ABL_{ABL}_ILD_{ILD}"
        data = simulation_results[stim_key]
        ddm_means.append(np.mean(data['ddm_rts_decided']))
        poisson_means.append(np.mean(data['poisson_rts_decided']))
    
    # Plot
    ax.plot(ild_vals, ddm_means, 'o-', linewidth=2, markersize=8, 
            label=f'DDM ABL={ABL}', alpha=0.7)
    ax.plot(ild_vals, poisson_means, 'x--', linewidth=2, markersize=10, 
            label=f'Poisson ABL={ABL}', alpha=0.7)

ax.set_xlabel('ILD (dB)', fontsize=12)
ax.set_ylabel('Mean RT (s)', fontsize=12)
ax.set_title('Mean RT vs ILD: DDM vs Poisson', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mean_rt_vs_ild_multistim.png', dpi=300, bbox_inches='tight')
print("✓ Saved: mean_rt_vs_ild_multistim.png")
plt.show()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

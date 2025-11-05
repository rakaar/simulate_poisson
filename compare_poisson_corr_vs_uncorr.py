# %%
# Compare Poisson Correlated vs Uncorrelated (Skellam) Models
# 
# For each ABL-ILD combination:
# 1. Get DDM left/right rates
# 2. Simulate Poisson Correlated (using BADS optimized params)
# 3. Simulate Poisson Uncorrelated (Skellam process with mu1, mu2 = DDM_rates/N)

# %%
# Imports
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
    simulate_poisson_rts
)
from simulators import simulate_skellam_trial

# %%
# Load the most recent multi-stimulus BADS results
pkl_pattern = 'bads_multistim_scaling_results_*.pkl'
pkl_files = glob.glob(pkl_pattern)

if not pkl_files:
    raise FileNotFoundError(f"No files matching '{pkl_pattern}' found in current directory")

# Get the most recent file
# latest_pkl = max(pkl_files, key=lambda x: Path(x).stat().st_mtime)
latest_pkl = "bads_multistim_scaling_results_20251103_121932.pkl"
print(f"Loading results from: {latest_pkl}")

# Define dummy functions for unpickling
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

# Extract optimized parameters for Poisson Correlated
N_opt = optimized_params['N']
k_opt = optimized_params['k']
c_opt = optimized_params['c']
theta_opt = optimized_params['theta']
rate_scaling_factor_opt = optimized_params['rate_scaling_factor']

# Extract DDM parameters
theta_ddm = ddm_params['theta']
Nr0_ddm = ddm_params['Nr0']
lam_ddm = ddm_params['lam']
ell_ddm = ddm_params['ell']

print("\n" + "="*70)
print("LOADED BADS OPTIMIZATION RESULTS")
print("="*70)
print(f"\nPoisson Corr params (BADS):")
print(f"  N:                   {N_opt}")
print(f"  k:                   {k_opt:.4f}")
print(f"  c:                   {c_opt:.6f}")
print(f"  theta:               {theta_opt:.4f}")
print(f"  rate_scaling_factor: {rate_scaling_factor_opt:.4f}")

print(f"\nDDM params:")
print(f"  theta:               {theta_ddm:.4f}")
print(f"  Nr0:                 {Nr0_ddm:.4f}")
print(f"  lambda:              {lam_ddm:.4f}")
print(f"  ell:                 {ell_ddm:.4f}")

print(f"\nPoisson Uncorr params:")
print(f"  N:                   {N_opt}")
print(f"  theta:               {theta_ddm:.4f} (DDM theta)")
print(f"  mu1, mu2:            DDM rates / {N_opt} (per-neuron rates)")

# %%
# Define stimulus conditions
ABL_values = [20, 40, 60]
ILD_values = [1, 2, 4, 8, 16]

# Simulation parameters
N_sim = int(50e3)  # 100K trials per condition
T = 20  # Max trial duration (seconds)

print("\n" + "="*70)
print("STIMULUS CONDITIONS")
print("="*70)
print(f"\nABL values: {ABL_values}")
print(f"ILD values: {ILD_values}")
print(f"Total conditions: {len(ABL_values) * len(ILD_values)}")
print(f"Trials per condition: {N_sim}")
print(f"Max trial duration: {T}s")

# %%
# Skellam (Poisson Uncorrelated) simulation function
def simulate_skellam_rts(N, mu1, mu2, theta, n_trials, seed=None, verbose=True):
    """
    Simulate Skellam process (uncorrelated Poisson) using N independent trials.
    
    For each trial, we simulate a Skellam random walk where evidence comes from
    N independent neurons, each contributing Poisson(mu1) - Poisson(mu2) jumps.
    
    This is equivalent to simulating a single Skellam trial with rates N*mu1 and N*mu2.
    
    Parameters:
    -----------
    N : int
        Number of neurons (for consistency with correlated model)
    mu1 : float
        Right rate per neuron (will be multiplied by N)
    mu2 : float
        Left rate per neuron (will be multiplied by N)
    theta : int
        Decision threshold
    n_trials : int
        Number of trials to simulate
    seed : int, optional
        Random seed
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    results : np.ndarray of shape (n_trials, 2)
        Column 0: reaction time (or nan if no decision)
        Column 1: choice (+1 or -1, or nan if no decision)
    """
    rng = np.random.default_rng(seed)
    
    # Total rates (summed across N neurons)
    total_mu1 = N * mu1
    total_mu2 = N * mu2
    
    if verbose:
        print(f"  Simulating {n_trials} Skellam trials...")
        print(f"  Per-neuron rates: mu1={mu1:.6f}, mu2={mu2:.6f}")
        print(f"  Total rates: {total_mu1:.6f} (right), {total_mu2:.6f} (left)")
        print(f"  Threshold: {theta}")
    
    def run_trial(trial_idx):
        """Run a single trial and return (rt, choice)"""
        local_seed = seed + trial_idx if seed is not None else None
        rt, choice = simulate_skellam_trial(total_mu1, total_mu2, theta, 
                                           rng=np.random.default_rng(local_seed))
        # Check if trial exceeded max time
        if rt > T:
            return [np.nan, np.nan]
        return [rt, choice]
    
    # Parallel simulation
    results = Parallel(n_jobs=-1)(
        delayed(run_trial)(i) for i in tqdm(range(n_trials), 
                                            desc='Skellam', 
                                            ncols=80, 
                                            disable=not verbose)
    )
    
    return np.array(results)

# %%
# Main simulation loop
print("\n" + "="*70)
print("RUNNING SIMULATIONS FOR ALL STIMULUS CONDITIONS")
print("="*70)

# Dictionary to store all results
all_results = {}

for ABL in ABL_values:
    for ILD in ILD_values:
        stim_key = f"ABL_{ABL}_ILD_{ILD}"
        
        print(f"\n{'='*70}")
        print(f"STIMULUS: {stim_key}")
        print(f"{'='*70}")
        
        # Step 1: Calculate DDM rates from ABL, ILD
        ddm_right_rate, ddm_left_rate = lr_rates_from_ABL_ILD(
            ABL, ILD, Nr0_ddm, lam_ddm, ell_ddm
        )
        
        print(f"\nDDM rates:")
        print(f"  Right rate: {ddm_right_rate:.6f}")
        print(f"  Left rate:  {ddm_left_rate:.6f}")
        
        # Step 2: Simulate Poisson Correlated
        print(f"\n--- Poisson Correlated ---")
        
        # Calculate Poisson correlated rates using scaling factor
        poisson_corr_right_rate = ddm_right_rate * rate_scaling_factor_opt
        poisson_corr_left_rate = ddm_left_rate * rate_scaling_factor_opt
        
        poisson_corr_params = {
            'N_right': N_opt,
            'N_left': N_opt,
            'c': c_opt,
            'r_right': poisson_corr_right_rate,
            'r_left': poisson_corr_left_rate,
            'theta': theta_opt,
            'T': T,
            'exponential_noise_scale': 0
        }
        
        print(f"Parameters:")
        print(f"  N: {N_opt}")
        print(f"  c: {c_opt:.6f}")
        print(f"  r_right: {poisson_corr_right_rate:.6f}")
        print(f"  r_left: {poisson_corr_left_rate:.6f}")
        print(f"  theta: {theta_opt:.4f}")
        
        poisson_corr_results = simulate_poisson_rts(
            poisson_corr_params, 
            n_trials=N_sim, 
            seed=42, 
            verbose=False
        )
        
        poisson_corr_decided_mask = ~np.isnan(poisson_corr_results[:, 0])
        poisson_corr_rts = poisson_corr_results[poisson_corr_decided_mask, 0]
        poisson_corr_choices = poisson_corr_results[poisson_corr_decided_mask, 1]
        
        mean_rt_corr = np.mean(poisson_corr_rts)
        p_right_corr = np.mean(poisson_corr_choices == 1)
        
        print(f"Results:")
        print(f"  Decided: {len(poisson_corr_rts)}/{N_sim}")
        print(f"  Mean RT: {mean_rt_corr:.4f}s")
        print(f"  P(right): {p_right_corr:.4f}")
        
        # Step 3: Simulate Poisson Uncorrelated (Skellam)
        print(f"\n--- Poisson Uncorrelated (Skellam) ---")
        
        # Per-neuron rates: DDM rates divided by N
        mu1_per_neuron = ddm_right_rate / N_opt
        mu2_per_neuron = ddm_left_rate / N_opt
        
        print(f"Parameters:")
        print(f"  N: {N_opt}")
        print(f"  mu1 (per neuron): {mu1_per_neuron:.6f}")
        print(f"  mu2 (per neuron): {mu2_per_neuron:.6f}")
        print(f"  Total mu1 (N*mu1): {N_opt * mu1_per_neuron:.6f}")
        print(f"  Total mu2 (N*mu2): {N_opt * mu2_per_neuron:.6f}")
        print(f"  theta: {theta_ddm:.4f}")
        
        poisson_uncorr_results = simulate_skellam_rts(
            N=N_opt,
            mu1=mu1_per_neuron,
            mu2=mu2_per_neuron,
            theta=int(theta_ddm),
            n_trials=N_sim,
            seed=42,
            verbose=False
        )
        
        poisson_uncorr_decided_mask = ~np.isnan(poisson_uncorr_results[:, 0])
        poisson_uncorr_rts = poisson_uncorr_results[poisson_uncorr_decided_mask, 0]
        poisson_uncorr_choices = poisson_uncorr_results[poisson_uncorr_decided_mask, 1]
        
        mean_rt_uncorr = np.mean(poisson_uncorr_rts)
        p_right_uncorr = np.mean(poisson_uncorr_choices == 1)
        
        print(f"Results:")
        print(f"  Decided: {len(poisson_uncorr_rts)}/{N_sim}")
        print(f"  Mean RT: {mean_rt_uncorr:.4f}s")
        print(f"  P(right): {p_right_uncorr:.4f}")
        
        # Comparison metrics
        print(f"\n--- Comparison ---")
        ks_stat, ks_pval = ks_2samp(poisson_corr_rts, poisson_uncorr_rts)
        rt_diff = mean_rt_corr - mean_rt_uncorr
        p_right_diff = p_right_corr - p_right_uncorr
        
        print(f"KS-statistic: {ks_stat:.6f}, p-value: {ks_pval:.6e}")
        print(f"Mean RT difference (Corr - Uncorr): {rt_diff:.4f}s")
        print(f"P(right) difference (Corr - Uncorr): {p_right_diff:.4f}")
        
        # Store results
        all_results[stim_key] = {
            'ABL': ABL,
            'ILD': ILD,
            'ddm_right_rate': ddm_right_rate,
            'ddm_left_rate': ddm_left_rate,
            # Poisson Correlated
            'poisson_corr_rts': poisson_corr_rts,
            'poisson_corr_choices': poisson_corr_choices,
            'mean_rt_corr': mean_rt_corr,
            'p_right_corr': p_right_corr,
            # Poisson Uncorrelated
            'poisson_uncorr_rts': poisson_uncorr_rts,
            'poisson_uncorr_choices': poisson_uncorr_choices,
            'mean_rt_uncorr': mean_rt_uncorr,
            'p_right_uncorr': p_right_uncorr,
            # Comparison
            'ks_stat': ks_stat,
            'ks_pval': ks_pval,
            'rt_diff': rt_diff,
            'p_right_diff': p_right_diff
        }

print("\n" + "="*70)
print("ALL SIMULATIONS COMPLETE")
print("="*70)

# %%
# Plot RTD comparisons in a grid (3 rows × 5 columns for 3 ABL × 5 ILD)
print("\n" + "="*70)
print("PLOTTING RTD COMPARISONS")
print("="*70)

fig, axes = plt.subplots(3, 5, figsize=(25, 15))

bins = np.arange(0, 2.01, 0.01)

for row_idx, ABL in enumerate(ABL_values):
    for col_idx, ILD in enumerate(ILD_values):
        stim_key = f"ABL_{ABL}_ILD_{ILD}"
        data = all_results[stim_key]
        ax = axes[row_idx, col_idx]
        
        # Plot histograms
        ax.hist(data['poisson_corr_rts'], bins=bins, density=True, 
                color='blue', alpha=0.5, label='Poisson Corr', histtype='step')
        ax.hist(data['poisson_uncorr_rts'], bins=bins, density=True, 
                color='red', alpha=0.5, label='Poisson Uncorr', histtype='step')
        
        # Add outlines
        ax.hist(data['poisson_corr_rts'], bins=bins, density=True, 
                color='blue', histtype='step', linewidth=2)
        ax.hist(data['poisson_uncorr_rts'], bins=bins, density=True, 
                color='red', histtype='step', linewidth=2)
        
        # Formatting
        ax.set_xlim(0, 0.8)
        ax.grid(True, alpha=0.3)
        
        # Labels only on edges
        if col_idx == 0:
            ax.set_ylabel(f'ABL={ABL}\nDensity', fontsize=10)
        else:
            ax.set_yticklabels([])
        
        if row_idx == 2:  # Bottom row
            ax.set_xlabel('RT (s)', fontsize=10)
        else:
            ax.set_xticklabels([])
        
        # Title on top row
        if row_idx == 0:
            ax.set_title(f'ILD={ILD}', fontsize=11, fontweight='bold')
        
        # Add text with metrics
        ax.text(0.98, 0.98, 
                f'KS={data["ks_stat"]:.3f}\n'
                f'ΔRT={data["rt_diff"]:.3f}\n'
                f'ΔP(R)={data["p_right_diff"]:.3f}',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Legend only on first subplot
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=9, loc='upper left')

plt.suptitle(f'Poisson Correlated vs Uncorrelated RTD Comparison\n'
             f'Corr: N={N_opt}, c={c_opt:.6f}, theta={theta_opt:.4f} | '
             f'Uncorr: N={N_opt}, theta={theta_ddm:.4f}',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('compare_poisson_corr_vs_uncorr_rtd.png', dpi=300, bbox_inches='tight')
print("✓ Saved: compare_poisson_corr_vs_uncorr_rtd.png")
plt.show()


# %%
# Plot mean RT comparison (all ABL values in single plot)
print("\n" + "="*70)
print("PLOTTING MEAN RT COMPARISON")
print("="*70)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Define colors for different ABL values
colors = ['tab:blue', 'tab:orange', 'tab:green']

for idx, ABL in enumerate(ABL_values):
    color = colors[idx]
    
    mean_rt_corr_list = []
    mean_rt_uncorr_list = []
    
    for ILD in ILD_values:
        stim_key = f"ABL_{ABL}_ILD_{ILD}"
        data = all_results[stim_key]
        mean_rt_corr_list.append(data['mean_rt_corr'])
        mean_rt_uncorr_list.append(data['mean_rt_uncorr'])
    
    # Plot
    ax.plot(ILD_values, mean_rt_corr_list, 'o-', color=color, 
            linewidth=2, markersize=8, label=f'Corr (ABL={ABL})', alpha=0.7)
    ax.plot(ILD_values, mean_rt_uncorr_list, 's--', color=color, 
            linewidth=2, markersize=8, label=f'Uncorr (ABL={ABL})', alpha=0.7)

# Formatting
ax.set_xlabel('ILD (dB)', fontsize=12)
ax.set_ylabel('Mean RT (s)', fontsize=12)
ax.set_title(f'Mean RT vs ILD: Poisson Correlated vs Uncorrelated\n'
             f'Corr: N={N_opt}, c={c_opt:.6f}, theta={theta_opt:.4f} | '
             f'Uncorr: N={N_opt}, theta={theta_ddm:.4f}',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='best', ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(ILD_values[0] - 0.5, ILD_values[-1] + 0.5)

plt.tight_layout()
plt.savefig('compare_poisson_corr_vs_uncorr_mean_rt.png', dpi=300, bbox_inches='tight')
print("✓ Saved: compare_poisson_corr_vs_uncorr_mean_rt.png")
plt.show()

# %%
# Chronometric: Mean RT vs ILD for all ABL values (1x1 plot)
print("\n" + "="*70)
print("PLOTTING CHRONOMETRIC CURVE (Mean RT vs ILD)")
print("="*70)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Define colors for different ABL values
colors = ['tab:blue', 'tab:orange', 'tab:green']

for idx, ABL in enumerate(ABL_values):
    color = colors[idx]
    
    mean_rt_corr_list = []
    mean_rt_uncorr_list = []
    
    for ILD in ILD_values:
        stim_key = f"ABL_{ABL}_ILD_{ILD}"
        data = all_results[stim_key]
        mean_rt_corr_list.append(data['mean_rt_corr'])
        mean_rt_uncorr_list.append(data['mean_rt_uncorr'])
    
    # Plot chronometric curves for this ABL
    ax.plot(ILD_values, mean_rt_corr_list, 'o-', color=color, 
            linewidth=2, markersize=8, label=f'Corr (ABL={ABL})', alpha=0.7)
    ax.plot(ILD_values, mean_rt_uncorr_list, 's--', color=color, 
            linewidth=2, markersize=8, label=f'Uncorr (ABL={ABL})', alpha=0.7)

# Formatting
ax.set_xlabel('ILD (dB)', fontsize=12)
ax.set_ylabel('Mean RT (s)', fontsize=12)
ax.set_title(f'Chronometric Curves: Mean RT vs ILD\n'
             f'Corr: N={N_opt}, c={c_opt:.6f}, theta={theta_opt:.4f} | '
             f'Uncorr: N={N_opt}, theta={theta_ddm:.4f}',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='best', ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(ILD_values[0] - 0.5, ILD_values[-1] + 0.5)

plt.tight_layout()
plt.savefig('compare_poisson_corr_vs_uncorr_chronometric.png', dpi=300, bbox_inches='tight')
print("✓ Saved: compare_poisson_corr_vs_uncorr_chronometric.png")
plt.show()

print("\n" + "="*70)
print("CHRONOMETRIC PLOT COMPLETE")
print("="*70)

# %%
# Psychometric: P(right) vs ILD for each ABL (1x3 plot)
# Need to simulate with extended ILD range: -16 to 16 in steps of 2
print("\n" + "="*70)
print("GENERATING PSYCHOMETRIC CURVE DATA")
print("="*70)

# Define extended ILD range for psychometric curves
ABL_values_psycho = [20, 40, 60]
ILD_values_psycho = np.arange(-16, 18, 2)  # -16 to 16 in steps of 2
n_trials_psychometric = int(50e3)  # 50K trials per condition

print(f"\nABL values: {ABL_values_psycho}")
print(f"ILD range: {ILD_values_psycho[0]} to {ILD_values_psycho[-1]} (step={ILD_values_psycho[1]-ILD_values_psycho[0]})")
print(f"Trials per condition: {n_trials_psychometric}")

# Store psychometric data
psychometric_data = {}

for ABL in ABL_values_psycho:
    print(f"\n{'='*70}")
    print(f"ABL = {ABL}")
    print(f"{'='*70}")
    
    p_right_corr_list = []
    p_right_uncorr_list = []
    
    for ILD in tqdm(ILD_values_psycho, desc=f'Psychometric ABL={ABL}', ncols=80):
        # Calculate DDM rates
        ddm_right_rate, ddm_left_rate = lr_rates_from_ABL_ILD(
            ABL, ILD, Nr0_ddm, lam_ddm, ell_ddm
        )
        
        # Poisson Correlated simulation
        poisson_corr_right_rate = ddm_right_rate * rate_scaling_factor_opt
        poisson_corr_left_rate = ddm_left_rate * rate_scaling_factor_opt
        
        poisson_corr_params = {
            'N_right': N_opt,
            'N_left': N_opt,
            'c': c_opt,
            'r_right': poisson_corr_right_rate,
            'r_left': poisson_corr_left_rate,
            'theta': theta_opt,
            'T': T,
            'exponential_noise_scale': 0
        }
        
        poisson_corr_results = simulate_poisson_rts(
            poisson_corr_params, 
            n_trials=n_trials_psychometric, 
            seed=42, 
            verbose=False
        )
        poisson_corr_decided_mask = ~np.isnan(poisson_corr_results[:, 0])
        poisson_corr_choices = poisson_corr_results[poisson_corr_decided_mask, 1]
        p_right_corr = np.mean(poisson_corr_choices == 1)
        
        # Poisson Uncorrelated (Skellam) simulation
        mu1_per_neuron = ddm_right_rate / N_opt
        mu2_per_neuron = ddm_left_rate / N_opt
        
        poisson_uncorr_results = simulate_skellam_rts(
            N=N_opt,
            mu1=mu1_per_neuron,
            mu2=mu2_per_neuron,
            theta=int(theta_ddm),
            n_trials=n_trials_psychometric,
            seed=42,
            verbose=False
        )
        
        poisson_uncorr_decided_mask = ~np.isnan(poisson_uncorr_results[:, 0])
        poisson_uncorr_choices = poisson_uncorr_results[poisson_uncorr_decided_mask, 1]
        p_right_uncorr = np.mean(poisson_uncorr_choices == 1)
        
        p_right_corr_list.append(p_right_corr)
        p_right_uncorr_list.append(p_right_uncorr)
    
    psychometric_data[ABL] = {
        'ILD_values': ILD_values_psycho,
        'p_right_corr': np.array(p_right_corr_list),
        'p_right_uncorr': np.array(p_right_uncorr_list)
    }
    
    print(f"  Completed ABL={ABL}: {len(ILD_values_psycho)} ILD conditions")

print(f"\n{'='*70}")
print("PSYCHOMETRIC DATA GENERATION COMPLETE")
print(f"{'='*70}\n")

# %%
# Plot psychometric curves (1x3 layout)
print("="*70)
print("PLOTTING PSYCHOMETRIC CURVES")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, ABL in enumerate(ABL_values_psycho):
    ax = axes[idx]
    data = psychometric_data[ABL]
    
    # Plot psychometric curves
    ax.plot(data['ILD_values'], data['p_right_corr'], 'o-', color='blue', 
            linewidth=2, markersize=6, label='Poisson Corr', alpha=0.7)
    ax.plot(data['ILD_values'], data['p_right_uncorr'], 's--', color='red', 
            linewidth=2, markersize=6, label='Poisson Uncorr', alpha=0.7)
    
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
    ax.set_xlim(ILD_values_psycho[0] - 1, ILD_values_psycho[-1] + 1)

plt.suptitle(f'Psychometric Curves: P(right) vs ILD\n'
             f'Corr: N={N_opt}, c={c_opt:.6f}, theta={theta_opt:.4f} | '
             f'Uncorr: N={N_opt}, theta={theta_ddm:.4f}',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('compare_poisson_corr_vs_uncorr_psychometric_full.png', dpi=300, bbox_inches='tight')
print("✓ Saved: compare_poisson_corr_vs_uncorr_psychometric_full.png")
plt.show()

print("\n" + "="*70)
print("PSYCHOMETRIC PLOT COMPLETE")
print("="*70)

print("\n" + "="*70)
print("ALL ANALYSIS COMPLETE")
print("="*70)
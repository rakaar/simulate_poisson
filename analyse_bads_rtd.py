# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import glob
from tqdm import tqdm
from bads_utils import lr_rates_from_ABL_ILD, simulate_single_ddm_trial
from joblib import Parallel, delayed

# %%
# Find and load the most recent psycho simulation data file
pkl_pattern = 'psycho_simulation_data_*.pkl'
pkl_files = glob.glob(pkl_pattern)

if not pkl_files:
    raise FileNotFoundError(f"No files matching '{pkl_pattern}' found in current directory")

# Get the most recent file
latest_pkl = max(pkl_files, key=lambda x: Path(x).stat().st_mtime)
print(f"Loading data from: {latest_pkl}")

with open(latest_pkl, 'rb') as f:
    simulation_data = pickle.load(f)

# Extract data
poisson_psychometric_data = simulation_data['poisson_psychometric_data']
ILD_values = simulation_data['stimulus']['ILD_values']
ABL_values = simulation_data['stimulus']['ABL_values']
parameters = simulation_data['parameters']
ddm_params = simulation_data['ddm_params']

print("\n=== LOADED SIMULATION DATA ===")
print(f"ABL values: {ABL_values}")
print(f"ILD values: {ILD_values}")
print(f"Parameters: N={parameters['N_opt']}, r={parameters['r_opt']:.4f}, "
      f"c={parameters['c_opt']:.6f}, theta={parameters['theta_opt']:.4f}")

# %%
# Group data by absolute ILD values
# For each ABL, group positive and negative ILDs together

print("\n=== GROUPING DATA BY ABSOLUTE ILD ===")

# Get unique absolute ILD values
abs_ILD_values = np.unique(np.abs(ILD_values))
print(f"Absolute ILD values: {abs_ILD_values}")

# Create dictionary to store RTs grouped by ABL and abs(ILD)
rtd_data = {}

for ABL in ABL_values:
    rtd_data[ABL] = {}
    
    for abs_ILD in abs_ILD_values:
        # Find all ILDs that match this absolute value
        if abs_ILD == 0:
            matching_ILDs = [0]
        else:
            matching_ILDs = [abs_ILD, -abs_ILD]
        
        # Collect all RTs for this abs(ILD)
        all_rts = []
        
        for ILD in matching_ILDs:
            if ILD in ILD_values:
                # Get results for this condition
                results = poisson_psychometric_data[ABL][ILD]
                
                # Extract RTs (column 0) for trials that resulted in a decision
                # Exclude no-decision trials (choice == 0)
                choices = results[:, 1]
                rts = results[:, 0]
                decision_mask = choices != 0
                
                rts_with_decision = rts[decision_mask]
                all_rts.append(rts_with_decision)
        
        # Concatenate all RTs for this abs(ILD)
        rtd_data[ABL][abs_ILD] = np.concatenate(all_rts)
        
        print(f"ABL={ABL} dB, |ILD|={abs_ILD} dB: {len(rtd_data[ABL][abs_ILD])} trials")

print("\n✓ Data grouped by absolute ILD!")

# %%
# Simulate DDM for all ABL/ILD combinations
print("\n=== SIMULATING DDM MODEL ===")

ddm_rtd_data = {}
N_trials_ddm = parameters['N_trials_per_condition']  # Match Poisson trial count
dt = 1e-4
dB = 1e-2
T = 2  # Match T_max from Poisson simulation
n_steps = int(T / dt)

for ABL in ABL_values:
    ddm_rtd_data[ABL] = {}
    
    for abs_ILD in tqdm(abs_ILD_values, desc=f'DDM ABL={ABL} dB', ncols=100):
        # For abs_ILD, we simulate both +ILD and -ILD and combine
        if abs_ILD == 0:
            ILD_list = [0]
        else:
            ILD_list = [abs_ILD, -abs_ILD]
        
        all_ddm_rts = []
        
        for ILD in ILD_list:
            # Calculate DDM rates
            ddm_right_rate, ddm_left_rate = lr_rates_from_ABL_ILD(
                ABL, ILD, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
            )
            
            mu_ddm = ddm_right_rate - ddm_left_rate
            sigma_sq_ddm = ddm_right_rate + ddm_left_rate
            sigma_ddm = np.sqrt(sigma_sq_ddm)
            
            # Simulate DDM trials in parallel
            tasks = [(i, mu_ddm, sigma_ddm, ddm_params['theta'], dt, dB, n_steps) 
                     for i in range(N_trials_ddm)]
            
            results = Parallel(n_jobs=30)(
                delayed(simulate_single_ddm_trial)(*task) for task in tasks
            )
            
            results_array = np.array(results)
            # Filter for decided trials only
            decision_mask = ~np.isnan(results_array[:, 0])
            ddm_rts_decided = results_array[decision_mask, 0]
            all_ddm_rts.append(ddm_rts_decided)
        
        # Concatenate all DDM RTs for this abs(ILD)
        ddm_rtd_data[ABL][abs_ILD] = np.concatenate(all_ddm_rts)
        print(f"  ABL={ABL} dB, |ILD|={abs_ILD} dB: {len(ddm_rtd_data[ABL][abs_ILD])} DDM trials")

print("\n✓ DDM simulations complete!")

# %%
# Plot reaction time distributions: 3x9 grid (3 ABL rows, 9 |ILD| columns)
# Each subplot shows both Poisson and DDM distributions

print("\n=== PLOTTING REACTION TIME DISTRIBUTIONS (3x9 GRID) ===")

# Define bins from 0 to 1.2 in steps of 0.01
bins = np.arange(0, 2, 0.01)

# Create 3x9 grid
fig, axes = plt.subplots(3, 9, figsize=(27, 12))

for row_idx, ABL in enumerate(ABL_values):
    for col_idx, abs_ILD in enumerate(abs_ILD_values):
        ax = axes[row_idx, col_idx]
        
        # Get Poisson and DDM RTs for this condition
        poisson_rts = rtd_data[ABL][abs_ILD]
        ddm_rts = ddm_rtd_data[ABL][abs_ILD]
        
        # Plot both distributions
        ax.hist(poisson_rts, bins=bins, alpha=0.7, color='blue', 
                label='Poisson', density=True, histtype='step', linewidth=2)
        ax.hist(ddm_rts, bins=bins, alpha=0.7, color='red', 
                label='DDM', density=True, histtype='step', linewidth=2)
        
        # Formatting
        ax.set_xlim(0, 0.8)
        
        # Only add labels to edge subplots
        if col_idx == 0:
            ax.set_ylabel(f'ABL={ABL}\nDensity', fontsize=9)
        if row_idx == 2:
            ax.set_xlabel(f'|ILD|={abs_ILD}\nRT (s)', fontsize=9)
        if row_idx == 0:
            ax.set_title(f'|ILD|={abs_ILD} dB', fontsize=10)
        
        # Add legend only to top-right subplot
        if row_idx == 0 and col_idx == 8:
            ax.legend(fontsize=8, loc='upper right')
        
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

plt.suptitle(f'Reaction Time Distributions: Poisson vs DDM\n'
             f'(Poisson: N={parameters["N_opt"]}, r={parameters["r_opt"]:.4f}, '
             f'c={parameters["c_opt"]:.6f}, θ={parameters["theta_opt"]:.4f})',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('rtd_poisson_vs_ddm_grid.png', dpi=300, bbox_inches='tight')
print("✓ Saved: rtd_poisson_vs_ddm_grid.png")
plt.show()

# %%
# Plot mean RT vs |ILD| for each ABL (both Poisson and DDM)
print("\n=== PLOTTING MEAN RT vs |ILD| ===")

fig, ax = plt.subplots(figsize=(12, 6))

# Define colors for different ABL values
abl_colors = plt.cm.tab10(np.linspace(0, 0.3, len(ABL_values)))

for idx, ABL in enumerate(ABL_values):
    # Poisson data
    mean_rts_poisson = []
    std_rts_poisson = []
    
    # DDM data
    mean_rts_ddm = []
    std_rts_ddm = []
    
    for abs_ILD in abs_ILD_values:
        # Poisson
        rts_poisson = rtd_data[ABL][abs_ILD]
        mean_rts_poisson.append(np.mean(rts_poisson))
        std_rts_poisson.append(np.std(rts_poisson)/np.sqrt(len(rts_poisson)))
        
        # DDM
        rts_ddm = ddm_rtd_data[ABL][abs_ILD]
        mean_rts_ddm.append(np.mean(rts_ddm))
        std_rts_ddm.append(np.std(rts_ddm)/np.sqrt(len(rts_ddm)))
    
    mean_rts_poisson = np.array(mean_rts_poisson)
    std_rts_poisson = np.array(std_rts_poisson)
    mean_rts_ddm = np.array(mean_rts_ddm)
    std_rts_ddm = np.array(std_rts_ddm)
    
    # Plot Poisson with circles
    ax.errorbar(abs_ILD_values, mean_rts_poisson, yerr=std_rts_poisson, 
                marker='o', markersize=8, linewidth=2, capsize=5,
                label=f'Poisson ABL={ABL} dB', alpha=0.8, color=abl_colors[idx])
    
    # Plot DDM with x markers
    ax.errorbar(abs_ILD_values, mean_rts_ddm, yerr=std_rts_ddm, 
                marker='x', markersize=10, linewidth=2, capsize=5,
                label=f'DDM ABL={ABL} dB', alpha=0.8, color=abl_colors[idx],
                linestyle='--')

# Formatting
ax.set_xlabel('|ILD| (dB)', fontsize=13)
ax.set_ylabel('Mean Reaction Time (s)', fontsize=13)
ax.set_title('Mean RT vs |ILD|: Poisson vs DDM', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='best', ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, max(abs_ILD_values) + 1)

plt.tight_layout()
plt.savefig('mean_rt_vs_abs_ILD_poisson_ddm.png', dpi=300, bbox_inches='tight')
print("✓ Saved: mean_rt_vs_abs_ILD_poisson_ddm.png")
plt.show()

# %%
# Summary statistics
print("\n=== SUMMARY STATISTICS ===")

for ABL in ABL_values:
    print(f"\nABL = {ABL} dB:")
    print(f"{'|ILD| (dB)':<12} {'Mean RT (s)':<15} {'Std RT (s)':<15} {'Median RT (s)':<15} {'N trials':<12}")
    print("-" * 75)
    
    for abs_ILD in abs_ILD_values:
        rts = rtd_data[ABL][abs_ILD]
        mean_rt = np.mean(rts)
        std_rt = np.std(rts)
        median_rt = np.median(rts)
        n_trials = len(rts)
        
        print(f"{abs_ILD:<12} {mean_rt:<15.4f} {std_rt:<15.4f} {median_rt:<15.4f} {n_trials:<12}")

print("\n✓ Analysis complete!")
# %%
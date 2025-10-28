# %%
"""
Check for negative RTs in simulation results.
Analyzes all pickle files to count and visualize negative reaction times.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import pandas as pd

# %%
# CONFIGURATION
version_number = '12'
results_folder_name = f'results_V{version_number}'

# %%
# Get the directory where this script is located
script_dir = Path(__file__).parent
results_folder = script_dir / results_folder_name

if not results_folder.exists():
    print(f"ERROR: Results folder '{results_folder}' does not exist!")
    import sys
    sys.exit(1)

print(f"Reading from: {results_folder}/")

# %%
# Collect statistics from all pickle files
pkl_files = list(results_folder.glob('*.pkl'))
print(f"Found {len(pkl_files)} pickle files\n")

results_list = []

for pkl_file in pkl_files:
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    params = data['params']
    poisson_rts = data['poisson']['results'][:, 0]
    decided_rts = poisson_rts[~np.isnan(poisson_rts)]
    
    # Count negative RTs
    n_negative = np.sum(decided_rts < 0)
    n_total = len(decided_rts)
    pct_negative = 100 * n_negative / n_total if n_total > 0 else 0
    
    # Get RT statistics
    min_rt = np.min(decided_rts) if len(decided_rts) > 0 else np.nan
    max_rt = np.max(decided_rts) if len(decided_rts) > 0 else np.nan
    mean_rt = np.mean(decided_rts) if len(decided_rts) > 0 else np.nan
    
    results_list.append({
        'c': params['c'],
        'corr_factor': params['corr_factor'],
        'exponential_noise_ms': params['exponential_noise_to_spk_time'] * 1000,
        'noise_mean_subtraction': params.get('noise_mean_subtraction', False),
        'theta_scaled': params['theta_scaled'],
        'N_total': n_total,
        'N_negative': n_negative,
        'pct_negative': pct_negative,
        'min_rt': min_rt,
        'max_rt': max_rt,
        'mean_rt': mean_rt,
    })

# Convert to DataFrame for easy analysis
df = pd.DataFrame(results_list)

# %%
# Display summary statistics
print("="*80)
print("SUMMARY: Negative RTs by Parameter Combination")
print("="*80)

# Sort by percentage of negative RTs (descending)
df_sorted = df.sort_values('pct_negative', ascending=False)

print(f"\nTotal combinations: {len(df)}")
print(f"Combinations with negative RTs: {np.sum(df['N_negative'] > 0)}")
print(f"\nTop 10 combinations with most negative RTs:")
print(df_sorted[['c', 'corr_factor', 'exponential_noise_ms', 'N_negative', 'pct_negative', 'min_rt']].head(10).to_string(index=False))

# %%
# Overall statistics
print(f"\n{'='*80}")
print("OVERALL STATISTICS")
print("="*80)
print(f"Total decisions across all combinations: {df['N_total'].sum()}")
print(f"Total negative RTs: {df['N_negative'].sum()}")
print(f"Overall percentage of negative RTs: {100 * df['N_negative'].sum() / df['N_total'].sum():.4f}%")
print(f"Min RT across all data: {df['min_rt'].min():.6f}s ({df['min_rt'].min()*1000:.3f}ms)")
print(f"Max RT across all data: {df['max_rt'].max():.6f}s")

# %%
# Group by noise level
print(f"\n{'='*80}")
print("STATISTICS BY NOISE LEVEL")
print("="*80)

grouped_by_noise = df.groupby('exponential_noise_ms').agg({
    'N_total': 'sum',
    'N_negative': 'sum',
    'min_rt': 'min',
    'mean_rt': 'mean'
}).reset_index()

grouped_by_noise['pct_negative'] = 100 * grouped_by_noise['N_negative'] / grouped_by_noise['N_total']

print(grouped_by_noise.to_string(index=False))

# %%
# Group by correlation coefficient
print(f"\n{'='*80}")
print("corr-c : negative check ")
print("="*80)

grouped_by_c = df.groupby('c').agg({
    'N_total': 'sum',
    'N_negative': 'sum',
    'min_rt': 'min',
    'mean_rt': 'mean'
}).reset_index()

grouped_by_c['pct_negative'] = 100 * grouped_by_c['N_negative'] / grouped_by_c['N_total']

print(grouped_by_c.to_string(index=False))

# %%
# Group by correlation factor
print(f"\n{'='*80}")
print("corr factor: negative check")
print("="*80)

grouped_by_corr = df.groupby('corr_factor').agg({
    'N_total': 'sum',
    'N_negative': 'sum',
    'min_rt': 'min',
    'mean_rt': 'mean'
}).reset_index()

grouped_by_corr['pct_negative'] = 100 * grouped_by_corr['N_negative'] / grouped_by_corr['N_total']

print(grouped_by_corr.to_string(index=False))

# %%
# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Percentage of negative RTs by noise level
ax = axes[0, 0]
noise_groups = df.groupby('exponential_noise_ms')['pct_negative'].mean()
ax.bar(noise_groups.index, noise_groups.values, color='steelblue', alpha=0.7)
ax.set_xlabel('Exponential Noise (ms)', fontsize=11)
ax.set_ylabel('% Negative RTs', fontsize=11)
ax.set_title('Percentage of Negative RTs by Noise Level', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Plot 2: Min RT by noise level
ax = axes[0, 1]
min_rt_by_noise = df.groupby('exponential_noise_ms')['min_rt'].min() * 1000  # Convert to ms
ax.bar(min_rt_by_noise.index, min_rt_by_noise.values, color='coral', alpha=0.7)
ax.set_xlabel('Exponential Noise (ms)', fontsize=11)
ax.set_ylabel('Min RT (ms)', fontsize=11)
ax.set_title('Minimum RT by Noise Level', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='RT = 0')
ax.legend()

# Plot 3: Number of negative RTs by c value
ax = axes[1, 0]
c_groups = df.groupby('c')['N_negative'].sum()
ax.bar(c_groups.index, c_groups.values, color='green', alpha=0.7)
ax.set_xlabel('Correlation Coefficient (c)', fontsize=11)
ax.set_ylabel('Number of Negative RTs', fontsize=11)
ax.set_title('Total Negative RTs by Correlation Coefficient', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Plot 4: Heatmap of negative RT percentage
ax = axes[1, 1]
pivot_table = df.pivot_table(values='pct_negative', 
                              index='exponential_noise_ms', 
                              columns='corr_factor',
                              aggfunc='mean')
im = ax.imshow(pivot_table.values, aspect='auto', cmap='YlOrRd', origin='lower')
ax.set_xticks(range(len(pivot_table.columns)))
ax.set_xticklabels(pivot_table.columns)
ax.set_yticks(range(len(pivot_table.index)))
ax.set_yticklabels([f'{x:.1f}' for x in pivot_table.index])
ax.set_xlabel('Correlation Factor', fontsize=11)
ax.set_ylabel('Exponential Noise (ms)', fontsize=11)
ax.set_title('% Negative RTs Heatmap', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='% Negative RTs')

plt.tight_layout()
plt.savefig(script_dir / 'negative_rts_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n{'='*80}")
print(f"Visualization saved to: {script_dir / 'negative_rts_analysis.png'}")
print("="*80)
plt.show()

# %%

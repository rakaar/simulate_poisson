# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import product
batch_name = 'LED8'
csv_path = f"/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/batch_csvs/batch_{batch_name}_valid_and_aborts.csv"
animal_id = 112
if not os.path.exists(csv_path):
    print(f"[ERROR] Data CSV not found: {csv_path}")
df = pd.read_csv(csv_path)
df_animal = df[df['animal'] == int(animal_id)].copy()

# Compute RT relative to stim: timed_fix - intended_fix (fallback TotalFixTime)
if 'timed_fix' in df_animal.columns:
    timed = df_animal['timed_fix']
else:
    timed = df_animal['TotalFixTime']
rt_rel = timed - df_animal['intended_fix']
df_animal['rt_rel'] = rt_rel

# %%
ABL = df_animal['ABL'].unique()
ILD = df_animal['ILD'].unique()


fig, axes = plt.subplots(nrows=len(ABL), ncols=len(ILD), figsize=(15, 10), sharex=True, sharey=True)

# Flatten axes array for easy iteration if it's 2D
if len(ABL) > 1 and len(ILD) > 1:
    axes = axes.flatten()

for i, (ab, il) in enumerate(product(ABL, ILD)):
    ax = axes[i]
    df_cond = df_animal[(df_animal['ABL'] == ab) & (df_animal['ILD'] == il)].copy()
    # remove RTs < t_stim and RTs < 0.3
    df_cond = df_cond[\
        ~( (df_cond['TotalFixTime'] < df_cond['intended_fix']) & (df_cond['TotalFixTime'] < 0.3))
        ] 
    
    
    ax.hist(data_to_plot, bins=np.arange(-2, 2, 0.05), density=True, histtype='step')
    ax.set_title(f'ABL={ab}, ILD={il}')
    ax.axvline(0, color='k', alpha=0.5)
    ax.axvline(0.073, color='k', alpha=0.5)

fig.supxlabel('Time Relative to Stimulus (s)')
fig.supylabel('Density')
fig.suptitle(f'Histograms of Reaction Times for Animal {animal_id} (Batch: {batch_name})')
plt.xlim(-0.5,0.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
# %%
# %%
import numpy as np
import matplotlib.pyplot as plt
from mgf_helper_utils import find_h0

# %%
# Poisson params (fixed)
N = 1000
N_values = [10000]
rho_values = [1e-4, 1e-3, 0.01, 0.05, 0.1]

# DDM params
lam = 1.3
l = 0.9
Nr0_base = 13.3

abl = 20
ild = 50  # Fixed ILD
dt = 1e-6

theta_poisson = 20

# %%
# Calculate rates at ILD = 50
r0 = Nr0_base / N
r_db = (2 * abl + ild) / 2
l_db = (2 * abl - ild) / 2
pr = 10 ** (r_db / 20)
pl = 10 ** (l_db / 20)

den = (pr ** (lam * l)) + (pl ** (lam * l))
rr = (pr ** lam) / den
rl = (pl ** lam) / den

rate_scalar_right = 1
rate_scalar_left = 1

r_right = r0 * rr * rate_scalar_right
r_left = r0 * rl * rate_scalar_left

print(f"r_right = {r_right:.6f}, r_left = {r_left:.6f}")

# %%
# Calculate h0 for each rho
h0_values = []
for rho in rho_values:
    h0 = find_h0(r_right, r_left, N, rho)
    h0_values.append(h0)
    print(f"rho = {rho}, h0 = {h0}")

# %%
# Plot h0 vs rho
plt.figure(figsize=(10, 6))
plt.plot(rho_values, h0_values, marker='o', markersize=10, linewidth=2.5, color='tab:blue')
plt.xlabel('ρ (rho)', fontsize=14, fontweight='bold')
plt.ylabel('h₀', fontsize=14, fontweight='bold')
plt.title(f'h₀ vs ρ\nN={N}, ILD={ild}, ABL={abl}, r_right={r_right:.4f}, r_left={r_left:.4f}, Rx={rate_scalar_right},Lx={rate_scalar_left}', 
          fontsize=14, fontweight='bold')
plt.axhline(-0.1, color='r', alpha=0.4, label='0.1')
plt.xscale('log')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rho_values, [f'{r}' for r in rho_values])
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Plot h0 vs (rho × N) for different N values - all curves on same plot
plt.figure(figsize=(12, 7))

colors = plt.cm.viridis(np.linspace(0, 0.9, len(N_values)))

for idx, N_val in enumerate(N_values):
    # Recalculate rates for this N
    r0_N = Nr0_base / N_val
    r_right_N = r0_N * rr * rate_scalar_right
    r_left_N = r0_N * rl * rate_scalar_left
    
    # Calculate h0 for each rho
    h0_vals = []
    rho_times_N = []
    for rho in rho_values:
        h0 = find_h0(r_right_N, r_left_N, N_val, rho)
        h0_vals.append(h0)
        rho_times_N.append(rho * N_val)
    
    plt.plot(rho_times_N, h0_vals, marker='o', markersize=8, linewidth=2, 
             color=colors[idx], label=f'N={N_val}')

plt.xlabel('ρ × N', fontsize=14, fontweight='bold')
plt.ylabel('h₀', fontsize=14, fontweight='bold')
plt.title(f'h₀ vs ρ×N for different N\nILD={ild}, ABL={abl}, Rx={rate_scalar_right}, Lx={rate_scalar_left}', 
          fontsize=14, fontweight='bold')
plt.axhline(-0.1, color='r', alpha=0.4, linestyle='--', label='h₀ = -0.1')
plt.xscale('log')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=11, loc='best')
plt.tight_layout()
plt.show()

# %%
# Plot h0 vs rho for ILD = 25 and ILD = 50
# ild_values = [25, 50, 75, 100]
# ild_colors = {25: 'tab:blue', 50: 'tab:orange', 75:'tab:green', 100: 'yellow'}
ild_values = [1,2,4,8,16]
cmap = plt.get_cmap('tab10')

ild_colors = {ratio: cmap(i) for i, ratio in enumerate(ild_values)}
plt.figure(figsize=(10, 6))

for ild_val in ild_values:
    # Recalculate rates for this ILD
    r_db_ild = (2 * abl + ild_val) / 2
    l_db_ild = (2 * abl - ild_val) / 2
    pr_ild = 10 ** (r_db_ild / 20)
    pl_ild = 10 ** (l_db_ild / 20)
    
    den_ild = (pr_ild ** (lam * l)) + (pl_ild ** (lam * l))
    rr_ild = (pr_ild ** lam) / den_ild
    rl_ild = (pl_ild ** lam) / den_ild
    
    # Use same N as defined at top (from N_values)
    N_plot = N_values[0]
    r0_ild = Nr0_base / N_plot
    r_right_ild = r0_ild * rr_ild * rate_scalar_right
    r_left_ild = r0_ild * rl_ild * rate_scalar_left
    
    # Calculate h0 for each rho
    h0_vals = []
    for rho in rho_values:
        h0 = find_h0(r_right_ild, r_left_ild, N_plot, rho)
        h0_vals.append(h0)
    
    plt.plot(rho_values, h0_vals, marker='o', markersize=10, linewidth=2.5, 
             color=ild_colors[ild_val], label=f'ILD={ild_val}')

plt.xlabel('ρ (rho)', fontsize=14, fontweight='bold')
plt.ylabel('h₀', fontsize=14, fontweight='bold')
plt.title(f'h₀ vs ρ for different ILDs\nN={N_plot}, ABL={abl}, Rx={rate_scalar_right}, Lx={rate_scalar_left}', 
          fontsize=14, fontweight='bold')
plt.axhline(-0.1, color='r', alpha=0.4, linestyle='--', label='h₀ = -0.1')
plt.xscale('log')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rho_values, [f'{r}' for r in rho_values])
plt.legend(fontsize=11, loc='best')
plt.tight_layout()
plt.show()

# %%
# Plot h0 ratios relative to ILD=25 vs rho
import matplotlib.pyplot as plt

ild_ratios = [1, 2, 4, 8, 16]  # ILDs to compare against ILD=25
ild_ref = 25
# Use a colormap for ratio colors
cmap = plt.get_cmap('tab10')
ratio_colors = {ratio: cmap(i) for i, ratio in enumerate(ild_ratios)}
N_plot = N_values[0]

# Calculate h0 for reference ILD=25
r_db_ref = (2 * abl + ild_ref) / 2
l_db_ref = (2 * abl - ild_ref) / 2
pr_ref = 10 ** (r_db_ref / 20)
pl_ref = 10 ** (l_db_ref / 20)
den_ref = (pr_ref ** (lam * l)) + (pl_ref ** (lam * l))
rr_ref = (pr_ref ** lam) / den_ref
rl_ref = (pl_ref ** lam) / den_ref
r0_ref = Nr0_base / N_plot
r_right_ref = r0_ref * rr_ref * rate_scalar_right
r_left_ref = r0_ref * rl_ref * rate_scalar_left

h0_ref_vals = []
for rho in rho_values:
    h0 = find_h0(r_right_ref, r_left_ref, N_plot, rho)
    h0_ref_vals.append(h0)

plt.figure(figsize=(10, 6))

for ild_val in ild_ratios:
    # Recalculate rates for this ILD
    r_db_ild = (2 * abl + ild_val) / 2
    l_db_ild = (2 * abl - ild_val) / 2
    pr_ild = 10 ** (r_db_ild / 20)
    pl_ild = 10 ** (l_db_ild / 20)
    
    den_ild = (pr_ild ** (lam * l)) + (pl_ild ** (lam * l))
    rr_ild = (pr_ild ** lam) / den_ild
    rl_ild = (pl_ild ** lam) / den_ild
    
    r0_ild = Nr0_base / N_plot
    r_right_ild = r0_ild * rr_ild * rate_scalar_right
    r_left_ild = r0_ild * rl_ild * rate_scalar_left
    
    # Calculate h0 ratio for each rho
    h0_ratios = []
    for idx, rho in enumerate(rho_values):
        h0 = find_h0(r_right_ild, r_left_ild, N_plot, rho)
        ratio = h0 / h0_ref_vals[idx]
        h0_ratios.append(ratio)
    
    plt.plot(rho_values, h0_ratios, marker='o', markersize=10, linewidth=2.5, 
             color=ratio_colors[ild_val], label=f'h₀(ILD={ild_val}) / h₀(ILD={ild_ref})')

plt.xlabel('ρ (rho)', fontsize=14, fontweight='bold')
plt.ylabel('h₀ ratio', fontsize=14, fontweight='bold')
plt.title(f'h₀ ratios relative to ILD={ild_ref}\nN={N_plot}, ABL={abl}, Rx={rate_scalar_right}, Lx={rate_scalar_left}', 
          fontsize=14, fontweight='bold')
plt.axhline(1.0, color='gray', alpha=0.5, linestyle='--', label='ratio = 1')
plt.xscale('log')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rho_values, [f'{r}' for r in rho_values])
plt.legend(fontsize=10, loc='best')
plt.tight_layout()
plt.show()

# %%
# Plot h0 vs (rho × N) for N = 500 and N = 1000
# Choose rho such that rho × N gives same x-axis values for both
N_compare = [500, 1000]
rho_times_N_target = [1e-4, 1e-3, 1e-2, 0.05, 0.1]
N_colors = {500: 'tab:blue', 1000: 'tab:orange'}
N_markers = {500: 'o', 1000: 's'}

plt.figure(figsize=(10, 6))

for N_val in N_compare:
    # Recalculate rates for this N
    r0_N = Nr0_base / N_val
    r_right_N = r0_N * rr * rate_scalar_right
    r_left_N = r0_N * rl * rate_scalar_left
    
    # Calculate h0 for each target rho×N value
    h0_vals = []
    for rhoN in rho_times_N_target:
        rho = rhoN / N_val  # Calculate rho to achieve target rho×N
        h0 = find_h0(r_right_N, r_left_N, N_val, rho)
        h0_vals.append(h0)
        print(f"N={N_val}, rho×N={rhoN}, rho={rho:.2e}, h0={h0:.4f}")
    
    plt.plot(rho_times_N_target, h0_vals, marker=N_markers[N_val], markersize=10, linewidth=2.5, 
             color=N_colors[N_val], label=f'N={N_val}')

plt.xlabel('ρ × N', fontsize=14, fontweight='bold')
plt.ylabel('h₀', fontsize=14, fontweight='bold')
plt.title(f'h₀ vs ρ×N (same x-values for different N)\nILD={ild}, ABL={abl}, Rx={rate_scalar_right}, Lx={rate_scalar_left}', 
          fontsize=14, fontweight='bold')
plt.axhline(-0.1, color='r', alpha=0.4, linestyle='--', label='h₀ = -0.1')
plt.xscale('log')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rho_times_N_target, [f'{x}' for x in rho_times_N_target])
plt.legend(fontsize=11, loc='best')
plt.tight_layout()
plt.show()

# %%
# Plot h0 * theta vs ILD for different rho values
theta_fixed = 20
ild_plot_values = [1, 2, 4, 8, 16, 32]
rho_plot_values =  10**np.arange(-6, 0, 0.1)

N_fixed = 1000

cmap = plt.get_cmap('tab10', 100)
rho_plot_colors = {rho: cmap(i) for i, rho in enumerate(rho_plot_values)}

plt.figure(figsize=(20, 10))

for rho in rho_plot_values:
    h0_theta_vals = []
    
    for ild_val in ild_plot_values:
        # Calculate rates for this ILD
        r_db_ild = (2 * abl + ild_val) / 2
        l_db_ild = (2 * abl - ild_val) / 2
        pr_ild = 10 ** (r_db_ild / 20)
        pl_ild = 10 ** (l_db_ild / 20)
        
        den_ild = (pr_ild ** (lam * l)) + (pl_ild ** (lam * l))
        rr_ild = (pr_ild ** lam) / den_ild
        rl_ild = (pl_ild ** lam) / den_ild
        
        r0_ild = Nr0_base / N_fixed
        r_right_ild = r0_ild * rr_ild * rate_scalar_right
        r_left_ild = r0_ild * rl_ild * rate_scalar_left
        
        h0 = find_h0(r_right_ild, r_left_ild, N_fixed, rho)
        h0_theta_vals.append(h0 * theta_fixed)
    
    plt.plot(ild_plot_values, -1*np.array(h0_theta_vals), marker='o', markersize=10, linewidth=2.5, 
             color=rho_plot_colors[rho], label=f'ρ={rho: .5f}')

plt.xlabel('ILD', fontsize=14, fontweight='bold')
plt.ylabel('h₀ × θ', fontsize=14, fontweight='bold')
plt.title(f'h₀ × θ vs ILD for different ρ\nN={N_fixed}, θ={theta_fixed}, ABL={abl}, Rx={rate_scalar_right}, Lx={rate_scalar_left}', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(ild_plot_values)
# plt.xlim(1,16)
# plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# %%
# Same plot but normalized by h0 at ILD=48 for each rho
cont_ild = np.arange(1, 32.1, 0.1)
ddm_logodds = np.tanh(lam * cont_ild/17.37)
ddm_logodds_norm = ddm_logodds / np.max(np.abs(ddm_logodds))
chosen_ild = ild_plot_values[-1]
plt.figure(figsize=(20, 10))
print(f'chosen_ild = {chosen_ild}')

for rho in rho_plot_values:
    h0_theta_vals = []
    
    for ild_val in ild_plot_values:
        # Calculate rates for this ILD
        r_db_ild = (2 * abl + ild_val) / 2
        l_db_ild = (2 * abl - ild_val) / 2
        pr_ild = 10 ** (r_db_ild / 20)
        pl_ild = 10 ** (l_db_ild / 20)
        
        den_ild = (pr_ild ** (lam * l)) + (pl_ild ** (lam * l))
        rr_ild = (pr_ild ** lam) / den_ild
        rl_ild = (pl_ild ** lam) / den_ild
        
        r0_ild = Nr0_base / N_fixed
        r_right_ild = r0_ild * rr_ild * rate_scalar_right
        r_left_ild = r0_ild * rl_ild * rate_scalar_left
        
        h0 = find_h0(r_right_ild, r_left_ild, N_fixed, rho)
        h0_theta_vals.append(h0 * theta_fixed)
    
    # h0 at ILD=48 is the last element
    h0_theta_at_48 = h0_theta_vals[-1]
    h0_theta_normalized = np.array(h0_theta_vals) / h0_theta_at_48
    
    plt.plot(ild_plot_values, h0_theta_normalized, marker='o', markersize=10, linewidth=2.5, 
             color=rho_plot_colors[rho], label=f'ρ={rho:.1e}')

plt.xlabel('ILD', fontsize=14, fontweight='bold')
plt.ylabel(f'(h₀ × θ) / (h₀ × θ at ILD={chosen_ild})', fontsize=14, fontweight='bold')
plt.title(f'Normalized h₀ × θ vs ILD (normalized by ILD={chosen_ild})\nN={N_fixed}, θ={theta_fixed}, ABL={abl}', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(ild_plot_values)
plt.plot(cont_ild, ddm_logodds_norm, linestyle='--', color='k', lw=2, label='DDM')
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()

# %%
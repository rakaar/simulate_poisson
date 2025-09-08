# VBMC fit, with theta from 1 to 50 fixed
# %%
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from pyvbmc import VBMC
import contextlib
import traceback
import argparse

from vbmc_skellam_utils import fpt_density_skellam, fpt_choice_skellam
# %%
pkl_folder = "poisson_fit_pkls_theta_fixed_logR_logL"
pkl_files = sorted([os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')])
# all_results = []
N_theta = 50
# Simple switch to control whether Gamma/Omega plots include only stable thetas
# Toggle this to True/False as desired.
ONLY_STABLE = False

# CLI arguments for flexibility
parser = argparse.ArgumentParser(description="Extract VBMC samples and stats across fixed-theta fits.")
parser.add_argument("--n-samples", type=int, default=int(100e3), help="Number of samples to draw from each VP.")
parser.add_argument("--theta-start", type=int, default=1, help="Starting theta index (inclusive, 1-based).")
parser.add_argument("--theta-end", type=int, default=N_theta, help="Ending theta index (inclusive, 1-based).")
parser.add_argument("--out-dir", type=str, default="poisson_fit_samples_theta_fixed_logR_logL", help="Output directory for samples and summary.")
args, unknown = parser.parse_known_args()

N_samples = int(args.n_samples)
theta_start = max(1, int(args.theta_start))
theta_end = min(N_theta, int(args.theta_end))

# Output directory for per-theta samples and summaries
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

# Summary arrays (length N_theta)
elbo = np.full(N_theta, np.nan)
elbo_sd = np.full(N_theta, np.nan)
e_log_joint = np.full(N_theta, np.nan)
e_log_joint_sd = np.full(N_theta, np.nan)
stable = np.zeros(N_theta, dtype=bool)
num_components = np.full(N_theta, np.nan)
has_file = np.zeros(N_theta, dtype=bool)
sample_base = [""] * N_theta  # base filename of saved samples per theta

for theta in range(theta_start, theta_end + 1):
    # vbmc_poisson_fit_SIM_theta36.pkl
    pkl_path = os.path.join(pkl_folder, f'vbmc_poisson_fit_SIM_theta{theta:02d}.pkl')
    if not os.path.exists(pkl_path):
        print(f"[WARN] Missing file for theta={theta:02d}: {pkl_path}")
        continue

    try:
        vp = VBMC.load(pkl_path)
    except Exception as err:
        print(f"[ERROR] Failed to load {pkl_path}: {err}")
        traceback.print_exc()
        continue

    # Draw samples from the variational posterior
    try:
        samples_obj = vp.vp.sample(N_samples)
        samples = samples_obj[0] if isinstance(samples_obj, (tuple, list)) else samples_obj
        # Convert back from log10-space to rate-space (R, L)
        R = 10 ** samples[:, 0]
        L = 10 ** samples[:, 1]

        # Save per-theta samples (as separate arrays)
        r_path = os.path.join(out_dir, f'theta{theta:02d}_R_samples.npy')
        l_path = os.path.join(out_dir, f'theta{theta:02d}_L_samples.npy')
        np.save(r_path, R)
        np.save(l_path, L)
        sample_base[theta - 1] = f'theta{theta:02d}'
        has_file[theta - 1] = True
    except Exception as err:
        print(f"[ERROR] Sampling/saving failed for theta={theta:02d}: {err}")
        traceback.print_exc()
        continue

    # Extract stats (handle both '*_sd' and '*_std' naming)
    try:
        stats = getattr(vp.vp, 'stats', None)
        if not isinstance(stats, dict):
            stats = getattr(vp, 'stats', None)
        if isinstance(stats, dict):
            elbo[theta - 1] = float(stats.get('elbo', np.nan))
            # prefer '*_sd' keys, fallback to '*_std'
            elbo_sd[theta - 1] = float(stats.get('elbo_sd', stats.get('elbo_std', np.nan)))
            e_log_joint[theta - 1] = float(stats.get('e_log_joint', np.nan))
            e_log_joint_sd[theta - 1] = float(stats.get('e_log_joint_sd', stats.get('e_log_joint_std', np.nan)))
            stable[theta - 1] = bool(stats.get('stable', False))
        # number of mixture components (robustly)
        try:
            w = getattr(vp.vp, 'weights', None)
            if w is not None:
                num_components[theta - 1] = int(np.asarray(w).shape[1])
            else:
                # fallback
                num_components[theta - 1] = int(getattr(vp.vp, 'K', np.nan))
        except Exception:
            num_components[theta - 1] = np.nan
    except Exception as err:
        print(f"[WARN] Could not extract stats for theta={theta:02d}: {err}")
        traceback.print_exc()

# Save summary across thetas
summary_npz_path = os.path.join(out_dir, 'summary_stats.npz')
np.savez(
    summary_npz_path,
    theta=np.arange(1, N_theta + 1),
    has_file=has_file,
    elbo=elbo,
    elbo_mean=elbo,  # alias for convenience
    elbo_sd=elbo_sd,
    elbo_std=elbo_sd,  # alias for convenience
    e_log_joint=e_log_joint,
    e_log_joint_sd=e_log_joint_sd,
    e_log_joint_std=e_log_joint_sd,  # alias for convenience
    stable=stable,
    num_components=num_components,
    N_samples=np.array(N_samples, dtype=np.int64),
)

print(elbo[:10])
print(e_log_joint[:10])

# %%
# Plot differences relative to theta=03 baseline
# (ELBO - ELBO[3]) and (E[log joint] - E[log joint][3]) vs theta
x_rel = np.arange(1, N_theta + 1)

# Baselines at theta=3 (index 2)
elbo_ref = elbo[2] if np.isfinite(elbo[2]) else np.nan
ej_ref = e_log_joint[2] if np.isfinite(e_log_joint[2]) else np.nan

# ELBO difference
mask_elbo_diff = has_file & np.isfinite(elbo) & np.isfinite(elbo_ref)
mask_elbo_diff_stable = mask_elbo_diff & stable
mask_elbo_diff_unstable = mask_elbo_diff & ~stable
if 'ONLY_STABLE' in globals() and ONLY_STABLE:
    mask_plot_elbo = mask_elbo_diff_stable
else:
    mask_plot_elbo = mask_elbo_diff

plt.figure(figsize=(10, 5))
width = 0.8  # bar width

if np.any(mask_elbo_diff_stable) and (not ('ONLY_STABLE' in globals() and ONLY_STABLE)):
    plt.bar(x_rel[mask_elbo_diff_stable], (elbo[mask_elbo_diff_stable] - elbo_ref), width=width, color='green', label='Stable', align='center')
if (not ('ONLY_STABLE' in globals() and ONLY_STABLE)) and np.any(mask_elbo_diff_unstable):
    plt.bar(x_rel[mask_elbo_diff_unstable], (elbo[mask_elbo_diff_unstable] - elbo_ref), width=width, color='red', label='Not stable', align='center')
if ('ONLY_STABLE' in globals() and ONLY_STABLE) and np.any(mask_plot_elbo):
    plt.bar(x_rel[mask_plot_elbo], (elbo[mask_plot_elbo] - elbo_ref), width=width, color='green', label='Stable', align='center')
plt.axhline(0.0, color='k', linewidth=1, alpha=0.5)
plt.xlabel('theta (fixed)')
plt.ylabel('ELBO - ELBO[3]')
plt.title('ELBO - ELBO[theta = 3] vs theta' + (' (only stable)' if ('ONLY_STABLE' in globals() and ONLY_STABLE) else ''))
plt.grid(True, alpha=0.3)
plt.legend()
if np.any(mask_plot_elbo):
    xt = x_rel[mask_plot_elbo]
    plt.xticks(xt)
    plt.xlim(xt.min() - 0.5, xt.max() + 0.5)
elbo_diff_plot_path = os.path.join(out_dir, 'elbo_diff_vs_theta.png')
plt.tight_layout()
plt.ylim(bottom=-200, top=25)

plt.savefig(elbo_diff_plot_path, dpi=150)
print(f"Saved plot: {elbo_diff_plot_path}")

# E[log joint] difference
mask_ej_diff = has_file & np.isfinite(e_log_joint) & np.isfinite(ej_ref)
mask_ej_diff_stable = mask_ej_diff & stable
mask_ej_diff_unstable = mask_ej_diff & ~stable
if 'ONLY_STABLE' in globals() and ONLY_STABLE:
    mask_plot_ej = mask_ej_diff_stable
else:
    mask_plot_ej = mask_ej_diff

plt.figure(figsize=(10, 5))
if np.any(mask_ej_diff_stable) and (not ('ONLY_STABLE' in globals() and ONLY_STABLE)):
    plt.bar(x_rel[mask_ej_diff_stable], (e_log_joint[mask_ej_diff_stable] - ej_ref), width=width, color='green', label='Stable', align='center')
if (not ('ONLY_STABLE' in globals() and ONLY_STABLE)) and np.any(mask_ej_diff_unstable):
    plt.bar(x_rel[mask_ej_diff_unstable], (e_log_joint[mask_ej_diff_unstable] - ej_ref), width=width, color='red', label='Not stable', align='center')
if ('ONLY_STABLE' in globals() and ONLY_STABLE) and np.any(mask_plot_ej):
    plt.bar(x_rel[mask_plot_ej], (e_log_joint[mask_plot_ej] - ej_ref), width=width, color='green', label='Stable', align='center')
plt.axhline(0.0, color='k', linewidth=1, alpha=0.5)
plt.xlabel('theta (fixed)')
plt.ylabel('E[log joint] - E[log joint][theta = 3]')
plt.title('E[log joint] - E[log joint][theta = 3] vs theta' + (' (only stable)' if ('ONLY_STABLE' in globals() and ONLY_STABLE) else ''))
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(bottom=-200, top=25)
if np.any(mask_plot_ej):
    xt = x_rel[mask_plot_ej]
    plt.xticks(xt)
    plt.xlim(xt.min() - 0.5, xt.max() + 0.5)
elog_diff_plot_path = os.path.join(out_dir, 'elog_diff_vs_theta.png')
plt.tight_layout()
plt.savefig(elog_diff_plot_path, dpi=150)
print(f"Saved plot: {elog_diff_plot_path}")

# %%

# 1) How many stable (converged)?
total_with_file = int(np.sum(has_file))
stable_count = int(np.sum(stable & has_file))
print(f"Stable (converged): {stable_count} / {total_with_file} with files; {stable_count} / {N_theta} overall")
if total_with_file > 0 and stable_count < total_with_file:
    unstable_thetas = np.where(has_file & ~stable)[0] + 1
    print(f"Non-stable thetas: {list(unstable_thetas)}")

# Common x-axis values (theta indices)
x = np.arange(1, N_theta + 1)

# 2) Plot ELBO with sd as error bars, color-coded by stability
mask_elbo = has_file & ~np.isnan(elbo) & ~np.isnan(elbo_sd)
mask_elbo_stable = mask_elbo & stable
mask_elbo_unstable = mask_elbo & ~stable

plt.figure(figsize=(10, 5))
if ONLY_STABLE:
    if np.any(mask_elbo_stable):
        plt.errorbar(
            x[mask_elbo_stable], elbo[mask_elbo_stable], yerr=elbo_sd[mask_elbo_stable],
            fmt='o', color='green', ecolor='green', elinewidth=1, capsize=2, label='Stable'
        )
else:
    if np.any(mask_elbo_stable):
        plt.errorbar(
            x[mask_elbo_stable], elbo[mask_elbo_stable], yerr=elbo_sd[mask_elbo_stable],
            fmt='o', color='green', ecolor='green', elinewidth=1, capsize=2, label='Stable'
        )
    if np.any(mask_elbo_unstable):
        plt.errorbar(
            x[mask_elbo_unstable], elbo[mask_elbo_unstable], yerr=elbo_sd[mask_elbo_unstable],
            fmt='o', color='red', ecolor='red', elinewidth=1, capsize=2, label='Not stable'
        )
plt.xlabel('theta (fixed)')
plt.ylabel('ELBO')
plt.title('ELBO vs theta (error bars = sd)' + (' (only stable)' if ONLY_STABLE else ''))
plt.ylim(bottom=-2400, top = -2150)
plt.grid(True, alpha=0.3)
plt.legend()
# Filter x-axis ticks/limits based on mask
xt_mask = mask_elbo_stable if ONLY_STABLE else mask_elbo
if np.any(xt_mask):
    xt = x[xt_mask]
    plt.xticks(xt)
    plt.xlim(xt.min() - 0.5, xt.max() + 0.5)
elbo_plot_path = os.path.join(out_dir, 'elbo_vs_theta.png')
plt.tight_layout()
plt.savefig(elbo_plot_path, dpi=150)
print(f"Saved plot: {elbo_plot_path}")

# 3) Plot E[log joint] with sd as error bars, color-coded by stability
mask_ej = has_file & ~np.isnan(e_log_joint) & ~np.isnan(e_log_joint_sd)
mask_ej_stable = mask_ej & stable
mask_ej_unstable = mask_ej & ~stable

plt.figure(figsize=(10, 5))
if ONLY_STABLE:
    if np.any(mask_ej_stable):
        plt.errorbar(
            x[mask_ej_stable], e_log_joint[mask_ej_stable], yerr=e_log_joint_sd[mask_ej_stable],
            fmt='o', color='green', ecolor='green', elinewidth=1, capsize=2, label='Stable'
        )
else:
    if np.any(mask_ej_stable):
        plt.errorbar(
            x[mask_ej_stable], e_log_joint[mask_ej_stable], yerr=e_log_joint_sd[mask_ej_stable],
            fmt='o', color='green', ecolor='green', elinewidth=1, capsize=2, label='Stable'
        )
    if np.any(mask_ej_unstable):
        plt.errorbar(
            x[mask_ej_unstable], e_log_joint[mask_ej_unstable], yerr=e_log_joint_sd[mask_ej_unstable],
            fmt='o', color='red', ecolor='red', elinewidth=1, capsize=2, label='Not stable'
        )
plt.xlabel('theta (fixed)')
plt.ylabel('E[log joint]')
plt.title('E[log joint] vs theta (error bars = sd)' + (' (only stable)' if ONLY_STABLE else ''))
plt.grid(True, alpha=0.3)
plt.ylim(bottom=-2400, top=-2150)

plt.legend()
# Filter x-axis ticks/limits based on mask
xt_mask_ej = mask_ej_stable if ONLY_STABLE else mask_ej
if np.any(xt_mask_ej):
    xt = x[xt_mask_ej]
    plt.xticks(xt)
    plt.xlim(xt.min() - 0.5, xt.max() + 0.5)
elog_plot_path = os.path.join(out_dir, 'e_log_joint_vs_theta.png')
plt.tight_layout()
plt.savefig(elog_plot_path, dpi=150)
print(f"Saved plot: {elog_plot_path}")

# %%
# 10x5 grid of histograms for R and L samples per theta
rows, cols = 10, 5
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 4.2))
for theta in range(1, N_theta + 1):
    r = (theta - 1) // cols
    c = (theta - 1) % cols
    ax = axes[r, c]

    if not has_file[theta - 1]:
        ax.set_visible(False)
        continue

    r_path = os.path.join(out_dir, f'theta{theta:02d}_R_samples.npy')
    l_path = os.path.join(out_dir, f'theta{theta:02d}_L_samples.npy')
    try:
        R = np.load(r_path)
        L = np.load(l_path)
        ax.hist(R, bins=50, density=True, alpha=0.5, color='tab:blue')
        ax.hist(L, bins=50, density=True, alpha=0.5, color='tab:orange')
    except Exception as e:
        ax.text(0.5, 0.5, 'Load error', transform=ax.transAxes, ha='center', va='center', fontsize=8)

    title_color = 'black' if stable[theta - 1] else 'red'
    ax.set_title(f'theta={theta:02d}', color=title_color, fontsize=15)
    ax.tick_params(labelsize=7)

# Add a single legend for colors
try:
    mlines  # type: ignore  # noqa: F821
except NameError:
    import matplotlib.lines as mlines  # type: ignore
r_handle = mlines.Line2D([], [], color='tab:blue', marker='s', linestyle='None', markersize=6, label='R')
l_handle = mlines.Line2D([], [], color='tab:orange', marker='s', linestyle='None', markersize=6, label='L')
fig.legend(handles=[r_handle, l_handle], loc='upper right')

fig.suptitle('Posterior histograms (density=True) of R (blue) and L (orange) per theta', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])
grid_plot_path = os.path.join(out_dir, 'rl_hist_grid_10x5.png')
plt.savefig(grid_plot_path, dpi=150)
print(f"Saved plot: {grid_plot_path}")

# %%
# Plot of R samples mean and L samples mean vs theta
ONLY_STABLE=False
R_mean = np.full(N_theta, np.nan)
L_mean = np.full(N_theta, np.nan)
R_std = np.full(N_theta, np.nan)
L_std = np.full(N_theta, np.nan)
for theta in range(1, N_theta + 1):
    if not has_file[theta - 1]:
        continue
    r_path = os.path.join(out_dir, f'theta{theta:02d}_R_samples.npy')
    l_path = os.path.join(out_dir, f'theta{theta:02d}_L_samples.npy')
    try:
        # Use memory mapping to avoid loading entire arrays eagerly if very large
        R = np.load(r_path, mmap_mode='r')
        L = np.load(l_path, mmap_mode='r')
        R_mean[theta - 1] = float(np.mean(R))
        L_mean[theta - 1] = float(np.mean(L))
        R_std[theta - 1] = float(np.std(R))
        L_std[theta - 1] = float(np.std(L))
    except Exception as e:
        print(f"[WARN] Could not compute means for theta={theta:02d}: {e}")

x_means = np.arange(1, N_theta + 1)
mask_means = has_file & ~np.isnan(R_mean) & ~np.isnan(L_mean) & ~np.isnan(R_std) & ~np.isnan(L_std)
if ONLY_STABLE:
    mask_means = mask_means & stable
plt.figure(figsize=(12, 5))
if np.any(mask_means):
    plt.errorbar(
        x_means[mask_means], R_mean[mask_means], yerr=R_std[mask_means],
        fmt='o', color='green', ecolor='green', elinewidth=1, capsize=2, label='R mean ± sd'
    )
    plt.errorbar(
        x_means[mask_means], L_mean[mask_means], yerr=L_std[mask_means],
        fmt='o', color='black', ecolor='black', elinewidth=1, capsize=2, label='L mean ± sd'
    )
# Thin red vertical lines for non-converged thetas
for t in (np.where(has_file & ~stable)[0] + 1):
    plt.axvline(t, color='red', ls='--', alpha=0.4)
plt.xlabel('theta (fixed)')
plt.ylabel('Mean ± sd of samples')
plt.title('Mean ± sd of R (green) and L (black) vs theta')
plt.grid(True, alpha=0.3)
plt.legend()
if np.any(mask_means):
    xt = x_means[mask_means]
    plt.xticks(xt)
    plt.xlim(xt.min() - 0.5, xt.max() + 0.5)
means_plot_path = os.path.join(out_dir, 'rl_means_vs_theta.png')
plt.tight_layout()
plt.savefig(means_plot_path, dpi=150)
print(f"Saved plot: {means_plot_path}")
# %%
# Plot log10 of means vs theta (same masking)
plt.figure(figsize=(12, 5))
if np.any(mask_means):
    plt.scatter(x_means[mask_means], np.log10(R_mean[mask_means]), color='green', label='log10 R mean')
    plt.scatter(x_means[mask_means], np.log10(L_mean[mask_means]), color='black', label='log10 L mean')
for t in (np.where(has_file & ~stable)[0] + 1):
    plt.axvline(t, color='red', ls='--', alpha=0.4)
plt.xlabel('theta (fixed)')
plt.ylabel('log10(mean of samples)')
plt.title('log10(mean) of R (green) and L (black) vs theta')
plt.grid(True, alpha=0.3)
plt.legend()
if np.any(mask_means):
    xt = x_means[mask_means]
    plt.xticks(xt)
    plt.xlim(xt.min() - 0.5, xt.max() + 0.5)
means_log_plot_path = os.path.join(out_dir, 'rl_means_log10_vs_theta.png')
plt.tight_layout()
plt.savefig(means_log_plot_path, dpi=150)
print(f"Saved plot: {means_log_plot_path}")

# ... (rest of the code remains the same)

# Compute Gamma and Omega from means (same mask)
den = R_mean + L_mean
gamma = np.full(N_theta, np.nan)
omega = np.full(N_theta, np.nan)
valid_go = mask_means & (den > 0)
ONLY_STABLE=True
# Toggle stable-only mask via ONLY_STABLE (set at top)
if ONLY_STABLE:
    valid_go = valid_go & stable
gamma[valid_go] = ((R_mean[valid_go] - L_mean[valid_go]) / den[valid_go]) * x_means[valid_go]
omega[valid_go] = den[valid_go] / (x_means[valid_go] ** 2)

# 1) Gamma vs theta (scatter)
plt.figure(figsize=(12, 5))
if np.any(valid_go):
    plt.scatter(x_means[valid_go], gamma[valid_go], color='purple', label='Gamma')
# True values from simulation setup
R_true, L_true, theta_true = 15.0, 12.0, 3
gamma_true_val = ((R_true - L_true) / (R_true + L_true)) * theta_true
# plt.axvline(theta_true, color='black', ls='-', alpha=0.8, linewidth=1.2, label=f'true theta={theta_true}')
# plt.scatter([theta_true], [gamma_true_val], color='black', marker='x', s=50, label=f'true gamma={gamma_true_val:.3f}')
plt.axhline(gamma_true_val, color='black', ls='-', alpha=0.8, linewidth=1.2, label=f'true gamma={gamma_true_val:.3f}')
for t in (np.where(has_file & ~stable)[0] + 1):
    plt.axvline(t, color='red', ls='--', alpha=0.4)
plt.xlabel('theta (fixed)')
plt.ylabel('Gamma')
plt.title('Gamma vs theta' + (' (only stable)' if ONLY_STABLE else ''))
plt.grid(True, alpha=0.3)
plt.legend()
if np.any(valid_go):
    xt = x_means[valid_go]
    plt.xticks(xt)
    plt.xlim(xt.min() - 0.5, xt.max() + 0.5)
gamma_plot_path = os.path.join(out_dir, 'gamma_vs_theta.png')
plt.tight_layout()
plt.savefig(gamma_plot_path, dpi=150)
print(f"Saved plot: {gamma_plot_path}")

# 2) Omega vs theta (scatter)
plt.figure(figsize=(12, 5))
if np.any(valid_go):
    plt.scatter(x_means[valid_go], omega[valid_go], color='blue', label='Omega')
# True values from simulation setup
R_true, L_true, theta_true = 15.0, 12.0, 3
omega_true_val = (R_true + L_true) / (theta_true ** 2)
# plt.axvline(theta_true, color='black', ls='-', alpha=0.8, linewidth=1.2, label=f'true theta={theta_true}')
# plt.scatter([theta_true], [omega_true_val], color='black', marker='x', s=50, label=f'true omega={omega_true_val:.3f}')
plt.axhline(omega_true_val, color='black', ls='-', alpha=0.8, linewidth=1.2, label=f'true omega={omega_true_val:.3f}')
for t in (np.where(has_file & ~stable)[0] + 1):
    plt.axvline(t, color='red', ls='--', alpha=0.4)
plt.xlabel('theta (fixed)')
plt.ylabel('Omega')
plt.title('Omega vs theta' + (' (only stable)' if ONLY_STABLE else ''))
plt.grid(True, alpha=0.3)
plt.legend()
if np.any(valid_go):
    xt = x_means[valid_go]
    plt.xticks(xt)
    plt.xlim(xt.min() - 0.5, xt.max() + 0.5)
omega_plot_path = os.path.join(out_dir, 'omega_vs_theta.png')
plt.tight_layout()
plt.savefig(omega_plot_path, dpi=150)
print(f"Saved plot: {omega_plot_path}")

# %%
# Simulated RT distributions: TRUE vs posterior MEAN per stable theta
# 1) Simulate once from TRUE params
# 2) For each stable theta, simulate from posterior MEAN (R_mean[theta], L_mean[theta], theta)
# 3) Plot probability-weighted mirrored step histograms (areas equal choice probabilities)

# Constants for TRUE simulation (from SIM code)
R_TRUE = 15.0
L_TRUE = 12.0
THETA_TRUE = 3
N_CMP_SIM = 50_000  # number of trials to simulate per condition

def _simulate_trial(mu1, mu2, theta, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x = 0
    time = 0.0
    total_rate = mu1 + mu2
    if total_rate <= 0:
        return np.nan, 0
    prob_up = mu1 / total_rate
    while abs(x) < theta:
        dt = rng.exponential(1.0 / total_rate)
        time += dt
        if rng.random() < prob_up:
            x += 1
        else:
            x -= 1
    choice = 1 if x >= theta else -1
    return time, choice

def _simulate_many(mu1, mu2, theta, n_trials, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    times = np.empty(n_trials, dtype=float)
    choices = np.empty(n_trials, dtype=int)
    for i in range(n_trials):
        t, ch = _simulate_trial(mu1, mu2, int(theta), rng=rng)
        times[i] = t
        choices[i] = ch
    return times, choices

# Simulate TRUE once
rng_true = np.random.default_rng(123)
rt_true_all, ch_true_all = _simulate_many(R_TRUE, L_TRUE, THETA_TRUE, N_CMP_SIM, rng=rng_true)

# Prepare output directory
dist_dir = os.path.join(out_dir, 'dist_compare')
os.makedirs(dist_dir, exist_ok=True)

# Fixed bins for comparability
t_max = 2.0
bins = np.arange(0.0, t_max, 0.01)

# Prepare a 10x5 grid figure (one subplot per theta)
rows, cols = 10, 5
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.2))

# Loop across requested thetas; only process stable ones with finite means
for theta in range(theta_start, theta_end + 1):
    idx = theta - 1
    r = (theta - 1) // cols
    c = (theta - 1) % cols
    ax = axes[r, c]

    if not (has_file[idx] and stable[idx]) or not (np.isfinite(R_mean[idx]) and np.isfinite(L_mean[idx])):
        ax.set_visible(False)
        continue

    Rm = float(R_mean[idx])
    Lm = float(L_mean[idx])

    # Simulate MEAN for this theta
    rng_mean = np.random.default_rng(123 + theta)
    rt_mean_all, ch_mean_all = _simulate_many(Rm, Lm, theta, N_CMP_SIM, rng=rng_mean)

    # Split by bound
    rt_true_up = rt_true_all[ch_true_all == 1]
    rt_true_lo = rt_true_all[ch_true_all == -1]
    rt_mean_up = rt_mean_all[ch_mean_all == 1]
    rt_mean_lo = rt_mean_all[ch_mean_all == -1]

    # Histograms (density) using common bins
    h_tu, edges = np.histogram(rt_true_up, bins=bins, density=True)
    h_mu, _     = np.histogram(rt_mean_up, bins=bins, density=True)
    h_tl, _     = np.histogram(rt_true_lo, bins=bins, density=True)
    h_ml, _     = np.histogram(rt_mean_lo, bins=bins, density=True)

    # Probability weights (empirical choice fractions)
    p_true_up = rt_true_up.size / max(rt_true_all.size, 1)
    p_true_lo = rt_true_lo.size / max(rt_true_all.size, 1)
    p_mean_up = rt_mean_up.size / max(rt_mean_all.size, 1)
    p_mean_lo = rt_mean_lo.size / max(rt_mean_all.size, 1)

    widths = np.diff(edges)
    area_tu = float(np.sum(h_tu * widths))
    area_tl = float(np.sum(h_tl * widths))
    area_mu = float(np.sum(h_mu * widths))
    area_ml = float(np.sum(h_ml * widths))

    eps = 1e-12
    h_tu_w = h_tu * (p_true_up / max(area_tu, eps))
    h_tl_w = h_tl * (p_true_lo / max(area_tl, eps))
    h_mu_w = h_mu * (p_mean_up / max(area_mu, eps))
    h_ml_w = h_ml * (p_mean_lo / max(area_ml, eps))

    # Plot on the subplot
    ax.axhline(0.0, color='k', linewidth=1)
    # Upper bound (positive)
    ax.step(edges[:-1], h_tu_w, where='post', label='TRUE upper (area=P(+1))', color='C0')
    ax.step(edges[:-1], h_mu_w, where='post', label='MEAN upper (area=P(+1))', color='C1')
    # Lower bound (negative)
    ax.step(edges[:-1], -h_tl_w, where='post', label='TRUE lower (area=P(-1))', color='C0', linestyle='--')
    ax.step(edges[:-1], -h_ml_w, where='post', label='MEAN lower (area=P(-1))', color='C1', linestyle='--')

    ymax_w = 1.1 * float(max(h_tu_w.max(), h_mu_w.max(), h_tl_w.max(), h_ml_w.max(), 1e-12))
    ax.set_ylim(-ymax_w, ymax_w)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel('RT')
    ax.set_ylabel('Prob. density')
    ax.set_title(f'theta={theta:02d}', fontsize=10)

# Single legend for the whole figure
try:
    mlines  # type: ignore  # noqa: F821
except NameError:
    import matplotlib.lines as mlines  # type: ignore
true_handle = mlines.Line2D([], [], color='C0', label='TRUE')
mean_handle = mlines.Line2D([], [], color='C1', label='MEAN')
fig.legend(handles=[true_handle, mean_handle], loc='upper right', ncol=2)

fig.suptitle('RT distributions by bound: TRUE vs MEAN (areas reflect choice probabilities)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])
grid_out_path = os.path.join(dist_dir, 'true_vs_mean_weighted_grid.png')
plt.savefig(grid_out_path, dpi=150)
print(f"Saved plot: {grid_out_path}")

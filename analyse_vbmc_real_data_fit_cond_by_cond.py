# %%
#!/usr/bin/env python3
import os
import argparse
import traceback
import numpy as np
import matplotlib.pyplot as plt
from pyvbmc import VBMC
import pandas as pd
from vbmc_skellam_utils import up_or_down_hit_fn, cum_pro_and_reactive_trunc_fn, up_or_down_hit_wrt_tstim, up_or_down_hit_wrt_tstim_V2, up_or_down_hit_wrt_tstim_V3


def extract_stats_from_vp(vp):
    """
    Robustly extract elbo, elbo_sd (fallback elbo_std), and stable
    from a loaded VBMC object, trying vp.vp.stats first, then vp.stats.
    Returns (elbo, elbo_sd, stable) as floats/bool or (np.nan, np.nan, False) on failure.
    """
    try:
        stats = getattr(vp.vp, 'stats', None)
        if not isinstance(stats, dict):
            stats = getattr(vp, 'stats', None)
        if isinstance(stats, dict):
            elbo = float(stats.get('elbo', np.nan))
            elbo_sd = float(stats.get('elbo_sd', stats.get('elbo_std', np.nan)))
            stable = bool(stats.get('stable', False))
            return elbo, elbo_sd, stable
    except Exception:
        pass
    return np.nan, np.nan, False


def load_condition_theta_series(root_dir, abl, ild, n_theta):
    """
    For a given (ABL, ILD), load up to n_theta PKLs and return arrays:
      - elbo[n_theta], elbo_sd[n_theta], stable[n_theta] (bool), has_file[n_theta] (bool)
    Missing files or load errors produce NaNs/False accordingly.
    """
    cond_dir = os.path.join(root_dir, f"ABL_{abl}_ILD_{ild}")
    elbo = np.full(n_theta, np.nan, dtype=float)
    elbo_sd = np.full(n_theta, np.nan, dtype=float)
    stable = np.zeros(n_theta, dtype=bool)
    has_file = np.zeros(n_theta, dtype=bool)

    if not os.path.isdir(cond_dir):
        # No directory at all, return empty arrays
        return elbo, elbo_sd, stable, has_file

    for t in range(1, n_theta + 1):
        pkl_path = os.path.join(cond_dir, f"ABL_{abl}_ILD_{ild}_theta{t:02d}.pkl")
        if not os.path.exists(pkl_path):
            continue
        try:
            vp = VBMC.load(pkl_path)
            e, e_sd, st = extract_stats_from_vp(vp)
            elbo[t - 1] = e
            elbo_sd[t - 1] = e_sd
            stable[t - 1] = st
            has_file[t - 1] = True
        except Exception as err:
            print(f"[WARN] Failed to load or extract stats from: {pkl_path}\n  -> {err}")
            traceback.print_exc()
            continue

    return elbo, elbo_sd, stable, has_file


def plot_elbo_grid(root_dir, abls, ilds, n_theta, out_name="elbo_grid_10x3.png", only_stable=False):
    """
    Create a grid of ELBO vs theta with error bars for each (ILD row, ABL col).
    - Green = stable points
    - Red   = unstable points (omitted if only_stable is True)
    """
    rows = len(ilds)
    cols = len(abls)
    # Slightly larger for readability
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 2.6), squeeze=False)

    # For legend handles
    stable_color = "green"
    unstable_color = "red"

    for r, ild in enumerate(ilds):
        for c, abl in enumerate(abls):
            ax = axes[r, c]
            elbo, elbo_sd, stable, has_file = load_condition_theta_series(root_dir, abl, ild, n_theta)
            x = np.arange(1, n_theta + 1)

            mask_elbo = has_file & np.isfinite(elbo) & np.isfinite(elbo_sd)
            mask_stable = mask_elbo & stable
            mask_unstable = mask_elbo & (~stable)

            if np.any(mask_stable):
                ax.errorbar(
                    x[mask_stable], elbo[mask_stable], yerr=elbo_sd[mask_stable],
                    fmt='o', color=stable_color, ecolor=stable_color,
                    elinewidth=1, capsize=2, markersize=3, label='Stable'
                )
            if (not only_stable) and np.any(mask_unstable):
                ax.errorbar(
                    x[mask_unstable], elbo[mask_unstable], yerr=elbo_sd[mask_unstable],
                    fmt='o', color=unstable_color, ecolor=unstable_color,
                    elinewidth=1, capsize=2, markersize=3, label='Not stable'
                )

            plot_mask = mask_stable if only_stable else mask_elbo
            if not np.any(plot_mask):
                msg = 'No stable data' if only_stable else 'No data'
                ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                        ha='center', va='center', fontsize=9, color='gray')

            ax.set_title(f"ABL={abl}, ILD={ild}", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Lighter tick density to reduce clutter
            ax.tick_params(labelsize=8)
            # Show axis labels only on leftmost column and bottom row for cleanliness
            if c == 0:
                ax.set_ylabel("ELBO", fontsize=9)
            if r == rows - 1:
                ax.set_xlabel("theta", fontsize=9)

    # Global legend
    # Build simple dummy handles so legend is always present
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', label='Stable',
               markerfacecolor=stable_color, markeredgecolor=stable_color, markersize=6)
    ]
    if not only_stable:
        handles.append(
            Line2D([0], [0], marker='o', color='w', label='Not stable',
                   markerfacecolor=unstable_color, markeredgecolor=unstable_color, markersize=6)
        )
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False)

    title_suffix = " (only stable)" if only_stable else ""
    fig.suptitle("ELBO vs theta by condition (error bars = sd)" + title_suffix, fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(root_dir, out_name)
    os.makedirs(root_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


# %%
def _load_animal_params(batch_name: str, animal_id: str):
    """
    Load proactive (V_A, theta_A, t_A_aff) and executive offsets (t_E_aff, del_go)
    from the per-animal results pickle, following the same conventions as the fit script.

    Returns dict with keys: V_A, theta_A, t_A_aff, t_E_aff, del_go
    """
    import pickle
    pkl_file = f"/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/results_{batch_name}_animal_{animal_id}.pkl"
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"Animal results pickle not found: {pkl_file}")
    with open(pkl_file, 'rb') as f:
        fit_results_data = pickle.load(f)

    # Maps as in fit script
    vbmc_aborts_param_keys_map = {
        'V_A_samples': 'V_A',
        'theta_A_samples': 'theta_A',
        't_A_aff_samp': 't_A_aff'
    }
    vbmc_vanilla_tied_param_keys_map = {
        'rate_lambda_samples': 'rate_lambda',
        'T_0_samples': 'T_0',
        'theta_E_samples': 'theta_E',
        'w_samples': 'w',
        't_E_aff_samples': 't_E_aff',
        'del_go_samples': 'del_go'
    }
    vbmc_norm_tied_param_keys_map = {
        **vbmc_vanilla_tied_param_keys_map,
        'rate_norm_l_samples': 'rate_norm_l'
    }

    V_A = theta_A = t_A_aff = np.nan
    t_E_aff_vals = []
    del_go_vals = []

    abort_keyname = "vbmc_aborts_results"
    if abort_keyname in fit_results_data:
        abort_samples = fit_results_data[abort_keyname]
        V_A = float(np.mean(abort_samples.get('V_A_samples', np.nan)))
        theta_A = float(np.mean(abort_samples.get('theta_A_samples', np.nan)))
        t_A_aff = float(np.mean(abort_samples.get('t_A_aff_samp', np.nan)))
    else:
        print(f"[WARN] {abort_keyname} not found in {pkl_file}")

    # vanilla tied
    tied_keyname = "vbmc_vanilla_tied_results"
    if tied_keyname in fit_results_data:
        tied_samples = fit_results_data[tied_keyname]
        if 't_E_aff_samples' in tied_samples:
            t_E_aff_vals.append(float(np.mean(tied_samples['t_E_aff_samples'])))
        if 'del_go_samples' in tied_samples:
            del_go_vals.append(float(np.mean(tied_samples['del_go_samples'])))
    # norm tied
    tied_keyname = "vbmc_norm_tied_results"
    if tied_keyname in fit_results_data:
        tied_samples = fit_results_data[tied_keyname]
        if 't_E_aff_samples' in tied_samples:
            t_E_aff_vals.append(float(np.mean(tied_samples['t_E_aff_samples'])))
        if 'del_go_samples' in tied_samples:
            del_go_vals.append(float(np.mean(tied_samples['del_go_samples'])))

    # average t_E_aff and del_go
    t_E_aff = float(np.mean(t_E_aff_vals)) if t_E_aff_vals else np.nan
    del_go = float(np.mean(del_go_vals)) if del_go_vals else np.nan

    return {
        'V_A': V_A,
        'theta_A': theta_A,
        't_A_aff': t_A_aff,
        't_E_aff': t_E_aff,
        'del_go': del_go,
    }


def _map_choice(row):
    """Map row to choice +1 / -1 using response_poke if available; fallback to success."""
    if 'response_poke' in row:
        try:
            rp = int(row['response_poke'])
            if rp in (2, 3):
                return 2 * rp - 5  # 3->+1, 2->-1
            if rp in (-1, 1):
                return rp
        except Exception:
            pass
    if 'success' in row and row['success'] in (-1, 1):
        return int(row['success'])
    return None


def _find_best_theta_and_means(root_dir, abl, ild, n_theta, only_stable, n_mu_samples):
    """
    Among available thetas for a condition, choose the best by highest ELBO
    (restricted to stable if only_stable). Return (theta_best, R_mean, L_mean).
    """
    elbo, elbo_sd, stable, has_file = load_condition_theta_series(root_dir, abl, ild, n_theta)
    mask = has_file & np.isfinite(elbo)
    if only_stable:
        mask = mask & stable
    if not np.any(mask):
        return None, np.nan, np.nan
    # choose argmax ELBO
    elbo_sel = np.where(mask, elbo, -np.inf)
    idx = int(np.argmax(elbo_sel))
    theta_best = idx + 1

    # Load VP and estimate means of R, L by sampling
    cond_dir = os.path.join(root_dir, f"ABL_{abl}_ILD_{ild}")
    pkl_path = os.path.join(cond_dir, f"ABL_{abl}_ILD_{ild}_theta{theta_best:02d}.pkl")
    try:
        vp = VBMC.load(pkl_path)
        samples_obj = vp.vp.sample(int(n_mu_samples))
        samples = samples_obj[0] if isinstance(samples_obj, (tuple, list)) else samples_obj
        # samples are in log10-space [logR, logL]
        R = 10 ** samples[:, 0]
        L = 10 ** samples[:, 1]
        R_mean = float(np.mean(R))
        L_mean = float(np.mean(L))
        return theta_best, R_mean, L_mean
    except Exception as e:
        print(f"[WARN] Failed to load VP or sample for {pkl_path}: {e}")
        return theta_best, np.nan, np.nan


def plot_rt_dists_grid(
    root_dir: str,
    batch_name: str,
    animal_id: str,
    abls: list,
    ilds: list,
    n_theta: int,
    out_name: str = "rt_dists_data_vs_theory_grid.png",
    only_stable: bool = True,
    n_mu_samples: int = 10000,
    bin_width: float = 0.01,
    rt_max: float = 1.0,
    c_A_trunc_time: float = 0.3,
):
    """
    Plot a grid (ILD rows x ABL cols) comparing DATA RT histograms vs THEORETICAL
    RT densities for each choice (+1 and -1). The theoretical curves use the
    condition's best (highest ELBO) stable theta and VP-mean rates (R,L), and
    proactive parameters loaded from the animal results pickle.
    """
    # Load animal params
    try:
        params = _load_animal_params(batch_name, animal_id)
    except Exception as e:
        print(f"[ERROR] Could not load animal params for {batch_name}, {animal_id}: {e}")
        return
    V_A = params['V_A']
    theta_A = int(params['theta_A']) if not np.isnan(params['theta_A']) else int(1)
    t_A_aff = params['t_A_aff']
    t_E_aff = params['t_E_aff']
    del_go = params['del_go']

    # Load data CSV and prepare RTs relative to stim
    csv_path = f"/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/batch_csvs/batch_{batch_name}_valid_and_aborts.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] Data CSV not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    df_animal = df[df['animal'] == int(animal_id)].copy()
    df_animal = df_animal[df_animal['success'].isin([1, -1])].copy()

    # Compute RT relative to stim: timed_fix - intended_fix (fallback TotalFixTime)
    if 'timed_fix' in df_animal.columns:
        timed = df_animal['timed_fix']
    else:
        timed = df_animal['TotalFixTime']
    rt_rel = timed - df_animal['intended_fix']
    df_animal['rt_rel'] = rt_rel
    df_animal = df_animal[(df_animal['rt_rel'] > 0) & (df_animal['rt_rel'] <= rt_max)].copy()

    # choice mapping
    df_animal['choice'] = df_animal.apply(_map_choice, axis=1)

    rows, cols = len(ilds), len(abls)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 2.6), squeeze=False)

    # Fixed time grid for theory
    t_grid = np.arange(0.0, rt_max, 0.001)
    eps = 1e-12

    for r, ild in enumerate(ilds):
        for c, abl in enumerate(abls):
            ax = axes[r, c]
            df_cond = df_animal[(df_animal['ABL'] == abl) & (df_animal['ILD'] == ild)]
            if df_cond.empty:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=9, color='gray')
                ax.set_title(f"ABL={abl}, ILD={ild}", fontsize=10)
                ax.grid(True, alpha=0.3)
                if c == 0:
                    ax.set_ylabel("Prob. density", fontsize=9)
                if r == rows - 1:
                    ax.set_xlabel("RT (s)", fontsize=9)
                continue

            # Data split by choice
            rts_pos = df_cond[df_cond['choice'] == 1]['rt_rel'].to_numpy()
            rts_neg = df_cond[df_cond['choice'] == -1]['rt_rel'].to_numpy()
            n_tot = rts_pos.size + rts_neg.size
            if n_tot == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=9, color='gray')
                ax.set_title(f"ABL={abl}, ILD={ild}", fontsize=10)
                ax.grid(True, alpha=0.3)
                if c == 0:
                    ax.set_ylabel("Prob. density", fontsize=9)
                if r == rows - 1:
                    ax.set_xlabel("RT (s)", fontsize=9)
                continue
            p_pos = rts_pos.size / n_tot
            p_neg = rts_neg.size / n_tot

            # Data histograms (density=True), then weight to match empirical fractions
            edges = np.arange(0.0, rt_max + bin_width, bin_width)
            if rts_pos.size > 0:
                h_pos, _ = np.histogram(rts_pos, bins=edges, density=True)
                area_pos = float(np.sum(h_pos * np.diff(edges)))
                h_pos_w = h_pos * (p_pos / max(area_pos, eps))
                ax.step(edges[:-1], h_pos_w, where='post', color='C0', label='DATA +1')
            if rts_neg.size > 0:
                h_neg, _ = np.histogram(rts_neg, bins=edges, density=True)
                area_neg = float(np.sum(h_neg * np.diff(edges)))
                h_neg_w = h_neg * (p_neg / max(area_neg, eps))
                ax.step(edges[:-1], -h_neg_w, where='post', color='C0', linestyle='--', label='DATA -1')

            # Theory using best stable theta and VP-mean R,L (via up_or_down_hit_wrt_tstim)
            theta_best, R_mean, L_mean = _find_best_theta_and_means(
                root_dir, abl, ild, n_theta, only_stable=only_stable, n_mu_samples=n_mu_samples
            )
            if theta_best is not None and np.isfinite(R_mean) and np.isfinite(L_mean) and np.isfinite(t_E_aff):
                try:
                    # Build t_pts and collect t_stim samples for this condition
                    t_pts = np.arange(-1.0, 2.0 + 0.001, 0.001)
                    t_stim_vals = df_cond['intended_fix'].to_numpy()
                    # Subsample to avoid oversizing inside util function
                    max_samples = 1000
                    if t_stim_vals.size > max_samples:
                        t_stim_vals = t_stim_vals[:max_samples]
                    # t_pts_0_1, up_theory_mean_norm, down_theory_mean_norm = up_or_down_hit_wrt_tstim(
                    #     t_pts, V_A, theta_A, t_A_aff, t_stim_vals, t_E_aff, del_go, R_mean, L_mean, int(theta_best), c_A_trunc_time
                    # )
                    # V2 version
                    t_pts_0_1 = np.arange(-5, 5+0.001, 0.001)
                    # up_theory_mean_norm, down_theory_mean_norm = up_or_down_hit_wrt_tstim_V2(t_pts_0_1, V_A, theta_A, t_A_aff, t_stim_vals, t_E_aff, del_go, R_mean, L_mean, int(theta_best), c_A_trunc_time)
                    up_theory_mean_norm, down_theory_mean_norm = up_or_down_hit_wrt_tstim_V3(t_pts_0_1, V_A, theta_A, t_A_aff, t_stim_vals, t_E_aff, del_go, R_mean, L_mean, int(theta_best), c_A_trunc_time)
                    ax.step(t_pts_0_1, up_theory_mean_norm, where='post', color='C1', label='THEORY +1')
                    ax.step(t_pts_0_1, -down_theory_mean_norm, where='post', color='C1', linestyle='--', label='THEORY -1')
                    print(f'R_mean = {R_mean:.3f}, L_mean = {L_mean:.3f}')
                    area_up = np.trapezoid(up_theory_mean_norm, t_pts_0_1)
                    area_down = np.trapezoid(down_theory_mean_norm, t_pts_0_1)
                    print(f"ABL = {abl}, ILD = {ild}, best θ = {theta_best:02d}, area_up = {area_up:.3f}, area_down = {area_down:.3f}")
                    print(f'Area up + down = {area_up + area_down:.3f}')
                    print(f'data up = {p_pos:.3f}, data down = {p_neg:.3f}')
                    ax.set_title(f"ABL={abl}, ILD={ild}, best θ={theta_best:02d}", fontsize=10)
                except Exception as e:
                    raise ValueError(f"[WARN] up_or_down_hit_wrt_tstim failed for ABL={abl}, ILD={ild}: {e}. Falling back to direct density.")
                    # Fallback: use direct hit densities with truncation as before
                    # trunc = (
                    #     cum_pro_and_reactive_trunc_fn(rt_max, c_A_trunc_time, V_A, theta_A, t_A_aff, 0.0, t_E_aff, R_mean, L_mean, int(theta_best))
                    #     - cum_pro_and_reactive_trunc_fn(0.0, c_A_trunc_time, V_A, theta_A, t_A_aff, 0.0, t_E_aff, R_mean, L_mean, int(theta_best))
                    # )
                    # trunc = float(trunc)
                    # dens_pos = np.array([
                    #     up_or_down_hit_fn(t, V_A, theta_A, t_A_aff, 0.0, t_E_aff, del_go, R_mean, L_mean, int(theta_best), +1)
                    #     for t in t_grid
                    # ]) / max(trunc, eps)
                    # dens_neg = np.array([
                    #     up_or_down_hit_fn(t, V_A, theta_A, t_A_aff, 0.0, t_E_aff, del_go, R_mean, L_mean, int(theta_best), -1)
                    #     for t in t_grid
                    # ]) / max(trunc, eps)
                    ax.step(t_grid, dens_pos, where='post', color='C1', label='THEORY +1')
                    ax.step(t_grid, -dens_neg, where='post', color='C1', linestyle='--', label='THEORY -1')
                    ax.set_title(f"ABL={abl}, ILD={ild}, best θ={theta_best:02d}", fontsize=10)
            else:
                ax.set_title(f"ABL={abl}, ILD={ild} (no best θ)", fontsize=10)

            ax.axhline(0.0, color='k', linewidth=1)
            ax.grid(True, alpha=0.3)
            if c == 0:
                ax.set_ylabel("Prob. density", fontsize=9)
            if r == rows - 1:
                ax.set_xlabel("RT (s)", fontsize=9)

    # Single legend
    try:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([], [], color='C0', label='DATA +1'),
            Line2D([], [], color='C0', linestyle='--', label='DATA -1'),
            Line2D([], [], color='C1', label='THEORY +1'),
            Line2D([], [], color='C1', linestyle='--', label='THEORY -1'),
        ]
        fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False)
    except Exception:
        pass

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(root_dir, out_name)
    os.makedirs(root_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


# %%
def plot_elbo_diff_grid(root_dir, abls, ilds, n_theta, out_name="elbo_diff_grid_10x3.png", only_stable=False):
    """
    For each condition (ILD row, ABL col), find the theta with the highest ELBO
    (respecting ONLY_STABLE if set), then plot a bar graph of (ELBO - ELBO_max)
    across theta. Color by stability (green stable, red unstable), omitting
    unstable bars if only_stable is True.
    """
    rows = len(ilds)
    cols = len(abls)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 2.6), squeeze=False)

    width = 0.8
    stable_color = "green"
    unstable_color = "red"

    for r, ild in enumerate(ilds):
        for c, abl in enumerate(abls):
            ax = axes[r, c]
            elbo, elbo_sd, stable, has_file = load_condition_theta_series(root_dir, abl, ild, n_theta)
            x = np.arange(1, n_theta + 1)

            mask_elbo = has_file & np.isfinite(elbo)
            mask_stable = mask_elbo & stable
            mask_unstable = mask_elbo & (~stable)

            # Choose baseline among allowed thetas
            baseline_mask = mask_stable if only_stable else mask_elbo
            if np.any(baseline_mask):
                # Set invalid entries to -inf to pick argmax safely
                elbo_for_max = np.where(baseline_mask, elbo, -np.inf)
                base_idx = int(np.argmax(elbo_for_max))
                base_theta = base_idx + 1
                base_val = elbo_for_max[base_idx]
            else:
                base_idx = None
                base_theta = None
                base_val = None

            if base_val is not None and np.isfinite(base_val):
                # Plot bars of differences
                diff = elbo - base_val
                plot_mask = mask_stable if only_stable else mask_elbo
                # Stable bars
                if np.any(plot_mask & stable):
                    idx = plot_mask & stable
                    ax.bar(x[idx], diff[idx], width=width, color=stable_color, label='Stable', align='center')
                # Unstable bars (only if not only_stable)
                if (not only_stable) and np.any(plot_mask & (~stable)):
                    idx = plot_mask & (~stable)
                    ax.bar(x[idx], diff[idx], width=width, color=unstable_color, label='Not stable', align='center')

                # Aesthetics
                ref_text = f", ref theta={base_theta:02d}" if base_theta is not None else ""
                ax.set_title(f"ABL={abl}, ILD={ild}{ref_text}", fontsize=10)
                ax.axhline(0.0, color='k', linewidth=1, alpha=0.5)
            else:
                msg = 'No stable data' if only_stable else 'No data'
                ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                        ha='center', va='center', fontsize=9, color='gray')
                ax.set_title(f"ABL={abl}, ILD={ild}", fontsize=10)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            if c == 0:
                ax.set_ylabel("ELBO - ELBO_max", fontsize=9)
            if r == rows - 1:
                ax.set_xlabel("theta", fontsize=9)
            ax.set_ylim(bottom=-10)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='s', color='w', label='Stable',
               markerfacecolor=stable_color, markeredgecolor=stable_color, markersize=8)
    ]
    if not only_stable:
        handles.append(
            Line2D([0], [0], marker='s', color='w', label='Not stable',
                   markerfacecolor=unstable_color, markeredgecolor=unstable_color, markersize=8)
        )
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False)

    title_suffix = " (only stable)" if only_stable else ""
    fig.suptitle("ELBO - ELBO_max by theta and condition" + title_suffix, fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

# Config cell (edit these as needed before running the cells below)
root_dir = "real_data_cond_wise_theta_fixed"
n_theta = 50
out_name = "elbo_grid_10x3.png"
only_stable = True
batch_name = "LED8"
animal_id = "112"
mu_samples = 10000
bin_width = 0.01
rt_max = 1.0
# ca_trunc_time = 0.3
ca_trunc_time = 0

abls = [20, 40, 60]
ilds = sorted([1, -1, 2, -2, 4, -4, 8, -8, 16, -16])
# %%
# Plot RT distributions: DATA vs THEORY per condition (10x3 grid)
rt_out = "rt_dists_data_vs_theory_grid.png"
plot_rt_dists_grid(
    root_dir, batch_name, animal_id,
    [20], ilds, n_theta,
    out_name=rt_out,
    only_stable=only_stable,
    n_mu_samples=mu_samples,
    bin_width=bin_width,
    rt_max=rt_max,
    c_A_trunc_time=ca_trunc_time,
)
# %%
# Plot ELBO vs theta per condition (10x3 grid)
plot_elbo_grid(root_dir, abls, ilds, n_theta, out_name=out_name, only_stable=only_stable)

# %%
# Plot ELBO - ELBO_max per condition (10x3 grid)
diff_out = (os.path.splitext(out_name)[0] + "_diff.png"
            if out_name.lower().endswith(('.png', '.jpg', '.jpeg', '.svg'))
            else "elbo_diff_grid_10x3.png")
plot_elbo_diff_grid(root_dir, abls, ilds, n_theta, out_name=diff_out, only_stable=only_stable)


# %%
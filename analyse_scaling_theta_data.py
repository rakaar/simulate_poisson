# %%
import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_results(pkl_path: str):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def get_abls(result_for_k: dict):
    """Extract sorted list of ABL values from results for a single k.
    Prefers the order [20, 40, 60] if present.
    """
    keys = list(result_for_k['discrete_ddm'].keys())
    abls = sorted({abl for (abl, _ild) in keys})
    preferred = [20, 40, 60]
    ordered = [a for a in preferred if a in abls] + [a for a in abls if a not in preferred]
    return ordered


def aggregate_times_for_abl(result_for_k: dict, abl: int):
    """Concatenate hitting times across all ILDs for the given ABL.
    Returns (discrete_times, continuous_times).
    """
    disc = result_for_k['discrete_ddm']
    cont = result_for_k['continuous_ddm']

    disc_arrays = []
    cont_arrays = []

    for (a, ild), rec in disc.items():
        if a == abl:
            if rec['pos_times'].size > 0:
                disc_arrays.append(rec['pos_times'])
            if rec['neg_times'].size > 0:
                disc_arrays.append(rec['neg_times'])

    for (a, ild), rec in cont.items():
        if a == abl:
            if rec['pos_times'].size > 0:
                cont_arrays.append(rec['pos_times'])
            if rec['neg_times'].size > 0:
                cont_arrays.append(rec['neg_times'])

    disc_all = np.concatenate(disc_arrays) if disc_arrays else np.array([])
    cont_all = np.concatenate(cont_arrays) if cont_arrays else np.array([])
    return disc_all, cont_all


def find_default_pkl(param_hint) -> str:
    candidates = []
    if param_hint:
        candidates.append(f'scale_theta_ddm_simulation_results_{param_hint}.pkl')
    candidates += [
        'scale_theta_ddm_simulation_results_norm.pkl',
        'scale_theta_ddm_simulation_results_vanilla.pkl',
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        'Could not locate results pkl. Provide --pkl or run scale_theta_and_check_when_poisson_breaks.py first.'
    )


def main():
    parser = argparse.ArgumentParser(description='Analyse RTDs across theta scaling factors and ABLs.')
    parser.add_argument('--pkl', type=str, default=None, help='Path to results pkl file.')
    parser.add_argument('--param', type=str, default=None, help='param_type hint for default pkl name (e.g., norm or vanilla).')
    parser.add_argument('--bins', type=int, default=100, help='Number of histogram bins per subplot.')
    parser.add_argument('--save', type=str, default=None, help='Path to save the output figure.')
    # Use parse_known_args to ignore ipykernel/VSCode-injected args like '-f=…json'
    args, _ = parser.parse_known_args()

    if args.pkl is None:
        pkl_path = find_default_pkl(args.param)
    else:
        pkl_path = args.pkl
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f'Provided pkl not found: {pkl_path}')

    results = load_results(pkl_path)

    # k values (theta scaling factors)
    try:
        ks = sorted(results.keys())
    except Exception:
        # In case keys are not directly comparable, coerce to float where possible
        ks = sorted(results.keys(), key=lambda x: float(x))

    # Determine ABLs from the first k
    abls = get_abls(results[ks[0]])

    n_rows = len(ks)
    n_cols = len(abls)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.6 * n_rows), sharey='row', constrained_layout=True)

    # Normalize axes to 2D array shape
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for i, k in enumerate(ks):
        result_k = results[k]

        # Precompute aggregated times and max time across the row for consistent x-limits/bins
        agg = {}
        row_max = 0.0
        for abl in abls:
            disc_all, cont_all = aggregate_times_for_abl(result_k, abl)
            agg[abl] = (disc_all, cont_all)
            if disc_all.size > 0:
                row_max = max(row_max, float(np.max(disc_all)))
            if cont_all.size > 0:
                row_max = max(row_max, float(np.max(cont_all)))
        if row_max <= 0:
            row_max = 1.0
        # bins = np.linspace(0, row_max, args.bins + 1)
        bins = np.arange(0,1, 0.005)

        for j, abl in enumerate(abls):
            ax = axes[i, j]
            disc_all, cont_all = agg[abl]

            if disc_all.size > 0:
                counts_disc, _ = np.histogram(disc_all, bins=bins, density=True)
                ax.step(bins[:-1], counts_disc, where='post', color='red', lw=2, label='Discrete')
            else:
                ax.text(0.5, 0.7, 'No discrete hits', transform=ax.transAxes, ha='center', va='center', fontsize=9, color='red')

            if cont_all.size > 0:
                counts_cont, _ = np.histogram(cont_all, bins=bins, density=True)
                ax.step(bins[:-1], counts_cont, where='post', color='black', lw=2, label='Continuous')
            else:
                ax.text(0.5, 0.3, 'No continuous hits', transform=ax.transAxes, ha='center', va='center', fontsize=9, color='black')

            ax.set_xlim(0, row_max)
            ax.grid(True, linestyle='--', alpha=0.3)

            if i == 0:
                ax.set_title(f'ABL = {abl} dB')
            if j == 0:
                ax.set_ylabel(f'k = {k}\nDensity')
            if i == n_rows - 1:
                ax.set_xlabel('Reaction Time (s)')

            if i == 0 and j == 0:
                ax.legend()
            
            ax.set_xlim(0,0.6)
    
    # Best-effort display for interactive sessions
    try:
        plt.show()
    except Exception:
        pass    

    base = os.path.splitext(os.path.basename(pkl_path))[0]
    out_path = args.save or f'RTDs_by_ABL_by_k_{base}.png'
    fig.suptitle('RT distributions aggregated across ILDs by ABL and theta scaling factor (k)', fontsize=16)
    fig.savefig(out_path, dpi=150)
    print(f'Saved figure to {out_path}')

    # Best-effort display for interactive sessions
    try:
        plt.show()
    except Exception:
        pass


if __name__ == '__main__':
    main()

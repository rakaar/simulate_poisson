import os
import sys
import pickle
import numpy as np
import pandas as pd

from vbmc_skellam_utils import fpt_density_skellam, fpt_cdf_skellam, fpt_choice_skellam


def load_t_E_aff(batch_name: str, animal_id: int) -> float:
    """
    Load t_E_aff from the same pickle path used in vbmc_fit_skellam.py.
    If unavailable, fall back to 0.0 with a warning.
    """
    pkl_file = f'/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/results_{batch_name}_animal_{animal_id}.pkl'
    try:
        with open(pkl_file, 'rb') as f:
            results = pickle.load(f)
        t_E_aff = results['vbmc_vanilla_tied_results']['t_E_aff_samples'].mean()
        print(f"Loaded t_E_aff from {pkl_file}: {t_E_aff:.6f}")
        return float(t_E_aff)
    except Exception as e:
        print(f"WARNING: Could not load t_E_aff from {pkl_file}: {e}\nUsing t_E_aff = 0.0 for test.")
        return 0.0


def prepare_dataframe(csv_path: str, animal_id: int, trunc_time: float) -> pd.DataFrame:
    """
    Apply the same filtering and column creation logic as in vbmc_fit_skellam.py.
    Returns the filtered dataframe.
    """
    df = pd.read_csv(csv_path)
    df_1 = df[
        (df['animal'] == animal_id) &
        (df['session_type'] == 1) &
        (df['LED_trial'].isin([0, np.nan])) &
        (df['repeat_trial'].isin([0, 2])) &
        (df['training_level'] == 16) &
        (df['success'].isin([1, -1]))
    ]
    df_1 = df_1.copy()
    df_1.loc[:, 'abs_ILD'] = df_1['ILD'].abs()
    df_1.loc[:, 'RTwrtStim'] = df_1['timed_fix'] - df_1['intended_fix']
    df_1 = df_1[(df_1['RTwrtStim'] > trunc_time) & (df_1['RTwrtStim'] < 1.0)]
    return df_1


def pick_subset(df_1: pd.DataFrame, n_rows: int, rng: np.random.Generator) -> tuple[pd.DataFrame, float, float]:
    """
    Pick an (ABL, ILD) group with at least n_rows rows. If none, fall back to largest group.
    Returns (df_subset, ABL, ILD).
    """
    groups = df_1.groupby(['ABL', 'ILD'])
    viable = [(k, len(g)) for k, g in groups if len(g) >= n_rows]
    if len(viable) == 0:
        # Fall back to the largest group
        k_max, _ = max(((k, len(g)) for k, g in groups), key=lambda x: x[1])
        g = groups.get_group(k_max)
        df_sub = g.sample(n=min(n_rows, len(g)), random_state=rng.integers(0, 2**32 - 1))
        return df_sub.reset_index(drop=True), float(k_max[0]), float(k_max[1])
    else:
        # Choose a random viable group
        idx = int(rng.integers(0, len(viable)))
        (ABL, ILD), _ = viable[idx]
        g = groups.get_group((ABL, ILD))
        df_sub = g.sample(n=n_rows, random_state=rng.integers(0, 2**32 - 1))
        return df_sub.reset_index(drop=True), float(ABL), float(ILD)


def compute_loglike_loop(df_sub: pd.DataFrame, R: float, L: float, theta_int: int, trunc_time: float, t_E_aff: float) -> float:
    cdf_trunc = 1 - fpt_cdf_skellam(trunc_time - t_E_aff, R, L, theta_int)
    loglike = 0.0
    for _, row in df_sub.iterrows():
        rt = row['RTwrtStim']
        choice = int(2 * row['response_poke'] - 5)
        trunc_factor = cdf_trunc * fpt_choice_skellam(R, L, theta_int, choice)
        if rt < trunc_time:
            p = 1e-50
        else:
            p = float(fpt_density_skellam(rt, R, L, theta_int)) * float(fpt_choice_skellam(R, L, theta_int, choice))
            if trunc_factor == 0:
                trunc_factor = 1e-80
            p /= trunc_factor
        if p <= 0:
            p = 1e-50
        lp = np.log(p)
        if np.isnan(lp):
            raise ValueError(
                f'log(p) is nan (loop) for params: (R={R}, L={L}, theta={theta_int}), '
                f'rt={rt}, choice={choice}, trunc_factor={trunc_factor}, cdf_trunc={cdf_trunc}, '
                f'rho(t)={fpt_density_skellam(rt, R, L, theta_int)}, '
                f'rho(+) = {fpt_choice_skellam(R, L, theta_int, choice)}'
            )
        loglike += lp
    return float(loglike)


def compute_loglike_vec(df_sub: pd.DataFrame, R: float, L: float, theta_int: int, trunc_time: float, t_E_aff: float) -> float:
    cdf_trunc = 1 - fpt_cdf_skellam(trunc_time - t_E_aff, R, L, theta_int)
    p_pos = fpt_choice_skellam(R, L, theta_int, 1)
    rts = df_sub['RTwrtStim'].to_numpy()
    choices = (2 * df_sub['response_poke'].to_numpy() - 5).astype(int)
    p_choice = np.where(choices == 1, p_pos, 1 - p_pos)
    rho_t = fpt_density_skellam(rts, R, L, theta_int)
    trunc_factor = cdf_trunc * p_choice
    trunc_factor = np.where(trunc_factor == 0, 1e-80, trunc_factor)
    p = np.where(rts < trunc_time, 1e-50, (rho_t * p_choice) / trunc_factor)
    p = np.where(p <= 0, 1e-50, p)
    logp = np.log(p)
    if np.any(np.isnan(logp)):
        bad_idx = int(np.where(np.isnan(logp))[0][0])
        bad_rt = float(rts[bad_idx])
        bad_choice = int(choices[bad_idx])
        bad_trunc = float(trunc_factor[bad_idx])
        raise ValueError(
            f'log(p) is nan (vec) for params: (R={R}, L={L}, theta={theta_int}), rt={bad_rt}, '
            f'choice={bad_choice}, trunc_factor={bad_trunc}, cdf_trunc={cdf_trunc}, '
            f'rho(t)={fpt_density_skellam(bad_rt, R, L, theta_int)}, '
            f'rho(+) = {fpt_choice_skellam(R, L, theta_int, bad_choice)}'
        )
    return float(np.sum(logp))


def main():
    rng = np.random.default_rng(0)

    # Match main script settings
    trunc_time = 0.085
    batch_name = 'LED8'
    animal_id = 112
    csv_path = f'out{batch_name}.csv'

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}")
        sys.exit(1)

    t_E_aff = load_t_E_aff(batch_name, animal_id)
    df_1 = prepare_dataframe(csv_path, animal_id, trunc_time)

    if len(df_1) == 0:
        print("ERROR: No rows after filtering; cannot test.")
        sys.exit(1)

    df_sub, ABL, ILD = pick_subset(df_1, n_rows=20, rng=rng)
    print('=' * 100)
    print(f"Using subset: ABL={ABL}, ILD={ILD}, n={len(df_sub)}")
    print('=' * 100)

    # Plausible bounds
    logR_plausible_bounds = [1.0, 3.8]
    logL_plausible_bounds = [1.0, 3.8]
    theta_plausible_bounds = [1.0, 45.0]

    n_param_sets = 5
    abs_diffs = []

    for i in range(n_param_sets):
        logR = rng.uniform(*logR_plausible_bounds)
        logL = rng.uniform(*logL_plausible_bounds)
        theta = rng.uniform(*theta_plausible_bounds)

        R = 10 ** logR
        L = 10 ** logL

        # Apply the same jitter and rounding to theta ONCE so both versions use identical theta
        jitter = rng.uniform(-0.5, 0.5)
        theta_j = int(round(theta + jitter))
        if theta_j < 1:
            theta_j = 1

        try:
            ll_loop = compute_loglike_loop(df_sub, R, L, theta_j, trunc_time, t_E_aff)
            ll_vec = compute_loglike_vec(df_sub, R, L, theta_j, trunc_time, t_E_aff)
        except Exception as e:
            print(f"Param set {i+1}/{n_param_sets} raised an exception: {e}")
            raise

        diff = ll_vec - ll_loop
        abs_diffs.append(abs(diff))
        print(f"Param set {i+1}: logR={logR:.4f}, logL={logL:.4f}, theta_base={theta:.2f}, theta_j={theta_j:2d} | "
              f"loop={ll_loop:.12f}, vec={ll_vec:.12f}, diff={diff:.12e}")

    print('-' * 100)
    print(f"Max |diff| across {n_param_sets} param sets: {max(abs_diffs):.12e}")
    tol = 1e-10
    if max(abs_diffs) < tol:
        print(f"SUCCESS: Vectorized and loop versions match within tolerance {tol}")
        sys.exit(0)
    else:
        print(f"WARNING: Differences exceed tolerance {tol}")
        sys.exit(2)


if __name__ == '__main__':
    main()

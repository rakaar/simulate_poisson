# make likelihood calculation vectorized

# %%
import numpy as np
import pandas as pd
from vbmc_skellam_utils import cum_pro_and_reactive_trunc_fn, up_or_down_hit_truncated_proactive_V2_fn
from time import perf_counter

V_A=1.1
theta_A=1
t_A_aff=0.1
dt = 1e-4
dB = 1e-2

t_E_aff = 0.073
del_go = 0.13

mu1=160
mu2=140
theta_E=11
batch_name = 'LED8'
animal_id = '112'
file_name = f'/home/rlab/raghavendra/ddm_data/fit_animal_by_animal/batch_csvs/batch_{batch_name}_valid_and_aborts.csv'
df = pd.read_csv(file_name)
df_animal = df[df['animal'] == int(animal_id)]
df_animal_success = df_animal[df_animal['success'].isin([1, -1])]
df_animal_success_rt_filter = df_animal_success[(df_animal_success['RTwrtStim'] <= 1) & (df_animal_success['RTwrtStim'] > 0)]
df_animal_cond_filter = df_animal_success_rt_filter[
                (df_animal_success_rt_filter['ABL'].isin([40])) & 
                (df_animal_success_rt_filter['ILD'].isin([16]))
            ]
def compute_loglike_trial(row, mu1, mu2, theta_E):
        # data
        c_A_trunc_time = 0.3
        if 'timed_fix' in row:
            rt = row['timed_fix']
        else:
            rt = row['TotalFixTime']
        
        t_stim = row['intended_fix']

        response_poke = row['response_poke']
        
        # trunc_factor_p_joint = cum_pro_and_reactive_trunc_fn(
        #                         t_stim + 1, c_A_trunc_time,
        #                         V_A, theta_A, t_A_aff,
        #                         t_stim, t_E_aff, gamma, omega, w, K_max) - \
        #                         cum_pro_and_reactive_trunc_fn(
        #                         t_stim, c_A_trunc_time,
        #                         V_A, theta_A, t_A_aff,
        #                         t_stim, t_E_aff, gamma, omega, w, K_max)

        trunc_factor_p_joint = cum_pro_and_reactive_trunc_fn(
                            t_stim + 1, c_A_trunc_time,
                            V_A, theta_A, t_A_aff,
                            t_stim, t_E_aff, mu1, mu2, theta_E) - \
                        cum_pro_and_reactive_trunc_fn(
                            t_stim, c_A_trunc_time,
                            V_A, theta_A, t_A_aff,
                            t_stim, t_E_aff, mu1, mu2, theta_E)

        choice = 2*response_poke - 5
        
        P_joint_rt_choice = up_or_down_hit_truncated_proactive_V2_fn(np.array([rt]), V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, choice)[0]
        

        
        P_joint_rt_choice_trunc = max(P_joint_rt_choice / (trunc_factor_p_joint + 1e-10), 1e-10)
        
        wt_log_like = np.log(P_joint_rt_choice_trunc)

        return wt_log_like




def sum_loglike_scalar(df):
    """Baseline: loop over rows (scalar calls)."""
    return sum(
        compute_loglike_trial(row, mu1, mu2, theta_E)
        for _, row in df.iterrows()
    )


def sum_loglike_grouped_vectorized(df):
    """Vectorize across trials by batching per unique (t_stim, choice).
    Uses array rts for each group so heavy functions are called once per group.
    """
    c_A_trunc_time = 0.3
    rt_col = 'timed_fix' if 'timed_fix' in df.columns else 'TotalFixTime'
    total = 0.0
    # Group by intended_fix and response_poke so that t_stim and choice are scalars within a group
    for (t_stim_val, response_poke_val), g in df.groupby(['intended_fix', 'response_poke']):
        rts = g[rt_col].to_numpy()
        choice = 2*response_poke_val - 5

        # Denominator truncation factor depends only on t_stim
        trunc = (
            cum_pro_and_reactive_trunc_fn(
                t_stim_val + 1, c_A_trunc_time,
                V_A, theta_A, t_A_aff,
                t_stim_val, t_E_aff, mu1, mu2, theta_E
            )
            - cum_pro_and_reactive_trunc_fn(
                t_stim_val, c_A_trunc_time,
                V_A, theta_A, t_A_aff,
                t_stim_val, t_E_aff, mu1, mu2, theta_E
            )
        )

        # Numerator vectorized over rts for this group
        P_joint = up_or_down_hit_truncated_proactive_V2_fn(
            rts, V_A, theta_A, t_A_aff, t_stim_val, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, choice
        )

        P_joint_trunc = np.maximum(P_joint / (trunc + 1e-10), 1e-10)
        total += np.log(P_joint_trunc).sum()

    return total


if __name__ == "__main__":
    # Warmup (optional)
    _ = sum_loglike_scalar(df_animal_cond_filter.head(5))
    _ = sum_loglike_grouped_vectorized(df_animal_cond_filter.head(5))

    t0 = perf_counter()
    ll_scalar = sum_loglike_scalar(df_animal_cond_filter)
    t1 = perf_counter()

    t2 = perf_counter()
    ll_grouped = sum_loglike_grouped_vectorized(df_animal_cond_filter)
    t3 = perf_counter()

    print(f"scalar_sum_loglike = {ll_scalar:.6f}, time = {t1 - t0:.4f}s")
    print(f"grouped_vectorized_sum_loglike = {ll_grouped:.6f}, time = {t3 - t2:.4f}s")
    print(f"diff = {abs(ll_scalar - ll_grouped):.3e}")
    if (t3 - t2) > 0:
        print(f"speedup x{(t1 - t0) / (t3 - t2):.2f}")
    else:
        print("vectorized time reported as 0, check timing.")
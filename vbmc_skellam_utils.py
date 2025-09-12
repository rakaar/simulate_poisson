import numpy as np
from math import erf
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.special import erf, erfcx

def fpt_density_skellam(t, mu1, mu2, theta):
    """
    Calculates the exact First Passage Time (FPT) density for a Skellam process.
    The process is dx = dN1 - dN2, with absorbing boundaries at +/- theta.

    Parameters
    ----------
    t : array_like
        Time points at which to evaluate the density.
    mu1 : float
        Rate of the upward Poisson process (N1).
    mu2 : float
        Rate of the downward Poisson process (N2).
    theta : int
        Symmetric absorbing boundary location (positive integer).

    Returns
    -------
    ft : np.ndarray
        PDF values at times `t`. For negative times, PDF is set to 0.
    """
    t = np.asarray(t, dtype=float)
    scalar_input = np.isscalar(t)
    t = np.atleast_1d(t).astype(float)

    if not (isinstance(theta, (int, np.integer)) and theta > 0):
        raise ValueError("Boundary theta must be a positive integer.")

    ft = np.zeros_like(t, dtype=float)
    
    # Create mask for non-negative times
    non_negative_mask = t >= 0
    
    # Only compute for non-negative times
    if np.any(non_negative_mask):
        t_positive = t[non_negative_mask]
        
        M = 2 * theta - 1  # size of interior state space
        j = np.arange(1, M + 1, dtype=float)  # 1..M

        # sum over odd k = 1,3,5,...,M
        for k in range(1, M + 1, 2):
            # eigenvalue
            lambda_k = -(mu1 + mu2) + 2.0 * np.sqrt(mu1 * mu2) * np.cos(k * np.pi / (2.0 * theta))

            # right/left eigenvectors (unnormalized)
            v_k = (mu1 / mu2) ** (j / 2.0) * np.sin(k * np.pi * j / (2.0 * theta))
            u_k = (mu2 / mu1) ** (j / 2.0) * np.sin(k * np.pi * j / (2.0 * theta))

            # projection coefficient c_k = (u_k' * 1) / (u_k' * v_k)
            numerator = np.sum(u_k)
            denominator = np.sum(u_k * v_k)

            if np.abs(denominator) < 1e-12:
                c_k = 0.0
            else:
                c_k = numerator / denominator

            # start state x=0 corresponds to the theta-th component (1-based) -> index theta-1 (0-based)
            v_k_theta = v_k[theta - 1]

            # amplitude
            A_k = -lambda_k * c_k * v_k_theta

            # mode contribution for positive times only
            ft[non_negative_mask] += A_k * np.exp(lambda_k * t_positive)

    # clamp tiny negative values due to floating point
    ft[ft < 0] = 0.0
    if scalar_input:
        return float(ft.item())
    return ft

def fpt_cdf_skellam(t, mu1, mu2, theta):
    """
    Calculates the exact FPT CDF for a Skellam process x(t) = N1(t) - N2(t)
    with absorbing boundaries at +/- theta, starting at x=0.

    Parameters
    ----------
    t : float or array-like
        Time(s) at which to evaluate the CDF.
    mu1 : float
        Rate of the upward Poisson process N1 (must be > 0).
    mu2 : float
        Rate of the downward Poisson process N2 (must be > 0).
    theta : int
        Symmetric absorbing boundary (positive integer).

    Returns
    -------
    cdf : float or ndarray
        P(T <= t) evaluated at the given t. Scalar if t is scalar; array otherwise.
        For negative times, CDF is set to 0.
    """
    # Convert and validate inputs
    t_arr = np.asarray(t, dtype=float)
    scalar_input = np.isscalar(t)
    t_arr = np.atleast_1d(t_arr).astype(float)
    if theta <= 0 or int(theta) != theta:
        raise ValueError("Boundary theta must be a positive integer.")
    if mu1 <= 0 or mu2 <= 0:
        raise ValueError("mu1 and mu2 must be positive.")

    theta = int(theta)
    # Initialize survival probability S_0(t)
    s0 = np.zeros_like(t_arr, dtype=float)
    
    # Create mask for non-negative times
    non_negative_mask = t_arr >= 0
    
    # Only compute for non-negative times
    if np.any(non_negative_mask):
        t_positive = t_arr[non_negative_mask]
        
        # Size of state space matrix and odd-mode loop
        M = 2 * theta - 1
        j = np.arange(1, M + 1, dtype=float)  # 1..M (MATLAB-style)

        # Precompute constants
        sqrt_mu1mu2 = np.sqrt(mu1 * mu2)
        ratio_12 = (mu1 / mu2) ** (j / 2.0)
        ratio_21 = (mu2 / mu1) ** (j / 2.0)

        # Sum over odd k = 1,3,5,...,M
        for k in range(1, M + 1, 2):
            # Step 1: eigenvalue (decay rate)
            lambda_k = -(mu1 + mu2) + 2.0 * sqrt_mu1mu2 * np.cos(k * np.pi / (2.0 * theta))

            # Step 2: right/left eigenvectors v_k, u_k
            sin_term = np.sin(k * np.pi * j / (2.0 * theta))
            v_k = ratio_12 * sin_term
            u_k = ratio_21 * sin_term

            # Step 3: projection coefficient c_k
            numerator = np.sum(u_k)
            denominator = np.sum(u_k * v_k)
            c_k = 0.0 if abs(denominator) < 1e-12 else (numerator / denominator)

            # Step 4: component at start state (x=0 maps to index theta in 1-based; theta-1 in 0-based)
            v_k_theta = v_k[theta - 1]

            # Step 5: amplitude B_k
            B_k = c_k * v_k_theta

            # Step 6: add this mode's contribution to S_0(t) for positive times only
            s0[non_negative_mask] += B_k * np.exp(lambda_k * t_positive)

    # CDF = 1 - S_0(t), clamped to [0, 1]
    # For negative times, CDF remains 0 (as initialized)
    cdf_arr = 1.0 - s0
    cdf_arr = np.clip(cdf_arr, 0.0, 1.0)
    cdf_arr[~non_negative_mask] = 0.0 

    # Return scalar if input was scalar
    if scalar_input:
        return float(cdf_arr.item())
    return cdf_arr

def fpt_choice_skellam(mu1, mu2, theta, choice):
    r = mu2 / mu1
    P_pos = (1 - r**theta) / (1 - r**(2*theta))
    if choice == 1:
        return P_pos
    elif choice == -1:
        return 1 - P_pos
    else:
        raise ValueError("choice must be +1 or -1")


def cum_pro_and_reactive_trunc_fn(
        t, c_A_trunc_time,
        V_A, theta_A, t_A_aff,
        t_stim, t_E_aff, mu1, mu2, theta_E):
    c_A = cum_A_t_fn(t-t_A_aff, V_A, theta_A)
    if c_A_trunc_time is not None:
        if t < c_A_trunc_time:
            c_A = 0
        else:
            c_A -= cum_A_t_fn(c_A_trunc_time - t_A_aff, V_A, theta_A)
            c_A  /= (1 - cum_A_t_fn(c_A_trunc_time - t_A_aff, V_A, theta_A))

    # c_E = CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t - t_stim - t_E_aff, gamma, omega, 1, w, K_max) + \
    #     CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t - t_stim - t_E_aff, gamma, omega, -1, w, K_max)

    dt = t - t_stim - t_E_aff
    if dt <= 0:
        c_E = 0.0
    else:
        c_E = fpt_cdf_skellam(dt, mu1, mu2, theta_E)
    
    return c_A + c_E - c_A * c_E


def Phi(x):
    """
    Define the normal cumulative distribution function Φ(x) using erf
    """
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def cum_A_t_fn(t, V_A, theta_A):
    """
    Proactive cdf, input time scalar, delays should be already subtracted before calling func
    """
    if t <= 0:
        return 0

    term1 = Phi(V_A * ((t) - (theta_A/V_A)) / np.sqrt(t))
    term2 = np.exp(2 * V_A * theta_A) * Phi(-V_A * ((t) + (theta_A / V_A)) / np.sqrt(t))
    
    return term1 + term2

# def rho_A_t_fn(t, V_A, theta_A):
#     """
#     Proactive PDF, takes input as scalar, delays should be already subtracted before calling func
#     """
#     if t <= 0:
#         return 0
#     return (theta_A*1/np.sqrt(2*np.pi*(t)**3))*np.exp(-0.5 * (V_A**2) * (((t) - (theta_A/V_A))**2)/(t))

def rho_A_t_fn(t, V_A, theta_A):
    """
    Proactive PDF, takes input as scalar or array, delays should be already subtracted before calling func
    """
    t = np.asarray(t)
    result = np.zeros_like(t, dtype=float)
    
    # Create mask for positive t values
    mask = t > 0
    
    # Only compute for positive t values
    if np.any(mask):
        t_positive = t[mask]
        result[mask] = (theta_A*1/np.sqrt(2*np.pi*(t_positive)**3))*np.exp(-0.5 * (V_A**2) * (((t_positive) - (theta_A/V_A))**2)/(t_positive))
    
    # Return scalar if input was scalar
    if np.isscalar(t):
        return float(result.item())
    
    return result

def truncated_rho_A_t_fn(t, V_A, theta_A, trunc_time):
    """
    Truncated proactive PDF, takes input as scalar or array, delays should be already subtracted before calling func
    Returns 0 for t < trunc_time, otherwise returns rho_A_t_fn value scaled by survival probability
    """
    t = np.asarray(t)
    result = np.zeros_like(t, dtype=float)
    
    # Create mask for t values >= trunc_time
    mask = t >= trunc_time
    
    # Only compute for t values >= trunc_time
    if np.any(mask):
        # Scale by survival probability: 1 - CDF at truncation time
        survival_prob = 1.0 - cum_A_t_fn(trunc_time, V_A, theta_A) + 1e-30
        # Avoid division by zero
        if survival_prob > 0:
            result[mask] = rho_A_t_fn(t[mask], V_A, theta_A) / survival_prob
    
    # Return scalar if input was scalar
    if np.isscalar(t):
        return float(result.item())
    
    return result

# def up_or_down_hit_fn(t, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, bound):
#     """
#     PDF of all RTs array irrespective of choice
#     bound == choice = +1/-1
#     """
#     t2 = t - t_stim - t_E_aff + del_go
#     t1 = t - t_stim - t_E_aff

#     P_A = rho_A_t_fn(t - t_A_aff, V_A, theta_A)
#     # prob_EA_hits_either_bound = CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t - t_stim - t_E_aff + del_go,\
#     #                                                                      gamma, omega, 1, w, K_max) \
#     #                          + CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t - t_stim - t_E_aff + del_go,\
#     #                                                                      gamma, omega, -1, w, K_max)
#     prob_EA_hits_either_bound = fpt_cdf_skellam(t - t_stim - t_E_aff + del_go, mu1, mu2, theta_E)
    
#     prob_EA_survives = 1 - prob_EA_hits_either_bound
#     random_readout_if_EA_surives = 0.5 * prob_EA_survives
#     # P_E_plus_or_minus_cum = CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t2, gamma, omega, bound, w, K_max) \
#     #                 - CDF_E_minus_small_t_NORM_omega_gamma_with_w_fn(t1, gamma, omega, bound, w, K_max)
#     p_choice = fpt_choice_skellam(mu1, mu2, theta_E, bound)
#     # Safe CDF values with negative times treated as 0
#     c2 = 0.0 if t2 <= 0 else fpt_cdf_skellam(t2, mu1, mu2, theta_E)
#     c1 = 0.0 if t1 <= 0 else fpt_cdf_skellam(t1, mu1, mu2, theta_E)
#     P_E_plus_or_minus_cum = p_choice * (c2 - c1)
    
#     # P_E_plus_or_minus = rho_E_minus_small_t_NORM_omega_gamma_with_w_fn(t-t_E_aff-t_stim, gamma, omega, bound, w, K_max)
#     dt_pdf = t - t_E_aff - t_stim
#     P_E_plus_or_minus = (0.0 if dt_pdf <= 0 else float(fpt_density_skellam(dt_pdf, mu1, mu2, theta_E))) * p_choice

#     C_A = cum_A_t_fn(t - t_A_aff, V_A, theta_A)
#     return (P_A*(random_readout_if_EA_surives + P_E_plus_or_minus_cum) + P_E_plus_or_minus*(1-C_A))


def up_or_down_hit_fn(t, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, bound):
    """
    PDF of all RTs array irrespective of choice
    bound == choice = +1/-1
    """
    t2 = t - t_stim - t_E_aff + del_go
    t1 = t - t_stim - t_E_aff

    P_A = rho_A_t_fn(t - t_A_aff, V_A, theta_A)
    prob_EA_hits_either_bound = fpt_cdf_skellam(t - t_stim - t_E_aff + del_go, mu1, mu2, theta_E)
    
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    random_readout_if_EA_surives = 0.5 * prob_EA_survives
    p_choice = fpt_choice_skellam(mu1, mu2, theta_E, bound)
    
    # Safe CDF values with negative times treated as 0 - vectorized version
    c2 = np.zeros_like(t2, dtype=float)
    c1 = np.zeros_like(t1, dtype=float)
    mask2 = t2 > 0
    mask1 = t1 > 0
    if np.any(mask2):
        c2[mask2] = fpt_cdf_skellam(t2[mask2], mu1, mu2, theta_E)
    if np.any(mask1):
        c1[mask1] = fpt_cdf_skellam(t1[mask1], mu1, mu2, theta_E)
    P_E_plus_or_minus_cum = p_choice * (c2 - c1)
    
    # P_E_plus_or_minus = rho_E_minus_small_t_NORM_omega_gamma_with_w_fn(t-t_E_aff-t_stim, gamma, omega, bound, w, K_max)
    dt_pdf = t - t_E_aff - t_stim
    P_E_plus_or_minus = np.zeros_like(dt_pdf, dtype=float)
    mask_pdf = dt_pdf > 0
    if np.any(mask_pdf):
        P_E_plus_or_minus[mask_pdf] = fpt_density_skellam(dt_pdf[mask_pdf], mu1, mu2, theta_E)
    P_E_plus_or_minus *= p_choice

    C_A = cum_A_t_fn(t - t_A_aff, V_A, theta_A)
    return (P_A*(random_readout_if_EA_surives + P_E_plus_or_minus_cum) + P_E_plus_or_minus*(1-C_A))


def up_or_down_hit_P_A_C_A_wrt_stim_fn(t_pts, P_A, C_A, t_E_aff, del_go, mu1, mu2, theta_E, bound):
    """
    PDF of all RTs array irrespective of choice
    bound == choice = +1/-1
    """
    t2 = t_pts - t_E_aff + del_go
    t1 = t_pts - t_E_aff

    # CDF for either bound with safe handling for t <= 0
    tmp = t_pts - t_E_aff + del_go
    prob_EA_hits_either_bound = np.zeros_like(t_pts, dtype=float)
    mask_any = tmp > 0
    if np.any(mask_any):
        prob_EA_hits_either_bound[mask_any] = fpt_cdf_skellam(tmp[mask_any], mu1, mu2, theta_E)
    
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    random_readout_if_EA_survives = 0.5 * prob_EA_survives

    p_choice = fpt_choice_skellam(mu1, mu2, theta_E, bound)
    # Safe CDF values with negative times treated as 0
    c2 = np.zeros_like(t_pts, dtype=float)
    c1 = np.zeros_like(t_pts, dtype=float)
    mask2 = t2 > 0
    mask1 = t1 > 0
    if np.any(mask2):
        c2[mask2] = fpt_cdf_skellam(t2[mask2], mu1, mu2, theta_E)
    if np.any(mask1):
        c1[mask1] = fpt_cdf_skellam(t1[mask1], mu1, mu2, theta_E)
    P_E_plus_or_minus_cum = p_choice * (c2 - c1)
    
    dt_pdf = t_pts - t_E_aff
    P_E_plus_or_minus = np.zeros_like(t_pts, dtype=float)
    mask_pdf = dt_pdf > 0
    if np.any(mask_pdf):
        P_E_plus_or_minus[mask_pdf] = fpt_density_skellam(dt_pdf[mask_pdf], mu1, mu2, theta_E)
    P_E_plus_or_minus *= p_choice

    return (P_A*(random_readout_if_EA_survives + P_E_plus_or_minus_cum) + P_E_plus_or_minus*(1-C_A))



def up_or_down_hit_wrt_tstim(t_pts, V_A, theta_A, t_A_aff, t_stim_samples, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time):
    # t_pts : -1 to 2: 1ms step
    N_theory = int(1e3)
    t_stim_arr = np.asarray(t_stim_samples, dtype=float)
    if t_stim_arr.size == 0:
        # Nothing to average; return zeros over [0,1]
        mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
        t_pts_0_1 = t_pts[mask_0_1]
        return t_pts_0_1, np.zeros_like(t_pts_0_1), np.zeros_like(t_pts_0_1)

    t_stim_subset = np.random.choice(t_stim_arr, size=N_theory, replace=True)

    P_A_samples = np.zeros((N_theory, len(t_pts)))
    for idx, t_stim in enumerate(t_stim_subset):
        # rho_A_t_fn expects scalar t; build with list comprehension
        P_A_samples[idx, :] = [rho_A_t_fn(t - t_stim - t_A_aff, V_A, theta_A) for t in t_pts]

    P_A_mean = np.mean(P_A_samples, axis=0)
    C_A_mean = cumtrapz(P_A_mean, t_pts, initial=0)

    trunc_fac_samples = np.zeros(N_theory)
    for idx, t_stim in enumerate(t_stim_subset):
        trunc_fac_samples[idx] = (
            cum_pro_and_reactive_trunc_fn(
                t_stim + 1, c_A_trunc_time,
                V_A, theta_A, t_A_aff,
                t_stim, t_E_aff, mu1, mu2, theta_E
            )
            - cum_pro_and_reactive_trunc_fn(
                t_stim, c_A_trunc_time,
                V_A, theta_A, t_A_aff,
                t_stim, t_E_aff, mu1, mu2, theta_E
            ) + 1e-10
        )
    trunc_factor = float(np.mean(trunc_fac_samples))

    up_mean = up_or_down_hit_P_A_C_A_wrt_stim_fn(t_pts, P_A_mean, C_A_mean, t_E_aff, del_go, mu1, mu2, theta_E, 1)
    down_mean = up_or_down_hit_P_A_C_A_wrt_stim_fn(t_pts, P_A_mean, C_A_mean, t_E_aff, del_go, mu1, mu2, theta_E, -1)

    mask_0_1 = (t_pts >= 0) & (t_pts <= 1)
    t_pts_0_1 = t_pts[mask_0_1]
    up_mean_0_1 = up_mean[mask_0_1]
    down_mean_0_1 = down_mean[mask_0_1]

    up_theory_mean_norm = up_mean_0_1 / trunc_factor
    down_theory_mean_norm = down_mean_0_1 / trunc_factor
    area_up = np.trapezoid(up_theory_mean_norm, t_pts_0_1)
    area_down = np.trapezoid(down_theory_mean_norm, t_pts_0_1)
    print(f"Area under up_theory_mean_norm: {area_up:.8f}")
    print(f"Area under down_theory_mean_norm: {area_down:.8f}")

    return t_pts_0_1, up_theory_mean_norm, down_theory_mean_norm



def up_or_down_hit_wrt_tstim_V2(t_pts_wrt_stim, V_A, theta_A, t_A_aff, t_stim_samples, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time):
    # t_pts : -1 to 2: 1ms step
    N_theory = int(5e3)
    t_stim_arr = np.asarray(t_stim_samples, dtype=float)
    t_stim_subset = np.random.choice(t_stim_arr, size=N_theory, replace=True)
    up_samples = np.zeros((N_theory, len(t_pts_wrt_stim)))
    down_samples = np.zeros((N_theory, len(t_pts_wrt_stim)))

    for idx, t_stim in enumerate(t_stim_subset):
        trunc_factor = cum_pro_and_reactive_trunc_fn(
                t_stim + 1, c_A_trunc_time,
                V_A, theta_A, t_A_aff,
                t_stim, t_E_aff, mu1, mu2, theta_E
            ) \
            - cum_pro_and_reactive_trunc_fn(
                t_stim, c_A_trunc_time,
                V_A, theta_A, t_A_aff,
                t_stim, t_E_aff, mu1, mu2, theta_E
            ) + 1e-10

        up1 = up_or_down_hit_fn(t_pts_wrt_stim + t_stim, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, 1)
        down1 = up_or_down_hit_fn(t_pts_wrt_stim + t_stim, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, -1)

        up_samples[idx, :] = up1 / trunc_factor
        down_samples[idx, :] = down1 / trunc_factor

    
    return np.mean(up_samples, axis=0), np.mean(down_samples, axis=0)


def Phi(x):
    """
    Define the normal cumulative distribution function Φ(x) using erf
    """
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def cum_A_t_fn(t, V_A, theta_A):
    """
    Proactive cdf, input time scalar or array, delays should be already subtracted before calling func
    """
    t = np.asarray(t)
    result = np.zeros_like(t, dtype=float)
    
    # Create mask for positive t values
    mask = t > 0
    
    # Only compute for positive t values
    if np.any(mask):
        t_positive = t[mask]
        term1 = Phi(V_A * ((t_positive) - (theta_A/V_A)) / np.sqrt(t_positive))
        term2 = np.exp(2 * V_A * theta_A) * Phi(-V_A * ((t_positive) + (theta_A / V_A)) / np.sqrt(t_positive))
        result[mask] = term1 + term2
    
    # Return scalar if input was scalar
    if np.isscalar(t):
        return float(result.item())
    
    return result

def truncated_cum_A_t_fn(t, V_A, theta_A, trunc_time):
    """
    Truncated proactive CDF, takes input as scalar or array, delays should be already subtracted before calling func
    Returns 0 for t < trunc_time, otherwise returns scaled cum_A_t_fn value
    """
    t = np.asarray(t)
    result = np.zeros_like(t, dtype=float)
    
    # Create mask for t values >= trunc_time
    mask = t >= trunc_time
    
    # Only compute for t values >= trunc_time
    if np.any(mask):
        # Get the CDF value at truncation time
        cdf_at_trunc = cum_A_t_fn(trunc_time, V_A, theta_A)
        # Scale by survival probability: (CDF(t) - CDF(trunc_time)) / (1 - CDF(trunc_time))
        survival_prob = 1.0 - cdf_at_trunc
        # Avoid division by zero
        if survival_prob > 0:
            result[mask] = (cum_A_t_fn(t[mask], V_A, theta_A) - cdf_at_trunc) / survival_prob

    
    # Return scalar if input was scalar
    if np.isscalar(t):
        return float(result.item())
    
    return result



def up_or_down_hit_wrt_tstim_V3(t_pts_wrt_stim, V_A, theta_A, t_A_aff, t_stim_samples, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time):
    N_theory = int(5e3)
    t_stim_arr = np.asarray(t_stim_samples, dtype=float)
    t_stim_subset = np.random.choice(t_stim_arr, size=N_theory, replace=True)
    up_samples = np.zeros((N_theory, len(t_pts_wrt_stim)))
    down_samples = np.zeros((N_theory, len(t_pts_wrt_stim)))

    for idx, t_stim in enumerate(t_stim_subset):
        up1 = up_or_down_hit_truncated_proactive_fn(t_pts_wrt_stim + t_stim, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, 1)
        down1 = up_or_down_hit_truncated_proactive_fn(t_pts_wrt_stim + t_stim, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, -1)

        area_up = np.trapezoid(up1, t_pts_wrt_stim)
        area_down = np.trapezoid(down1, t_pts_wrt_stim)

        total_area = area_up + area_down
        # print(f'total area = {total_area}')

        up_samples[idx, :] = up1 / total_area
        down_samples[idx, :] = down1 / total_area

    
    return np.mean(up_samples, axis=0), np.mean(down_samples, axis=0)

def up_or_down_hit_truncated_proactive_fn(t, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, proactive_trunc_time, bound):
    t2 = t - t_stim - t_E_aff + del_go
    t1 = t - t_stim - t_E_aff

    P_A = truncated_rho_A_t_fn(t - t_A_aff, V_A, theta_A, proactive_trunc_time)
    prob_EA_hits_either_bound = fpt_cdf_skellam(t - t_stim - t_E_aff + del_go, mu1, mu2, theta_E)
    
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    random_readout_if_EA_surives = 0.5 * prob_EA_survives
    p_choice = fpt_choice_skellam(mu1, mu2, theta_E, bound)
    
    # Safe CDF values with negative times treated as 0 - vectorized version
    c2 = np.zeros_like(t2, dtype=float)
    c1 = np.zeros_like(t1, dtype=float)
    mask2 = t2 > 0
    mask1 = t1 > 0
    if np.any(mask2):
        c2[mask2] = fpt_cdf_skellam(t2[mask2], mu1, mu2, theta_E)
    if np.any(mask1):
        c1[mask1] = fpt_cdf_skellam(t1[mask1], mu1, mu2, theta_E)
    P_E_plus_or_minus_cum = p_choice * (c2 - c1)
    
    # P_E_plus_or_minus = rho_E_minus_small_t_NORM_omega_gamma_with_w_fn(t-t_E_aff-t_stim, gamma, omega, bound, w, K_max)
    dt_pdf = t - t_E_aff - t_stim
    P_E_plus_or_minus = np.zeros_like(dt_pdf, dtype=float)
    mask_pdf = dt_pdf > 0
    if np.any(mask_pdf):
        P_E_plus_or_minus[mask_pdf] = fpt_density_skellam(dt_pdf[mask_pdf], mu1, mu2, theta_E)
    P_E_plus_or_minus *= p_choice

    # C_A = cum_A_t_fn(t - t_A_aff, V_A, theta_A)
    C_A = truncated_cum_A_t_fn(t - t_A_aff, V_A, theta_A, proactive_trunc_time)
    return (P_A*(random_readout_if_EA_surives + P_E_plus_or_minus_cum) + P_E_plus_or_minus*(1-C_A))


def up_or_down_hit_truncated_proactive_V2_fn(t, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, proactive_trunc_time, bound):
    t2 = t - t_stim - t_E_aff + del_go
    t1 = t - t_stim - t_E_aff
    if proactive_trunc_time > t_stim:
        proactive_trunc_time = 0

    P_A = truncated_rho_A_t_fn(t - t_A_aff, V_A, theta_A, proactive_trunc_time)
    prob_EA_hits_either_bound = fpt_cdf_skellam(t - t_stim - t_E_aff + del_go, mu1, mu2, theta_E)
    
    prob_EA_survives = 1 - prob_EA_hits_either_bound
    random_readout_if_EA_surives = 0.5 * prob_EA_survives
    p_choice = fpt_choice_skellam(mu1, mu2, theta_E, bound)
    
    # Safe CDF values with negative times treated as 0 - vectorized version
    c2 = np.zeros_like(t2, dtype=float)
    c1 = np.zeros_like(t1, dtype=float)
    mask2 = t2 > 0
    mask1 = t1 > 0
    if np.any(mask2):
        c2[mask2] = fpt_cdf_skellam(t2[mask2], mu1, mu2, theta_E)
    if np.any(mask1):
        c1[mask1] = fpt_cdf_skellam(t1[mask1], mu1, mu2, theta_E)
    P_E_plus_or_minus_cum = p_choice * (c2 - c1)
    
    # P_E_plus_or_minus = rho_E_minus_small_t_NORM_omega_gamma_with_w_fn(t-t_E_aff-t_stim, gamma, omega, bound, w, K_max)
    dt_pdf = t - t_E_aff - t_stim
    P_E_plus_or_minus = np.zeros_like(dt_pdf, dtype=float)
    mask_pdf = dt_pdf > 0
    if np.any(mask_pdf):
        P_E_plus_or_minus[mask_pdf] = fpt_density_skellam(dt_pdf[mask_pdf], mu1, mu2, theta_E)
    P_E_plus_or_minus *= p_choice

    # C_A = cum_A_t_fn(t - t_A_aff, V_A, theta_A)
    C_A = truncated_cum_A_t_fn(t - t_A_aff, V_A, theta_A, proactive_trunc_time)
    return (P_A*(random_readout_if_EA_surives + P_E_plus_or_minus_cum) + P_E_plus_or_minus*(1-C_A))

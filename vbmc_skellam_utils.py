import numpy as np

def fpt_density_skellam(t, mu1, mu2, theta):
    """
    Calculates the exact First Passage Time (FPT) density for a Skellam process.
    The process is dx = dN1 - dN2, with absorbing boundaries at +/- theta.

    Parameters
    ----------
    t : array_like
        Time points at which to evaluate the density (must be >= 0).
    mu1 : float
        Rate of the upward Poisson process (N1).
    mu2 : float
        Rate of the downward Poisson process (N2).
    theta : int
        Symmetric absorbing boundary location (positive integer).

    Returns
    -------
    ft : np.ndarray
        PDF values at times `t`.
    """
    t = np.asarray(t, dtype=float)

    if np.any(t < 0):
        raise ValueError("Time t cannot be negative.")
    if not (isinstance(theta, (int, np.integer)) and theta > 0):
        raise ValueError("Boundary theta must be a positive integer.")

    ft = np.zeros_like(t, dtype=float)
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

        # mode contribution
        ft += A_k * np.exp(lambda_k * t)

    # clamp tiny negative values due to floating point
    ft[ft < 0] = 0.0
    return ft

def fpt_cdf_skellam(t, mu1, mu2, theta):
    """
    Calculates the exact FPT CDF for a Skellam process x(t) = N1(t) - N2(t)
    with absorbing boundaries at +/- theta, starting at x=0.

    Parameters
    ----------
    t : float or array-like
        Time(s) at which to evaluate the CDF (must be >= 0).
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
    """
    # Convert and validate inputs
    t_arr = np.asarray(t, dtype=float)
    if np.any(t_arr < 0):
        raise ValueError("Time t cannot be negative.")
    if theta <= 0 or int(theta) != theta:
        raise ValueError("Boundary theta must be a positive integer.")
    if mu1 <= 0 or mu2 <= 0:
        raise ValueError("mu1 and mu2 must be positive.")

    theta = int(theta)
    # Initialize survival probability S_0(t)
    s0 = np.zeros_like(t_arr, dtype=float)

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

        # Step 6: add this mode's contribution to S_0(t)
        s0 += B_k * np.exp(lambda_k * t_arr)

    # CDF = 1 - S_0(t), clamped to [0, 1]
    cdf_arr = 1.0 - s0
    cdf_arr = np.clip(cdf_arr, 0.0, 1.0)

    # Return scalar if input was scalar
    if np.isscalar(t):
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



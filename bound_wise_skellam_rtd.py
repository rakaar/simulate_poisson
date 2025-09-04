# %%
import numpy as np
import matplotlib.pyplot as plt


# ---------- Theory: split first-passage-time densities ----------
def fpt_split_skellam(t, mu1, mu2, theta):
    """
    Unconditional split first-passage-time densities for Skellam walk with absorbing bounds ±theta.
    Returns f_plus(t), f_minus(t), where integrals are the hit probabilities of +θ and -θ.
    """
    t = np.asarray(t, dtype=float)
    if np.any(t < 0):
        raise ValueError("t must be ≥ 0")
    if not (isinstance(theta, (int, np.integer)) and theta > 0):
        raise ValueError("theta must be a positive integer")

    r = np.sqrt(mu1 / mu2)
    ks = np.arange(1, 2 * theta, 2)                    # odd k: 1,3,...,2θ-1
    alpha = ks * np.pi / (2 * theta)
    lam = -(mu1 + mu2) + 2 * np.sqrt(mu1 * mu2) * np.cos(alpha)
    sin_alpha = np.sin(alpha)
    sgn = (-1) ** ((ks - 1) // 2)                      # (-1)^m with k=2m+1

    # Sum over modes (vectorized over t)
    terms = (sgn * sin_alpha)[:, None] * np.exp(lam[:, None] * t[None, :])
    S = np.sum(terms, axis=0)

    f_plus  = (mu1 / theta) * (r ** (theta - 1))   * S
    f_minus = (mu2 / theta) * (r ** (-(theta - 1))) * S

    # Clamp tiny negatives from roundoff
    f_plus[f_plus < 0] = 0.0
    f_minus[f_minus < 0] = 0.0
    return f_plus, f_minus


def split_prob(mu1, mu2, theta):
    """Closed-form splitting probabilities (hit +θ vs −θ)."""
    if np.isclose(mu1, mu2):
        return 0.5, 0.5
    rho = mu2 / mu1
    p_plus = (1 - rho ** theta) / (1 - rho ** (2 * theta))
    return p_plus, 1 - p_plus


# ---------- Simulation ----------
def simulate_fpt_with_side(mu1, mu2, theta, rng=None):
    """
    Event-driven simulation: returns (time, +1 if upper bound hit else -1).
    """
    if rng is None:
        rng = np.random.default_rng()

    x = 0
    t = 0.0
    total_rate = mu1 + mu2
    p_up = mu1 / total_rate

    while abs(x) < theta:
        t += rng.exponential(1.0 / total_rate)
        x += 1 if (rng.random() < p_up) else -1

    return t, (1 if x == theta else -1)


# ---------- Demo / Plot ----------
def main():
    # Parameters (same as your MATLAB snippet)
    mu1_true   = 5.0
    mu2_true   = 20.0
    theta_true = 3
    num_trials = 500_000

    # Time axis for theory curves
    t_max = 2.0
    t = np.linspace(0.0, t_max, 600)

    # Theoretical (unconditional) split densities
    f_plus_u, f_minus_u = fpt_split_skellam(t, mu1_true, mu2_true, theta_true)
    p_plus, p_minus = split_prob(mu1_true, mu2_true, theta_true)

    # Convert to conditional (given which bound is hit) for overlay with per-group histograms
    f_plus_cond = f_plus_u / p_plus
    f_minus_cond = f_minus_u / p_minus

    # Simulate
    rng = np.random.default_rng()
    times_up = []
    times_dn = []
    for _ in range(num_trials):
        tt, side = simulate_fpt_with_side(mu1_true, mu2_true, theta_true, rng)
        if side == 1:
            times_up.append(tt)
        else:
            times_dn.append(tt)
    times_up = np.array(times_up, dtype=float)
    times_dn = np.array(times_dn, dtype=float)

    # Report split probabilities (sim vs theory)
    sim_p_plus = len(times_up) / num_trials
    print(f"Hit +θ probability: theory = {p_plus:.4f},   simulation = {sim_p_plus:.4f}")
    print(f"Hit -θ probability: theory = {p_minus:.4f},  simulation = {1 - sim_p_plus:.4f}")

    # Binning for histograms
    bins = np.arange(0.0, t_max + 0.01, 0.01)

    # Plot: Upper-bound hits (conditional density)
    plt.figure(figsize=(7, 4), facecolor="white")
    plt.hist(times_up, bins=bins, density=True, alpha=0.5, edgecolor="none", label="Sim (upper hits)")
    plt.plot(t, f_plus_cond, linewidth=1.8, label="Theory (upper | hit)")
    plt.xlabel("Time")
    plt.ylabel("Density (conditional)")
    plt.title(f"Upper bound hitting times (θ={theta_true}, μ₁={mu1_true}, μ₂={mu2_true})")
    plt.legend()
    plt.tight_layout()

    # Plot: Lower-bound hits (conditional density)
    plt.figure(figsize=(7, 4), facecolor="white")
    plt.hist(times_dn, bins=bins, density=True, alpha=0.5, edgecolor="none", label="Sim (lower hits)")
    plt.plot(t, f_minus_cond, linewidth=1.8, label="Theory (lower | hit)")
    plt.xlabel("Time")
    plt.ylabel("Density (conditional)")
    plt.title(f"Lower bound hitting times (θ={theta_true}, μ₁={mu1_true}, μ₂={mu2_true})")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()

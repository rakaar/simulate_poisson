# %%
import numpy as np
import matplotlib.pyplot as plt

# --- theory (your functions) ---
def fpt_split_skellam_start(t, mu1, mu2, theta, x0):
    t = np.asarray(t, float)
    if np.any(t < 0): raise ValueError("t must be ≥ 0")
    if not (isinstance(theta, (int, np.integer)) and theta > 0):
        raise ValueError("theta must be a positive integer")
    if not (isinstance(x0, (int, np.integer)) and abs(x0) < theta):
        raise ValueError("|x0| must be < theta")

    M = 2*theta - 1
    j0 = x0 + theta
    r = np.sqrt(mu1/mu2)

    ks = np.arange(1, M+1)                  # all modes
    alpha = ks * np.pi / (2*theta)
    lam = -(mu1 + mu2) + 2*np.sqrt(mu1*mu2) * np.cos(alpha)
    sin_alpha = np.sin(alpha)
    sin_alpha_j0 = np.sin(alpha * j0)
    e = np.exp(lam[:, None] * t[None, :])   # (M, |t|)

    # p(1,t)
    r_pow_minus = r ** ((1 - j0)/2.0)
    S_minus = np.sum((sin_alpha * sin_alpha_j0)[:, None] * e, axis=0)
    f_minus = (mu2 / theta) * r_pow_minus * S_minus

    # p(M,t)
    upper_sign = (-1) ** (ks + 1)
    r_pow_plus = r ** ((M - j0)/2.0)
    S_plus = np.sum((upper_sign * sin_alpha * sin_alpha_j0)[:, None] * e, axis=0)
    f_plus = (mu1 / theta) * r_pow_plus * S_plus

    f_plus[f_plus < 0] = 0.0
    f_minus[f_minus < 0] = 0.0
    return f_plus, f_minus

def split_prob_start(mu1, mu2, theta, x0):
    if np.isclose(mu1, mu2):
        p_plus = (x0 + theta) / (2*theta)
        return p_plus, 1 - p_plus
    rho = mu2 / mu1
    p_plus = (1 - rho ** (x0 + theta)) / (1 - rho ** (2*theta))
    return p_plus, 1 - p_plus

# --- simulation ---
def simulate_fpt_with_side_start(mu1, mu2, theta, x0, rng=None):
    if rng is None: rng = np.random.default_rng()
    x = int(x0); t = 0.0
    tot = mu1 + mu2; p_up = mu1 / tot
    while abs(x) < theta:
        t += rng.exponential(1.0 / tot)
        x += 1 if (rng.random() < p_up) else -1
    return t, (1 if x == theta else -1)

# --- helper: ECDF ---
def ecdf(x):
    x = np.sort(x)
    y = np.linspace(0, 1, len(x), endpoint=False) + 1/len(x)
    return x, y

def main():
    mu1, mu2, theta, x0 = 12.0, 9.0, 3, 1
    num_trials = 50_000
    t_max = 2.0

    # theory (unconditional split densities)
    t = np.linspace(0.0, t_max, 2000)
    f_plus_u, f_minus_u = fpt_split_skellam_start(t, mu1, mu2, theta, x0)
    p_plus, p_minus = split_prob_start(mu1, mu2, theta, x0)

    # conditional (full) densities
    f_plus_cond_full  = f_plus_u / p_plus
    f_minus_cond_full = f_minus_u / p_minus

    # truncated-normalization on [0, t_max] to match histogram (density=True)
    dt = t[1] - t[0]
    Z_plus_trunc  = np.trapz(f_plus_cond_full, t)
    Z_minus_trunc = np.trapz(f_minus_cond_full, t)

    print(f"Mass within [0, {t_max}] | upper-hit cond: {Z_plus_trunc:.4f}")
    print(f"Mass within [0, {t_max}] | lower-hit cond: {Z_minus_trunc:.4f}")

    f_plus_cond_trunc  = f_plus_cond_full  / Z_plus_trunc
    f_minus_cond_trunc = f_minus_cond_full / Z_minus_trunc

    # simulate
    rng = np.random.default_rng()
    up, dn = [], []
    for _ in range(num_trials):
        tt, s = simulate_fpt_with_side_start(mu1, mu2, theta, x0, rng)
        (up if s == 1 else dn).append(tt)
    up = np.asarray(up); dn = np.asarray(dn)
    sim_p_plus = up.size / num_trials
    print(f"P(+θ): theory={p_plus:.4f}  sim={sim_p_plus:.4f}")
    print(f"P(-θ): theory={p_minus:.4f}  sim={1-sim_p_plus:.4f}")

    # hist settings
    bins = np.arange(0.0, t_max + 0.01, 0.01)

    # Upper (conditional, truncated)
    plt.figure(figsize=(7,4), facecolor="white")
    plt.hist(up, bins=bins, density=True, alpha=0.5, edgecolor="none", label="Sim (upper hits)")
    plt.plot(t, f_plus_cond_trunc, lw=1.8, label="Theory (upper | hit), renorm on [0,t_max]")
    plt.xlabel("Time"); plt.ylabel("Density")
    plt.title(f"Upper hits (θ={theta}, μ₁={mu1}, μ₂={mu2}, x₀={x0})")
    plt.legend(); plt.tight_layout()

    # Lower (conditional, truncated)
    plt.figure(figsize=(7,4), facecolor="white")
    plt.hist(dn, bins=bins, density=True, alpha=0.5, edgecolor="none", label="Sim (lower hits)")
    plt.plot(t, f_minus_cond_trunc, lw=1.8, label="Theory (lower | hit), renorm on [0,t_max]")
    plt.xlabel("Time"); plt.ylabel("Density")
    plt.title(f"Lower hits (θ={theta}, μ₁={mu1}, μ₂={mu2}, x₀={x0})")
    plt.legend(); plt.tight_layout()

    # ECDF vs theoretical CDF (nice diagnostic)
    from numpy import cumsum
    F_plus_trunc  = np.cumsum(f_plus_cond_trunc) * dt
    F_minus_trunc = np.cumsum(f_minus_cond_trunc) * dt

    xu, yu = ecdf(up[up <= t_max])
    xd, yd = ecdf(dn[dn <= t_max])

    plt.figure(figsize=(7,4), facecolor="white")
    plt.plot(t, F_plus_trunc, label="Theory CDF (upper | hit, truncated)")
    plt.step(xu, yu, where="post", alpha=0.7, label="ECDF (upper hits)", ls='--')
    plt.xlabel("Time"); plt.ylabel("CDF"); plt.title("Upper-hit CDF check"); plt.legend(); plt.tight_layout()

    plt.figure(figsize=(7,4), facecolor="white")
    plt.plot(t, F_minus_trunc, label="Theory CDF (lower | hit, truncated)", ls='--')
    plt.step(xd, yd, where="post", alpha=0.7, label="ECDF (lower hits)")
    plt.xlabel("Time"); plt.ylabel("CDF"); plt.title("Lower-hit CDF check"); plt.legend(); plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()

# %%
# after you simulate `up` and `dn` and compute p_plus, p_minus
t_max = 2.0
bins = np.arange(0.0, t_max + 0.01, 0.01)
bw = bins[1] - bins[0]

# theory (unconditional) on [0, t_max]
t = np.linspace(0, t_max, 1500)
f_plus_u, f_minus_u = fpt_split_skellam_start(t, mu1, mu2, theta, x0)

# scale hist height to probability per second
plt.figure(figsize=(7,4), facecolor="white")
counts, _, _ = plt.hist(up, bins=bins, density=False, alpha=0.45, edgecolor="none",
                        label="Sim (upper hits, unnormalized)")
plt.plot(t, f_plus_u, lw=1.8, label="Theory f₊(t) (unconditional)")
# rescale bars to density units:
plt.gca().patches.clear()  # clear bars
plt.bar(bins[:-1], counts/(num_trials*bw), width=bw, alpha=0.45, align="edge",
        label="Sim scaled to density (area = P(hit +θ))")
plt.xlabel("Time"); plt.ylabel("Probability density")
plt.title("Upper: unconditional density (area = P(hit +θ))")
plt.legend(); plt.tight_layout()

plt.figure(figsize=(7,4), facecolor="white")
counts, _, _ = plt.hist(dn, bins=bins, density=False, alpha=0.45, edgecolor="none",
                        label="Sim (lower hits, unnormalized)")
plt.plot(t, f_minus_u, lw=1.8, label="Theory f₋(t) (unconditional)")
plt.gca().patches.clear()
plt.bar(bins[:-1], counts/(num_trials*bw), width=bw, alpha=0.45, align="edge",
        label="Sim scaled to density (area = P(hit −θ))")
plt.xlabel("Time"); plt.ylabel("Probability density")
plt.title("Lower: unconditional density (area = P(hit −θ))")
plt.legend(); plt.tight_layout()


# %%

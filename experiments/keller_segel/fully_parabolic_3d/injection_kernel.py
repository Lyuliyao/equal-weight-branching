"""injection_kernel.py -- exact cross-species decay-injection reaction substep for
the fully parabolic-parabolic Keller-Segel chemical equation  v_t = alpha u - beta v
(reaction-only substep, u* frozen).  Exact update of the measure:

    mu_v^{n+1} = e^{-beta tau} mu_v^*  +  (alpha/beta)(1 - e^{-beta tau}) mu_u^* .

Implemented as a genuine two-species particle kernel (NO (alpha u - beta v)/v
multiplicative rate, NO division by v, NO positivity floor):

  DECAY    each transported v-particle survives w.p.  e^{-beta tau}, else dies.
  INJECT   each transported u-particle creates K_i new v-particles at its own
           location X_i*, with conditional mean
               m_inj = (alpha/beta)(1 - e^{-beta tau}) * (omega_u/omega_v).
           Default = minimum-variance stochastic rounding:
               K_i = floor(m_inj) + Bernoulli(m_inj - floor(m_inj)).
           Optional Poisson variance-baseline:  K_i ~ Poisson(m_inj).

Conditional means (given the transported clouds):
  E[surviving v]   = e^{-beta tau} mu_v^*
  E[injected v]    = (alpha/beta)(1 - e^{-beta tau})(omega_u/omega_v) mu_u^*  (in particle count)
                   = (alpha/beta)(1 - e^{-beta tau}) mu_u^*                   (in mass, since
                     injected mass = omega_v * count)
so E[mu_v^{n+1}] is exactly the boxed update.  For alpha=beta=1, omega_u=omega_v:
m_inj = 1 - e^{-tau} < 1  => pure Bernoulli injection.
"""
import numpy as np


def injection_mean(alpha, beta, tau, omega_u, omega_v):
    """Conditional mean number of v-offspring per transported u-particle."""
    return (alpha / beta) * (1.0 - np.exp(-beta * tau)) * (omega_u / omega_v)


def survival_prob(beta, tau):
    return float(np.exp(-beta * tau))


def stochastic_round(m, u01):
    """Minimum-variance integer kernel: floor(m) + Bernoulli(frac); u01 ~ U(0,1).
    Variance = frac(1-frac) <= 1/4, the minimum among integer kernels with mean m."""
    fl = np.floor(m)
    return (fl + (u01 < (m - fl))).astype(np.int64)


def decay_inject(Xu_star, Yv_star, alpha, beta, tau, omega_u, omega_v, rng,
                 kernel="minvar"):
    """One exact decay-injection substep.

    Xu_star : (Nu,3) transported u-particle positions (frozen u*).
    Yv_star : (Nv,3) transported v-particle positions.
    rng     : numpy Generator.
    kernel  : "minvar" (default, stochastic rounding) or "poisson" (variance baseline).

    Returns (Yv_new (Nv',3), info dict).  New v-particles carry mass omega_v each.
    """
    Xu = np.asarray(Xu_star, dtype=np.float64)
    Yv = np.asarray(Yv_star, dtype=np.float64)
    Nu = Xu.shape[0]

    # ---- DECAY: survive with prob e^{-beta tau} ----
    p_surv = survival_prob(beta, tau)
    if Yv.shape[0] > 0:
        surv = rng.random(Yv.shape[0]) < p_surv
        Yv_surv = Yv[surv]
        n_death = int((~surv).sum())
    else:
        Yv_surv = Yv.reshape(0, 3)
        n_death = 0

    # ---- INJECT: each u spawns K_i offspring at its own location ----
    m_inj = injection_mean(alpha, beta, tau, omega_u, omega_v)
    if Nu > 0:
        if kernel == "minvar":
            Ki = stochastic_round(np.full(Nu, m_inj), rng.random(Nu))
        elif kernel == "poisson":
            Ki = rng.poisson(m_inj, size=Nu).astype(np.int64)
        else:
            raise ValueError(f"unknown injection kernel {kernel!r}")
        n_birth = int(Ki.sum())
        Yv_new = np.repeat(Xu, Ki, axis=0) if n_birth > 0 else Xu[:0]
    else:
        n_birth = 0
        Yv_new = Xu[:0]

    Yv_out = np.concatenate([Yv_surv, Yv_new], axis=0) if (Yv_surv.shape[0] or n_birth) \
        else np.zeros((0, 3))
    info = dict(p_surv=p_surv, m_inj=float(m_inj), n_birth=n_birth, n_death=n_death,
                Nv_in=int(Yv.shape[0]), Nv_out=int(Yv_out.shape[0]))
    return Yv_out, info

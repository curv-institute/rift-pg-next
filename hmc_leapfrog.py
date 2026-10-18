"""
One gauge leapfrog trajectory on the six-staple 4D SU(2) engine.

Gauge weight, unchanged from rift_4d.py:
    dS = -beta * Re Tr((U' - U) @ staple)
with no factor 1/2. Stored <P> is (1/2) Re Tr U_p.
Lengths are not in the Hamiltonian.

Dirac links are identity. apply_hop does not read U, and no fermion
residual is mapped onto the link algebra. S_fermion is therefore
independent of U and cancels in Delta H.

The leapfrog Hamiltonian is
    H = K(P) + S_gauge(beta) + S_fermion
with K = (1/2) sum p_a^2. Acceptance is the Metropolis check on Delta H.
"""
import numpy as np

from dirac_strong_blocks import lanczos_min
from rift_4d import mean_plaquette, random_su2, staple_4d

# su(2) generators i * sigma. These are the link algebra, not the Dirac basis.
SIGMA = (
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
)


def cold_links(L):
    U = np.zeros((4, L, L, L, L, 2, 2), dtype=complex)
    U[:] = np.eye(2)
    return U


def S_gauge(U, beta):
    """S = -beta * sum_{mu<nu, x} Re Tr U_p. No 1/2, each plaquette once."""
    L = U.shape[1]
    acc = 0.0
    for mu in range(4):
        for nu in range(mu + 1, 4):
            for x in np.ndindex(L, L, L, L):
                x = tuple(x)
                x_mu = list(x)
                x_mu[mu] = (x_mu[mu] + 1) % L
                x_nu = list(x)
                x_nu[nu] = (x_nu[nu] + 1) % L
                pl = (U[mu][x] @ U[nu][tuple(x_mu)]
                      @ U[mu][tuple(x_nu)].conj().T @ U[nu][x].conj().T)
                acc += np.real(np.trace(pl))
    return -beta * acc


def staple_coeff(U):
    """coeff_k = Re Tr(i sigma_k U staple), so dS = -beta * coeff · dphi."""
    L = U.shape[1]
    coeff = np.empty((4, L, L, L, L, 3))
    for mu in range(4):
        for x in np.ndindex(L, L, L, L):
            M = U[mu][x] @ staple_4d(U, mu, x, L)
            for k in range(3):
                coeff[(mu,) + x + (k,)] = np.real(np.trace(1j * SIGMA[k] @ M))
    return coeff


def exp_i_psigma(p, eps):
    """exp(i eps p·sigma) in SU(2)."""
    n = float(np.linalg.norm(p))
    if n < 1e-14:
        return np.eye(2, dtype=complex)
    c = np.cos(eps * n)
    s = np.sin(eps * n) / n
    p1, p2, p3 = (float(p[0]), float(p[1]), float(p[2]))
    return np.array([
        [c + 1j * p3 * s, (1j * p1 + p2) * s],
        [(1j * p1 - p2) * s, c - 1j * p3 * s],
    ], dtype=complex)


def evolve_links(U, p, eps):
    U_new = U.copy()
    L = U.shape[1]
    for mu in range(4):
        for x in np.ndindex(L, L, L, L):
            U_new[mu][x] = exp_i_psigma(p[(mu,) + x], eps) @ U[mu][x]
    return U_new


def kinetic(p):
    return 0.5 * float(np.sum(p * p))


def leapfrog(U, p, beta, eps, n_steps):
    """p <- p - (dt) * dS/dphi, U <- exp(i eps p·sigma) U."""
    p = np.array(p, copy=True)
    p += (eps / 2.0) * beta * staple_coeff(U)
    for step in range(n_steps):
        U = evolve_links(U, p, eps)
        dt = eps if step < n_steps - 1 else eps / 2.0
        p += dt * beta * staple_coeff(U)
    return U, p


def check_action_matches_engine(beta=2.5, L=4):
    """A finite link change must reproduce the engine's dS, with no 1/2."""
    rng = np.random.default_rng(1)
    U = cold_links(L)
    assert abs(mean_plaquette(U, L) - 1.0) < 1e-12
    # Cold start is a critical point of S_gauge: the staple force vanishes.
    assert np.max(np.abs(staple_coeff(U))) < 1e-10
    for n in range(4):
        mu = int(rng.integers(0, 4))
        x = tuple(int(v) for v in rng.integers(0, L, size=4))
        old = U[mu][x].copy()
        prop = random_su2(rng, 0.4) @ old
        dS_engine = -beta * np.real(np.trace((prop - old) @ staple_4d(U, mu, x, L)))
        U2 = U.copy()
        U2[mu][x] = prop
        dS_action = S_gauge(U2, beta) - S_gauge(U, beta)
        if not np.allclose(dS_action, dS_engine, rtol=1e-9, atol=1e-7):
            raise SystemExit(
                f"action disagrees with engine dS: {dS_action} vs {dS_engine}"
            )
        # Infinitesimal left move along i sigma_k matches -beta * coeff.
        # Gradient at the deformed link, where it is not sitting at a critical point.
        # Centered difference: the cold start has vanishing first derivative, so a
        # one-sided step there is O(delta) and does not test the staple force.
        U_def = U2
        old_d = prop
        M = old_d @ staple_4d(U_def, mu, x, L)
        for k in range(3):
            delta = 1e-6
            plus = exp_i_psigma(np.eye(3)[k], delta)
            minus = exp_i_psigma(np.eye(3)[k], -delta)
            U_plus = U_def.copy()
            U_minus = U_def.copy()
            U_plus[mu][x] = plus @ old_d
            U_minus[mu][x] = minus @ old_d
            numeric = (S_gauge(U_plus, beta) - S_gauge(U_minus, beta)) / (2 * delta)
            analytic = -beta * np.real(np.trace(1j * SIGMA[k] @ M))
            if not np.allclose(numeric, analytic, rtol=1e-5, atol=1e-5):
                raise SystemExit(
                    f"staple force disagrees at mu={mu} k={k}: "
                    f"{numeric} vs {analytic}"
                )


def check_leapfrog_conserves(beta=2.5, L=4):
    """A short trajectory must keep Delta H well below the action change."""
    rng = np.random.default_rng(2)
    U = cold_links(L)
    p = rng.normal(size=U.shape[:5] + (3,))
    S0 = S_gauge(U, beta)
    K0 = kinetic(p)
    U1, p1 = leapfrog(U, p, beta, eps=1e-3, n_steps=4)
    dS = S_gauge(U1, beta) - S0
    dK = kinetic(p1) - K0
    dH = dK + dS
    if not np.isfinite(dH) or abs(dH) > 1e-4:
        raise SystemExit(f"leapfrog Delta H is not small: dH={dH} dS={dS} dK={dK}")


def run_one(L=4, beta=2.5, mass=0.1, eps=0.05, n_steps=4, seed=42):
    check_action_matches_engine(beta=beta, L=L)
    check_leapfrog_conserves(beta=beta, L=L)
    rng = np.random.default_rng(seed)
    U = cold_links(L)
    p = rng.normal(size=(4, L, L, L, L, 3))
    S0 = S_gauge(U, beta)
    K0 = kinetic(p)
    U1, p1 = leapfrog(U, p, beta, eps, n_steps)
    S1 = S_gauge(U1, beta)
    K1 = kinetic(p1)
    # S_fermion(U) does not depend on U and cancels.
    dH = (K1 + S1) - (K0 + S0)
    if not np.isfinite(dH):
        raise SystemExit(f"non-finite Delta H: {dH}")
    if dH <= 0.0 or rng.random() < np.exp(-dH):
        acceptance = 1
        U_keep = U1
    else:
        acceptance = 0
        U_keep = U
    mean_P = float(mean_plaquette(U_keep, L))
    min_E = lanczos_min(mass, L=L, k=20)
    return {
        "L": L,
        "beta": beta,
        "mass": mass,
        "acceptance": acceptance,
        "mean_P": mean_P,
        "min_abs_E": min_E,
        "dH": float(dH),
        "dS": float(S1 - S0),
        "dK": float(K1 - K0),
        "seed": seed,
        "eps": eps,
        "n_steps": n_steps,
    }


def write_row(path, row):
    with open(path, "w", encoding="utf-8") as out:
        out.write("L,beta,mass,acceptance,mean_P,min_abs_E\n")
        out.write(
            f"{row['L']},{row['beta']},{row['mass']},{row['acceptance']},"
            f"{row['mean_P']:.16g},{row['min_abs_E']:.16g}\n"
        )


if __name__ == "__main__":
    row = run_one()
    write_row("hmc_one_row.csv", row)
    print(row, flush=True)

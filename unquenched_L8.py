"""
Four cold unquenched trajectories at L=8.

apply_hop is the identity-link operator in dirac_strong_blocks.py. It is not
multiplied by the gauge field. The Euclidean norm of its CG residual after 5
iterations, from one complex Gaussian source with seed 42, sets a scale. Each
leapfrog step draws one 3-vector p = scale * normal(3), maps it with
exp_i_psigma, and left-multiplies that single SU(2) matrix onto every link.
That 3-vector is not a contraction of the spin-color residual onto the links.

Gauge weight is unchanged: dS = -beta * Re Tr((U' - U) @ staple), no factor 1/2.
Stored <P> is (1/2) Re Tr U_p.
"""
import numpy as np

from dirac_strong_blocks import apply_hop
from hmc_leapfrog import (
    S_gauge,
    cold_links,
    evolve_links,
    exp_i_psigma,
    kinetic,
    mean_plaquette,
    staple_coeff,
)

CG_ITERS = 5
MASS = 0.1
SOURCE_SEED = 42


def complex_gaussian_source(L, seed=SOURCE_SEED):
    rng = np.random.default_rng(seed)
    shape = (L, L, L, L, 4, 2)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def cg_residual(source, mass, L, n_iter=CG_ITERS):
    """Euclidean norm of source - apply_hop(x) after n_iter CG steps, x0 = 0."""
    x = np.zeros_like(source)
    residual = np.array(source, copy=True)
    direction = np.array(residual, copy=True)
    rho = np.vdot(residual, residual)
    for _ in range(n_iter):
        hopped = apply_hop(direction, mass, L)
        alpha = rho / np.vdot(direction, hopped)
        x = x + alpha * direction
        residual = residual - alpha * hopped
        rho_new = np.vdot(residual, residual)
        direction = residual + (rho_new / rho) * direction
        rho = rho_new
    true = source - apply_hop(x, mass, L)
    return float(np.linalg.norm(true))


def left_multiply_all(U, V):
    """Left-multiply one SU(2) matrix onto every link."""
    return np.einsum("ij,...jk->...ik", V, U)


def leapfrog_global(U, momentum, beta, eps, n_steps, rng, scale):
    """Existing gauge leapfrog, plus one global left multiply per step."""
    momentum = np.array(momentum, copy=True)
    momentum += (eps / 2.0) * beta * staple_coeff(U)
    for step in range(n_steps):
        U = evolve_links(U, momentum, eps)
        algebra = scale * rng.normal(size=3)
        U = left_multiply_all(U, exp_i_psigma(algebra, eps))
        dt = eps if step < n_steps - 1 else eps / 2.0
        momentum += dt * beta * staple_coeff(U)
    return U, momentum


def run_trajectory(seed, residual, scale, L, beta, mass, eps, n_steps):
    rng = np.random.default_rng(seed)
    U = cold_links(L)
    momentum = rng.normal(size=(4, L, L, L, L, 3))
    s0 = S_gauge(U, beta)
    k0 = kinetic(momentum)
    U1, momentum1 = leapfrog_global(U, momentum, beta, eps, n_steps, rng, scale)
    s1 = S_gauge(U1, beta)
    k1 = kinetic(momentum1)
    dH = (k1 + s1) - (k0 + s0)
    if not np.isfinite(dH):
        raise SystemExit(f"non-finite Delta H for seed {seed}: {dH}")
    if dH <= 0.0 or rng.random() < np.exp(-dH):
        acceptance = 1
        kept = U1
    else:
        acceptance = 0
        kept = U
    return {
        "L": L,
        "beta": beta,
        "mass": mass,
        "eps": eps,
        "n_steps": n_steps,
        "seed": seed,
        "cg_iters": CG_ITERS,
        "residual": residual,
        "acceptance": acceptance,
        "mean_P": float(mean_plaquette(kept, L)),
        "dH": float(dH),
    }


def write_csv(rows, path="unquenched_L8.csv"):
    with open(path, "w", encoding="utf-8") as out:
        out.write(
            "L,beta,mass,eps,n_steps,seed,cg_iters,residual,acceptance,mean_P\n"
        )
        for row in rows:
            out.write(
                f"{row['L']},{row['beta']},{row['mass']},{row['eps']},"
                f"{row['n_steps']},{row['seed']},{row['cg_iters']},"
                f"{row['residual']:.16g},{row['acceptance']},{row['mean_P']:.16g}\n"
            )


def main():
    L = 8
    beta = 2.5
    eps = 0.01
    n_steps = 20
    probe = cold_links(2)
    algebra = np.array([0.2, -0.1, 0.05])
    mapped = exp_i_psigma(algebra, eps)
    if not np.allclose(left_multiply_all(probe, mapped)[0, 0, 0, 0, 0], mapped):
        raise SystemExit("global left multiply did not apply exp_i_psigma")
    source = complex_gaussian_source(L, SOURCE_SEED)
    residual = cg_residual(source, MASS, L, CG_ITERS)
    if not np.isfinite(residual) or residual < 0.0:
        raise SystemExit(f"CG residual is not a Euclidean norm: {residual}")
    scale = residual / max(MASS, 0.05)
    rows = []
    for seed in range(42, 46):
        row = run_trajectory(seed, residual, scale, L, beta, MASS, eps, n_steps)
        rows.append(row)
        print(row, flush=True)
    write_csv(rows)
    return rows


if __name__ == "__main__":
    main()

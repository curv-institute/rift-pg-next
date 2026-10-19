"""
One cold trajectory whose fermion force and fermion action share one operator.

D, D dagger, and F^a are the functions in unquenched_force_L8.py. The solve of
D dagger D eta = phi is the same routine for the finite-difference check and
for the leapfrog. phi is drawn once and held fixed.

L=4, beta=2.5, mass=0.1, and n_steps=4 are the gauge leapfrog conservation
check. eps=1e-4 is the step at which that integrator keeps |Delta H| much
smaller than 1 once the fermion force is included. The momentum update is
beta * staple - 2 F, because lambda^a = -i sigma^a / 2 maps the generator
step alpha onto the existing i sigma coordinate by phi = -alpha / 2.
"""
import numpy as np

from hmc_leapfrog import (
    S_gauge,
    cold_links,
    evolve_links,
    kinetic,
    mean_plaquette,
    staple_coeff,
)
from rift_4d import random_su2
from unquenched_force_L8 import (
    apply_A,
    exp_lambda,
    fermion_force_on_link,
)

L = 4
BETA = 2.5
MASS = 0.1
EPS = 1e-4
N_STEPS = 4
SEED = 42
REL_TOL = 1e-12
MAX_CG = 4000
FORCE_LINKS = (
    (0, (0, 0, 0, 0)),
    (1, (1, 0, 0, 0)),
    (2, (0, 1, 2, 3)),
    (3, (2, 2, 1, 0)),
    (0, (3, 1, 2, 2)),
    (1, (2, 3, 0, 1)),
)


def complex_gaussian(size, rng):
    shape = (size, size, size, size, 4, 2)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def solve_eta(phi, U, mass, size):
    """Tight CG for (D†D) eta = phi. Same routine for the check and the trajectory."""
    eta = np.zeros_like(phi)
    residual = np.array(phi, copy=True)
    direction = np.array(residual, copy=True)
    rho = np.vdot(residual, residual)
    phi_norm = np.linalg.norm(phi)
    rel = 1.0
    used = 0
    for used in range(1, MAX_CG + 1):
        if abs(rho) < 1e-30:
            break
        hopped = apply_A(direction, U, mass, size)
        alpha = rho / np.vdot(direction, hopped)
        eta = eta + alpha * direction
        residual = residual - alpha * hopped
        rho_new = np.vdot(residual, residual)
        direction = residual + (rho_new / rho) * direction
        rho = rho_new
        if used % 20 == 0 or used == MAX_CG:
            rel = float(np.linalg.norm(apply_A(eta, U, mass, size) - phi) / phi_norm)
            if rel < REL_TOL:
                return eta, rel, used
    rel = float(np.linalg.norm(apply_A(eta, U, mass, size) - phi) / phi_norm)
    if rel >= REL_TOL:
        raise SystemExit(f"CG residual {rel} after {used} iterations")
    return eta, rel, used


def fermion_action(phi, eta):
    return float(np.real(np.vdot(phi, eta)))


def finite_difference_force(phi, U, mu, x, mass, size, delta=1e-6):
    numeric = np.zeros(3)
    base = U[mu][x].copy()
    for a in range(3):
        plus = np.array(U, copy=True)
        minus = np.array(U, copy=True)
        plus[mu][x] = exp_lambda(np.eye(3)[a], delta) @ base
        minus[mu][x] = exp_lambda(np.eye(3)[a], -delta) @ base
        eta_p, _, _ = solve_eta(phi, plus, mass, size)
        eta_m, _, _ = solve_eta(phi, minus, mass, size)
        s_p = fermion_action(phi, eta_p)
        s_m = fermion_action(phi, eta_m)
        numeric[a] = -(s_p - s_m) / (2.0 * delta)
    return numeric


def validate_force():
    rng = np.random.default_rng(1)
    U = cold_links(L)
    for mu in range(4):
        for site in np.ndindex(L, L, L, L):
            U[mu][site] = random_su2(rng, 0.15) @ U[mu][site]
    phi = complex_gaussian(L, rng)
    eta, rel, used = solve_eta(phi, U, MASS, L)
    print(f"check solve rel residual {rel:.3e} in {used} iterations", flush=True)
    rows = []
    worst = 0.0
    for mu, site in FORCE_LINKS:
        analytic = fermion_force_on_link(U, eta, mu, site, L, MASS)
        numeric = finite_difference_force(phi, U, mu, site, MASS, L)
        denom = max(float(np.linalg.norm(numeric)), float(np.linalg.norm(analytic)))
        error = float(np.linalg.norm(analytic - numeric) / denom) if denom > 0.0 else 0.0
        worst = max(worst, error)
        rows.append((mu, site, analytic, numeric, error))
        print(
            f"link mu={mu} x={site} rel={error:.3e} "
            f"F={analytic} d={numeric}",
            flush=True,
        )
    if worst >= 1e-4:
        raise SystemExit(f"force check failed, worst relative error {worst}")
    print(f"force check passed, worst relative error {worst:.3e}", flush=True)
    return rel, worst, rows


def hamiltonian(U, momentum, phi, mass, size, beta):
    eta, rel, used = solve_eta(phi, U, mass, size)
    energy = kinetic(momentum) + S_gauge(U, beta) + fermion_action(phi, eta)
    return energy, eta, rel, used


def leapfrog(U, momentum, phi, beta, mass, size, eps, n_steps):
    """Staple force plus fermion force, one momentum, existing drift."""
    momentum = np.array(momentum, copy=True)
    eta, _, _ = solve_eta(phi, U, mass, size)
    # dS/dφ = 2 F because φ = -α/2 for U = exp(α λ) U = exp(i φ·σ) U.
    force = beta * staple_coeff(U) - 2.0 * fermion_force_field(U, eta, mass, size)
    momentum += (eps / 2.0) * force
    for step in range(n_steps):
        U = evolve_links(U, momentum, eps)
        eta, _, _ = solve_eta(phi, U, mass, size)
        force = beta * staple_coeff(U) - 2.0 * fermion_force_field(U, eta, mass, size)
        dt = eps if step < n_steps - 1 else eps / 2.0
        momentum += dt * force
    return U, momentum


def fermion_force_field(U, eta, mass, size):
    from unquenched_force_L8 import all_forces

    return all_forces(U, eta, mass, size)


def run_trajectory(check_residual):
    rng = np.random.default_rng(SEED)
    phi = complex_gaussian(L, rng)
    U = cold_links(L)
    momentum = rng.normal(size=(4, L, L, L, L, 3))
    h0, _, rel0, n0 = hamiltonian(U, momentum, phi, MASS, L, BETA)
    print(f"trajectory start residual {rel0:.3e} in {n0} iterations", flush=True)
    U1, momentum1 = leapfrog(U, momentum, phi, BETA, MASS, L, EPS, N_STEPS)
    h1, _, rel1, n1 = hamiltonian(U1, momentum1, phi, MASS, L, BETA)
    dH = h1 - h0
    print(f"trajectory end residual {rel1:.3e} in {n1} iterations", flush=True)
    print(f"delta_H {dH:.16g}", flush=True)
    if not np.isfinite(dH) or abs(dH) >= 1e-3:
        raise SystemExit(f"|delta_H| is not much smaller than 1: {dH}")
    if dH <= 0.0 or rng.random() < np.exp(-dH):
        acceptance = 1
        kept = U1
    else:
        acceptance = 0
        kept = U
    mean_p = float(mean_plaquette(kept, L))
    row = {
        "L": L,
        "beta": BETA,
        "mass": MASS,
        "eps": EPS,
        "n_steps": N_STEPS,
        "seed": SEED,
        "cg_residual": max(check_residual, rel0, rel1),
        "acceptance": acceptance,
        "mean_P": mean_p,
        "delta_H": float(dH),
    }
    print(row, flush=True)
    with open("unquenched_force_gate.csv", "w", encoding="utf-8") as out:
        out.write(
            "L,beta,mass,eps,n_steps,seed,cg_residual,acceptance,mean_P,delta_H\n"
        )
        out.write(
            f"{row['L']},{row['beta']},{row['mass']},{row['eps']},{row['n_steps']},"
            f"{row['seed']},{row['cg_residual']:.16g},{row['acceptance']},"
            f"{row['mean_P']:.16g},{row['delta_H']:.16g}\n"
        )
    return row


def main():
    rel, worst, _rows = validate_force()
    run_trajectory(rel)
    print(f"worst_force_rel {worst:.16g}", flush=True)


if __name__ == "__main__":
    main()

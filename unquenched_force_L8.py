"""
One cold trajectory with a fermion force and the existing staple force.

Spin projectors are the G projectors in dirac_strong_blocks.py, including the
backward hop. Color is multiplied by U on the forward hop and by U dagger on
the backward hop. D dagger is the adjoint of that operator. Generators are
lambda^a = -i sigma^a / 2. Each leapfrog step left-multiplies
exp(eps * sum_a F^a lambda^a). The staple gauge force stays in the leapfrog.

This force sketch was not in the recorded repository.
"""
import sys

import numpy as np

from dirac_strong_blocks import G, apply_hop
from hmc_leapfrog import (
    S_gauge,
    cold_links,
    evolve_links,
    kinetic,
    mean_plaquette,
    staple_coeff,
)
from rift_4d import random_su2

SIGMA = (
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
)
LAM = tuple(-0.5j * si for si in SIGMA)


def _proj(mu):
    return 0.5 * (np.eye(4, dtype=complex) - G[mu])


def _color(link, psi):
    """link is (2, 2) or (..., 2, 2); psi is (..., 4, 2)."""
    return np.einsum("...ij,...sj->...si", link, psi)


def apply_D(psi, U, mass, L):
    """Wilson hop with the existing projectors and the gauge links."""
    out = (4.0 + mass) * psi
    for mu in range(4):
        proj = _proj(mu)
        fwd = np.roll(psi, -1, axis=mu)
        fwd = _color(U[mu], fwd)
        fwd = np.einsum("ab,...bc->...ac", proj, fwd)
        out -= fwd
        bwd = np.roll(psi, +1, axis=mu)
        udag = np.roll(np.conjugate(np.swapaxes(U[mu], -1, -2)), 1, axis=mu)
        bwd = _color(udag, bwd)
        bwd = np.einsum("ab,...bc->...ac", proj.conj().T, bwd)
        out -= bwd
    return out


def apply_D_dagger(phi, U, mass, L):
    """Adjoint of apply_D for the vdot inner product. Not an elementwise conj."""
    out = (4.0 + mass) * phi
    for mu in range(4):
        proj = _proj(mu)
        q = proj.conj().T
        # (H dagger phi)(y) = U(y-mu) dagger P dagger phi(y-mu)
        phi_m = np.roll(phi, +1, axis=mu)
        pd = np.einsum("ab,...bc->...ac", proj.conj().T, phi_m)
        udag = np.roll(np.conjugate(np.swapaxes(U[mu], -1, -2)), 1, axis=mu)
        out -= _color(udag, pd)
        # (B dagger phi)(y) = U(y) Q dagger phi(y+mu)
        phi_p = np.roll(phi, -1, axis=mu)
        qd = np.einsum("ab,...bc->...ac", q.conj().T, phi_p)
        out -= _color(U[mu], qd)
    return out


def apply_A(psi, U, mass, L):
    return apply_D_dagger(apply_D(psi, U, mass, L), U, mass, L)


def cg_solve(phi, U, mass, L, iters):
    """Solve (D dagger D) eta = phi. Runs `iters` steps from eta = 0."""
    eta = np.zeros_like(phi)
    residual = np.array(phi, copy=True)
    direction = np.array(residual, copy=True)
    rho = np.vdot(residual, residual)
    for _ in range(iters):
        if abs(rho) < 1e-30:
            break
        hopped = apply_A(direction, U, mass, L)
        denom = np.vdot(direction, hopped)
        alpha = rho / denom
        eta = eta + alpha * direction
        residual = residual - alpha * hopped
        rho_new = np.vdot(residual, residual)
        direction = residual + (rho_new / rho) * direction
        rho = rho_new
    return eta


def _shift(x, mu, L):
    y = list(x)
    y[mu] = (y[mu] + 1) % L
    return tuple(y)


def _local_force(U, eta, xi, mu, x, L):
    """F^a = - d/da Re phi† (D†D)^{-1} phi, with xi = D eta."""
    proj = _proj(mu)
    q = proj.conj().T
    xp = _shift(x, mu, L)
    link = U[mu][x]
    eta_x = eta[x]
    eta_xp = eta[xp]
    xi_x = xi[x]
    xi_xp = xi[xp]
    force = np.zeros(3)
    for a, lam in enumerate(LAM):
        fwd_vec = np.einsum("ab,cd,de,be->ac", proj, lam, link, eta_xp)
        bwd_vec = np.einsum("ab,cd,de,be->ac", q, link.conj().T, lam, eta_x)
        fwd_amp = np.vdot(xi_x, fwd_vec)
        bwd_amp = np.vdot(xi_xp, bwd_vec)
        force[a] = -2.0 * np.real(fwd_amp) + 2.0 * np.real(bwd_amp)
    return force


def fermion_force_on_link(U, eta, mu, x, L, mass):
    """3-vector F^a on one link. eta = (D†D)^{-1} phi."""
    xi = apply_D(eta, U, mass, L)
    return _local_force(U, eta, xi, mu, tuple(x), L)


def all_forces(U, eta, mass, L):
    xi = apply_D(eta, U, mass, L)
    force = np.zeros((4, L, L, L, L, 3))
    for mu in range(4):
        for x in np.ndindex(L, L, L, L):
            force[(mu,) + x] = _local_force(U, eta, xi, mu, x, L)
    return force


def exp_lambda(force_vec, eps):
    """exp(eps * sum_a F^a lambda^a) by a scaled Taylor series. Not exp_i_psigma."""
    matrix = np.zeros((2, 2), dtype=complex)
    for a in range(3):
        matrix += float(force_vec[a]) * LAM[a]
    matrix *= eps
    scale = np.linalg.norm(matrix, ord=1)
    squares = 0
    while scale > 0.5 and squares < 40:
        matrix *= 0.5
        scale *= 0.5
        squares += 1
    term = np.eye(2, dtype=complex)
    out = np.eye(2, dtype=complex)
    for k in range(1, 16):
        term = term @ matrix / k
        out = out + term
    for _ in range(squares):
        out = out @ out
    return out


def fermion_link_update(U, phi, mass, L, eps, cg_iters):
    eta = cg_solve(phi, U, mass, L, cg_iters)
    force = all_forces(U, eta, mass, L)
    updated = np.array(U, copy=True)
    for mu in range(4):
        for x in np.ndindex(L, L, L, L):
            updated[mu][x] = exp_lambda(force[(mu,) + x], eps) @ U[mu][x]
    return updated


def leapfrog_with_fermion(U, momentum, phi, beta, mass, L, eps, n_steps, cg_iters):
    momentum = np.array(momentum, copy=True)
    momentum += (eps / 2.0) * beta * staple_coeff(U)
    for step in range(n_steps):
        U = evolve_links(U, momentum, eps)
        U = fermion_link_update(U, phi, mass, L, eps, cg_iters)
        dt = eps if step < n_steps - 1 else eps / 2.0
        momentum += dt * beta * staple_coeff(U)
    return U, momentum


def fermion_action(phi, U, mass, L, eta):
    return float(np.real(np.vdot(phi, eta)))


def complex_gaussian(L, rng):
    shape = (L, L, L, L, 4, 2)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _dense_matrix(apply, U, mass, L):
    n = (L ** 4) * 4 * 2
    columns = np.empty((n, n), dtype=complex)
    eye = np.eye(n, dtype=complex)
    for i in range(n):
        psi = eye[:, i].reshape(L, L, L, L, 4, 2)
        columns[:, i] = apply(psi, U, mass, L).ravel()
    return columns


def _solve_dense(phi, U, mass, L):
    dmat = _dense_matrix(apply_D, U, mass, L)
    ddmat = _dense_matrix(apply_D_dagger, U, mass, L)
    amat = ddmat @ dmat
    eta = np.linalg.solve(amat, phi.ravel())
    return eta.reshape(phi.shape), amat


def check_operator_and_force(mass=0.1):
    """Steps 1–3. Returns the analytic and finite-difference 3-vectors."""
    L = 2
    rng = np.random.default_rng(1)
    psi = complex_gaussian(L, rng)
    ident = cold_links(L)
    hop_err = np.max(np.abs(apply_D(psi, ident, mass, L) - apply_hop(psi, mass, L)))
    if hop_err > 1e-12:
        raise SystemExit(f"identity-link D does not match apply_hop: {hop_err}")
    print(f"identity-link D matches apply_hop, max abs {hop_err:.3e}", flush=True)

    U = cold_links(L)
    for mu in range(4):
        for x in np.ndindex(L, L, L, L):
            U[mu][x] = random_su2(rng, 0.8)
    phi = complex_gaussian(L, rng)
    left = np.vdot(phi, apply_D(psi, U, mass, L))
    right = np.vdot(apply_D_dagger(phi, U, mass, L), psi)
    adj_err = abs(left - right)
    if adj_err > 1e-8:
        raise SystemExit(f"D dagger is not the adjoint: {left} vs {right}")
    print(f"D dagger matches the adjoint, abs diff {adj_err:.3e}", flush=True)

    eta, _amat = _solve_dense(phi, U, mass, L)
    residual = np.linalg.norm(apply_A(eta, U, mass, L) - phi)
    print(f"dense (D†D) residual {residual:.3e}", flush=True)
    mu, x = 0, (0, 0, 0, 0)
    analytic = fermion_force_on_link(U, eta, mu, x, L, mass)
    delta = 1e-6
    numeric = np.zeros(3)
    base_link = U[mu][x].copy()
    for a, lam in enumerate(LAM):
        plus = np.array(U, copy=True)
        minus = np.array(U, copy=True)
        step = exp_lambda(np.eye(3)[a], delta)
        step_m = exp_lambda(np.eye(3)[a], -delta)
        plus[mu][x] = step @ base_link
        minus[mu][x] = step_m @ base_link
        eta_p, _ = _solve_dense(phi, plus, mass, L)
        eta_m, _ = _solve_dense(phi, minus, mass, L)
        s_p = float(np.real(np.vdot(phi, eta_p)))
        s_m = float(np.real(np.vdot(phi, eta_m)))
        numeric[a] = - (s_p - s_m) / (2.0 * delta)
        del lam
    diff = analytic - numeric
    print("analytic F", analytic, flush=True)
    print("minus dS    ", numeric, flush=True)
    print("difference  ", diff, flush=True)
    scale = max(1.0, float(np.max(np.abs(numeric))))
    if np.max(np.abs(diff)) > 1e-4 * scale:
        raise SystemExit("fermion force does not match the finite difference")
    print("force matches the centered difference", flush=True)
    return analytic, numeric, diff


def run_trajectory():
    L = 8
    beta = 2.5
    mass = 0.1
    eps = 0.01
    n_steps = 4
    cg_iters = 20
    seed = 42
    rng = np.random.default_rng(seed)
    phi = complex_gaussian(L, rng)
    U = cold_links(L)
    momentum = rng.normal(size=(4, L, L, L, L, 3))
    eta0 = cg_solve(phi, U, mass, L, cg_iters)
    s0 = S_gauge(U, beta)
    k0 = kinetic(momentum)
    sf0 = fermion_action(phi, U, mass, L, eta0)
    U1, momentum1 = leapfrog_with_fermion(
        U, momentum, phi, beta, mass, L, eps, n_steps, cg_iters
    )
    eta1 = cg_solve(phi, U1, mass, L, cg_iters)
    s1 = S_gauge(U1, beta)
    k1 = kinetic(momentum1)
    sf1 = fermion_action(phi, U1, mass, L, eta1)
    dH = (k1 + s1 + sf1) - (k0 + s0 + sf0)
    if not np.isfinite(dH):
        raise SystemExit(f"non-finite Delta H: {dH}")
    if dH <= 0.0 or rng.random() < np.exp(-dH):
        acceptance = 1
        kept = U1
    else:
        acceptance = 0
        kept = U
    row = {
        "L": L,
        "beta": beta,
        "mass": mass,
        "eps": eps,
        "n_steps": n_steps,
        "seed": seed,
        "acceptance": acceptance,
        "mean_P": float(mean_plaquette(kept, L)),
        "dH": float(dH),
    }
    print(row, flush=True)
    path = "unquenched_force_L8.csv"
    with open(path, "w", encoding="utf-8") as out:
        out.write("L,beta,mass,eps,n_steps,seed,acceptance,mean_P\n")
        out.write(
            f"{row['L']},{row['beta']},{row['mass']},{row['eps']},{row['n_steps']},"
            f"{row['seed']},{row['acceptance']},{row['mean_P']:.16g}\n"
        )
    return row


def main():
    check_operator_and_force()
    run_trajectory()


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        raise

"""Hasenbusch split of det(D†D). μ² shifts the determinant, not the quark mass.

det(W) = det(W+μ²) det(W (W+μ²)^{-1}), W = D†D.
S_h = φ† (W+μ²)^{-1} φ, S_r = χ† (W+μ²) W^{-1} χ.
The dense Hermitian square root is used only to draw φ and χ.
Forces and the molecular dynamics stay matrix-free, with the same D and the
same true-residual tolerance as the unshifted solver.
"""
import numpy as np

from hmc_leapfrog import S_gauge, evolve_links, kinetic, staple_coeff
from unquenched_force_L8 import all_forces, apply_A, exp_lambda
from unquenched_force_gate import (
    BETA,
    L,
    MASS,
    MAX_CG,
    REL_TOL,
    _residual_collapsed,
    solve_eta,
)


def apply_shifted(psi, U, mass, size, mu2):
    return apply_A(psi, U, mass, size) + mu2 * psi


def solve_shifted(phi, U, mass, size, mu2):
    """CG for (D†D + μ²) psi = phi. Same restart and tolerance as solve_eta."""
    psi = np.zeros_like(phi)
    residual = np.array(phi, copy=True)
    direction = np.array(residual, copy=True)
    rho = np.vdot(residual, residual)
    phi_norm = np.linalg.norm(phi)
    if phi_norm == 0.0:
        return psi, 0.0, 0
    rel = 1.0
    used = 0
    for used in range(1, MAX_CG + 1):
        if abs(rho) < 1e-30:
            residual = phi - apply_shifted(psi, U, mass, size, mu2)
            direction = np.array(residual, copy=True)
            rho = np.vdot(residual, residual)
            rel = float(np.linalg.norm(residual) / phi_norm)
            if rel < REL_TOL:
                return psi, rel, used
            if abs(rho) < 1e-30:
                continue
        hopped = apply_shifted(direction, U, mass, size, mu2)
        alpha = rho / np.vdot(direction, hopped)
        psi = psi + alpha * direction
        residual = residual - alpha * hopped
        rho_new = np.vdot(residual, residual)
        direction = residual + (rho_new / rho) * direction
        rho = rho_new
        if used % 20 == 0 or used == MAX_CG:
            true = phi - apply_shifted(psi, U, mass, size, mu2)
            rel = float(np.linalg.norm(true) / phi_norm)
            if rel < REL_TOL:
                return psi, rel, used
            if _residual_collapsed(rho, residual, true):
                residual = true
                direction = np.array(true, copy=True)
                rho = np.vdot(residual, residual)
    rel = float(np.linalg.norm(apply_shifted(psi, U, mass, size, mu2) - phi) / phi_norm)
    if rel >= REL_TOL:
        raise SystemExit(f"shifted CG residual {rel} after {used} iterations")
    return psi, rel, used


def action_h(phi, U, mu2):
    psi, rel, used = solve_shifted(phi, U, MASS, L, mu2)
    return float(np.real(np.vdot(phi, psi))), psi, rel, used


def action_r(chi, U, mu2):
    zeta, rel, used = solve_eta(chi, U, MASS, L)
    value = float(np.real(np.vdot(chi, chi)) + mu2 * np.real(np.vdot(chi, zeta)))
    return value, zeta, rel, used


def force_h(U, psi):
    """-d S_h / dα. Same link formula as the unshifted force, on ψ with ξ = Dψ."""
    return all_forces(U, psi, MASS, L)


def force_r(U, zeta, mu2):
    """-d S_r / dα. ||χ||² is gauge-invariant, so only the μ² χ† W^{-1} χ term remains."""
    return mu2 * all_forces(U, zeta, MASS, L)


def rms(force):
    return float(np.sqrt(np.mean(force * force)))


def gauge_force(U):
    return BETA * staple_coeff(U)


def dense_spectrum(U):
    """Eigenpairs of the Hermitian part of W. Used only for the pseudofermion draw."""
    n = L ** 4 * 4 * 2
    matrix = np.empty((n, n), dtype=complex)
    for index in range(n):
        basis = np.zeros(n, dtype=complex)
        basis[index] = 1.0
        field = basis.reshape(L, L, L, L, 4, 2)
        matrix[:, index] = apply_A(field, U, MASS, L).reshape(-1)
        if index % 256 == 0:
            print(f"dense W column {index}/{n}", flush=True)
    matrix = 0.5 * (matrix + matrix.conj().T)
    evals, evecs = np.linalg.eigh(matrix)
    print(f"W eigenvalues min={evals[0]:.6e} max={evals[-1]:.6e}", flush=True)
    return evals, evecs


def _apply_fn(evals, evecs, values, field):
    coeffs = evecs.conj().T @ field.reshape(-1)
    return (evecs @ (values * coeffs)).reshape(field.shape)


def draw_hasenbusch(evals, evecs, eta, xi, mu2):
    """φ = (W+μ²)^{1/2} η, χ = W^{1/2} (W+μ²)^{-1/2} ξ."""
    safe = np.maximum(evals, 0.0)
    shifted = safe + mu2
    phi = _apply_fn(evals, evecs, np.sqrt(shifted), eta)
    chi = _apply_fn(evals, evecs, np.sqrt(safe / shifted), xi)
    return phi, chi


def _shift_link(U, mu, site, generator, delta):
    varied = np.array(U, copy=True)
    varied[mu][site] = exp_lambda(np.eye(3)[generator], delta) @ U[mu][site]
    return varied


def finite_difference_h(phi, U, mu, site, mu2, delta=1e-6):
    numeric = np.zeros(3)
    for generator in range(3):
        plus = _shift_link(U, mu, site, generator, delta)
        minus = _shift_link(U, mu, site, generator, -delta)
        s_plus, _, _, _ = action_h(phi, plus, mu2)
        s_minus, _, _, _ = action_h(phi, minus, mu2)
        numeric[generator] = -(s_plus - s_minus) / (2.0 * delta)
    return numeric


def finite_difference_r(chi, U, mu, site, mu2, delta=1e-6):
    numeric = np.zeros(3)
    for generator in range(3):
        plus = _shift_link(U, mu, site, generator, delta)
        minus = _shift_link(U, mu, site, generator, -delta)
        s_plus, _, _, _ = action_r(chi, plus, mu2)
        s_minus, _, _, _ = action_r(chi, minus, mu2)
        numeric[generator] = -(s_plus - s_minus) / (2.0 * delta)
    return numeric


def relative_error(analytic, numeric):
    denom = max(float(np.linalg.norm(numeric)), float(np.linalg.norm(analytic)))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(analytic - numeric) / denom)


def momentum_force(U, fermion_force):
    return gauge_force(U) - 2.0 * fermion_force


def leapfrog(U, momentum, force_fn, eps, n_steps):
    momentum = np.array(momentum, copy=True)
    momentum += (eps / 2.0) * force_fn(U)
    for step in range(n_steps):
        U = evolve_links(U, momentum, eps)
        dt = eps if step < n_steps - 1 else eps / 2.0
        momentum += dt * force_fn(U)
    return U, momentum


def leapfrog_nested(U, momentum, fine_fn, coarse_fn, eps, n_inner, n_outer):
    """Coarse kicks of the small force, fine leapfrog of the large force plus the gauge force."""
    momentum = np.array(momentum, copy=True)
    delta = n_inner * eps
    momentum += (delta / 2.0) * coarse_fn(U)
    for outer in range(n_outer):
        momentum += (eps / 2.0) * fine_fn(U)
        for inner in range(n_inner):
            U = evolve_links(U, momentum, eps)
            dt = eps if inner < n_inner - 1 else eps / 2.0
            momentum += dt * fine_fn(U)
        dt_coarse = delta if outer < n_outer - 1 else delta / 2.0
        momentum += dt_coarse * coarse_fn(U)
    return U, momentum


def fine_per_coarse(n_fine, ratio):
    """Largest divisor of n_fine that does not exceed the force ratio."""
    if not np.isfinite(ratio):
        return n_fine
    target = max(2, int(round(ratio)))
    chosen = 1
    for divisor in range(1, n_fine + 1):
        if n_fine % divisor == 0 and divisor <= target:
            chosen = divisor
    return chosen


def hamiltonian(U, momentum, phi, chi, mu2):
    s_g = float(S_gauge(U, BETA))
    s_h, psi, rel_h, n_h = action_h(phi, U, mu2)
    s_r, zeta, rel_r, n_r = action_r(chi, U, mu2)
    energy = kinetic(momentum) + s_g + s_h + s_r
    return energy, s_g, s_h, s_r, psi, zeta, rel_h, rel_r, n_h, n_r

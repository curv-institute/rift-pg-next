"""
RIFT-PG 4D Metropolis core.

Same constants as the 2D engine. Gauge weight is
dS = -beta * Re Tr((U' - U) @ staple), with no factor 1/2.
The staple for U_mu(x), summed over nu != mu, is
  forward  U_nu(x+mu) U_mu(x+nu)^dagger U_nu(x)^dagger
  backward U_nu(x+mu-nu)^dagger U_mu(x-nu)^dagger U_nu(x-nu)
Plaquettes are measured on every site, with no stride.
"""
import numpy as np
import time

C = 0.15
XI_STAR = 0.25
LAMBDA_H = 0.5
V2 = 1.0


def random_su2(rng, eps=0.15):
    a = np.array([1.0, 0.0, 0.0, 0.0]) + eps * rng.normal(size=4)
    a /= np.linalg.norm(a)
    return np.array([[a[0] + 1j * a[1], a[2] + 1j * a[3]],
                     [-a[2] + 1j * a[3], a[0] - 1j * a[1]]], dtype=complex)


def _at(x, direction, delta, L):
    y = list(x)
    y[direction] = (y[direction] + delta) % L
    return tuple(y)


def staple_4d(U, mu, x, L):
    """Six staples. Indices follow the forward and backward formulas above."""
    stap = np.zeros((2, 2), dtype=complex)
    x = tuple(x)
    for nu in range(4):
        if nu == mu:
            continue
        x_mu = _at(x, mu, +1, L)
        x_nu = _at(x, nu, +1, L)
        x_mnu = _at(x, nu, -1, L)
        x_mu_mnu = _at(x_mu, nu, -1, L)
        # U_nu(x+mu) U_mu(x+nu)^dagger U_nu(x)^dagger
        stap += (U[nu][x_mu] @ U[mu][x_nu].conj().T @ U[nu][x].conj().T)
        # U_nu(x+mu-nu)^dagger U_mu(x-nu)^dagger U_nu(x-nu)
        stap += (U[nu][x_mu_mnu].conj().T @ U[mu][x_mnu].conj().T @ U[nu][x_mnu])
    return stap


def lengths_and_Xi(U):
    """ell and population variance over the 4 L^4 links."""
    L = U.shape[1]
    ell = np.empty((4, L, L, L, L))
    eye = np.eye(2)
    for mu in range(4):
        for x in np.ndindex(L, L, L, L):
            dev = np.sum(np.abs(U[mu][x] - eye) ** 2)
            ell[(mu,) + x] = np.exp(-C * min(dev, 5.0))
    return ell, float(np.var(ell))


def higgs_action_local(h, x, U, ell, L):
    """Potential plus forward and backward hops in all four directions."""
    pot = LAMBDA_H * (np.abs(h[x]) ** 2 - V2) ** 2
    kin = 0.0
    x = tuple(x)
    for mu in range(4):
        xp = _at(x, mu, +1, L)
        link = U[mu][x][0, 0]
        kin += ell[(mu,) + x] * np.abs(h[xp] - link * h[x]) ** 2
        xm = _at(x, mu, -1, L)
        link_b = U[mu][xm][0, 0]
        kin += ell[(mu,) + xm] * np.abs(h[x] - link_b * h[xm]) ** 2
    return pot + kin


def mean_plaquette(U, L):
    """<(1/2) Re Tr U_p> over every site and every mu < nu. No stride."""
    acc = 0.0
    n = 0
    for mu in range(4):
        for nu in range(mu + 1, 4):
            for x in np.ndindex(L, L, L, L):
                x = tuple(x)
                x_mu = _at(x, mu, +1, L)
                x_nu = _at(x, nu, +1, L)
                # U_mu(x) U_nu(x+mu) U_mu(x+nu)^dagger U_nu(x)^dagger
                pl = (U[mu][x] @ U[nu][x_mu]
                      @ U[mu][x_nu].conj().T @ U[nu][x].conj().T)
                acc += np.real(np.trace(pl)) / 2
                n += 1
    return acc / n


def run_4d(L=6, beta=3.0, eps=0.15, eps_h=0.15,
           n_therm=4, n_meas=4, seed=42):
    rng = np.random.default_rng(seed)
    U = np.zeros((4, L, L, L, L, 2, 2), dtype=complex)
    U[:] = np.eye(2)
    h = np.ones((L, L, L, L), dtype=complex)
    Ps, ell_means, Xis, hvevs = [], [], [], []
    t0 = time.time()
    for step in range(n_therm + n_meas):
        for mu in range(4):
            for x in np.ndindex(L, L, L, L):
                old = U[mu][x].copy()
                prop = random_su2(rng, eps) @ old
                dS = -beta * np.real(np.trace((prop - old) @ staple_4d(U, mu, x, L)))
                if dS <= 0 or rng.random() < np.exp(-dS):
                    U[mu][x] = prop
        ell, _ = lengths_and_Xi(U)
        for x in np.ndindex(L, L, L, L):
            old_h = h[x]
            prop_h = old_h + eps_h * (rng.normal() + 1j * rng.normal())
            dS = higgs_action_local(h, x, U, ell, L)
            h[x] = prop_h
            dS = higgs_action_local(h, x, U, ell, L) - dS
            if dS > 0 and rng.random() > np.exp(-dS):
                h[x] = old_h
        if step >= n_therm:
            Ps.append(mean_plaquette(U, L))
            ell, Xi = lengths_and_Xi(U)
            ell_means.append(float(ell.mean()))
            Xis.append(Xi)
            hvevs.append(float(np.mean(np.abs(h))))
    Ps = np.asarray(Ps)
    ell_means = np.asarray(ell_means)
    Xis = np.asarray(Xis)
    hvevs = np.asarray(hvevs)
    return {
        "L": L,
        "beta": beta,
        "mean_P": float(Ps.mean()),
        "mean_ell": float(ell_means.mean()),
        "mean_Xi": float(Xis.mean()),
        "mean_h": float(hvevs.mean()),
        "geometric": bool(Xis.mean() < XI_STAR),
        "n_therm": n_therm,
        "n_meas": n_meas,
        "seed": seed,
        "time_s": time.time() - t0,
    }


def write_scan(path="continuum_scan_4d.csv", n_therm=4, n_meas=4, seed=42):
    """Three points only: (6, 3.0), (6, 4.0), (8, 3.0)."""
    points = ((6, 3.0), (6, 4.0), (8, 3.0))
    rows = []
    with open(path, "w", encoding="utf-8") as out:
        out.write(
            f"# engine=python-4d n_therm={n_therm} n_meas={n_meas} "
            f"seed={seed} eps=0.15 eps_h=0.15\n"
        )
        out.write(
            "# staple: forward U_nu(x+mu) U_mu(x+nu)^+ U_nu(x)^+ ; "
            "backward U_nu(x+mu-nu)^+ U_mu(x-nu)^+ U_nu(x-nu)\n"
        )
        out.write("L,beta,mean_P,mean_ell,mean_Xi,mean_h,geometric\n")
        out.flush()
        for L, beta in points:
            row = run_4d(L=L, beta=beta, n_therm=n_therm, n_meas=n_meas, seed=seed)
            rows.append(row)
            print(row, flush=True)
            flag = "true" if row["geometric"] else "false"
            out.write(
                f"{row['L']},{row['beta']},{row['mean_P']:.16g},"
                f"{row['mean_ell']:.16g},{row['mean_Xi']:.16g},"
                f"{row['mean_h']:.16g},{flag}\n"
            )
            out.flush()
    return rows


if __name__ == "__main__":
    write_scan()

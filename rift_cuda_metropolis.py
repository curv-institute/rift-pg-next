"""
CUDA checkerboard Metropolis for the 4D SU(2) engine in rift_4d.py.

The gauge weight is the one already in that file,
    dS = -beta * Re Tr((U' - U) @ staple),
with no factor 1/2. The staple is the same six-link sum. Stored <P> is
(1/2) Re Tr U_p. Lengths, Xi, and the quenched Higgs use the formulas
already in rift_4d.py.

A sweep updates every link once. Sites of one parity are updated together:
a staple for U_mu(x) reads U_mu only at x±nu, which has the other parity,
so those reads do not hit a link this launch is writing. The accept/reject
test is the engine's test. This is a different update order from the serial
ndindex loop, with the same proposal distribution and the same dS.
"""
import math
import sys
import time

import numpy as np
from numba import cuda, float64
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_normal_float64,
    xoroshiro128p_uniform_float64,
)

from rift_4d import (
    C,
    LAMBDA_H,
    V2,
    XI_STAR,
    higgs_action_local,
    lengths_and_Xi,
    mean_plaquette,
    random_su2,
    staple_4d,
)

THREADS = 256
# Several thousand measurement sweeps after thermalization.
N_THERM = 3000
N_MEAS = 4000
SEED = 42
EPS = 0.15
EPS_H = 0.15
POINTS = ((16, 3.0), (16, 4.0), (24, 3.5))


def decode_half_host(idx, parity, L):
    """idx-th site whose coordinates sum to `parity` mod 2. L must be even."""
    half = L // 2
    i3 = idx // half
    it = idx - i3 * half
    x = i3 % L
    y = (i3 // L) % L
    z = i3 // (L * L)
    if ((x + y + z) & 1) == (parity & 1):
        t = 2 * it
    else:
        t = 2 * it + 1
    return x, y, z, t


@cuda.jit(device=True)
def shift(x, y, z, t, mu, delta, L):
    if mu == 0:
        x = x + delta
        if x >= L:
            x = 0
        elif x < 0:
            x = L - 1
    elif mu == 1:
        y = y + delta
        if y >= L:
            y = 0
        elif y < 0:
            y = L - 1
    elif mu == 2:
        z = z + delta
        if z >= L:
            z = 0
        elif z < 0:
            z = L - 1
    else:
        t = t + delta
        if t >= L:
            t = 0
        elif t < 0:
            t = L - 1
    return x, y, z, t


@cuda.jit(device=True)
def decode_half(idx, parity, L):
    half = L // 2
    i3 = idx // half
    it = idx - i3 * half
    x = i3 % L
    y = (i3 // L) % L
    z = i3 // (L * L)
    if ((x + y + z) & 1) == (parity & 1):
        t = 2 * it
    else:
        t = 2 * it + 1
    return x, y, z, t


@cuda.jit(device=True)
def load(U, mu, x, y, z, t):
    return (
        U[mu, x, y, z, t, 0, 0],
        U[mu, x, y, z, t, 0, 1],
        U[mu, x, y, z, t, 1, 0],
        U[mu, x, y, z, t, 1, 1],
    )


@cuda.jit(device=True)
def dag(m):
    a, b, c, d = m
    return a.conjugate(), c.conjugate(), b.conjugate(), d.conjugate()


@cuda.jit(device=True)
def mm(p, q):
    a, b, c, d = p
    e, f, g, h = q
    return a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h


@cuda.jit(device=True)
def staple(U, mu, x, y, z, t, L):
    s0 = 0.0 + 0.0j
    s1 = 0.0 + 0.0j
    s2 = 0.0 + 0.0j
    s3 = 0.0 + 0.0j
    for nu in range(4):
        if nu != mu:
            x1, y1, z1, t1 = shift(x, y, z, t, mu, 1, L)
            x2, y2, z2, t2 = shift(x, y, z, t, nu, 1, L)
            x3, y3, z3, t3 = shift(x, y, z, t, nu, -1, L)
            x4, y4, z4, t4 = shift(x1, y1, z1, t1, nu, -1, L)
            # U_nu(x+mu) U_mu(x+nu)^dagger U_nu(x)^dagger
            fwd = mm(
                mm(load(U, nu, x1, y1, z1, t1), dag(load(U, mu, x2, y2, z2, t2))),
                dag(load(U, nu, x, y, z, t)),
            )
            # U_nu(x+mu-nu)^dagger U_mu(x-nu)^dagger U_nu(x-nu)
            bwd = mm(
                mm(dag(load(U, nu, x4, y4, z4, t4)), dag(load(U, mu, x3, y3, z3, t3))),
                load(U, nu, x3, y3, z3, t3),
            )
            s0 = s0 + fwd[0] + bwd[0]
            s1 = s1 + fwd[1] + bwd[1]
            s2 = s2 + fwd[2] + bwd[2]
            s3 = s3 + fwd[3] + bwd[3]
    return s0, s1, s2, s3


@cuda.jit(device=True)
def link_ell(U, mu, x, y, z, t, c_len):
    u00 = U[mu, x, y, z, t, 0, 0]
    u01 = U[mu, x, y, z, t, 0, 1]
    u10 = U[mu, x, y, z, t, 1, 0]
    u11 = U[mu, x, y, z, t, 1, 1]
    d00r = u00.real - 1.0
    d00i = u00.imag
    d11r = u11.real - 1.0
    d11i = u11.imag
    dev = (
        d00r * d00r
        + d00i * d00i
        + u01.real * u01.real
        + u01.imag * u01.imag
        + u10.real * u10.real
        + u10.imag * u10.imag
        + d11r * d11r
        + d11i * d11i
    )
    if dev > 5.0:
        dev = 5.0
    return math.exp(-c_len * dev)


@cuda.jit(device=True)
def apply_accept(U, beta, R0, R1, R2, R3, uni, mu, x, y, z, t, L):
    old = load(U, mu, x, y, z, t)
    prop = mm((R0, R1, R2, R3), old)
    diff = (prop[0] - old[0], prop[1] - old[1], prop[2] - old[2], prop[3] - old[3])
    prod = mm(diff, staple(U, mu, x, y, z, t, L))
    dS = -beta * (prod[0].real + prod[3].real)
    accept = dS <= 0.0
    if not accept:
        accept = uni < math.exp(-dS)
    if accept:
        U[mu, x, y, z, t, 0, 0] = prop[0]
        U[mu, x, y, z, t, 0, 1] = prop[1]
        U[mu, x, y, z, t, 1, 0] = prop[2]
        U[mu, x, y, z, t, 1, 1] = prop[3]


@cuda.jit
def sweep_given(U, beta, props, unifs, mu, parity, L):
    idx = cuda.grid(1)
    n_half = (L * L * L * L) // 2
    if idx >= n_half:
        return
    x, y, z, t = decode_half(idx, parity, L)
    apply_accept(
        U,
        beta,
        props[idx, 0, 0],
        props[idx, 0, 1],
        props[idx, 1, 0],
        props[idx, 1, 1],
        unifs[idx],
        mu,
        x,
        y,
        z,
        t,
        L,
    )


@cuda.jit
def sweep_rng(U, beta, eps, rng_states, mu, parity, L):
    idx = cuda.grid(1)
    n_half = (L * L * L * L) // 2
    if idx >= n_half:
        return
    x, y, z, t = decode_half(idx, parity, L)
    site = ((x * L + y) * L + z) * L + t
    g0 = xoroshiro128p_normal_float64(rng_states, site)
    g1 = xoroshiro128p_normal_float64(rng_states, site)
    g2 = xoroshiro128p_normal_float64(rng_states, site)
    g3 = xoroshiro128p_normal_float64(rng_states, site)
    a0 = 1.0 + eps * g0
    a1 = eps * g1
    a2 = eps * g2
    a3 = eps * g3
    nrm = math.sqrt(a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3)
    a0 /= nrm
    a1 /= nrm
    a2 /= nrm
    a3 /= nrm
    uni = xoroshiro128p_uniform_float64(rng_states, site)
    apply_accept(
        U,
        beta,
        a0 + 1j * a1,
        a2 + 1j * a3,
        -a2 + 1j * a3,
        a0 - 1j * a1,
        uni,
        mu,
        x,
        y,
        z,
        t,
        L,
    )


@cuda.jit(device=True)
def higgs_action(h_here, U, h, x, y, z, t, L, c_len, lambda_h, v2):
    mag2 = h_here.real * h_here.real + h_here.imag * h_here.imag
    pot = lambda_h * (mag2 - v2) * (mag2 - v2)
    kin = 0.0
    for mu in range(4):
        x1, y1, z1, t1 = shift(x, y, z, t, mu, 1, L)
        link = U[mu, x, y, z, t, 0, 0]
        hp = h[x1, y1, z1, t1]
        hop_r = hp.real - (link.real * h_here.real - link.imag * h_here.imag)
        hop_i = hp.imag - (link.real * h_here.imag + link.imag * h_here.real)
        kin += link_ell(U, mu, x, y, z, t, c_len) * (hop_r * hop_r + hop_i * hop_i)
        x2, y2, z2, t2 = shift(x, y, z, t, mu, -1, L)
        link_b = U[mu, x2, y2, z2, t2, 0, 0]
        hm = h[x2, y2, z2, t2]
        hopb_r = h_here.real - (link_b.real * hm.real - link_b.imag * hm.imag)
        hopb_i = h_here.imag - (link_b.real * hm.imag + link_b.imag * hm.real)
        kin += link_ell(U, mu, x2, y2, z2, t2, c_len) * (hopb_r * hopb_r + hopb_i * hopb_i)
    return pot + kin


@cuda.jit
def higgs_given(U, h, props_r, props_i, unifs, parity, L, c_len, lambda_h, v2):
    idx = cuda.grid(1)
    n_half = (L * L * L * L) // 2
    if idx >= n_half:
        return
    x, y, z, t = decode_half(idx, parity, L)
    old = h[x, y, z, t]
    prop = props_r[idx] + 1j * props_i[idx]
    dS = higgs_action(prop, U, h, x, y, z, t, L, c_len, lambda_h, v2) - higgs_action(
        old, U, h, x, y, z, t, L, c_len, lambda_h, v2
    )
    if not (dS > 0.0 and unifs[idx] > math.exp(-dS)):
        h[x, y, z, t] = prop


@cuda.jit
def higgs_rng(U, h, eps_h, rng_states, parity, L, c_len, lambda_h, v2):
    idx = cuda.grid(1)
    n_half = (L * L * L * L) // 2
    if idx >= n_half:
        return
    x, y, z, t = decode_half(idx, parity, L)
    site = ((x * L + y) * L + z) * L + t
    g1 = xoroshiro128p_normal_float64(rng_states, site)
    g2 = xoroshiro128p_normal_float64(rng_states, site)
    uni = xoroshiro128p_uniform_float64(rng_states, site)
    old = h[x, y, z, t]
    prop = (old.real + eps_h * g1) + 1j * (old.imag + eps_h * g2)
    dS = higgs_action(prop, U, h, x, y, z, t, L, c_len, lambda_h, v2) - higgs_action(
        old, U, h, x, y, z, t, L, c_len, lambda_h, v2
    )
    if not (dS > 0.0 and uni > math.exp(-dS)):
        h[x, y, z, t] = prop


@cuda.jit(device=True)
def site_plaq(U, x, y, z, t, L):
    acc = 0.0
    for mu in range(4):
        for nu in range(mu + 1, 4):
            x1, y1, z1, t1 = shift(x, y, z, t, mu, 1, L)
            x2, y2, z2, t2 = shift(x, y, z, t, nu, 1, L)
            pl = mm(
                mm(
                    mm(load(U, mu, x, y, z, t), load(U, nu, x1, y1, z1, t1)),
                    dag(load(U, mu, x2, y2, z2, t2)),
                ),
                dag(load(U, nu, x, y, z, t)),
            )
            acc += 0.5 * (pl[0].real + pl[3].real)
    return acc


@cuda.jit
def measure_kernel(U, h, partial, L, c_len):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    stride = cuda.blockDim.x * cuda.gridDim.x
    acc_p = 0.0
    acc_e = 0.0
    acc_e2 = 0.0
    acc_h = 0.0
    n = L * L * L * L
    i = bx * cuda.blockDim.x + tx
    while i < n:
        x = i % L
        y = (i // L) % L
        z = (i // (L * L)) % L
        t = i // (L * L * L)
        acc_p += site_plaq(U, x, y, z, t, L)
        for mu in range(4):
            ell = link_ell(U, mu, x, y, z, t, c_len)
            acc_e += ell
            acc_e2 += ell * ell
        hv = h[x, y, z, t]
        acc_h += math.sqrt(hv.real * hv.real + hv.imag * hv.imag)
        i += stride
    s0 = cuda.shared.array(THREADS, dtype=float64)
    s1 = cuda.shared.array(THREADS, dtype=float64)
    s2 = cuda.shared.array(THREADS, dtype=float64)
    s3 = cuda.shared.array(THREADS, dtype=float64)
    s0[tx] = acc_p
    s1[tx] = acc_e
    s2[tx] = acc_e2
    s3[tx] = acc_h
    cuda.syncthreads()
    offset = THREADS // 2
    while offset > 0:
        if tx < offset:
            s0[tx] += s0[tx + offset]
            s1[tx] += s1[tx + offset]
            s2[tx] += s2[tx + offset]
            s3[tx] += s3[tx + offset]
        cuda.syncthreads()
        offset >>= 1
    if tx == 0:
        partial[bx, 0] = s0[0]
        partial[bx, 1] = s1[0]
        partial[bx, 2] = s2[0]
        partial[bx, 3] = s3[0]


@cuda.jit
def staple_kernel(U, out, L):
    n = L * L * L * L
    i = cuda.grid(1)
    if i >= n:
        return
    x = i % L
    y = (i // L) % L
    z = (i // (L * L)) % L
    t = i // (L * L * L)
    for mu in range(4):
        s0, s1, s2, s3 = staple(U, mu, x, y, z, t, L)
        out[mu, x, y, z, t, 0, 0] = s0
        out[mu, x, y, z, t, 0, 1] = s1
        out[mu, x, y, z, t, 1, 0] = s2
        out[mu, x, y, z, t, 1, 1] = s3


@cuda.jit
def decode_kernel(out, parity, L):
    idx = cuda.grid(1)
    n_half = (L * L * L * L) // 2
    if idx >= n_half:
        return
    x, y, z, t = decode_half(idx, parity, L)
    out[idx, 0] = x
    out[idx, 1] = y
    out[idx, 2] = z
    out[idx, 3] = t


def _half_grid(L):
    n_half = (L * L * L * L) // 2
    blocks = (n_half + THREADS - 1) // THREADS
    return n_half, blocks


def _measure_blocks(L):
    n = L * L * L * L
    return min(1024, (n + THREADS - 1) // THREADS)


_PARTIAL = {}


def measure_means(U_d, h_d, L):
    blocks = _measure_blocks(L)
    partial = _PARTIAL.get((L, blocks))
    if partial is None:
        partial = cuda.device_array((blocks, 4), dtype=np.float64)
        _PARTIAL[(L, blocks)] = partial
    measure_kernel[blocks, THREADS](U_d, h_d, partial, np.int64(L), np.float64(C))
    host = partial.copy_to_host().sum(axis=0)
    n = float(L ** 4)
    mean_p = float(host[0] / (6.0 * n))
    mean_ell = float(host[1] / (4.0 * n))
    mean_ell2 = float(host[2] / (4.0 * n))
    mean_h = float(host[3] / n)
    xi = mean_ell2 - mean_ell * mean_ell
    return mean_p, mean_ell, xi, mean_h


def cold_fields(L):
    U = np.zeros((4, L, L, L, L, 2, 2), dtype=np.complex128)
    U[:] = np.eye(2)
    h = np.ones((L, L, L, L), dtype=np.complex128)
    return U, h


def gauge_sweep(U_d, beta, rng_states, L):
    _n_half, blocks = _half_grid(L)
    for mu in range(4):
        for parity in (0, 1):
            sweep_rng[blocks, THREADS](
                U_d,
                np.float64(beta),
                np.float64(EPS),
                rng_states,
                np.int64(mu),
                np.int64(parity),
                np.int64(L),
            )


def higgs_sweep(U_d, h_d, rng_states, L):
    _n_half, blocks = _half_grid(L)
    for parity in (0, 1):
        higgs_rng[blocks, THREADS](
            U_d,
            h_d,
            np.float64(EPS_H),
            rng_states,
            np.int64(parity),
            np.int64(L),
            np.float64(C),
            np.float64(LAMBDA_H),
            np.float64(V2),
        )


def run_point(L, beta, n_therm, n_meas, seed=SEED):
    U, h = cold_fields(L)
    U_d = cuda.to_device(U)
    h_d = cuda.to_device(h)
    rng_states = create_xoroshiro128p_states(L ** 4, seed=seed)
    sum_p = 0.0
    sum_ell = 0.0
    sum_xi = 0.0
    sum_h = 0.0
    t0 = time.time()
    total = n_therm + n_meas
    for step in range(total):
        gauge_sweep(U_d, beta, rng_states, L)
        higgs_sweep(U_d, h_d, rng_states, L)
        recording = step >= n_therm
        if recording or (step + 1) % 200 == 0 or step == 0:
            mean_p, mean_ell, xi, mean_h = measure_means(U_d, h_d, L)
            if recording:
                sum_p += mean_p
                sum_ell += mean_ell
                sum_xi += xi
                sum_h += mean_h
            if (step + 1) % 200 == 0 or step == 0 or step + 1 == total:
                elapsed = time.time() - t0
                phase = "meas" if recording else "therm"
                print(
                    f"L={L} beta={beta} step={step + 1}/{total} {phase} "
                    f"P={mean_p:.6f} ell={mean_ell:.6f} Xi={xi:.6e} "
                    f"|h|={mean_h:.6f} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    row = {
        "L": L,
        "beta": beta,
        "mean_P": sum_p / n_meas,
        "mean_ell": sum_ell / n_meas,
        "mean_Xi": sum_xi / n_meas,
        "mean_h": sum_h / n_meas,
        "geometric": bool((sum_xi / n_meas) < XI_STAR),
        "n_therm": n_therm,
        "n_meas": n_meas,
        "seed": seed,
        "time_s": time.time() - t0,
    }
    print(row, flush=True)
    return row


def fit_p_infinity(rows):
    """Unique quadratic in 1/beta through the three (beta, <P>) rows.

    The intercept is <P>(beta -> infinity) for this fit only.
    """
    design = np.empty((len(rows), 3), dtype=np.float64)
    target = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        inv = 1.0 / float(row["beta"])
        design[i, 0] = 1.0
        design[i, 1] = inv
        design[i, 2] = inv * inv
        target[i] = float(row["mean_P"])
    coeff = np.linalg.solve(design, target)
    return float(coeff[0]), float(coeff[1]), float(coeff[2])


def _fail(msg):
    raise SystemExit(msg)


def verify():
    if not cuda.is_available():
        _fail("no CUDA device")
    for L in (4, 16):
        for parity in (0, 1):
            n_half = (L ** 4) // 2
            seen = set()
            for idx in range(n_half):
                x, y, z, t = decode_half_host(idx, parity, L)
                if ((x + y + z + t) & 1) != parity:
                    _fail(f"host decode parity mismatch L={L}")
                seen.add((x, y, z, t))
            if len(seen) != n_half:
                _fail(f"host decode is not bijective L={L} parity={parity}")
            out = np.empty((n_half, 4), dtype=np.int32)
            blocks = (n_half + THREADS - 1) // THREADS
            decode_kernel[blocks, THREADS](out, np.int64(parity), np.int64(L))
            got = out.copy() if not hasattr(out, "copy_to_host") else out
            # kernel wrote a host array; numba copies it back on sync
            cuda.synchronize()
            for idx in range(0, n_half, max(1, n_half // 32)):
                x, y, z, t = decode_half_host(idx, parity, L)
                if tuple(int(v) for v in got[idx]) != (x, y, z, t):
                    _fail(f"device decode mismatch at {idx}: {got[idx]} vs {(x, y, z, t)}")
    print("decode ok", flush=True)

    rng = np.random.default_rng(1)
    L = 4
    U, h = cold_fields(L)
    for mu in range(4):
        for site in np.ndindex(L, L, L, L):
            U[(mu,) + site] = random_su2(rng, 0.4)
    h = rng.normal(size=(L, L, L, L)) + 1j * rng.normal(size=(L, L, L, L))

    U_d = cuda.to_device(U)
    out = np.empty_like(U)
    n = L ** 4
    staple_kernel[(n + THREADS - 1) // THREADS, THREADS](U_d, out, np.int64(L))
    cuda.synchronize()
    worst = 0.0
    for mu in range(4):
        for site in np.ndindex(L, L, L, L):
            cpu = staple_4d(U, mu, site, L)
            gpu = out[(mu,) + site]
            worst = max(worst, float(np.max(np.abs(cpu - gpu))))
    if worst > 1e-10:
        _fail(f"staple mismatch {worst}")
    print(f"staple ok max_abs={worst:.3e}", flush=True)

    h_d = cuda.to_device(h)
    gpu_p, gpu_ell, gpu_xi, gpu_h = measure_means(U_d, h_d, L)
    cpu_p = float(mean_plaquette(U, L))
    ell, cpu_xi = lengths_and_Xi(U)
    cpu_ell = float(ell.mean())
    cpu_h = float(np.mean(np.abs(h)))
    if abs(gpu_p - cpu_p) > 1e-10 or abs(gpu_ell - cpu_ell) > 1e-10:
        _fail(f"observable mismatch P {gpu_p} {cpu_p} ell {gpu_ell} {cpu_ell}")
    if abs(gpu_xi - cpu_xi) > 1e-10 or abs(gpu_h - cpu_h) > 1e-10:
        _fail(f"observable mismatch Xi {gpu_xi} {cpu_xi} h {gpu_h} {cpu_h}")
    print(
        f"observables ok P={gpu_p:.6f} ell={gpu_ell:.6f} Xi={gpu_xi:.6e} |h|={gpu_h:.6f}",
        flush=True,
    )

    # One checkerboard gauge update against the engine dS, all four directions.
    beta = 3.0
    U_work = U.copy()
    U_d = cuda.to_device(U_work)
    for mu in range(4):
        for parity in (0, 1):
            n_half = (L ** 4) // 2
            props = np.empty((n_half, 2, 2), dtype=np.complex128)
            unifs = rng.random(n_half)
            for idx in range(n_half):
                props[idx] = random_su2(rng, EPS)
            blocks = (n_half + THREADS - 1) // THREADS
            sweep_given[blocks, THREADS](
                U_d,
                np.float64(beta),
                props,
                unifs,
                np.int64(mu),
                np.int64(parity),
                np.int64(L),
            )
            cuda.synchronize()
            for idx in range(n_half):
                x, y, z, t = decode_half_host(idx, parity, L)
                old = U_work[mu, x, y, z, t].copy()
                prop = props[idx] @ old
                dS = -beta * np.real(np.trace((prop - old) @ staple_4d(U_work, mu, (x, y, z, t), L)))
                if dS <= 0.0 or unifs[idx] < np.exp(-dS):
                    U_work[mu, x, y, z, t] = prop
            gpu_U = U_d.copy_to_host()
            if not np.allclose(gpu_U, U_work, rtol=1e-10, atol=1e-10):
                _fail(f"checkerboard gauge mismatch mu={mu} parity={parity}")
            U_d = cuda.to_device(U_work)
    print("checkerboard gauge ok", flush=True)

    ell, _xi = lengths_and_Xi(U_work)
    h_work = h.copy()
    h_d = cuda.to_device(h_work)
    U_d = cuda.to_device(U_work)
    for parity in (0, 1):
        n_half = (L ** 4) // 2
        props_r = rng.normal(size=n_half)
        props_i = rng.normal(size=n_half)
        unifs = rng.random(n_half)
        # The engine adds eps_h * (normal + i normal) to the old field.
        blocks = (n_half + THREADS - 1) // THREADS
        # Build the proposed field values the way the kernel does: old + eps_h * g.
        proposed = np.empty(n_half, dtype=np.complex128)
        for idx in range(n_half):
            x, y, z, t = decode_half_host(idx, parity, L)
            proposed[idx] = h_work[x, y, z, t] + EPS_H * (props_r[idx] + 1j * props_i[idx])
        higgs_given[blocks, THREADS](
            U_d,
            h_d,
            np.ascontiguousarray(np.real(proposed)),
            np.ascontiguousarray(np.imag(proposed)),
            unifs,
            np.int64(parity),
            np.int64(L),
            np.float64(C),
            np.float64(LAMBDA_H),
            np.float64(V2),
        )
        cuda.synchronize()
        for idx in range(n_half):
            x, y, z, t = decode_half_host(idx, parity, L)
            site = (x, y, z, t)
            old_h = h_work[site]
            dS = higgs_action_local(h_work, site, U_work, ell, L)
            h_work[site] = proposed[idx]
            dS = higgs_action_local(h_work, site, U_work, ell, L) - dS
            if dS > 0.0 and unifs[idx] > np.exp(-dS):
                h_work[site] = old_h
        gpu_h = h_d.copy_to_host()
        if not np.allclose(gpu_h, h_work, rtol=1e-10, atol=1e-10):
            _fail(f"higgs mismatch parity={parity}")
    print("higgs ok", flush=True)

    U_id, h_id = cold_fields(L)
    p, ell_m, xi, h_m = measure_means(cuda.to_device(U_id), cuda.to_device(h_id), L)
    if abs(p - 1.0) > 1e-12 or abs(ell_m - 1.0) > 1e-12 or abs(xi) > 1e-12 or abs(h_m - 1.0) > 1e-12:
        _fail(f"cold observables {p} {ell_m} {xi} {h_m}")
    print("cold start ok", flush=True)


def write_scan(path="continuum_gpu.csv", n_therm=N_THERM, n_meas=N_MEAS, seed=SEED):
    if not cuda.is_available():
        _fail("no CUDA device")
    rows = []
    with open(path, "w", encoding="utf-8") as out:
        out.write(
            f"# engine=numba-cuda-4d schedule=checkerboard n_therm={n_therm} "
            f"n_meas={n_meas} seed={seed} eps={EPS} eps_h={EPS_H}\n"
        )
        out.write(
            "# dS = -beta * Re Tr((U' - U) @ staple), no factor 1/2. "
            "Stored <P> = (1/2) Re Tr U_p.\n"
        )
        out.write(
            "# staple: forward U_nu(x+mu) U_mu(x+nu)^+ U_nu(x)^+ ; "
            "backward U_nu(x+mu-nu)^+ U_mu(x-nu)^+ U_nu(x-nu)\n"
        )
        out.write(
            "# lengths, Xi, and quenched Higgs are the rift_4d.py formulas. "
            "GPU is NVIDIA GeForce RTX 3090.\n"
        )
        out.write("L,beta,mean_P,mean_ell,mean_Xi,mean_h,geometric\n")
        out.flush()
        for L, beta in POINTS:
            row = run_point(L, beta, n_therm, n_meas, seed)
            rows.append(row)
            flag = "true" if row["geometric"] else "false"
            out.write(
                f"{row['L']},{row['beta']},{row['mean_P']:.16g},"
                f"{row['mean_ell']:.16g},{row['mean_Xi']:.16g},"
                f"{row['mean_h']:.16g},{flag}\n"
            )
            out.flush()
        intercept, slope, quad = fit_p_infinity(rows)
        out.write(
            "# <P>(beta->inf) intercept is the constant term of the unique fit "
            "<P> = a + b/beta + c/beta^2 through these three rows only.\n"
        )
        out.write(
            f"# intercept_a={intercept:.16g} b={slope:.16g} c={quad:.16g}\n"
        )
        out.flush()
    return rows, (intercept, slope, quad)


def main(argv):
    if "--check" in argv:
        verify()
        return
    n_therm = N_THERM
    n_meas = N_MEAS
    if "--short" in argv:
        n_therm, n_meas = 20, 30
    rows, coeff = write_scan(n_therm=n_therm, n_meas=n_meas)
    print("fit_intercept", coeff[0], "b", coeff[1], "c", coeff[2], flush=True)
    print("rows", rows, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

import numpy as np


def gamma():
    s1 = np.array([[0.,1],[1,0]], dtype=complex)
    s2 = np.array([[0,-1j],[1j,0]], dtype=complex)
    s3 = np.array([[1.,0],[0,-1]], dtype=complex)
    z  = np.zeros((2,2), dtype=complex)
    I2 = np.eye(2, dtype=complex)
    def blk(a,b,c,d):
        return np.block([[a,b],[c,d]])
    return [blk(z, -1j*s1, 1j*s1, z),
            blk(z, -1j*s2, 1j*s2, z),
            blk(z, -1j*s3, 1j*s3, z),
            blk(z,  I2,    I2,    z)]
G = gamma()   # list of four 4×4 matrices


def apply_hop(psi, mass, L):
    """psi shape (L,L,L,L,4,3) or (L,L,L,L,4,2)"""
    out = (4.0 + mass) * psi
    shifts = [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]
    for mu, sh in enumerate(shifts):
        fwd = np.roll(psi, -1, axis=mu)
        bwd = np.roll(psi, +1, axis=mu)
        proj = 0.5 * (np.eye(4) - G[mu])
        fwd_s = np.einsum('ab,...bc->...ac', proj, fwd)
        bwd_s = np.einsum('ab,...bc->...ac', proj.conj().T, bwd)
        out -= fwd_s + bwd_s
    return out


def lanczos_min(mass, L=4, k=20):
    rng = np.random.default_rng(0)
    shape = (L,L,L,L,4,2)          # SU(2); use 3 for SU(3)
    q = rng.normal(size=shape) + 1j*rng.normal(size=shape)
    q /= np.linalg.norm(q)
    alphas, betas = [], []
    q_prev = np.zeros_like(q)
    beta = 0.0
    for i in range(k):
        v = apply_hop(q, mass, L) - beta * q_prev
        alpha = np.real(np.vdot(q, v))
        v -= alpha * q
        beta = np.linalg.norm(v)
        alphas.append(alpha)
        if i < k-1:
            betas.append(beta)
        q_prev = q
        q = v / beta if beta > 1e-14 else q
    T = np.diag(alphas) + np.diag(betas,1) + np.diag(betas,-1)
    return float(np.min(np.abs(np.linalg.eigvalsh(T))))

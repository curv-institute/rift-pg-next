# rift-pg-next
Next stage for RIFT-PG: production continuum and dynamical HMC.

`hmc_leapfrog.py` runs one gauge leapfrog on the six-staple 4D SU(2) engine (L=4, β=2.5, ε=0.05, 4 steps, cold identity links). Dirac links are identity, and the fermion operator is not coupled to the link update. The measured row is `hmc_one_row.csv`.
Eight independent cold starts, seeds 42–49, same ε and step count, are in `hmc_rows.csv`. Every proposal was rejected. The kept configuration stays cold, so ⟨P⟩ = 1 and min|E| is unchanged.
`hmc_accept.csv` repeats those seeds at ε=0.01 and 20 steps, so τ=ε n_steps stays 0.2; four proposals were accepted.

`continuum_gpu.csv` is the Numba-CUDA checkerboard Metropolis scan on an RTX 3090. The staple, the gauge weight, the stored plaquette, the lengths, Ξ, and the quenched Higgs are the formulas in `rift_4d.py`. The rows are L=16 β=3.0, L=16 β=4.0, and L=24 β=3.5, each with 3000 thermalization sweeps and 4000 measurement sweeps, seed 42. The intercept ⟨P⟩(β→∞) = 0.9992106117646175 is the constant term of the unique fit ⟨P⟩ = a + b/β + c/β² through these three rows only.
`unquenched_L8.csv` is four cold L=8 trajectories at β=2.5, ε=0.01, and 20 steps: the Euclidean norm of the CG residual of the identity-link `apply_hop` after 5 iterations sets the scale of one 3-vector per leapfrog step, which `exp_i_psigma` maps to one SU(2) matrix left-multiplied onto every link.
`unquenched_force_L8.csv` is one cold L=8 trajectory at β=2.5, ε=0.01, and 4 steps; this fermion-force sketch was not in the recorded repository.

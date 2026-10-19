# rift-pg-next
Next stage for RIFT-PG: production continuum and dynamical HMC.

`hmc_leapfrog.py` runs one gauge leapfrog on the six-staple 4D SU(2) engine (L=4, β=2.5, ε=0.05, 4 steps, cold identity links). Dirac links are identity, and the fermion operator is not coupled to the link update. The measured row is `hmc_one_row.csv`.
Eight independent cold starts, seeds 42–49, same ε and step count, are in `hmc_rows.csv`. Every proposal was rejected. The kept configuration stays cold, so ⟨P⟩ = 1 and min|E| is unchanged.
`hmc_accept.csv` repeats those seeds at ε=0.01 and 20 steps, so τ=ε n_steps stays 0.2; four proposals were accepted.

`continuum_gpu.csv` is the Numba-CUDA checkerboard Metropolis scan on an RTX 3090. The staple, the gauge weight, the stored plaquette, the lengths, Ξ, and the quenched Higgs are the formulas in `rift_4d.py`. The rows are L=16 β=3.0, L=16 β=4.0, and L=24 β=3.5, each with 3000 thermalization sweeps and 4000 measurement sweeps, seed 42. The intercept ⟨P⟩(β→∞) = 0.9992106117646175 is the constant term of the unique fit ⟨P⟩ = a + b/β + c/β² through these three rows only.
`unquenched_L8.csv` is four cold L=8 trajectories at β=2.5, ε=0.01, and 20 steps: the Euclidean norm of the CG residual of the identity-link `apply_hop` after 5 iterations sets the scale of one 3-vector per leapfrog step, which `exp_i_psigma` maps to one SU(2) matrix left-multiplied onto every link.
`unquenched_force_L8.csv` is one cold L=8 trajectory at β=2.5, ε=0.01, and 4 steps; this fermion-force sketch was not in the recorded repository.
`unquenched_force_gate.csv` is one cold L=4 trajectory at β=2.5, ε=10^{-4}, and 4 steps; the fermion force and Re φ†η use the same operator and the same tight solve.
`unquenched_force_L4_eps1e4.csv` is three cold L=4 trajectories at β=2.5, ε=10^{-4}, and seed 42, with 20, 100, and 400 steps; the fermion force and Re φ†η use the same operator and the same tight solve.
`unquenched_force_L4_chain.csv` is eight trajectories on one cold L=4 chain at β=2.5, ε=10^{-4}, and 400 steps, seed 42. φ and the momenta are redrawn each trajectory. The CG replaces a collapsed recursive residual with φ − D†D η and still exits only when the true residual is below 10^{-12}.
`unquenched_force_L4_tau4e-4.csv` is one trajectory of length τ=4×10^{-4}, with ε=4×10^{-4}/280 and 280 steps, from the accepted trajectory-1 endpoint of that chain. φ and the initial momenta are the ones held fixed for the coarser τ=4×10^{-4} runs.
`hasenbusch.py` splits det(D†D) with one Hasenbusch parameter μ². The quark mass in D stays 0.1. `hasenbusch_L4.csv` holds the τ=4×10^{-4} trajectories on the trajectory-1 endpoint, for μ² = 0.01, 0.1, 1, and 10, whose |ΔH| is below 0.1.
`hasenbusch_L4_tau004.csv` is one μ²=0.1 trajectory of length τ=0.04 on that endpoint, with ε=10^{-3} and 40 steps.
`hasenbusch_L4_chain.csv` continues from that endpoint with the same D, mass 0.1, μ²=0.1, ε=10^{-3}, and 40 steps. Both pseudofermions and the momenta are redrawn each trajectory. ⟨ℓ⟩ and Ξ are the `lengths_and_Xi` formulas on the kept links. ⟨|h|⟩ is the quenched Higgs on those frozen links, using the local action in `rift_4d.py`. The first eight trajectories are all accepted, and every |ΔH| is below 0.1.

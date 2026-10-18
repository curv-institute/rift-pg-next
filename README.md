# rift-pg-next
Next stage for RIFT-PG: production continuum and dynamical HMC.

`hmc_leapfrog.py` runs one gauge leapfrog on the six-staple 4D SU(2) engine (L=4, β=2.5, ε=0.05, 4 steps, cold identity links). Dirac links are identity, and the fermion operator is not coupled to the link update. The measured row is `hmc_one_row.csv`.
Eight independent cold starts, seeds 42–49, same ε and step count, are in `hmc_rows.csv`. Every proposal was rejected. The kept configuration stays cold, so ⟨P⟩ = 1 and min|E| is unchanged.

# rift-pg-next
Next stage for RIFT-PG: production continuum and dynamical HMC.

`hmc_leapfrog.py` runs one gauge leapfrog on the six-staple 4D SU(2) engine (L=4, β=2.5, ε=0.05, 4 steps, cold identity links). Dirac links are identity, and the fermion operator is not coupled to the link update. The measured row is `hmc_one_row.csv`.

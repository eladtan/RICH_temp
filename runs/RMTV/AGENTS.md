# Execution policy

The user requires simulations to be submitted through the cluster scheduler.
Do not launch RMTV simulations directly on gateway/login nodes, including
diagnostic runs. Use Slurm `sbatch`; the requested configuration is partition
`bigrun`, 16 CPU cores unless the user specifies otherwise.

The already-running local 24^3 convergence job is explicitly authorized to
finish. Do not stop it when applying this policy to new runs.

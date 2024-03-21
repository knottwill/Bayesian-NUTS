"""!@file

@brief Script producing all figures and results from the report.
"""
import numpy as np
from src.simulation import *

np.random.seed(0)

#############
# Part iii: Simulation Study
#############

print('Running simulation study for part iii...')
sample_sizes = np.logspace(4, 6.5, 25, dtype=int)
M_samples = 50

# simulation study for standard Cauchy distribution
study_1 = run_simulation_study(
    alpha=0,
    beta=1,
    sample_sizes=sample_sizes,
    M_samples=M_samples
)

# simulation study for Cauchy distribution with location parameter alpha=1, scale parameter beta=2
study_2 = run_simulation_study(
    alpha=1,
    beta=2,
    sample_sizes=sample_sizes,
    M_samples=M_samples
)

fig = plot_simulation_results(study_1, study_2)
save_path = 'plots/simulation.png'
fig.savefig(save_path)
print(f"Saved simulation study (part iii) plot to {save_path}")
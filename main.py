"""!@file main.py

@brief Script producing all figures and results from the report.
"""
import numpy as np
import arviz as az
import os 
from os.path import join
from time import time
import argparse

t0 = time()

from src.simulation import run_simulation_study, plot_simulation_results
from src.sampling import sample_part_v, sample_part_vii
from src.statistics import compute_posterior_statistics
from src.visualisations import *

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="Lighthouse dataset (containing flash locations and intensities)")
parser.add_argument("--output_dir", default='plots', type=str, help="output directory to save figures in")
args = parser.parse_known_args()[0]
os.makedirs(args.output_dir, exist_ok=True) # create output directory if it doesn't exist

# load observed data
data = np.loadtxt(args.data)
x_observed = data[:, 0] # flash positions
I_observed = data[:, 1] # flash intensities

# set random seed
SEED = 42
np.random.seed(SEED)

# function for saving figures
def save_fig(fig, save_path):
    fig.savefig(save_path, bbox_inches='tight')
    print(f"Figure saved at {save_path}")

#############
# Part iii: Simulation Study
#############

print('\nRunning simulation study for part iii...')
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

# plot results
fig = plot_simulation_results(study_1, study_2)
save_fig(fig, join(args.output_dir, 'simulation-iii.png'))

#############
# Part V
#############
print("\n#######################")
print("-----  Part V  -------\n")

idata_v = sample_part_v(
    x_observed=x_observed,
    chains=4,
    draws=10000, # number of draws per chain after tuning & burn-in
    tune=500, # number of tuning steps
    burnin=50, # number of burn-in steps
    SEED=SEED
)

alpha = idata_v.posterior['alpha'].values
beta = idata_v.posterior['beta'].values
N = len(alpha.flatten())

# Calculate the Gelman-Rubin statistic for alpha and beta, specifying the split method
print(f"\nGelman-Rubin statistic for alpha: {az.rhat(alpha, method='split')}")
print(f"Gelman-Rubin statistic for beta: {az.rhat(beta, method='split')}\n")

# Compute mean, std, mcse and integrated autocorrelation time for alpha and beta
alpha_mean, alpha_std, alpha_mcse, alpha_tau = compute_posterior_statistics(alpha)
beta_mean, beta_std, beta_mcse, beta_tau = compute_posterior_statistics(beta)

print(f"Estimated integrated autocorrelation time for alpha: {alpha_tau}")
print(f"Estimated integrated autocorrelation time for beta: {beta_tau}\n")

print("Parameter Quotes (Part V):")
print(f"alpha = {alpha_mean:.5f} ± {alpha_std:.5f} (mean ± std), with standard error on the mean of {alpha_mcse:.5f}")
print(f"beta = {beta_mean:.5f} ± {beta_std:.5f} (mean ± std), with standard error on the mean of {beta_mcse:.5f}\n")

# plot traces for the first 500 samples of the first chain
fig = plot_traces(
    alpha=alpha[0][:500],
    beta=beta[0][:500],
)
save_fig(fig, join(args.output_dir, 'traces-v.png'))

# plot marginal distributions
fig = plot_marginals(alpha, beta)
save_fig(fig, join(args.output_dir, 'marginals-v.png'))

# plot joint distribution
fig = plot_joint(alpha, beta)
save_fig(fig, join(args.output_dir, 'joint-v.png'))

# plot autocorrelations
fig = plot_autocorrelations(alpha, beta)
save_fig(fig, join(args.output_dir, 'autocorrelations-v.png'))

#############
# Part VII
#############
print("\n######################")
print("----- Part VII -------\n")

idata_vii = sample_part_vii(
    x_observed=x_observed, 
    I_observed=I_observed, 
    chains=4, # number of chains
    draws=10000, # number of draws per chain after tuning & burn-in
    tune=500, # number of tuning steps
    burnin=50, # number of burn-in steps
    SEED=SEED)

alpha = idata_vii.posterior['alpha'].values
beta = idata_vii.posterior['beta'].values
I_0 = idata_vii.posterior['I_0'].values
N = len(alpha.flatten())

# Calculate the Gelman-Rubin statistic for alpha and beta, specifying the split method
print(f"\nGelman-Rubin statistic for alpha: {az.rhat(alpha, method='split')}")
print(f"Gelman-Rubin statistic for beta: {az.rhat(beta, method='split')}")
print(f"Gelman-Rubin statistic for I_0: {az.rhat(I_0, method='split')}\n")

# Compute mean, std, mcse and integrated autocorrelation time for alpha, beta and I_0
alpha_mean, alpha_std, alpha_mcse, alpha_tau = compute_posterior_statistics(alpha)
beta_mean, beta_std, beta_mcse, beta_tau = compute_posterior_statistics(beta)
I_0_mean, I_0_std, I_0_mcse, I_0_tau = compute_posterior_statistics(I_0)

print(f"Estimated integrated autocorrelation time for alpha: {alpha_tau}")
print(f"Estimated integrated autocorrelation time for beta: {beta_tau}")
print(f"Estimated integrated autocorrelation time for I_0: {I_0_tau}\n")

print("Parameter Quotes (Part VII):")
print(f"alpha = {alpha_mean:.5f} ± {alpha_std:.5f} (mean ± std), with standard error on the mean of {alpha_mcse:.5f}")
print(f"beta = {beta_mean:.5f} ± {beta_std:.5f} (mean ± std), with standard error on the mean of {beta_mcse:.5f}")
print(f"I_0 = {I_0_mean:.5f} ± {I_0_std:.5f} (mean ± std), with standard error on the mean of {I_0_mcse:.5f}\n")

# plot traces for the first 500 samples of the first chain
fig = plot_traces(
    alpha=alpha[0][:500],
    beta=beta[0][:500],
    I_0=I_0[0][:500],
)
save_fig(fig, join(args.output_dir, 'traces-vii.png'))

# plot marginal distributions
fig = plot_marginals(alpha, beta, I_0)
save_fig(fig, join(args.output_dir, 'marginals-vii.png'))

# plot joint distributions (corner plot)
fig = plot_joint_corner(alpha, beta, I_0)
save_fig(fig, join(args.output_dir, 'joint-corner-vii.png'))

# plot autocorrelations
fig = plot_autocorrelations(alpha, beta, I_0)
save_fig(fig, join(args.output_dir, 'autocorrelations-vii.png'))

print(f"\nTotal time taken: {time() - t0:.5f} seconds")
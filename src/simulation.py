"""!@file simulation.py

@brief Contains all functionality for running the simulation study discussed in part (iii) of the report.

@details The simulation study is designed to compare the maximum liklihood estimate (MLE) and the sample
mean, as an estimator for the location parameter, \f$ \alpha \f$, of the Cauchy distribution. For a given
sample size, N, the study generates M samples of size N from a Cauchy distribution (with location 
parameter \f$ \alpha \f$ and scale parameter \f$ \beta \f$ specified by the user). For each sample,
the MLE is computed using the Newton-Raphson method, and the sample mean is computed. The study then
computes the mean squared error (MSE) of the MLE and the sample mean, over all M samples. 
This process is repeated for a range of sample sizes, N.

We do this for two sets of parameters (Standard cauchy: \f$ \alpha = 0, \beta = 1\f$ and \f$ \alpha = 1, \beta = 2\f$), 
and plot the results.
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import newton
from tqdm import tqdm

def cauchy_sample(N, alpha, beta):
    """!
    @brief Generate a sample of size N from a Cauchy distribution with location parameter alpha and scale parameter beta
    
    @details This is done by generating uniform random numbers in (-pi/2, pi/2) and transforming them to Cauchy random numbers 
    using the equation x = alpha + beta*tan(theta), where theta is the uniform random number.
    """
    # generate uniform random numbers in (-pi/2, pi/2)
    theta = np.pi*(np.random.rand(N) - 1/2)
    return alpha + beta*np.tan(theta)

def find_MLE(alpha_0, x, beta):
    """!
    @brief Find maximum liklihood estimate (MLE) of alpha for Cauchy distribution, using Newton-Raphson method

    @details The log-likelihood function for the Cauchy distribution is given by:
    \f[
    l(\alpha) = N\log(\beta) - N\log(\pi) - \sum_{k=1}^{N}\log(\beta^2 + \left(x_k - \alpha\right)^2)
    \f]
    where \f$ x_i \f$ are the sample points. The first derivative of the log-likelihood function is:
    \f[
    \frac{\partial l}{\partial\alpha} = \sum_{k=1}^{N}\frac{2(x_k - \alpha)}{\beta^2 + (x_k - \alpha)^2}
    \f]
    The second derivative of the log-likelihood function is:
    \f[
    \frac{{\partial^2}l}{{\partial\alpha^2}} = -2\sum_{k=1}^{N}\frac{\beta^2 - (x_k - \alpha)^2}{(\beta^2 + (x_k - \alpha)^2)^2}
    \f]
    The MLE of \f$ \alpha \f$ is found by solving the equation \f$ \frac{dl}{d\alpha} = 0 \f$ using the Newton-Raphson method.

    @param alpha_0: initial guess for MLE
    @param x: sample points
    @param beta: scale parameter of Cauchy distribution

    @return mle: MLE of alpha
    """
    
    # define first and second derivatives of log-likelihood function
    def l1(alpha):
        """ First Derivative of log-likelihood function """
        numer = 2*(x - alpha)
        denom = beta**2 + (x - alpha)**2
        return np.sum(numer/denom)

    def l2(alpha):
        """ Second Derivative of log-likelihood function """
        numer = -2*(beta**2 - (x - alpha)**2)
        denom = (beta**2 + (x - alpha)**2)**2
        return np.sum(numer/denom)
    
    # compute MLE using Newton-Raphson method
    mle = newton(func=l1, x0=alpha_0, fprime=l2)
    
    return mle

def run_simulation_study(alpha, beta, sample_sizes, M_samples):
    """! 
    @brief Run simulation study for part iii of report

    @details Runs the study described in the module docstring: For a range of sample sizes,
    we generate M_samples samples, compute the MLE and sample mean for each sample, and
    compute the mean squared error (MSE) of the MLE and sample mean over all samples.

    @param alpha: location parameter of Cauchy distribution
    @param beta: scale parameter of Cauchy distribution
    @param sample_sizes: sample sizes to simulate
    @param M_samples: number of samples to simulate for each sample size

    @return results: dictionary containing results of simulation study
    """

    MLE_MSE, sample_mean_MSE = [], []
    for N in tqdm(sample_sizes): # loop over sample sizes

        MLE, sample_means = np.zeros(M_samples), np.zeros(M_samples)
        for i in range(M_samples): # loop over samples

            # generate sample and compute sample mean & MLE
            x_sample = cauchy_sample(N, alpha, beta)
            sample_means[i] = np.mean(x_sample)
            MLE[i] = find_MLE(0, x_sample, beta)

        # compute mean squared error
        mle_mse = np.mean((MLE - alpha)**2)
        sample_mean_mse = np.mean((sample_means - alpha)**2)
        MLE_MSE.append(mle_mse)
        sample_mean_MSE.append(sample_mean_mse)
        
    # put results in dictionary
    results = {'Sample Size': sample_sizes, 'MLE MSE': MLE_MSE, 'Sample Mean MSE': sample_mean_MSE, 'alpha': alpha, 'beta': beta}

    return results

def plot_simulation_results(study_1, study_2):
    """!
    @brief Plot results of two simulation studies from part iii of report

    @param study_1: dictionary containing results of simulation study for standard Cauchy distribution
    @param study_2: dictionary containing results of simulation study for Cauchy distribution with alpha=1, beta=2

    @return fig: figure object
    """

    # on 4 axes, plot the Mean and Std of the MLE and sample mean
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # plot MLE MSE for first study
    ax[0, 0].scatter(study_1['Sample Size'], study_1['MLE MSE'], label=rf"$\alpha={study_1['alpha']}, \beta={study_1['beta']}$")
    ax[0, 0].axhline(y=0, color='r', linestyle='--')
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_xlabel('Sample Size', fontsize=16)
    ax[0, 0].set_ylabel('MLE MSE', fontsize=16)
    ax[0, 0].text(0, 1.05, "(a)", fontsize=16, transform=ax[0, 0].transAxes, fontweight="bold")
    ax[0, 0].legend()

    # plot sample mean MSE for first study
    ax[0, 1].scatter(study_1['Sample Size'], study_1['Sample Mean MSE'], label=rf"$\alpha={study_1['alpha']}, \beta={study_1['beta']}$")
    ax[0, 1].axhline(y=0, color='r', linestyle='--')
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_yscale('log')
    ax[0, 1].set_xlabel('Sample Size', fontsize=16)
    ax[0, 1].set_ylabel('Sample Mean MSE', fontsize=16)
    ax[0, 1].text(0, 1.05, "(b)", fontsize=16, transform=ax[0, 1].transAxes, fontweight="bold")
    ax[0, 1].legend()

    # plot MLE MSE for second study
    ax[1, 0].scatter(study_2['Sample Size'], study_2['MLE MSE'], label=rf"$\alpha={study_2['alpha']}, \beta={study_2['beta']}$")
    ax[1, 0].axhline(y=0, color='r', linestyle='--')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_xlabel('Sample Size', fontsize=16)
    ax[1, 0].set_ylabel('MLE MSE', fontsize=16)
    ax[1, 0].text(0, 1.05, "(c)", fontsize=16, transform=ax[1, 0].transAxes, fontweight="bold")
    ax[1, 0].legend()

    # plot sample mean MSE for second study
    ax[1, 1].scatter(study_2['Sample Size'], study_2['Sample Mean MSE'], label=rf"$\alpha={study_2['alpha']}, \beta={study_2['beta']}$")
    ax[1, 1].axhline(y=0, color='r', linestyle='--')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_yscale('log')
    ax[1, 1].set_xlabel('Sample Size', fontsize=16)
    ax[1, 1].set_ylabel('Sample Mean MSE', fontsize=16)
    ax[1, 1].text(0, 1.05, "(d)", fontsize=16, transform=ax[1, 1].transAxes, fontweight="bold")
    ax[1, 1].legend()

    plt.tight_layout()

    return fig
"""!@file statistics.py

@brief Module containing function to compute the posterior statistics of the chains of a single parameter.
"""

import numpy as np
from emcee.autocorr import integrated_time

def compute_posterior_statistics(chains):
    """!
    @brief Compute the posterior statistics for a given set of chains (of 1 parameter).

    @details Given a set of markov chains with the burn-in removed, this function:
    - estimates IAT
    - computes the effective sample size
    - computes the posterior mean and standard deviation
    - computes the Markov chain standard error on the mean using the formula: mcse = std / sqrt(ESS)
    We do not thin the chains since it is usually considered best practice to use all samples, 
    except to with memory restrictions (see https://www.researchgate.net/publication/230547481_On_thinning_of_chains_in_MCMC). 
    Additionally, using the NUTS sampler has an extremely low autocorrelation for this study (< 2), hence
    thinning would actually cause the loss of valuable statistics, leading to a less accurate estimation of the posterior statistics.

    @param chains: The set of markov chains with the burn-in removed.

    @return mean: The posterior mean.
    @return std: The posterior standard deviation.
    @return mcse: The Markov chain standard error on the mean.
    @return tau: The integrated autocorrelation time.
    """

    # Estimate integrated autocorrelation time
    taus = [integrated_time(chain, tol=0) for chain in chains]
    tau = np.mean(taus)

    # Flatten the chains
    chains = chains.flatten()

    # Compute the effective sample size
    ESS = len(chains) / tau

    # Compute the posterior mean and standard deviation
    mean = np.mean(chains)
    std = np.std(chains, ddof=1)

    # Markov chain standard error on the mean
    mcse = std / np.sqrt(ESS)

    return mean, std, mcse, tau

"""!@file sampling.py

@brief Functions for sampling from the posterior of parts v and vii in the report, using NUTS sampler.
"""

import pymc as pm

def sample_part_v(x_observed, chains=4, tune=500, burnin=50, draws=50000, SEED=None):
    """! 
    @brief Sampling from the posterior of part v in the report, using NUTS sampler.

    @details This function samples from the posterior of alpha and beta, given the observed positions
    of flashes, x_observed. We use an improper flat prior over the reals for alpha, and an improper flat
    prior over the POSITIVE reals for beta. The likelihood is a Cauchy distribution. We use the NUTS sampler
    to sample from the posterior.

    @param x_observed (np.array): observed positions of flashes
    @param chains (int): number of chains
    @param tune (int): number of tuning steps
    @param burnin (int): number of burn-in steps (after tuning)
    @param draws (int): number of draws (steps after burn-in)
    @param SEED (int): random seed

    @return idata (az.InferenceData): Object containing the traces and other information.
    """

    with pm.Model():

        # prior for alpha (improper flat prior over the reals)
        alpha = pm.Flat('alpha')

        # prior for beta (improper flat prior over the positive reals)
        beta = pm.HalfFlat('beta')

        # Likelihood for x (Cauchy distribution)
        x = pm.Cauchy('x', alpha=alpha, beta=beta, observed=x_observed)

        # Initialise NUTS sampler
        initial_points, step = pm.init_nuts(init='auto', 
                                            chains=chains, 
                                            random_seed=SEED)

        # Sample from the posterior
        idata = pm.sample(draws=burnin + draws,  
                          tune=tune, 
                          step=step, 
                          initvals=initial_points, 
                          random_seed=SEED)

        # discard burn-in
        idata = idata.sel(draw=slice(burnin, None))

    return idata


def sample_part_vii(x_observed, I_observed, chains=4, tune=200, burnin=50, draws=50000, SEED=None):
    """!
    @brief Sampling from the posterior of part vii in the report, using NUTS sampler.

    @details This function samples from the posterior of alpha, beta, and I_0, given the observed positions
    of flashes, x_observed, and the observed intensities of flashes, I_observed. We use an improper flat prior over the reals for alpha,
    an improper flat prior over the POSITIVE reals for beta, and improper Jeffreys prior for I_0 (prior ~ 1/I_0). Jeffrey's prior is created
    by using an improper flat prior over the reals for log(I_0), then transforming it to I_0 using the exponential function.
    We sample from the posterior using the NUTS sampler (tuning then burn-in then draws). 

    @param x_observed (np.array): observed positions of flashes
    @param I_observed (np.array): observed intensities of flashes
    @param chains (int): number of chains
    @param tune (int): number of tuning steps
    @param burnin (int): number of burn-in steps (after tuning)
    @param draws (int): number of draws (steps after burn-in)
    @param SEED (int): random seed

    @return idata (az.InferenceData): Object containing the traces and other information.
    """

    with pm.Model():

        # prior for alpha
        alpha = pm.Flat('alpha') # non-informative

        # prior for beta
        beta = pm.HalfFlat('beta') # non-informative

        # prior for I_0 
        log_I_0 = pm.Flat('log_I_0')
        I_0 = pm.Deterministic('I_0', pm.math.exp(log_I_0)) # jeffreys prior - non-informative

        # Likelihood for x (Cauchy distribution)
        x = pm.Cauchy('x', alpha=alpha, beta=beta, observed=x_observed)

        # Likelihood for I given x (Log-normal distribution)
        mu = pm.math.log(I_0 / (beta**2 + (x_observed - alpha)**2))
        I = pm.LogNormal('I', mu=mu, sigma=1, observed=I_observed)

        # Initialise NUTS sampler
        initial_points, step = pm.init_nuts(init='auto', 
                                            chains=chains, 
                                            random_seed=SEED)

        # Sample from the posterior
        idata = pm.sample(draws=burnin + draws,  
                          tune=tune, 
                          step=step, 
                          initvals=initial_points, 
                          random_seed=SEED)

        # discard burn-in
        idata = idata.sel(draw=slice(burnin, None))

    return idata
        
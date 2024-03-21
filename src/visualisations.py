"""!@file visualisations.py

@brief Module containing all visualisation functions for the posterior samples of part v and vii.

@details The module contains functions to plot the marginal distributions, joint distributions, traces, 
and autocorrelations of the parameters alpha, beta, and I_0.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

def plot_marginals(alpha, beta, I_0=None):
    """!
    @brief Plot the marginal distributions of alpha, beta, and (optionally) I_0.

    @param alpha: The alpha parameter samples of shape (n_chains, n_samples)
    @param beta: The beta parameter samples of shape (n_chains, n_samples)
    @param I_0: The I_0 parameter samples of shape (n_chains, n_samples)

    @return fig: The figure object
    """

    # flatten alpha & beta (all samples from all chains used for plotting)
    alpha, beta = alpha.flatten(), beta.flatten()

    n_parameters = 3 if I_0 is not None else 2
    fig, ax = plt.subplots(1, n_parameters, figsize=(5*n_parameters, 5))

    # Marginal distribution of alpha
    sns.histplot(alpha, stat='density', ax=ax[0], kde=True)
    ax[0].set_xlabel(r'Location $\alpha$', fontsize=18)
    ax[0].set_ylabel('Density', fontsize=18)
    ax[0].text(0.02, 0.89, "(a)", fontsize=16, transform=ax[0].transAxes, fontweight="bold")

    # Marginal distribution of beta
    sns.histplot(beta, stat='density', ax=ax[1], kde=True, color='#ff7f0e')
    ax[1].set_xlabel(r'Distance $\beta$', fontsize=18)
    ax[1].set_ylabel('Density', fontsize=18)
    ax[1].text(0.02, 0.89, "(b)", fontsize=16, transform=ax[1].transAxes, fontweight="bold")

    # Marginal distribution of I_0
    if I_0 is not None:
        I_0 = I_0.flatten()
        sns.histplot(I_0, stat='density', ax=ax[2], kde=True, color='#2ca02c')
        ax[2].set_xlabel(r'Source Intensity $I_0$', fontsize=18)
        ax[2].set_ylabel('Density', fontsize=18)
        ax[2].text(0.02, 0.89, "(c)", fontsize=16, transform=ax[2].transAxes, fontweight="bold")

    plt.tight_layout()

    return fig


def plot_joint(alpha, beta):
    """!
    @brief Plot 2D joint distribution of alpha and beta

    @details The joint distribution is plotted as a scatter plot overlaid by a 2D histogram.
    """
    # flatten alpha & beta (all samples from all chains used for plotting)
    alpha, beta = alpha.flatten(), beta.flatten()

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(x=alpha, y=beta, color='black', alpha=1, s=2) # scatter plot

    # 2D histogram
    sns.histplot(x=alpha, y=beta, bins=40, stat='density', cbar=True, thresh=0.02, cmap='viridis', ax=ax, 
                cbar_kws={'label': 'Probability Density'},
                )
    ax.set_xlabel(r'Location $\alpha$', fontsize=18)
    ax.set_ylabel(r'Distance $\beta$', fontsize=18)
    ax.figure.axes[-1].yaxis.label.set_size(16) # set colorbar label size

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()  

    return fig


def plot_joint_corner(alpha, beta, I_0):
    """!
    @brief Plot 3 2D joint distributions between alpha, beta, and I_0 (corner plot).

    @details The joint distributions are plotted as scatter plots overlaid by 2D histograms.
    """
    # flatten alpha, beta, I_0 (all samples from all chains used for plotting)
    alpha, beta, I_0 = alpha.flatten(), beta.flatten(), I_0.flatten()

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Hide the top left subplot as we won't use it
    axs[0,1].axis('off')

    # Beta vs Alpha
    axs[0,0].scatter(x=alpha, y=beta, color='black', alpha=1, s=2)
    sns.histplot(x=alpha, y=beta, bins=40, stat='density', cbar=True, thresh=0.02, cmap='viridis', ax=axs[0,0], 
                 cbar_kws={'label': 'Probability Density'})
    axs[0,0].set_xlabel('')
    axs[0,0].set_xticklabels('')
    axs[0,0].set_ylabel(r'Distance $\beta$', fontsize=18)
    axs[0,0].figure.axes[-1].yaxis.label.set_size(16)
    axs[0,0].text(0.02, 0.89, "(a)", fontsize=16, transform=axs[0,0].transAxes, fontweight="bold")

    # I_0 vs Alpha
    axs[1, 0].scatter(x=alpha, y=I_0, color='black', alpha=1, s=2)
    sns.histplot(x=alpha, y=I_0, bins=40, stat='density', cbar=True, thresh=0.005, cmap='viridis', ax=axs[1, 0])
    axs[1, 0].set_xlabel(r'Location $\alpha$', fontsize=18)
    axs[1, 0].set_ylabel(r'Source Intensity $I_0$', fontsize=18)
    axs[1, 0].figure.axes[-1].yaxis.label.set_size(16)
    axs[1, 0].text(0.02, 0.89, "(c)", fontsize=16, transform=axs[1,0].transAxes, fontweight="bold")

    # I_0 vs Beta
    axs[1, 1].scatter(x=beta, y=I_0, color='black', alpha=1, s=2)
    sns.histplot(x=beta, y=I_0, bins=40, stat='density', cbar=True, thresh=0.005, cmap='viridis', ax=axs[1, 1], 
                 cbar_kws={'label': 'Probability Density'})
    axs[1, 1].set_xlabel(r'Distance $\beta$', fontsize=18)
    axs[1, 1].set_ylabel('')
    axs[1, 1].set_yticklabels('')
    axs[1, 1].figure.axes[-1].yaxis.label.set_size(16)
    axs[1, 1].text(0.02, 0.89, "(d)", fontsize=16, transform=axs[1,1].transAxes, fontweight="bold")
    
    fig.tight_layout()

    return fig


def plot_traces(alpha, beta, I_0=None):
    """!
    @brief Plot the traces of alpha, beta, and (optionally) I_0.

    @details The parameters should be just a single section of one chain, 
    for a more appealing/interpretable visualisation.

    @param alpha: The alpha parameter samples of shape (n_samples,)
    @param beta: The beta parameter samples of shape (n_samples,)
    @param I_0: The I_0 parameter samples of shape (n_samples)

    @return fig: The figure object
    """

    n_parameters = 3 if I_0 is not None else 2
    fig, ax = plt.subplots(n_parameters, 1, figsize=(7, 3*n_parameters))

    # Trace of alpha
    ax[0].plot(alpha, label=r'$\alpha$')
    ax[0].set_xlabel('')
    ax[0].set_xticklabels('')
    ax[0].set_ylabel(r'Location $\alpha$', fontsize=18)
    ax[0].text(0.02, 0.89, "(a)", fontsize=16, transform=ax[0].transAxes, fontweight="bold")

    # Trace of beta
    ax[1].plot(beta, label=r'$\beta$', color='#ff7f0e')
    ax[1].set_xlabel('Iteration', fontsize=18)
    ax[1].set_ylabel(r'Distance $\beta$', fontsize=18)
    ax[1].text(0.02, 0.89, "(b)", fontsize=16, transform=ax[1].transAxes, fontweight="bold")

    # Trace of I_0
    if I_0 is not None:
        ax[1].set_xlabel('') # take x label off ax[1]

        ax[2].plot(I_0, label=r'$I_0$', color='#2ca02c')
        ax[2].set_xlabel('Iteration', fontsize=18)
        ax[2].set_ylabel(r'Source Intensity $I_0$', fontsize=18)
        ax[2].text(0.02, 0.89, "(c)", fontsize=16, transform=ax[2].transAxes, fontweight="bold")

    plt.tight_layout()

    return fig


def plot_autocorrelations(alpha, beta, I_0=None, maxlag=30):
    """!
    @brief Plot the autocorrelation of the chains for alpha, beta, and I_0.

    @details The autocorrelation is plotted for each chain, up to a maximum lag of `maxlag`.

    @param alpha: The alpha parameter samples of shape (n_chains, n_samples)
    @param beta: The beta parameter samples of shape (n_chains, n_samples)
    @param I_0: The I_0 parameter samples of shape (n_chains, n_samples)
    @param maxlag: The maximum lag to plot the autocorrelation up to

    @return fig: The figure object
    """

    n_parameters = 3 if I_0 is not None else 2
    fig, ax = plt.subplots(1, n_parameters, figsize=(5*n_parameters, 5))

    # Autocorrelation of alpha
    autocorr_alpha = az.autocorr(alpha) # shape (n_chains, n_samples)
    for chain, rho in enumerate(autocorr_alpha):

        rho = rho[:maxlag] # only plot up to maxlag

        ax[0].plot(rho, label=f'Chain {chain+1}')
        ax[0].axhline(0, color='black', lw=0.5, ls='--') # add horizontal line at 0
        ax[0].set_xlabel('Lag', fontsize=18)
        ax[0].set_ylabel(r'$\rho_\alpha$', fontsize=18)
        ax[0].text(0.15, 0.89, "(a)", fontsize=16, transform=ax[0].transAxes, fontweight="bold")
        ax[0].legend()

    # Autocorrelation of beta
    autocorr_beta = az.autocorr(beta) # shape (n_chains, n_samples)
    for chain, rho in enumerate(autocorr_beta):

        rho = rho[:maxlag] # only plot up to maxlag

        ax[1].plot(rho, label=f'Chain {chain+1}')
        ax[1].axhline(0, color='black', lw=0.5, ls='--')
        ax[1].set_xlabel('Lag', fontsize=18)
        ax[1].set_ylabel(r'$\rho_\beta$', fontsize=18)
        ax[1].text(0.15, 0.89, "(b)", fontsize=16, transform=ax[1].transAxes, fontweight="bold")
        ax[1].legend()

    # Autocorrelation of I_0
    if I_0 is not None:
        autocorr_I_0 = az.autocorr(I_0) # shape (n_chains, n_samples)
        for chain, rho in enumerate(autocorr_I_0):
            rho = rho[:maxlag]
            ax[2].plot(rho, label=f'Chain {chain+1}')
            ax[2].axhline(0, color='black', lw=0.5, ls='--')
            ax[2].set_xlabel('Lag', fontsize=18)
            ax[2].set_ylabel(r'$\rho_{I_0}$', fontsize=18)
            ax[2].text(0.15, 0.89, "(c)", fontsize=16, transform=ax[2].transAxes, fontweight="bold")
            ax[2].legend()
    
    return fig
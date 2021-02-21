import numpy as np


def _whiten(Y):
    return Y

def _initial_guess(Y):
    Xi_i = []
    A_i = []
    epsilon_i1 = []
    epsilon_i2 = []
    s = []
    return Xi_i, A_i, epsilon_i1, epsilon_i2, s

def compute_hpica(Y, n_components, n_iter=10, whiten=True, random_state=None):
    """
    Y needs to be a list of datasets of shape (n_samples, n_features)

    Fits two-level model as introduced Guo et al 2013. 

    The first, subject-specific, level is defined with:

    y_i (v) = A_i * Xi_i (v) + epsilon_i1 (v),

    where

    v denotes sample index (v for voxel in fMRI),
    i denotes subject index,
    y_i denotes subject-specific observations,
    A_i denotes subject-specific mixing matrix,
    Xi_i denotes subject-specific source signals, and
    epsilon_i1 denotes gaussian noise unexplained noise.

    epsilon_i1 ~ N(0, sigma_1), and 
    sigma_1 = std_1*I (i.e isotropic covariance).

    The second level is defined with:

    Xi_i (v) = s (v) + epsilon_i2 (v),

    where

    epsilon_i2 denotes subject-specific deviation 
      from population-level source signals. 
      epsilon_i2 ~ N(0, sigma_2), and
      sigma_2 is a diagonal matrix with each entry
      representing between-subject variability for
      corresponding source.
    s denotes population-level source signals, with 
      each source signal is specified with a mixture of Gaussians,
    

    We need to solve for Xi_i, A_i, epsilon_i1, epsilon_i2, 
      and s (which includes means, standard deviations and the 
      membership variable z of gaussian mixtures). 
      These are solved with an EM algorithm.

    """

    # Demean and whiten the input variables
    if whiten:
        Y = _whiten(Y)

    # Get initial guess
    Xi_i, A_i, epsilon_i1, epsilon_i2, s = _initial_guess(Y)
    
    # Use EM-algorithm
    for idx in range(n_iter):
        
        # E-step
        Q = None

        # M-step
        # ...
        Xi_i = Xi_i
        A_i = A_i
        epsilon_i1 = epsilon_i1
        epsilon_i2 = epsilon_i2
        s = s

    return Xi_i, A_i, epsilon_i1, epsilon_i2, s


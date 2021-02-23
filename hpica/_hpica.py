import numpy as np

from sklearn.decomposition import PCA


def _whiten(Y, n_sources):
    """
    """
    n_subjects = len(Y)
    wh_means = []
    wh_matrix = []
    Ybar = []
    for i in range(n_subjects):
        pca = PCA(n_components=n_sources, whiten=True)
        Ybar.append(pca.fit_transform(Y[i]))
        wh_means.append(pca.mean_)
        wh_matrix.append(pca.components_)

    return Ybar, wh_means, wh_matrix

def _initial_guess(Y, X, n_sources, random_state, n_gaussians=3):
    """
    """
    n_subjects = len(Y)
    n_samples = Y[0].shape[1]
    n_covariates = X.shape[1]

    # generate params for gaussian mixtures
    mus = []
    stds = []
    pis = []
    for idx in range(n_gaussians):
        mus.append(np.zeros(n_sources))
        stds.append(np.ones(n_sources))
        pis.append(np.ones(n_sources)/n_gaussians)

    mus = np.array(mus).T
    stds = np.array(stds).T
    pis = np.array(pis).T

    # generate beta 
    beta = random_state.normal(0, 1, size=(n_covariates, n_sources))

    # generate subject-specific deviations from population
    gamma_i_stds = np.ones((n_subjects, n_sources)) 

    # generate subject-specific noise
    epsilon_i_stds = np.ones((n_subjects, n_sources))

    # generate mixing matrix
    A_i = random_state.normal(0, 1, size=(n_subjects, n_sources, n_sources))
        
    return mus, stds, pis, beta, epsilon_i_stds, gamma_i_stds, A_i

def _source_from_mixture(mus, stds, pis, n_samples):
    """
    """
    n_gaussians = len(mus)
    n_sources = len(mus[0])

    s_ = []
    for v in range(n_samples):
        s_v = []
        for l in range(n_sources):
            val = sum([pis[j][l]*random_state.normal(mus[j][l], stds[j][l]) 
                       for j in range(n_gaussians)])
            s_v.append(val)
        s_.append(s_v)
    s_ = np.array(s_)
    return s_


def compute_hpica(Y, X, n_components, n_iter=10, whiten=True, random_state=None):
    """
    Y needs to be a list of datasets of shape (n_samples, n_features)

    X needs to be a covariate matrix of shape (n_subjects, n_covariates)

    Fits two-level model as introduced Shi et al 2016. 

    The first, subject-specific, level is defined with:

    y_i (v) = A_i * s_i (v) + epsilon_i (v),

    where

    v denotes sample index (v for voxel in fMRI),
    i denotes subject index,
    y_i denotes subject-specific observations,
    A_i denotes subject-specific mixing matrix,
    s_i denotes subject-specific source signals, and
    epsilon_i denotes unexplained gaussian noise.

    The second level is defined with:

    s_i (v) = s (v) + (beta (v)).T @ x_i + gamma_i (v),

    where

    gamma_i denotes subject-specific deviation 
      from population-level source signals. 
    s denotes population-level source signals, with 
      each source signal is specified with a mixture of Gaussians,
    x_i is a covariate vector containing subject-specific 
      characteristics, and
    beta maps the x_i to the correct space.


    """
    n_sources = n_components

    if not random_state:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        pass
    else:
        raise Exception('Unsupported random_state')

    # Demean and whiten the input variables
    if whiten:
        Y, wh_means, wh_matrix = _whiten(Y, n_sources)

    # Get initial guess
    mus, stds, pis, beta, epsilon_i_stds, gamma_i_stds, A_i = _initial_guess(
        Y, X, n_sources, random_state=random_state)

    import pdb; pdb.set_trace()

    # Use EM-algorithm
    for idx in range(n_iter):
        pass
        
        # In the E-step, some conditional expectations are determined.

        # Determine p[ss(v) | yy(v), z(v); theta]
        # Determine p[z(v) | yy(v); theta]
        # Determine p[ss(v), z(v) | yy(v); theta]
        # Determine p[ss(v) | yy(v); theta]

        # Determine E[S(v) | y(v); theta]
        # Determine E[S(v) * S(v).T | y(v); theta]

        # In the M-step, we update parameters.

        # Update beta
        # Update A_i
        # Update epsilon_i_stds
        # Update gamma_i_stds
        # Update pis
        # Update mus
        # Update stds

    # return results
    return None


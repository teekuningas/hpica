import time

import numpy as np
import itertools

from decimal import Decimal

from numpy.linalg import pinv
from numpy.linalg import inv
from scipy.linalg import block_diag
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal


def _sym_decorrelation(w_):
    """ Uses classic method of orthogonalization via matrix square roots
    """
    return np.dot(w_, sqrtm(pinv(np.dot(np.conj(w_.T), w_))))


def _whiten_deterministic(Ybar, n_sources, random_state):
    """ Uses deterministic PCA to whiten the data
    """
    from sklearn.decomposition import PCA
    n_subjects = len(Ybar)
    wh_means = []
    wh_matrix = []

    Y = []
    for i in range(n_subjects):
        pca = PCA(n_components=n_sources, whiten=True, random_state=random_state)
        Y.append(pca.fit_transform(Ybar[i].T).T)  # (n_sources, n_samples)
        wh_means.append(pca.mean_)  # (n_features)
        wh_matrix.append((pca.components_ / pca.singular_values_[:, np.newaxis]) * 
                         np.sqrt(Ybar[i].shape[1]))  # (n_sources, n_features), 

        # Check that all is fine
        np.testing.assert_allclose(wh_matrix[i] @ (Ybar[i] - wh_means[i][:, np.newaxis]), Y[i], atol=0.1)
 
    return Y, wh_means, wh_matrix


def _initial_guess_ica(Y, n_gaussians, random_state):
    """ Use fastica solution on feature-space 
    concatenated data as a starting guess

    """
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import FastICA

    n_sources = Y[0].shape[0]
    n_features = Y[0].shape[0]
    n_subjects = len(Y)

    sources = []
    mixing = []

    ica = FastICA(whiten=True, n_components=n_sources, random_state=random_state)

    # find sources and mixing and scale correctly
    sources = ica.fit_transform(np.vstack(Y).T).T
    sources = sources / np.std(sources)
    factor = np.std(np.vstack(Y)) / np.std(ica.components_.T @ sources)
    mixing = ica.components_.T * factor
    mixing = np.array(np.split(mixing, n_subjects))

    # Generate mixing matrices based on the FastICA results
    A = np.array(mixing)

    # Generate mus, pis and vars by fitting GaussianMixture with sklearn
    mus = np.zeros((n_sources, n_gaussians))
    pis = np.zeros((n_sources, n_gaussians))
    vars_ = np.zeros((n_sources, n_gaussians))

    gm = GaussianMixture(n_components=n_gaussians, random_state=random_state)
    for s_idx, s in enumerate(sources):
        gm.fit(s[:, np.newaxis])
        mus[s_idx, :] = gm.means_[:, 0]
        vars_[s_idx, :] = gm.covariances_[:, 0, 0]
        pis[s_idx, :] = gm.weights_

    # generate cov for subject-specific deviations from population
    D = np.eye(n_sources) * 0.01

    # generate cov for subject-specific noise
    E = np.eye(n_sources) * 0.01

    return mus, vars_, pis, E, D, A


def _initial_guess_ica_match(Y, n_gaussians, random_state):
    """ Use fastica solutions combined with post-hoc 
    component matching as a starting guess
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import FastICA

    n_sources = Y[0].shape[0]
    n_features = Y[0].shape[0]
    n_subjects = len(Y)

    sources = []
    mixing = []

    first_ica = FastICA(whiten=False, random_state=random_state)
    sources.append(first_ica.fit_transform(Y[0].T).T)
    mixing.append(first_ica.components_.T)

    # Sort (with cov as a metric) sources and mixings of other 
    # subjects so that they match the first subject (uses cov as a metric).
    for i in range(n_subjects-1):
        ica = FastICA(whiten=False, random_state=random_state)

        srcs = ica.fit_transform(Y[i+1].T).T
        mxng = ica.components_.T  # (n_features x n_sources)
        sorted_srcs = []
        sorted_mixing = []
        for l in range(n_sources):
            cov = sources[0][l] @ srcs.T
            idx = np.argmax(np.abs(cov))
            sgn = np.sign(cov[idx])
            sorted_srcs.append(sgn*srcs[idx])
            sorted_mixing.append(sgn*mxng[:, idx])

        sources.append(np.array(sorted_srcs))
        mixing.append(np.array(sorted_mixing).T)

    # Generate mixing matrices based on the FastICA results
    A = np.array(mixing)

    # Generate mus, pis and vars by fitting GaussianMixture with sklearn
    mus = np.zeros((n_sources, n_gaussians))
    pis = np.zeros((n_sources, n_gaussians))
    vars_ = np.zeros((n_sources, n_gaussians))

    gm = GaussianMixture(n_components=n_gaussians, random_state=random_state)
    for s_idx, s in enumerate(sources[0]):
        gm.fit(s[:, np.newaxis])
        mus[s_idx, :] = gm.means_[:, 0]
        vars_[s_idx, :] = gm.covariances_[:, 0, 0]
        pis[s_idx, :] = gm.weights_

    # generate cov for subject-specific deviations from population
    D = np.eye(n_sources) * 0.01

    # generate cov for subject-specific noise
    E = np.eye(n_sources) * 0.01

    return mus, vars_, pis, E, D, A

def _initial_guess_random(Y, n_sources, n_gaussians, random_state):
    """ Makes a random guess. Usually works poorly and 
    converges slowly to a nonglobal optimum.
    """
    n_subjects = len(Y)
    n_samples = Y[0].shape[1]
    n_features = Y[0].shape[0]

    # Generate params for gaussian mixtures
    mus = random_state.normal(size=(n_sources, n_gaussians))
    vars_ = 0.1 + np.abs(random_state.normal(size=(n_sources, n_gaussians)))
    pis = np.abs(random_state.normal(size=(n_sources, n_gaussians)))
    pis = pis / np.sum(pis, axis=1)[:, np.newaxis]

    # Generate cov for subject-specific deviations from population
    D = np.diag(np.abs(random_state.normal(size=n_sources)))

    # generate cov for subject-specific noise
    E = np.abs(random_state.normal())*np.eye(n_features)

    # generate mixing matrix
    A = random_state.normal(size=(n_subjects, n_features, n_sources))
        
    return mus, vars_, pis, E, D, A


def compute_hpica(Ybar, 
                  n_components=10, 
                  n_gaussians=3,
                  whiten='deterministic',
                  algorithm='exact',
                  initial_guess='ica',
                  random_state=None, 
                  eps=1e-9, 
                  n_iter=10, 
                  verbose=True):
    """
    Compute hierarhical probabilistic ICA from Guo et al. 2013

    Params:
    
    Ybar: a list of datasets of shape (n_features, n_samples),
    n_components: number of estimated components,
    n_gaussians: Number of gaussians in the source distribution model,
    whiten: 'deterministic' or None (prewhitened),
    algorithm: 'exact' or 'subspace',
    initial_guess: 'random' or 'ica' (with data concatenation) or 'ica_match' (with component matching)
    random_state: None, int or RandomState,
    eps: quit after ||theta_new-theta|| / ||theta|| < eps,
    n_iter: quit after n_iter,
    verbose: show prints,

    """
    n_sources = n_components
    n_subjects = len(Ybar)
    n_samples = Ybar[0].shape[1]
    n_features = Ybar[0].shape[0]

    if not random_state:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        pass
    else:
        raise Exception('Unsupported random_state')

    # Demean and whiten the input variables
    # Note that whitening matrix unmixes and A mixes..
    if whiten == 'deterministic':
        Y, wh_means, wh_matrix = _whiten_deterministic(Ybar, n_sources=n_sources, random_state=random_state)
        n_features = Y[0].shape[0]
    else:
        if Ybar[0].shape[0] != n_sources:
            raise Exception('n_features should be equal to n_sources if whiten=None')
        Y = Ybar

    # Get a initial guess
    if initial_guess == 'random':
        mus, vars_, pis, E, D, A = _initial_guess_random(
            Y, n_sources, n_gaussians, random_state)
    elif initial_guess == 'ica':
        mus, vars_, pis, E, D, A = _initial_guess_ica(
            Y, n_gaussians, random_state)
    elif initial_guess == 'ica_match':
        mus, vars_, pis, E, D, A = _initial_guess_ica_match(
            Y, n_gaussians, random_state)
    else:
        raise Exception('Unsupported initial_guess')

    start_time = time.time()

    # Introduce helper matrices to handle joint distributions
    B = np.kron(np.ones(n_subjects)[:, np.newaxis], np.eye(n_sources))
    J = np.block([np.zeros((n_sources, n_sources*n_subjects)), np.eye(n_sources)])
    Q = np.block([np.eye(n_subjects*n_sources), B])
    P = np.block([[Q], [J]])
    KU = np.kron(np.ones(n_subjects+1)[:, np.newaxis], np.eye(n_sources))

    if algorithm == 'subspace':
        # z space has (m-1)*q + 1 elements
        z_space = [(0, 0, 0, 0)]
        for j in range(1, n_gaussians):
            for l in range(n_sources):
                z = [0, 0, 0, 0]
                z[l] = j
                z_space.append(tuple(z))
    elif algorithm == 'exact':
        # z space has m^q elements
        z_space = list(
            itertools.product(*[range(n_gaussians) for lst in 
                                range(n_sources)]))
    else:
        raise Exception('Not implemented yet')

    for iter_idx in range(n_iter):

        # Start with the E-step, where
        # we compute E(s|Y,z) and p(z|Y)
        # that will easily give us
        # the expectations needed in the M-step.
        # We do not actually compute the conditional expectation
        # of the complete log likelihood as it is not
        # needed in the M-step.

        # Replicated E and D matrices
        KE = np.kron(np.eye(n_subjects), E)
        KD = np.kron(np.eye(n_subjects), D)

        # A-dependent helper matrices
        AA = block_diag(*A)
        AAB = AA @ B
        R = AA @ KD @ AA.T + KE
        X = np.block([AA, AAB])

        # Compute full joint distribution of p(s|Y,z) to facilitiate use of
        # p(s|Y,z)-based expectations in the M-step. p(z|Y) is need in the M-step too.
        # These three are collected for each v and z.
        E_s_Y_z = []
        Var_s_Y_z = []
        p_z_Y = []

        for v in range(n_samples):

            # YY is a column vector with subjects' data concatenated
            YY = np.concatenate([Y[i][:, v] for i in range(n_subjects)])[:, np.newaxis]

            E_s_Y_z_v = []
            Var_s_Y_z_v = []
            p_z_Y_v = []
            for z in z_space:

                # Initialize mu, pi and sigma conditional to z.
                Sigma_z = np.zeros((n_sources, n_sources))
                pi_z = np.zeros((n_sources, 1))
                mu_z = np.zeros((n_sources, 1))
                for l in range(n_sources):
                    Sigma_z[l, l] = vars_[l, z[l]]
                    pi_z[l] = pis[l, z[l]]
                    mu_z[l] = mus[l, z[l]]

                # Here we introduce p(gamma|z,Y) which has
                # error terms of a collapsed model
                Sigma_gamma = block_diag(KD, Sigma_z)
                Sigma_gamma_z_Y = pinv(X.T @ pinv(KE) @ X + pinv(Sigma_gamma))
                Eeta_gamma_z_Y = Sigma_gamma_z_Y @ X.T @ pinv(KE) @ (YY - AAB @ mu_z)

                # Using that, compute p(s|Y,z), where
                # s has subject-specific sources and
                # the population sources concatenated,
                # ((n_subjects*n_sources + n_sources) x 1)
                E_s_Y_z_v.append(P @ Eeta_gamma_z_Y + KU @ mu_z)
                Var_s_Y_z_v.append(P @ Sigma_gamma_z_Y @ P.T)

                # Then compute p(z|Y), which can be used 
                # to integrate z out of the expectations.

                g_mu = AAB @ mu_z
                g_sigma = (AAB @ Sigma_z @ AAB.T + R) 

                numer_log = multivariate_normal.logpdf(YY[:, 0], g_mu[:, 0], g_sigma)
                numer_factor = pi_z.prod()

                denom_logs = []
                denom_factors = []
                for z_denom in z_space:

                    Sigma_z_denom = np.zeros((n_sources, n_sources))
                    pi_z_denom = np.zeros((n_sources, 1))
                    mu_z_denom = np.zeros((n_sources, 1))
                    for l in range(n_sources):
                        Sigma_z_denom[l, l] = vars_[l, z_denom[l]]
                        pi_z_denom[l] = pis[l, z_denom[l]]
                        mu_z_denom[l] = mus[l, z_denom[l]]

                    g_mu_denom = AAB @ mu_z_denom
                    g_sigma_denom = AAB @ Sigma_z_denom @ AAB.T + R

                    denom_logs.append(multivariate_normal.logpdf(YY[:, 0], g_mu_denom[:, 0], g_sigma_denom))
                    denom_factors.append(pi_z_denom.prod())

                # Compute the numer / denom part with high precision library to not get
                # overflows
                prob = (Decimal(numer_factor) * Decimal(numer_log).exp() /
                        sum([Decimal(denom_factors[ii])*Decimal(denom_logs[ii]).exp() 
                             for ii in range(len(denom_logs))]))
                p_z_Y_v.append(float(prob))

            E_s_Y_z.append(E_s_Y_z_v)
            Var_s_Y_z.append(Var_s_Y_z_v)
            p_z_Y.append(p_z_Y_v)

        if verbose:
            print("Elapsed in E: " + str(time.time() - start_time))
        start_time = time.time()

        # Update parameters (M-step).
        # All of these are kindly derived in the Guo et al. 2013.
        A_new = np.zeros(A.shape)
        for i in range(n_subjects):
            first = []
            second = []
            for v in range(n_samples):
                E_ksi_i_Y = []
                E_ksi2_i_Y = []
                for z_idx in range(len(z_space)):
                    E_ksi_i_Y.append(p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][i*n_sources:(i+1)*n_sources])
                    E_ksi2_i_Y.append(
                        p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][i*n_sources:(i+1)*n_sources] * E_s_Y_z[v][z_idx][i*n_sources:(i+1)*n_sources].T +
                        p_z_Y[v][z_idx] * Var_s_Y_z[v][z_idx][i*n_sources:(i+1)*n_sources, i*n_sources:(i+1)*n_sources] 
                    )
                E_ksi_i_Y = np.sum(E_ksi_i_Y, axis=0)
                E_ksi2_i_Y = np.sum(E_ksi2_i_Y, axis=0)

                first.append(Y[i][:, v, np.newaxis] @ E_ksi_i_Y.T)
                second.append(E_ksi2_i_Y)
            # Y = As <=> (Y @ s.T) @ inv((s @ s.T)) = A
            A_new[i] = np.sum(first, axis=0) @ pinv(np.sum(second, axis=0))
            A_new[i] = _sym_decorrelation(A_new[i])

        E_new = np.eye(n_features)
        elems = []
        for i in range(n_subjects):
            for v in range(n_samples):
                E_ksi_i_Y = []
                E_ksi2_i_Y = []
                for z_idx in range(len(z_space)):
                    E_ksi_i_Y.append(p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][i*n_sources:(i+1)*n_sources])
                    E_ksi2_i_Y.append(
                        p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][i*n_sources:(i+1)*n_sources] * E_s_Y_z[v][z_idx][i*n_sources:(i+1)*n_sources].T +
                        p_z_Y[v][z_idx] * Var_s_Y_z[v][z_idx][i*n_sources:(i+1)*n_sources, i*n_sources:(i+1)*n_sources] 
                    )
                E_ksi_i_Y = np.sum(E_ksi_i_Y, axis=0)
                E_ksi2_i_Y = np.sum(E_ksi2_i_Y, axis=0)
                Y_i_v = Y[i][:, v, np.newaxis]

                elems.append(Y_i_v.T @ Y_i_v - 2 * Y_i_v.T @ A_new[i] @ E_ksi_i_Y + 
                             np.trace(A_new[i].T @ A_new[i] @ E_ksi2_i_Y))
        E_new = E_new * np.sum(elems) / (n_features * n_subjects * n_samples)

        D_new = np.zeros(D.shape)
        for l in range(n_sources):
            elems = []
            for i in range(n_subjects):
                for v in range(n_samples):
                    E_ksi_il_2_Y = []
                    E_s_l_2_Y = []
                    E_ksi_il_s_l_Y = []
                    for z_idx in range(len(z_space)):
                        E_ksi_il_2_Y.append(p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][i*n_sources+l]**2 + 
                                            p_z_Y[v][z_idx] * Var_s_Y_z[v][z_idx][i*n_sources+l, i*n_sources+l])
                        E_s_l_2_Y.append(p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][n_subjects*n_sources+l]**2 + 
                                         p_z_Y[v][z_idx] * Var_s_Y_z[v][z_idx][n_subjects*n_sources+l, n_subjects*n_sources+l])
                        E_ksi_il_s_l_Y.append(p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][n_subjects*n_sources+l]*E_s_Y_z[v][z_idx][i*n_sources+l] +
                                              p_z_Y[v][z_idx] * Var_s_Y_z[v][z_idx][n_subjects*n_sources+l, i*n_sources+l])
                    E_ksi_il_2_Y = np.sum(E_ksi_il_2_Y, axis=0)
                    E_s_l_2_Y = np.sum(E_s_l_2_Y, axis=0)
                    E_ksi_il_s_l_Y = np.sum(E_ksi_il_s_l_Y, axis=0)
                    elems.append(E_ksi_il_2_Y - 2*E_ksi_il_s_l_Y + E_s_l_2_Y)
            D_new[l, l] = np.sum(elems) / (n_subjects * n_samples)

        pis_new = np.zeros(pis.shape)
        for l in range(n_sources):
            for j in range(n_gaussians):
                v_sum = []
                for v in range(n_samples):
                    z_sum = []
                    for z_idx, z in enumerate(z_space):
                        if z[l] == j:
                            z_sum.append(p_z_Y[v][z_idx])
                    v_sum.append(np.sum(z_sum))

                pis_new[l, j] = np.sum(v_sum) / n_samples
    
        mus_new = np.zeros(mus.shape)
        for l in range(n_sources):
            for j in range(n_gaussians):
                v_sum = []
                for v in range(n_samples):
                    z_sum = []
                    for z_idx, z in enumerate(z_space):
                        if z[l] == j:
                            z_sum.append(p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][n_subjects*n_sources+l])
                    v_sum.append(np.sum(z_sum))

                mus_new[l, j] = np.sum(v_sum) / (n_samples*pis_new[l, j])

        vars_new = np.zeros(vars_.shape)
        for l in range(n_sources):
            for j in range(n_gaussians):
                v_sum = []
                for v in range(n_samples):
                    z_sum = []
                    for z_idx, z in enumerate(z_space):
                        if z[l] == j:
                            z_sum.append(p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][n_subjects*n_sources + l]**2 + 
                                         p_z_Y[v][z_idx] * Var_s_Y_z[v][z_idx][n_subjects*n_sources + l, n_subjects*n_sources + l])
                    v_sum.append(np.sum(z_sum))
                vars_new[l, j] = np.sum(v_sum) / (n_samples*pis_new[l, j]) - mus_new[l, j]**2

        if verbose: 
            print("Elapsed in M: " + str(time.time() - start_time))

        # test if converged
        theta_new = np.concatenate([A_new.flatten(), 
                                    E_new.flatten(), 
                                    D_new.flatten(), 
                                    pis_new.flatten(), 
                                    mus_new.flatten(), 
                                    vars_new.flatten()], axis=0)
        theta = np.concatenate([A.flatten(), 
                                E.flatten(), 
                                D.flatten(), 
                                pis.flatten(), 
                                mus.flatten(), 
                                vars_.flatten()], axis=0)

        distance = np.linalg.norm(theta_new - theta) / np.linalg.norm(theta)

        if verbose:
            print("Distance: " + str(distance) + ", (iter " + str(iter_idx+1) + ")")

        if verbose:
            for i in range(n_subjects):
                print("Distance (A" + str(i+1) + "): " + 
                      str(np.linalg.norm(A_new[i].flatten() - A[i].flatten()) / 
                          np.linalg.norm(A[i].flatten())))
            print("Distance (E): " + str(np.linalg.norm(E_new.flatten() - E.flatten()) / 
                                         np.linalg.norm(E.flatten())))
            print("Distance (D): " + str(np.linalg.norm(D_new.flatten() - D.flatten()) / 
                                         np.linalg.norm(D.flatten())))
            print("Distance (pis): " + str(np.linalg.norm(pis_new.flatten() - pis.flatten()) / 
                                           np.linalg.norm(pis.flatten())))
            print("Distance (mus): " + str(np.linalg.norm(mus_new.flatten() - mus.flatten()) / 
                                           np.linalg.norm(mus.flatten())))
            print("Distance (vars_): " + str(np.linalg.norm(vars_new.flatten() - vars_.flatten()) / 
                                             np.linalg.norm(vars_.flatten())))

            for i in range(n_subjects):
                print("A" + str(i+1) + ": ")
                print(str(A_new[i]))

            print("pis: ") 
            print(str(pis_new))

            print("mus: ")
            print(str(mus_new))

            print("vars: ")
            print(str(vars_new))

            print("E: ")
            print(str(np.diag(E_new)))

            print("D: ")
            print(str(np.diag(D_new)))

        A = A_new
        E = E_new
        D = D_new
        pis = pis_new
        mus = mus_new
        vars_ = vars_new

        if distance < eps:
            break

    return A_new, E_new, D_new, pis_new, mus_new, vars_new, wh_means, wh_matrix


import time
import itertools

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from decimal import Decimal

from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

from numpy.linalg import pinv
from scipy.linalg import block_diag
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal


def _sym_decorrelation(w_):
    """ Classic method of orthogonalization via matrix square roots
    """
    return np.dot(w_, sqrtm(pinv(np.dot(np.conj(w_.T), w_))))


def _whiten_deterministic(Ybar, n_sources, random_state):
    """ Uses deterministic PCA to whiten data.
    """
    n_subjects = len(Ybar)
    wh_mean = []
    wh_matrix = []

    Y = []
    for i in range(n_subjects):
        pca = PCA(n_components=n_sources, whiten=True, random_state=random_state)
        Y.append(pca.fit_transform(Ybar[i].T).T)  # (n_sources, n_samples)
        wh_mean.append(pca.mean_)  # (n_features)
        wh_matrix.append((pca.components_ / pca.singular_values_[:, np.newaxis]) * 
                         np.sqrt(Ybar[i].shape[1]))  # (n_sources, n_features), 

        # Check that all is fine
        np.testing.assert_allclose(wh_matrix[i] @ (Ybar[i] - wh_mean[i][:, np.newaxis]), Y[i], atol=0.1)
 
    return Y, wh_mean, wh_matrix


def _initial_guess_ica(Y, X, n_gaussians, init_values, random_state):
    """ Use fastica solution on feature-space concatenated data as a 
    starting guess for A_is. Other values are set to fine defaults.
    You may provide your own init values through init_values dictionary.

    """
    n_sources = Y[0].shape[0]
    n_samples = Y[0].shape[1]
    n_subjects = len(Y)
    n_covariates = X.shape[1]

    if init_values.get('A') is None:

        ica = FastICA(whiten=True, n_components=n_sources, random_state=random_state)

        # find sources and mixing and scale correctly
        sources = ica.fit_transform(np.vstack(Y).T).T
        sources = sources / np.std(sources)
        factor = np.std(np.vstack(Y)) / np.std(ica.components_.T @ sources)
        mixing = ica.components_.T * factor
        mixing = np.array(np.split(mixing, n_subjects))

        # Generate mixing matrices based on the FastICA results
        A = np.array(mixing)
    else:
        A = init_values.get('A')

    if init_values.get('mus') is None:
        if n_gaussians == 2:
            mus = np.tile([0,1], (n_sources, 1))
        elif n_gaussians == 3:
            mus = np.tile([0,1,-1], (n_sources, 1))
        else:
            raise Exception('Automatic initialization of mus supports only 2 or 3 gaussians')
    else:
        mus = init_values.get('mus')

    if init_values.get('pis') is None:
        if n_gaussians == 2:
            pis = np.tile([0.9, 0.1], (n_sources, 1))
        elif n_gaussians == 3:
            pis = np.tile([0.9, 0.05, 0.05], (n_sources, 1))
        else:
            raise Exception('Automatic initialization of pis supports only 2 or 3 gaussians')
    else:
        pis = init_values.get('pis')

    if init_values.get('vars') is None:
        if n_gaussians == 2:
            vars_ = np.tile([1.0, 1.0], (n_sources, 1))
        elif n_gaussians == 3:
            vars_ = np.tile([1.0, 1.0, 1.0], (n_sources, 1))
        else:
            raise Exception('Automatic initialization of vars supports only 2 or 3 gaussians')
    else:
        vars_ = init_values.get('vars')

    # generate cov for subject-specific deviations from population
    if init_values.get('D') is None:
        D = np.eye(n_sources) * 0.01
    else:
        D = init_values.get('D')

    # generate cov for subject-specific noise
    if init_values.get('E') is None:
        E = np.eye(n_sources) * 0.01
    else:
        E = init_values.get('E')

    # generate cov for subject-specific noise
    if init_values.get('Beta') is None:
        Beta = random_state.normal(size=(n_samples, n_covariates, n_sources)) * 0.01
    else:
        Beta = init_values.get('Beta')

    return mus, vars_, pis, E, D, A, Beta


def _compute_hpica(Ybar, 
                   X,
                   n_components=10, 
                   n_gaussians=3,
                   whiten='deterministic',
                   algorithm='exact',
                   init_values={},
                   random_state=None, 
                   eps=1e-9, 
                   n_iter=10, 
                   store_intermediate_results=True,
                   verbose=True):
    """ Implements data preparation and the EM algorithm.
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
        Y, wh_mean, wh_matrix = _whiten_deterministic(Ybar, n_sources=n_sources, random_state=random_state)
        n_features = Y[0].shape[0]
    else:
        if Ybar[0].shape[0] != n_sources:
            raise Exception('n_features should be equal to n_sources if whiten=None')
        Y = Ybar

    # Get a initial guess
    mus, vars_, pis, E, D, A, Beta = _initial_guess_ica(
        Y, X, n_gaussians, init_values, random_state)

    intermediate_results = {
        'A': [],
        'E': [],
        'D': [],
        'Beta': [],
        'pis': [],
        'mus': [],
        'vars': [],
        'E_s_Y': [],
    }

    start_time = time.time()

    # Introduce helper matrices to handle joint distributions

    U = np.kron(np.ones(n_subjects)[:, np.newaxis], np.eye(n_sources)) 
    R = np.block([np.eye(n_subjects*n_sources), U])

    P_upper = np.block([np.eye(n_subjects*n_sources), U])
    P_lower = np.block([np.zeros((n_sources, n_sources*n_subjects)), np.eye(n_sources)])
    P = np.block([[P_upper], [P_lower]])

    XX = np.concatenate(X)[:, np.newaxis]

    if algorithm == 'subspace':
        # Subspace algorithm works similarly to the exact algorithm
        # but limits the search to a reasonable subspace.
        # This subspace has (n_gaussians-1)*n_sources + 1 elements.
        z_space = [tuple([0]*n_sources)]
        for j in range(1, n_gaussians):
            for l in range(n_sources):
                z = [0]*n_sources
                z[l] = j
                z_space.append(tuple(z))
    elif algorithm == 'exact':
        # The exact algorithm searches through the whole z space,
        # which can quickly become large.
        # The z space has n_gaussians^n_sources elements.
        z_space = list(
            itertools.product(*[range(n_gaussians) for lst in 
                                range(n_sources)]))
    else:
        raise Exception("Algorithm must be either 'subspace' or 'exact'.")

    for iter_idx in range(n_iter):

        # Start with the E-step, where
        # we compute E(s|Y,z) and p(z|Y)
        # that will easily give us
        # the expectations needed in the M-step.
        # We do not actually compute the conditional expectation
        # of the complete log likelihood as it is not
        # needed in the M-step.

        KE = np.kron(np.eye(n_subjects), E)
        KD = np.kron(np.eye(n_subjects), D)

        AA = block_diag(*A)

        # Compute full joint distribution of p(s|Y,z) to facilitiate use of
        # p(s|Y,z)-based expectations in the M-step. p(z|Y) is need in the M-step too.
        # These three are collected for each v and z.
        E_s_Y_z = []
        Var_s_Y_z = []
        p_z_Y = []

        for v in range(n_samples):

            YY = np.concatenate([Y[i][:, v] for i in range(n_subjects)])[:, np.newaxis]

            B = np.kron(np.eye(n_subjects), Beta[v].T)

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

                Q_z = np.block([[B @ XX + U @ mu_z], [mu_z]])

                Gamma_z = block_diag(KD, Sigma_z)

                # I think the supplementary materials of Shi et al 2016 
                # have mixed up something here.
                # Thus instead of AA.T @ Y we are going to use just
                # YY and instead of B, U, R, we are going to
                # use these:
                AB = AA @ B
                AU = AA @ U
                AR = AA @ R

                Sigma_r_Y = pinv(AR.T @ pinv(KE) @ AR + pinv(Gamma_z))
                mu_r_Y = Sigma_r_Y @ AR.T @ pinv(KE) @ (YY - AB @ XX - AU @ mu_z)

                # Using this distribution, compute p(s|Y,z), where
                # s has subject-specific sources and
                # the population sources concatenated,
                # ((n_subjects*n_sources + n_sources) x 1)
                
                E_s_Y_z_v.append(P @ mu_r_Y + Q_z)
                Var_s_Y_z_v.append(P @ Sigma_r_Y @ P.T)

                # Then compute p(z|Y), which can be used 
                # to integrate z out of the expectations.

                g_mu = AB @ XX + AU @ mu_z
                g_sigma = AR @ Gamma_z @ AR.T + KE

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

                    Gamma_z_denom = block_diag(KD, Sigma_z_denom)

                    g_mu_denom = AB @ XX + AU @ mu_z_denom
                    g_sigma_denom = AR @ Gamma_z_denom @ AR.T + KE

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

        # Compute E_s_Y so that it can be used e.g for inference.
        E_s_Y = []
        for i in range(n_subjects):
            E_s_i_Y_i = []
            for v in range(n_samples):
                E_s_i_Y_i_v = []
                for z_idx in range(len(z_space)):
                    E_s_i_Y_i_v.append(p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][i*n_sources:(i+1)*n_sources])
                E_s_i_Y_i.append(np.sum(E_s_i_Y_i_v, axis=0))
            E_s_Y.append(E_s_i_Y_i)

        E_s_0_Y = []
        for v in range(n_samples):
            E_s_0_Y_v = []
            for z_idx in range(len(z_space)):
                E_s_0_Y_v.append(p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][n_subjects*n_sources:])
            E_s_0_Y.append(np.sum(E_s_0_Y_v, axis=0))
        E_s_Y.append(E_s_0_Y)

        E_s_Y = np.array(E_s_Y)[:, :, :, 0]

        # Update parameters (M-step).
        # All of these are kindly derived both in the Guo et al. 2013
        # and Shi et al. 2016.
        Beta_new = np.zeros(Beta.shape)
        for v in range(n_samples):
            first_elems = []
            second_elems = []
            for i in range(n_subjects):
                first_elems.append(X[i][:, np.newaxis] @ X[i][np.newaxis, :])

                E_s_i_Y = []
                E_s_0_Y = []
                for z_idx in range(len(z_space)):
                    E_s_i_Y.append(p_z_Y[v][z_idx] * 
                                   E_s_Y_z[v][z_idx][i*n_sources:(i+1)*n_sources])
                    E_s_0_Y.append(p_z_Y[v][z_idx] * 
                                   E_s_Y_z[v][z_idx][n_subjects*n_sources:])
                E_s_i_Y = np.sum(E_s_i_Y, axis=0)
                E_s_0_Y = np.sum(E_s_0_Y, axis=0)

                second_elems.append(X[i][:, np.newaxis] @ (E_s_i_Y.T - E_s_0_Y.T))

            Beta_new[v] = pinv(np.sum(first_elems, axis=0)) @ np.sum(second_elems, axis=0)


        A_new = np.zeros(A.shape)
        for i in range(n_subjects):
            first = []
            second = []
            for v in range(n_samples):
                E_s_i_Y = []
                E_s2_i_Y = []
                for z_idx in range(len(z_space)):
                    E_s_i_Y.append(p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][i*n_sources:(i+1)*n_sources])
                    E_s2_i_Y.append(
                        p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][i*n_sources:(i+1)*n_sources] * E_s_Y_z[v][z_idx][i*n_sources:(i+1)*n_sources].T +
                        p_z_Y[v][z_idx] * Var_s_Y_z[v][z_idx][i*n_sources:(i+1)*n_sources, i*n_sources:(i+1)*n_sources] 
                    )
                E_s_i_Y = np.sum(E_s_i_Y, axis=0)
                E_s2_i_Y = np.sum(E_s2_i_Y, axis=0)

                first.append(Y[i][:, v, np.newaxis] @ E_s_i_Y.T)
                second.append(E_s2_i_Y)
            # Y = As <=> (Y @ s.T) @ inv((s @ s.T)) = A
            A_new[i] = np.sum(first, axis=0) @ pinv(np.sum(second, axis=0))
            A_new[i] = _sym_decorrelation(A_new[i])


        E_new = np.eye(n_features)
        elems = []
        for i in range(n_subjects):
            for v in range(n_samples):
                E_s_i_Y = []
                E_s2_i_Y = []
                for z_idx in range(len(z_space)):
                    E_s_i_Y.append(p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][i*n_sources:(i+1)*n_sources])
                    E_s2_i_Y.append(
                        p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][i*n_sources:(i+1)*n_sources] * E_s_Y_z[v][z_idx][i*n_sources:(i+1)*n_sources].T +
                        p_z_Y[v][z_idx] * Var_s_Y_z[v][z_idx][i*n_sources:(i+1)*n_sources, i*n_sources:(i+1)*n_sources] 
                    )
                E_s_i_Y = np.sum(E_s_i_Y, axis=0)
                E_s2_i_Y = np.sum(E_s2_i_Y, axis=0)
                Y_i_v = Y[i][:, v, np.newaxis]

                elems.append(Y_i_v.T @ Y_i_v - 2 * Y_i_v.T @ A_new[i] @ E_s_i_Y + 
                             np.trace(A_new[i].T @ A_new[i] @ E_s2_i_Y))
        E_new = E_new * np.sum(elems) / (n_features * n_subjects * n_samples)

        D_new = np.zeros(D.shape)
        for l in range(n_sources):
            elems = []
            for i in range(n_subjects):
                for v in range(n_samples):
                    E_s_il_Y = []
                    E_s_0l_Y = []
                    E_s_il_2_Y = []
                    E_s_0l_2_Y = []
                    E_s_il_s_0l_Y = []
                    for z_idx in range(len(z_space)):
                        E_s_il_Y.append(p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][i*n_sources+l])
                        E_s_0l_Y.append(p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][n_subjects*n_sources+l])
                        E_s_il_2_Y.append(p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][i*n_sources+l]**2 + 
                                          p_z_Y[v][z_idx] * Var_s_Y_z[v][z_idx][i*n_sources+l, i*n_sources+l])
                        E_s_0l_2_Y.append(p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][n_subjects*n_sources+l]**2 + 
                                          p_z_Y[v][z_idx] * Var_s_Y_z[v][z_idx][n_subjects*n_sources+l, n_subjects*n_sources+l])
                        E_s_il_s_0l_Y.append(p_z_Y[v][z_idx] * E_s_Y_z[v][z_idx][n_subjects*n_sources+l]*E_s_Y_z[v][z_idx][i*n_sources+l] +
                                             p_z_Y[v][z_idx] * Var_s_Y_z[v][z_idx][n_subjects*n_sources+l, i*n_sources+l])
                    E_s_il_Y = np.sum(E_s_il_Y, axis=0)
                    E_s_0l_Y = np.sum(E_s_0l_Y, axis=0)
                    E_s_il_2_Y = np.sum(E_s_il_2_Y, axis=0)
                    E_s_0l_2_Y = np.sum(E_s_0l_2_Y, axis=0)
                    E_s_il_s_0l_Y = np.sum(E_s_il_s_0l_Y, axis=0)
                    first_beta_term = (Beta_new[v][:, l][np.newaxis, :] @ X[i][:, np.newaxis] @ 
                                       X[i][np.newaxis, :] @ Beta_new[v][:, l][:, np.newaxis])
                    second_beta_term = 2*(E_s_0l_Y - E_s_il_Y) @ X[i][np.newaxis, :] @ Beta_new[v][:, l][:, np.newaxis]
                    elems.append(E_s_il_2_Y - 2*E_s_il_s_0l_Y + E_s_0l_2_Y + 
                                 first_beta_term + second_beta_term)
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
            print("Distance (Beta): " + str(np.linalg.norm(Beta_new.flatten() - Beta.flatten()) / 
                                            np.linalg.norm(Beta.flatten())))
            print("Distance (pis): " + str(np.linalg.norm(pis_new.flatten() - pis.flatten()) / 
                                           np.linalg.norm(pis.flatten())))
            print("Distance (mus): " + str(np.linalg.norm(mus_new.flatten() - mus.flatten()) / 
                                           np.linalg.norm(mus.flatten())))
            print("Distance (vars_): " + str(np.linalg.norm(vars_new.flatten() - vars_.flatten()) / 
                                             np.linalg.norm(vars_.flatten())))

            for i in range(n_subjects):
                print("A" + str(i+1) + ": ")
                print(str(A_new[i]))

            print("Beta: ")
            print(str(Beta_new.squeeze()))

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
        Beta = Beta_new
        pis = pis_new
        mus = mus_new
        vars_ = vars_new

        if store_intermediate_results:
            intermediate_results['A'].append(A)
            intermediate_results['E'].append(E)
            intermediate_results['D'].append(D)
            intermediate_results['Beta'].append(Beta)
            intermediate_results['pis'].append(pis)
            intermediate_results['mus'].append(mus)
            intermediate_results['vars'].append(vars_)
            intermediate_results['E_s_Y'].append(E_s_Y)

        if distance < eps:
            break

    return (Beta_new, A_new, E_new, D_new, pis_new, mus_new, vars_new, 
            wh_mean, wh_matrix, E_s_Y, intermediate_results)


class HPICA:
    """
    Compute hierarchical probabilistic ICA from Shi et al. 2016

    Parameters
    ----------
    n_components : int
        Number of estimated components.
    n_gaussians : int
        Number of gaussians in the source distribution model.
    whiten : str or None
        If 'deterministic', uses PCA to whiten the data before
        fitting ICA. If None, the data is assumed to be
        prewhitened.
    algorithm : str
        Can either be 'exact' or 'subspace'.
    init_values : dict
        Values for initialization can be passed here, can be e.g
        {'mus': np.tile([-1, 1], (n_sources, 1))}.
    random_state: None, int or RandomState,
        Seed for reproducibility.
    eps : float
        Quit after ||theta_new-theta|| / ||theta|| < eps.
    n_iter : int
        Quit after n_iter.
    store_intermediate_results : bool
        Whether to store parameter estimates
        from every iteration.
    verbose: bool
        Whether to print verbosely.
    """

    def __init__(self, 
                 n_components=10, 
                 n_gaussians=3,
                 whiten='deterministic',
                 algorithm='exact',
                 init_values={},
                 random_state=None, 
                 eps=1e-9, 
                 n_iter=10,
                 store_intermediate_results=True,
                 verbose=True):
 
        self._n_components = n_components
        self._n_gaussians = n_gaussians
        self._whiten = whiten
        self._algorithm = algorithm
        self._init_values = init_values
        self._random_state = random_state
        self._eps = eps
        self._n_iter = n_iter
        self._store_intermediate_results = store_intermediate_results
        self._verbose = verbose
        
        self._is_fit = False

    def fit(self, Ybar, X=None):
        """ Uses EM to estimate the model.

        Parameters
        ----------
        Ybar : list
            List of datasets. Each element should be np.array of shape (n_features, n_samples),
            that is, if this is spatial ICA, the second dimension should be the spatial one.
        X : np.array or None
            An array containing the covariate data. If not None, should
            be of shape (n_subjects, n_covariates).
        """
        # If no covariates given, use a zero vector.
        if X is None:
            X = np.zeros((len(Ybar), 1))

        results = _compute_hpica(
            Ybar, 
            X, 
            n_components=self._n_components, 
            n_gaussians=self._n_gaussians,
            whiten=self._whiten,
            algorithm=self._algorithm,
            init_values=self._init_values,
            random_state=self._random_state, 
            eps=self._eps, 
            n_iter=self._n_iter, 
            store_intermediate_results=self._store_intermediate_results,
            verbose=self._verbose)

        self._Beta = results[0]
        self._A = results[1]
        self._E = results[2]
        self._D = results[3]
        self._pis = results[4]
        self._mus = results[5]
        self._vars = results[6]
        self._wh_mean = results[7]
        self._wh_matrix = results[8]
        self._E_s_Y = results[9]
        self._intermediate_results = results[10]

        self._is_fit = True

    def infer(self, Ybar, X):
        """ Computes variance estimator for Beta,
        and uses it to provide hypothesis tests.
        """
        if not self._is_fit:
            raise Exception('Must fit first.')

        if X is None:
            raise Exception('Cannot infer without proper X.')

        n_subjects = len(Ybar)
        n_samples = Ybar[0].shape[-1]
        n_sources = self._n_components

        SE = []
        for v in range(n_samples):
            sum_elems = []
            for i in range(n_subjects):
                Y_i = self._wh_matrix[i] @ Ybar[i]
                A_i = self._A[i]
                X_i = np.kron(X[i][np.newaxis, :], np.eye(n_sources))

                summand = (A_i.T @ Y_i[:, v][:, np.newaxis] -
                           self._E_s_Y[-1][v][:, np.newaxis] -
                           (X_i @ np.concatenate(self._Beta[v]))[:, np.newaxis])
                sum_elems.append(summand @ summand.T)
            W_v = (1/n_subjects) * np.sum(sum_elems, axis=0)

            sum_elems = []
            for i in range(n_subjects):
                Y_i = self._wh_matrix[i] @ (Ybar[i] - self._wh_mean[i][:, np.newaxis])
                A_i = self._A[i]
                X_i = np.kron(X[i][np.newaxis, :], np.eye(n_sources))

                sum_elems.append(X_i.T @ pinv(W_v) @ X_i)

            res = (1/n_subjects) * pinv(np.sum(sum_elems, axis=0))

            SE_v = np.array([np.sqrt(res[ix, ix]) for ix in range(len(res))]).reshape(self._Beta[v].shape)
            SE.append(SE_v)
        SE = np.array(SE)

        return self._Beta, SE


    def plot_evolution(self):
        """ Plots how source distribution model evolves on
        each iteration.
        """
        if not self._is_fit:
            raise Exception('Must fit first.')
        if not self._store_intermediate_results:
            raise Excpetion('Must store intermediate results.')

        res = self._intermediate_results

        n_rows_on_page = 8
        n_rows_total = len(res['pis'])
        n_cols = res['pis'][0].shape[0]
        n_figs = self._n_iter // n_rows_on_page + 1
        for fig_idx in range(n_figs):
            start_idx = fig_idx * n_rows_on_page
            end_idx = min(((fig_idx + 1) * n_rows_on_page), n_rows_total)
            n_rows = end_idx - start_idx

            fig, axes = plt.subplots(n_rows, n_cols,
                                     squeeze=False, constrained_layout=True)
            fig.suptitle('Iterations {0} - {1}'.format(
                fig_idx*n_rows_on_page+1,
                min((fig_idx+1)*n_rows_on_page, n_rows_total)))

            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    ax = axes[row_idx, col_idx]

                    mus = res['mus'][start_idx + row_idx]
                    pis = res['pis'][start_idx + row_idx]
                    vars_ = res['vars'][start_idx + row_idx]

                    # Create the hist plots by sampling from the source distrubution model.
                    samples = []
                    n_samples = 1000
                    for sample_idx in range(n_samples):
                        gaussian_idx = self._random_state.choice(range(pis.shape[1]), p=pis[col_idx])
                        samples.append(self._random_state.normal(mus[col_idx][gaussian_idx], 
                                                                 vars_[col_idx][gaussian_idx]))

                    sns.histplot(samples, ax=ax)

    @property
    def sources(self):
        if not self._is_fit:
            raise Exception('Must fit first.')

        return self._E_s_Y[:-1]

    @property
    def mixing(self):
        if not self._is_fit:
            raise Exception('Must fit first.')

        return self._A
        
    @property
    def wh_mean(self):
        if not self._is_fit:
            raise Exception('Must fit first.')

        return self._wh_mean

    @property
    def wh_matrix(self):
        if not self._is_fit:
            raise Exception('Must fit first.')

        return self._wh_matrix

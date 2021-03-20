import time

import numpy as np
import itertools

from numpy.linalg import pinv
from numpy.linalg import inv

from scipy.linalg import block_diag
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal

from sklearn.decomposition import PCA


def _sym_decorrelation(w_):
    """
    """
    return np.dot(w_, sqrtm(pinv(np.dot(np.conj(w_.T), w_))))


def _mvn_pdf(x, mu, cov):
    """ Naive but faster implementation than the scipy one """
    part1 = 1 / (((2*np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)))
    part2 = (-1/2) * ((x-mu).T.dot(inv(cov))).dot((x-mu))
    return part1 * np.exp(part2)

def _mvn_pdf_scipy(x, mu, cov):
    """ Scipy implementation """
    return multivariate_normal.pdf(x, mu, cov)

def _whiten(Ybar, n_sources):
    """
    """
    n_subjects = len(Ybar)
    wh_means = []
    wh_matrix = []
    Y = []
    for i in range(n_subjects):
        pca = PCA(n_components=n_sources, whiten=True)
        Y.append(pca.fit_transform(Ybar[i].T).T)  # (n_components, n_samples)
        wh_means.append(pca.mean_)  # (n_features)
        wh_matrix.append((pca.components_ / pca.singular_values_[:, np.newaxis]) * 
                         np.sqrt(Ybar[i].shape[1]))  # (n_components, n_features), 

        # Check that all is fine
        np.testing.assert_allclose(wh_matrix[i] @ (Ybar[i] - wh_means[i][:, np.newaxis]), Y[i], atol=0.1)
 
    return Y, wh_means, wh_matrix

def _initial_guess(Y, n_sources, n_gaussians, random_state):
    """
    """
    n_subjects = len(Y)
    n_samples = Y[0].shape[1]
    n_features = Y[0].shape[0]

    # generate params for gaussian mixtures
    mus = random_state.normal(size=(n_sources, n_gaussians))
    vars_ = 0.1 + np.abs(random_state.normal(size=(n_sources, n_gaussians)))
    pis = np.abs(random_state.normal(size=(n_sources, n_gaussians)))
    pis = pis / np.sum(pis, axis=1)[:, np.newaxis]

    # generate cov for subject-specific deviations from population
    D = np.diag(np.abs(random_state.normal(size=n_sources)))

    # generate cov for subject-specific noise
    E = np.abs(random_state.normal())*np.eye(n_features)

    # generate mixing matrix
    A = random_state.normal(size=(n_subjects, n_features, n_sources))
        
    return mus, vars_, pis, E, D, A


def compute_hpica(Ybar, n_components=10, n_iter=10, n_gaussians=2,
                  whiten=True, random_state=None, eps=0.001):
    """
    Ybar needs to be a list of datasets of shape (n_features, n_samples)
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

    if whiten:
        Y, wh_means, wh_matrix = _whiten(Ybar, n_sources=n_sources)
        n_features = Y[0].shape[0]
    else:
        Y = Ybar

    # Get initial guess
    mus, vars_, pis, E, D, A = _initial_guess(
        Y, n_sources, n_gaussians, random_state)

    start_time = time.time()

    # Use EM-algorithm (exact version)
    B = np.kron(np.ones(n_subjects)[:, np.newaxis], np.eye(n_sources))
    J = np.block([np.zeros((n_sources, n_sources*n_subjects)), np.eye(n_sources)])
    Q = np.block([np.eye(n_subjects*n_sources), B])
    P = np.block([[Q], [J]])

    KE = np.kron(np.eye(n_subjects), E)
    KD = np.kron(np.eye(n_subjects), D)
    KU = np.kron(np.ones(n_subjects+1)[:, np.newaxis], np.eye(n_sources))

    z_space = list(
        itertools.product(*[range(n_gaussians) for lst in 
                            range(n_sources)]))
    for iter_idx in range(n_iter):

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

            YY = np.concatenate([Y[i][:, v] for i in range(n_subjects)])[:, np.newaxis]

            E_s_Y_z_v = []
            Var_s_Y_z_v = []
            p_z_Y_v = []
            for z in z_space:

                Sigma_z = np.zeros((n_sources, n_sources))
                pi_z = np.zeros((n_sources, 1))
                mu_z = np.zeros((n_sources, 1))
                for l in range(n_sources):
                    Sigma_z[l, l] = vars_[l, z[l]]
                    pi_z[l] = pis[l, z[l]]
                    mu_z[l] = mus[l, z[l]]

                Sigma_gamma = block_diag(KD, Sigma_z)
                Sigma_gamma_z_Y = pinv(X.T @ pinv(KE) @ X + 
                                       pinv(Sigma_gamma))
                Eeta_gamma_z_Y = (Sigma_gamma_z_Y @ X.T @ 
                                  pinv(KE) @ 
                                  (YY - AAB @ mu_z))

                E_s_Y_z_v.append(P @ Eeta_gamma_z_Y + KU @ mu_z)
                Var_s_Y_z_v.append(P @ Sigma_gamma_z_Y @ P.T)

                g_mu = AAB @ mu_z
                g_sigma = (AAB @ Sigma_z @ AAB.T + R) 
                numer = pi_z.prod() * _mvn_pdf(YY[:, 0], g_mu[:, 0], g_sigma)

                denom_elems = []
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
                    prob = _mvn_pdf(YY[:, 0], g_mu_denom[:, 0], g_sigma_denom)
                    denom_elems.append(pi_z_denom.prod() * prob)

                p_z_Y_v.append(numer / np.sum(denom_elems))

            E_s_Y_z.append(E_s_Y_z_v)
            Var_s_Y_z.append(Var_s_Y_z_v)
            p_z_Y.append(p_z_Y_v)

        print("Elapsed in E: " + str(time.time() - start_time))
        start_time = time.time()

        # Update parameters (M-step):
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
            A_new[i] = np.sum(first, axis=0) @ pinv(np.sum(second, axis=0))
            A_new[i] = _sym_decorrelation(A_new[i])

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
        E_new = np.eye(E.shape[0]) * np.sum(elems) / (n_features * n_subjects * n_samples)

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
            D_new[l, l] = np.sum(elems, axis=0) / (n_subjects * n_samples)

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
        print("Distance: " + str(distance) + ", (iter " + str(iter_idx+1) + ")")
        if distance < eps:
            break

        print("Distance (A): " + str(np.linalg.norm(A_new.flatten() - A.flatten()) / np.linalg.norm(A.flatten())))
        print("Distance (E): " + str(np.linalg.norm(E_new.flatten() - E.flatten()) / np.linalg.norm(E.flatten())))
        print("Distance (D): " + str(np.linalg.norm(D_new.flatten() - D.flatten()) / np.linalg.norm(D.flatten())))
        print("Distance (pis): " + str(np.linalg.norm(pis_new.flatten() - pis.flatten()) / np.linalg.norm(pis.flatten())))
        print("Distance (mus): " + str(np.linalg.norm(mus_new.flatten() - mus.flatten()) / np.linalg.norm(mus.flatten())))
        print("Distance (vars_): " + str(np.linalg.norm(vars_new.flatten() - vars_.flatten()) / np.linalg.norm(vars_.flatten())))

        print("pis_new: ") 
        print(str(pis_new))

        print("mus_new: ")
        print(str(mus_new))

        print("vars_new: ")
        print(str(vars_new))

        print("E_new: ")
        print(str(E_new))

        print("D_new: ")
        print(str(D_new))

        A = A_new
        E = E_new
        D = D_new
        pis = pis_new
        mus = mus_new
        vars_ = vars_new

    return A_new, E_new, D_new, pis_new, mus_new, vars_new, wh_means, wh_matrix


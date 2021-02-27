import numpy as np
import itertools

from scipy.linalg import block_diag
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal as gaussian

from sklearn.decomposition import PCA


def _sym_decorrelation(w_):
    """
    """
    return np.dot(w_, sqrtm(np.linalg.pinv(np.dot(np.conj(w_.T), w_))))

def _whiten(Y, n_sources):
    """
    """
    n_subjects = len(Y)
    wh_means = []
    wh_matrix = []
    Ybar = []
    for i in range(n_subjects):
        pca = PCA(n_components=n_sources, whiten=True)
        Ybar.append(pca.fit_transform(Y[i]))  # (n_samples, n_components)
        wh_means.append(pca.mean_)  # (n_features)
        wh_matrix.append(pca.components_)  # (n_components, n_features)
 
    return Ybar, wh_means, wh_matrix

def _initial_guess(Y, X, n_sources, n_gaussians, random_state):
    """
    """
    n_subjects = len(Y)
    n_samples = Y[0].shape[0]
    n_covariates = X.shape[1]

    # generate params for gaussian mixtures
    mus = []
    stds = []
    pis = []
    for idx in range(n_gaussians):
        stds.append(np.ones(n_sources))
        mus.append(np.zeros(n_sources))
        pis.append(np.ones(n_sources)/n_gaussians)

    # stds = np.arange(1, n_sources*n_gaussians+1).reshape(n_gaussians, n_sources)

    mus = np.array(mus).T
    stds = np.array(stds).T
    pis = np.array(pis).T

    # generate beta 
    beta = random_state.normal(0, 1, size=(n_covariates, n_sources, n_samples))

    # generate cov for subject-specific deviations from population
    D = np.eye(n_sources)

    # generate cov for subject-specific noise
    E = np.eye(n_sources)

    # generate mixing matrix
    A_i = random_state.normal(0, 1, size=(n_subjects, n_sources, n_sources))
        
    return mus, stds, pis, beta, E, D, A_i


def compute_hpica(Y, n_components=10, X=None, n_iter=10, n_gaussians=3,
                  whiten=True, random_state=None, eps=0.001):
    """
    Y needs to be a list of datasets of shape (n_samples, n_features)

    X needs to be a covariate matrix of shape (n_subjects, n_covariates)

    Fits two-level model as introduced Shi et al 2016. 

    The first, subject-specific, level is defined with:

    y_i (v) = A_i * s_i (v) + e_i (v),

    where

    v denotes sample index (v for voxel in fMRI),
    i denotes subject index,
    y_i denotes subject-specific observations,
    A_i denotes subject-specific mixing matrix,
    s_i denotes subject-specific source signals, and
    e_i denotes unexplained gaussian noise.

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
    n_subjects = len(Y)
    n_samples = Y[0].shape[0]
    n_features = Y[0].shape[1]

    if not random_state:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        pass
    else:
        raise Exception('Unsupported random_state')

    if X is None:
        # X = np.zeros((n_subjects, 1))
        X = random_state.normal(size=(n_subjects, 2))

    # Demean and whiten the input variables
    if whiten:
        Y, wh_means, wh_matrix = _whiten(Y, n_sources)

    # Get initial guess
    mus, stds, pis, beta, E, D, A_i = _initial_guess(
        Y, X, n_sources, n_gaussians, random_state)

    import pdb; pdb.set_trace()

    # Use EM-algorithm (exact version)
    for iter_idx in range(n_iter):
        # In the E-step, some conditional expectations are determined.
        # First, init new variables (notation follows closely the paper):

        z_space = list(itertools.product(*[range(n_gaussians) for lst in range(n_sources)]))

        YY = np.concatenate(Y, axis=1).T
        XX = np.concatenate(X)

        I_Nq = np.eye(n_subjects*n_sources)
        I_N = np.eye(n_subjects)
        I_q = np.eye(n_sources)

        U = np.kron(np.ones(n_subjects)[:, np.newaxis], I_q)
        R = np.block([I_Nq, np.kron(np.ones(n_subjects)[:, np.newaxis], I_q)])
        A = block_diag(*A_i)
        P = np.block([[I_Nq, U], [np.zeros((n_sources, n_subjects*n_sources)), I_q]])
        Yps = np.kron(I_N, E)
        Yps_inv = np.linalg.pinv(Yps)

        # These contain expectations (for every v) that are needed everywhere in the M-step
        E_s_YY = []
        E_s2_YY = []

        # These contain values for every v and every z over m^q dimensional space
        # And are needed in the M-step to compute some marginals
        p_z_YY = []
        E_s_YY_z = []

        for v in range(n_samples):
            E_s_YY_sum = 0
            E_s2_YY_sum_1 = 0
            E_s2_YY_sum_2 = 0

            p_z_YY_v = []
            E_s_YY_z_v = []

            for z in z_space:
                B = np.kron(I_N, beta[:, :, v].T)

                Sigma_z = np.diag([stds[l, z[l]] for l in range(n_sources)])
                mu_z = np.array([mus[l, z[l]] for l in range(n_sources)])
                pi_z = np.array([pis[l, z[l]] for l in range(n_sources)])

                Q_z = np.concatenate([B @ XX + U @ mu_z, mu_z])

                Gamma_z = block_diag(np.kron(I_N, D), Sigma_z)
                Gamma_z_inv = np.linalg.pinv(Gamma_z)
                Sigma_r_YY = np.linalg.pinv(R.T @ Yps_inv @ R + Gamma_z_inv)
                mu_r_YY = (Sigma_r_YY @ R.T @ Yps_inv) @ (A.T @ YY[:, v] - B @ XX - U @ mu_z)

                E_s_YY_z_v_z = P @ mu_r_YY + Q_z
                E_s_YY_z_v.append(E_s_YY_z_v_z)
                 
                Var_s_YY_z = P @ Sigma_r_YY @ P.T

                # Next, figure out p[z|y; theta] for the current z (out of m^q z's)
                numer = np.prod(pi_z) * gaussian(
                    B @ XX + U @ mu_z, R @ Gamma_z @ R.T + Yps).pdf(A.T @ YY[:, v])
                sum_denom = 0
                # Computing denom involves summing over the whole space of z's
                for z_denom in z_space:
                    Sigma_z_denom = np.diag([stds[l, z_denom[l]] for l in range(n_sources)])
                    mu_z_denom = np.array([mus[l, z_denom[l]] for l in range(n_sources)])
                    pi_z_denom = np.array([pis[l, z_denom[l]] for l in range(n_sources)])
                    Gamma_z_denom = block_diag(np.kron(I_N, D), Sigma_z_denom)
                    sum_denom += np.prod(pi_z_denom) * gaussian(
                        B @ XX + U @ mu_z_denom, R @ Gamma_z_denom @ R.T + Yps).pdf(A.T @ YY[:, v])
                p_z_YY_v_z = numer / sum_denom
                p_z_YY_v.append(p_z_YY_v_z)

                E_s_YY_sum += p_z_YY_v_z * E_s_YY_z_v_z
                E_s2_YY_sum_1 += p_z_YY_v_z * np.outer(E_s_YY_z_v_z, E_s_YY_z_v_z)
                E_s2_YY_sum_2 += p_z_YY_v_z * Var_s_YY_z

            E_s_YY.append(E_s_YY_sum)
            E_s2_YY.append(E_s2_YY_sum_1 + E_s2_YY_sum_2)

            p_z_YY.append(p_z_YY_v)
            E_s_YY_z.append(E_s_YY_z_v)

        # In the M-step, we update parameters.

        # Update beta
        beta_new = np.zeros(beta.shape)
        for v in range(n_samples):
            sum_1 = np.sum([np.outer(X[i], X[i]) for i in range(n_subjects)], axis=0)

            sum_2 = 0
            for i in range(n_subjects):
                sum_2 += np.outer(
                    X[i], (E_s_YY[v][i*n_sources:(i+1)*n_sources] - E_s_YY[v][-n_sources:]))

            beta_new[:, :, v] = np.linalg.pinv(sum_1) @ sum_2

        # Update A_i
        A_i_new = np.zeros(A_i.shape)
        for i in range(n_subjects):
            sum_1 = 0
            sum_2 = 0
            for v in range(n_samples):
                sum_1 += np.outer(Y[i][v], E_s_YY[v][i*n_sources:(i+1)*n_sources])
                sum_2 += E_s2_YY[v][i*n_sources:(i+1)*n_sources,i*n_sources:(i+1)*n_sources]

            A_i_new[i] = sum_1 * np.linalg.pinv(sum_2)
            A_i_new[i] = _sym_decorrelation(A_i_new[i])

        # Update E
        sum_ = 0
        for v in range(n_samples):
            for i in range(n_subjects):
                part_1 = (np.dot(Y[i][v], Y[i][v]) - 
                          (2*Y[i][v]) @ A_i_new[i] @ E_s_YY[v][i*n_sources:(i+1)*n_sources])
                part_2 = np.trace(A_i_new[i].T @ A_i_new[i] @ 
                                  E_s2_YY[v][i*n_sources:(i+1)*n_sources,i*n_sources:(i+1)*n_sources])
                sum_ += part_1 + part_2

        E_new = (sum_/(n_subjects*n_features*n_samples))*np.eye(n_sources)

        # Update D
        D_new = np.zeros(D.shape)
        for l in range(n_sources):
            sum_ = 0
            for v in range(n_samples):
                for i in range(n_subjects):
                    E_s_il_YY = E_s_YY[v][i * n_subjects + l]
                    E_s_0l_YY = E_s_YY[v][-n_sources + l]

                    sum_ += E_s_il_YY**2 + E_s_0l_YY**2
                    sum_ -= 2*E_s_il_YY*E_s_0l_YY
                    sum_ += beta[:, l, v] @ np.outer(X[i], X[i]) @ beta[:, l, v]
                    sum_ += 2*(E_s_0l_YY - E_s_il_YY) * (X[i] @ beta[:, l, v])
            D_new[l, l] = (1/(n_subjects*n_samples))*sum_


        # Update pis
        pis_new = np.zeros(pis.shape)
        for l in range(n_sources):
            for j in range(n_gaussians):
                sum_ = 0
                for v in range(n_samples): 
                    sum_ += np.sum([p_z_YY[v][z_idx] for z_idx, z in enumerate(z_space) 
                                    if z[l] == j])

                pis_new[l, j] = (1/n_samples)*sum_

        # Update mus
        mus_new = np.zeros(mus.shape)
        for l in range(n_sources):
            for j in range(n_gaussians):
                sum_ = 0
                for v in range(n_samples):
                    # here is some cancelling out of p[z_l(v) = j|y(v); theta], so looks 
                    # simpler than in the paper
                    for z_idx, z in enumerate(z_space):
                        if z[l] == j:
                            first_part = p_z_YY[v][z_idx]
                            second_part = E_s_YY_z[v][z_idx][-n_sources + l]
                            sum_ += first_part * second_part
                mus_new[l, j] = (1 / (n_samples * pis_new[l, j])) * sum_

        # Update stds
        stds_new = np.zeros(stds.shape)
        for l in range(n_sources):
            for j in range(n_gaussians):
                sum_ = 0
                for v in range(n_samples):

                    numer = 0
                    for z_idx, z in enumerate(z_space):
                        if z[l] == j:
                            numer += p_z_YY[v][z_idx] * E_s_YY_z[v][z_idx][-n_sources + l]

                    p_z_l_j_YY = np.sum([p_z_YY[v][z_idx] for z_idx, z in enumerate(z_space) 
                                    if z[l] == j])
                    E_s_0l_z_l_j_y = numer / p_z_l_j_YY

                    sum_ += p_z_l_j_YY * E_s_0l_z_l_j_y**2
                stds_new[l, j] = (sum_ / (n_samples*pis_new[l, j])) - mus_new[l, j]**2
 
        # test if converged
        theta_new = np.concatenate([beta_new.flatten(), 
                                    A_i_new.flatten(), 
                                    E_new.flatten(), 
                                    D_new.flatten(), 
                                    pis_new.flatten(), 
                                    mus_new.flatten(), 
                                    stds_new.flatten()], axis=0)
        theta = np.concatenate([beta.flatten(), 
                                A_i.flatten(), 
                                E.flatten(), 
                                D.flatten(), 
                                pis.flatten(), 
                                mus.flatten(), 
                                stds.flatten()], axis=0)
        distance = np.linalg.norm(theta_new - theta) / np.linalg.norm(theta)
        print("Distance: " + str(distance) + ", (iter " + str(iter_idx+1) + ")")
        if distance < eps:
            break

        beta = beta_new
        A_i = A_i_new
        E = E_new
        D = D_new
        pis = pis_new
        mus = mus_new
        stds = stds_new

    return beta, A_i, E, D, pis, mus, stds, wh_means, wh_matrix


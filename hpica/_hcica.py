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

def _initial_guess(Y, X, n_sources, n_gaussians, random_state):
    """
    """
    n_subjects = len(Y)
    n_samples = Y[0].shape[1]
    n_covariates = X.shape[0]

    # generate params for gaussian mixtures
    mus = random_state.normal(size=(n_sources, n_gaussians))
    vars_ = 0.1 + np.abs(random_state.normal(size=(n_sources, n_gaussians)))
    pis = np.abs(random_state.normal(size=(n_sources, n_gaussians)))
    pis = pis / np.sum(pis, axis=1)[:, np.newaxis]

    # generate beta 
    beta = random_state.normal(0, 1, size=(n_covariates, n_sources, n_samples))

    # generate cov for subject-specific deviations from population
    D = np.diag(np.abs(random_state.normal(size=n_sources)))

    # generate cov for subject-specific noise
    E = np.abs(random_state.normal())*np.eye(n_sources)

    # generate mixing matrix
    A_i = random_state.normal(size=(n_subjects, n_sources, n_sources))
        
    return mus, vars_, pis, beta, E, D, A_i


def compute_hcica(Ybar, n_components=10, X=None, n_iter=10, n_gaussians=3,
                  whiten=True, random_state=None, eps=0.001):
    """
    Ybar needs to be a list of datasets of shape (n_features, n_samples)

    X needs to be a covariate matrix of shape (n_covariates, n_subjects) or None

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

    if X is None:
        # X = np.zeros((n_subjects, 1))
        X = random_state.normal(size=(2, n_subjects))

    # Demean and whiten the input variables
    if whiten:
        Y, wh_means, wh_matrix = _whiten(Ybar, n_sources)

    # Get initial guess
    mus, vars_, pis, beta, E, D, A_i = _initial_guess(
        Y, X, n_sources, n_gaussians, random_state)

    # Note that whitening matrix unmixes and A_i mixes..

    # Use EM-algorithm (exact version)
    for iter_idx in range(n_iter):
        # In the E-step, some conditional expectations are determined.
        # First, init new variables (notation follows closely the paper):

        z_space = list(itertools.product(*[range(n_gaussians) for lst in range(n_sources)]))

        YY = np.concatenate(Y, axis=0)  # (n_subjects * n_sources, n_samples)
        XX = np.concatenate(X.T)  # (n_covariates * n_subjects)

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

            print("Sample " + str(v+1))
            E_s_YY_sum = 0
            E_s2_YY_sum_1 = 0
            E_s2_YY_sum_2 = 0

            p_z_YY_v = []
            E_s_YY_z_v = []

            for z in z_space:
                B = np.kron(I_N, beta[:, :, v].T)

                Sigma_z = np.diag([vars_[l, z[l]] for l in range(n_sources)])
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
                    Sigma_z_denom = np.diag([vars_[l, z_denom[l]] for l in range(n_sources)])
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
            sum_1 = np.sum([np.outer(X[:, i], X[:, i]) for i in range(n_subjects)], axis=0)

            sum_2 = 0
            for i in range(n_subjects):
                sum_2 += np.outer(
                    X[:, i], (E_s_YY[v][i*n_sources:(i+1)*n_sources] - E_s_YY[v][-n_sources:]))

            beta_new[:, :, v] = np.linalg.pinv(sum_1) @ sum_2

        # Update A_i
        A_i_new = np.zeros(A_i.shape)
        for i in range(n_subjects):
            sum_1 = 0
            sum_2 = 0
            for v in range(n_samples):
                sum_1 += np.outer(Y[i][:, v], E_s_YY[v][i*n_sources:(i+1)*n_sources])
                sum_2 += E_s2_YY[v][i*n_sources:(i+1)*n_sources,i*n_sources:(i+1)*n_sources]

            A_i_new[i] = sum_1 * np.linalg.pinv(sum_2)
            A_i_new[i] = _sym_decorrelation(A_i_new[i])

        # Update E
        sum_ = 0
        for v in range(n_samples):
            for i in range(n_subjects):
                part_1 = (np.dot(Y[i][:, v], Y[i][:, v]) - 
                          (2*Y[i][:, v]) @ A_i_new[i] @ E_s_YY[v][i*n_sources:(i+1)*n_sources])
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
                    sum_ += beta[:, l, v] @ np.outer(X[:, i], X[:, i]) @ beta[:, l, v]
                    sum_ += 2*(E_s_0l_YY - E_s_il_YY) * (X[:, i] @ beta[:, l, v])
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

        # Update vars_
        vars_new = np.zeros(vars_.shape)
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
                    E_s_0l_z_l_j_YY = numer / p_z_l_j_YY

                    sum_ += p_z_l_j_YY * E_s_0l_z_l_j_YY**2
                vars_new[l, j] = np.abs((sum_ / (n_samples*pis_new[l, j])) - mus_new[l, j]**2)
 
        # test if converged
        theta_new = np.concatenate([beta_new.flatten(), 
                                    A_i_new.flatten(), 
                                    E_new.flatten(), 
                                    D_new.flatten(), 
                                    pis_new.flatten(), 
                                    mus_new.flatten(), 
                                    vars_new.flatten()], axis=0)
        theta = np.concatenate([beta.flatten(), 
                                A_i.flatten(), 
                                E.flatten(), 
                                D.flatten(), 
                                pis.flatten(), 
                                mus.flatten(), 
                                vars_.flatten()], axis=0)
        distance = np.linalg.norm(theta_new - theta) / np.linalg.norm(theta)
        print("Distance: " + str(distance) + ", (iter " + str(iter_idx+1) + ")")
        if distance < eps:
            break

        print("Distance (beta): " + str(np.linalg.norm(beta_new.flatten() - beta.flatten()) / np.linalg.norm(beta.flatten())))
        print("Distance (A_i): " + str(np.linalg.norm(A_i_new.flatten() - A_i.flatten()) / np.linalg.norm(A_i.flatten())))
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

        beta = beta_new
        A_i = A_i_new
        E = E_new
        D = D_new
        pis = pis_new
        mus = mus_new
        vars_ = vars_new

    return beta_new, A_i_new, E_new, D_new, pis_new, mus_new, vars_new, wh_means, wh_matrix


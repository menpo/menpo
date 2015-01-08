import numpy as np
from numpy.testing import assert_almost_equal
from menpo.math import eigenvalue_decomposition, pca, ipca

# Positive semi-definite matrix
cov_matrix = np.array([[3, 1], [1, 3]])
# Data values taken from:
# http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf
# Tested values are equal
large_samples_data_matrix = np.array([[2.5, 2.4],
                                      [0.5, 0.7],
                                      [2.2, 2.9],
                                      [1.9, 2.2],
                                      [3.1, 3.0],
                                      [2.3, 2.7],
                                      [2.0, 1.6],
                                      [1.0, 1.1],
                                      [1.5, 1.6],
                                      [1.1, 0.9]])
centered_eigenvectors_s = np.array([[0.6778734, 0.73517866],
                                    [-0.73517866, 0.6778734]])
non_centered_eigenvectors_s = np.array([[0.68647784, 0.72715072],
                                        [-0.72715072, 0.68647784]])
mean_vector_s = np.array([1.81, 1.91])
eigenvalues_no_centre_s = np.array([8.97738481, 0.04928186])
eigenvalues_centered_s = np.array([1.28402771, 0.0490834])

centered_eigenvectors_f = np.array([[-0.09901475, 0.19802951, 0.69310328,
                                     0.29704426, -0.09901475, 0.39605902,
                                     -0.39605902, 0.09901475, 0.09901475,
                                     -0.19802951]])
non_centered_eigenvectors_f = np.array(
    [[0.38507927, 0.09481302, 0.40261598, 0.32307722, 0.4794398, 0.39407387,
      0.28217662, 0.16535718, 0.24399096, 0.15681507],
     [-0.25575629, 0.17561812, 0.58718113, 0.19319469, -0.29239933, 0.27184299,
      -0.5344514, 0.04200527, 0.01146941, -0.27333287]])
mean_vector_f = np.array([2.45, 0.6, 2.55, 2.05, 3.05,
                          2.5, 1.8, 1.05, 1.55, 1.])
eigenvalues_no_centre_f = np.array([80.79646326, 0.44353674])
eigenvalues_centered_f = np.array([0.51])


def pcd_samples_nocentre_test():
    output = pca(large_samples_data_matrix, centre=False)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_no_centre_s)
    assert_almost_equal(eigenvectors, non_centered_eigenvectors_s)
    assert_almost_equal(mean_vector, [0.0, 0.0])


def pcd_samples_yescentre_test():
    output = pca(large_samples_data_matrix, centre=True)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_centered_s)
    assert_almost_equal(eigenvectors, centered_eigenvectors_s)
    assert_almost_equal(mean_vector, mean_vector_s)


def pcd_features_nocentre_test():
    output = pca(large_samples_data_matrix.T, centre=False)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_no_centre_f)
    assert_almost_equal(eigenvectors, non_centered_eigenvectors_f)
    assert_almost_equal(mean_vector, np.zeros(10))


def pcd_features_nocentre_inplace_test():
    # important to copy as this will now destructively effect the input data
    # matrix (due to inplace)
    output = pca(large_samples_data_matrix.T.copy(), centre=False,
                 inplace=True)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_no_centre_f)
    assert_almost_equal(eigenvectors, non_centered_eigenvectors_f)
    assert_almost_equal(mean_vector, np.zeros(10))


def pcd_features_yescentre_test():
    output = pca(large_samples_data_matrix.T, centre=True)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_centered_f)
    assert_almost_equal(eigenvectors, centered_eigenvectors_f)
    assert_almost_equal(mean_vector, mean_vector_f)


def eigenvalue_decomposition_default_epsilon_test():
    pos_eigenvectors, pos_eigenvalues = eigenvalue_decomposition(cov_matrix)

    assert_almost_equal(pos_eigenvalues, [4.0, 2.0])
    sqrt_one_over_2 = np.sqrt(2.0) / 2.0
    assert_almost_equal(pos_eigenvectors, [[sqrt_one_over_2, -sqrt_one_over_2],
                                           [sqrt_one_over_2, sqrt_one_over_2]])


def eigenvalue_decomposition_large_epsilon_test():
    pos_eigenvectors, pos_eigenvalues = eigenvalue_decomposition(cov_matrix,
                                                                 eps=0.5)

    assert_almost_equal(pos_eigenvalues, [4.0])
    sqrt_one_over_2 = np.sqrt(2.0) / 2.0
    assert_almost_equal(pos_eigenvectors,
                        [[sqrt_one_over_2], [sqrt_one_over_2]])


def ipca_samples_yescentre_test():
    n_a = large_samples_data_matrix.shape[0] / 2
    A = large_samples_data_matrix[:n_a, :]
    U_a, l_a, m_a = pca(A, centre=True)

    B = large_samples_data_matrix[n_a:, :]
    i_U, i_l, i_m = ipca(B, U_a, l_a, n_a, m_a=m_a)

    b_U, b_l, b_m = pca(large_samples_data_matrix, centre=True)

    assert_almost_equal(np.abs(i_U), np.abs(b_U))
    assert_almost_equal(i_l, b_l)
    assert_almost_equal(i_m, b_m)


def ipca_samples_nocentre_test():
    n_a = large_samples_data_matrix.shape[0] / 2
    A = large_samples_data_matrix[:n_a, :]
    U_a, l_a, m_a = pca(A, centre=False)

    B = large_samples_data_matrix[n_a:, :]
    i_U, i_l, i_m = ipca(B, U_a, l_a, n_a, m_a=m_a)

    b_U, b_l, b_m = pca(large_samples_data_matrix, centre=False)

    assert_almost_equal(np.abs(i_U), np.abs(b_U))
    assert_almost_equal(i_l, b_l)
    assert_almost_equal(i_m, b_m)


def ipca_features_yescentre_test():
    C = np.vstack((large_samples_data_matrix.T, large_samples_data_matrix.T))

    n_a = C.shape[0] / 2
    A = C[:n_a, :]
    U_a, l_a, m_a = pca(A, centre=True)

    B = C[n_a:, :]
    i_U, i_l, i_m = ipca(B, U_a, l_a, n_a, m_a=m_a)

    b_U, b_l, b_m = pca(C, centre=True)

    assert_almost_equal(np.abs(i_U), np.abs(b_U))
    assert_almost_equal(i_l, b_l)
    assert_almost_equal(i_m, b_m)


def ipca_features_nocentre_test():
    C = np.vstack((large_samples_data_matrix.T, large_samples_data_matrix.T))

    n_a = C.shape[0] / 2
    A = C[:n_a, :]
    U_a, l_a, m_a = pca(A, centre=False)

    B = C[n_a:, :]
    i_U, i_l, i_m = ipca(B, U_a, l_a, n_a, m_a=m_a)

    b_U, b_l, b_m = pca(C, centre=False)

    assert_almost_equal(np.abs(i_U), np.abs(b_U))
    assert_almost_equal(i_l, b_l)
    assert_almost_equal(i_m, b_m)

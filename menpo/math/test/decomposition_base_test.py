import numpy as np
from numpy.testing import assert_almost_equal
from menpo.math import eigenvalue_decomposition, \
    principal_component_decomposition

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
eigenvalues_no_centre_no_bias_s = np.array([8.97738481, 0.04928186])
eigenvalues_centered_biased_s = np.array([1.15562494, 0.04417506])
eigenvalues_no_centre_biased_s = np.array([8.07964633, 0.04435367])
eigenvalues_centered_no_bias_s = np.array([1.28402771, 0.0490834])

centered_eigenvectors_f = np.array([[-0.09901475, 0.19802951, 0.69310328,
                                     0.29704426, -0.09901475, 0.39605902,
                                     -0.39605902, 0.09901475, 0.09901475,
                                     -0.19802951]])
centered_eigenvectors_biased_f = np.array([[-0.13864839, 0.27729678,
                                            0.97053872, 0.41594517,
                                            -0.13864839, 0.55459355,
                                            -0.55459355, 0.13864839,
                                            0.13864839, -0.27729678]])
non_centered_eigenvectors_biased_f = np.array(
    [[0.04284044, 0.01054804, 0.04479142, 0.03594266, 0.05333815,
      0.0438411, 0.03139242, 0.01839615, 0.02714423, 0.01744583],
     [-0.3840268, 0.26369659, 0.88167249, 0.29008842, -0.43904756,
      0.40818153, -0.802497, 0.06307234, 0.0172217, -0.41041863]])
non_centered_eigenvectors_f = np.array(
    [[0.38507927, 0.09481302, 0.40261598, 0.32307722, 0.4794398, 0.39407387,
      0.28217662, 0.16535718, 0.24399096, 0.15681507],
     [-0.25575629, 0.17561812, 0.58718113, 0.19319469, -0.29239933, 0.27184299,
      -0.5344514, 0.04200527, 0.01146941, -0.27333287]])
mean_vector_f = np.array([2.45, 0.6, 2.55, 2.05, 3.05,
                          2.5, 1.8, 1.05, 1.55, 1.])
eigenvalues_no_centre_no_bias_f = np.array([80.79646326, 0.44353674])
eigenvalues_centered_biased_f = np.array([0.255])
eigenvalues_no_centre_biased_f = np.array([40.39823163, 0.22176837])
eigenvalues_centered_no_bias_f = np.array([0.51])


# whiten,centre,bias (samples)
# 000
def pcd_samples_nowhiten_nocentre_nobias_test():
    output = principal_component_decomposition(large_samples_data_matrix,
                                               centre=False, whiten=False,
                                               bias=False)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_no_centre_no_bias_s)
    assert_almost_equal(eigenvectors, non_centered_eigenvectors_s)
    assert_almost_equal(mean_vector, [0.0, 0.0])


# 001
def pcd_samples_nowhiten_nocentre_yesbias_test():
    output = principal_component_decomposition(large_samples_data_matrix,
                                               centre=False, bias=True,
                                               whiten=False)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_no_centre_biased_s)
    assert_almost_equal(eigenvectors, non_centered_eigenvectors_s)
    assert_almost_equal(mean_vector, [0.0, 0.0])


# 010
def pcd_samples_nowhiten_yescentre_nobias_test():
    output = principal_component_decomposition(large_samples_data_matrix,
                                               whiten=False, centre=True,
                                               bias=False)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_centered_no_bias_s)
    assert_almost_equal(eigenvectors, centered_eigenvectors_s)
    assert_almost_equal(mean_vector, mean_vector_s)


# 011
def pcd_samples_nowhiten_yescentre_yesbias_test():
    output = principal_component_decomposition(large_samples_data_matrix,
                                               bias=True, centre=True,
                                               whiten=False)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_centered_biased_s)
    assert_almost_equal(eigenvectors, centered_eigenvectors_s)
    assert_almost_equal(mean_vector, mean_vector_s)


# 100
def pcd_samples_yeswhiten_nocentre_nobias_test():
    output = principal_component_decomposition(large_samples_data_matrix,
                                               centre=False, whiten=True,
                                               bias=False)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_no_centre_no_bias_s)
    assert_almost_equal(eigenvectors.T / np.sqrt(1.0 / eigenvalues),
                        non_centered_eigenvectors_s.T)
    assert_almost_equal(mean_vector, [0.0, 0.0])


# 101
def pcd_samples_yeswhiten_nocentre_yesbias_test():
    output = principal_component_decomposition(large_samples_data_matrix,
                                               bias=True, centre=False,
                                               whiten=True)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_no_centre_biased_s)
    assert_almost_equal(eigenvectors.T / np.sqrt(1.0 / eigenvalues),
                        non_centered_eigenvectors_s.T)
    assert_almost_equal(mean_vector, [0.0, 0.0])


# 110
def pcd_samples_yeswhiten_yescentre_nobias_test():
    output = principal_component_decomposition(large_samples_data_matrix,
                                               whiten=True, centre=True,
                                               bias=False)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_centered_no_bias_s)
    assert_almost_equal(eigenvectors.T / np.sqrt(1.0 / eigenvalues),
                        centered_eigenvectors_s.T)
    assert_almost_equal(mean_vector, mean_vector_s)


# 111
def pcd_samples_yeswhiten_yescentre_yesbias_test():
    output = principal_component_decomposition(large_samples_data_matrix,
                                               whiten=True, centre=True,
                                               bias=True)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_centered_biased_s)
    assert_almost_equal(eigenvectors.T / np.sqrt(1.0 / eigenvalues),
                        centered_eigenvectors_s.T)
    assert_almost_equal(mean_vector, mean_vector_s)


# whiten,centre,bias (features)
# 000
def pcd_features_nowhiten_nocentre_nobias_test():
    output = principal_component_decomposition(large_samples_data_matrix.T,
                                               centre=False, whiten=False,
                                               bias=False)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_no_centre_no_bias_f)
    assert_almost_equal(eigenvectors, non_centered_eigenvectors_f)
    assert_almost_equal(mean_vector, np.zeros(10))


def pcd_features_nowhiten_nocentre_nobias_inplace_test():
    # important to copy as this will now destructively effect the input data
    # matrix (due to inplace)
    output = principal_component_decomposition(large_samples_data_matrix.T.copy(),
                                               centre=False, whiten=False,
                                               bias=False, inplace=True)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_no_centre_no_bias_f)
    assert_almost_equal(eigenvectors, non_centered_eigenvectors_f)
    assert_almost_equal(mean_vector, np.zeros(10))


# 001
def pcd_features_nowhiten_nocentre_yesbias_test():
    output = principal_component_decomposition(large_samples_data_matrix.T,
                                               centre=False, bias=True,
                                               whiten=False)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_no_centre_biased_f)
    assert_almost_equal(eigenvectors, non_centered_eigenvectors_f)
    assert_almost_equal(mean_vector, np.zeros(10))


# 010
def pcd_features_nowhiten_yescentre_nobias_test():
    output = principal_component_decomposition(large_samples_data_matrix.T,
                                               whiten=False, centre=True,
                                               bias=False)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_centered_no_bias_f)
    assert_almost_equal(eigenvectors, centered_eigenvectors_f)
    assert_almost_equal(mean_vector, mean_vector_f)


# 011
def pcd_features_nowhiten_yescentre_yesbias_test():
    output = principal_component_decomposition(large_samples_data_matrix.T,
                                               bias=True, centre=True,
                                               whiten=False)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_centered_biased_f)
    assert_almost_equal(eigenvectors, centered_eigenvectors_f)
    assert_almost_equal(mean_vector, mean_vector_f)


# 100
def pcd_features_yeswhiten_nocentre_nobias_test():
    output = principal_component_decomposition(large_samples_data_matrix.T,
                                               centre=False, whiten=True,
                                               bias=False)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_no_centre_no_bias_f)
    assert_almost_equal(eigenvectors.T / np.sqrt(1.0 / eigenvalues),
                        non_centered_eigenvectors_f.T)
    assert_almost_equal(mean_vector, np.zeros(10))


# 101
def pcd_features_yeswhiten_nocentre_yesbias_test():
    output = principal_component_decomposition(large_samples_data_matrix.T,
                                               bias=True, centre=False,
                                               whiten=True)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_no_centre_biased_f)
    assert_almost_equal(eigenvectors, non_centered_eigenvectors_biased_f)
    assert_almost_equal(mean_vector, np.zeros(10))


# 110
def pcd_features_yeswhiten_yescentre_nobias_test():
    output = principal_component_decomposition(large_samples_data_matrix.T,
                                               whiten=True, centre=True,
                                               bias=False)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_centered_no_bias_f)
    assert_almost_equal(eigenvectors, centered_eigenvectors_biased_f)
    assert_almost_equal(mean_vector, mean_vector_f)


# 111
def pcd_features_yeswhiten_yescentre_yesbias_test():
    output = principal_component_decomposition(large_samples_data_matrix.T,
                                               whiten=True, centre=True,
                                               bias=True)
    eigenvectors, eigenvalues, mean_vector = output

    assert_almost_equal(eigenvalues, eigenvalues_centered_biased_f)
    assert_almost_equal(eigenvectors, centered_eigenvectors_biased_f)
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

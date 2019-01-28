import numpy as np
def L2_distance(X, Y):
    """
    Computes pairwise distance between each vector of the two lists. The vectors has to be of same length.

    Eq. || X_i - Y_j || ^ 2 = || X_i || ^ 2 + || Y_j || ^ 2 - 2* transpose(X_i) * Y_j

    :param X: List of vectors
    :param Y: List of vectors
    :return D: A matrix of dimension len(X) * len(Y), D_ij containing the L2 distance between X_i and Y_j
    """

    X_square = sum(X * X, 0).reshape(X.shape[1], 1)
    Y_square = sum(Y * Y, 0)

    return X_square + Y_square - 2 * np.matmul(np.transpose(X), Y)


if __name__ == '__main__':
    # Unit test for L2_distance

    X = np.array([[1, 2, 3], [4, 5, 6]])
    Y = np.array([[4, 5, 6, 11], [7, 8, 9, 10]])
    D = L2_distance(X, Y)
    expected_D = np.array([[18, 32, 50, 136], [8, 18, 32, 106], [2, 8, 18, 80]])
    assert ((D == expected_D).sum() == 12), "Test failed"

    D = L2_distance(X, X)
    expected_D = np.array([[0, 2, 8], [2, 0, 2], [8, 2, 0]])
    assert ((D == expected_D).sum() == 9), "Test failed"

    D = L2_distance(Y, Y)
    expected_D = np.array([[0, 2, 8, 58], [2, 0, 2, 40], [8, 2, 0, 26], [58, 40, 26, 0]])
    assert ((D == expected_D).sum() == 16), "Test failed"

    print("Test passed")

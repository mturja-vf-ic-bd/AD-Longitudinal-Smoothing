import numpy as np
def L2_distance(X, Y):
    """
    Computes pairwise distance between each vector of the two lists. The vectors has to be of same length.

    Eq. || X_i - Y_j || ^ 2 = || X_i || ^ 2 + || Y_j || ^ 2 - 2* transpose(X_i) * Y_j

    :param X: List of vectors
    :param Y: List of vectors
    :return D: A matrix of dimension len(X) * len(Y), D_ij containing the L2 distance between X_i and Y_j
    """

    X_square = sum(X * X, 0) # dimension = (1, X.shape[1])
    Y_square = sum(Y * Y, 0) # dimension = (1, Y.shape[1])

    X_square_tiled = np.transpose(np.tile(X_square, (Y.shape[1], 1))) # copy same row Y.shape[1] times
    Y_square_tiled = np.tile(Y_square, (X.shape[1], 1))  # copy same row X.shape[1] times

    # Now both are of X.shape[1] * Y.shape[1] dimension

    return X_square_tiled + Y_square_tiled - 2 * np.matmul(np.transpose(X), Y)


if __name__ == '__main__':
    # Unit test for L2_distance

    X = np.array([[1, 2, 3], [4, 5, 6]])
    Y = np.array([[4, 5, 6, 11], [7, 8, 9, 10]])
    D = L2_distance(X, Y)
    expected_D = np.array([[18, 32, 50, 136], [8, 18, 32, 106], [2, 8, 18, 80]])

    assert ((D == expected_D).sum() == 12), "Test failed"
    print("Test passed")

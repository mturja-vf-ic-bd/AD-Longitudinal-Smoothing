from compare_network import *
import numpy as np


class RBF:
    def __init__(self, sigma, lamb, debug):
        self.sigma = sigma
        self.lambda_m = lamb
        self.debug = debug

    def rbf_optimization(self, S):
        n = len(S)
        W = np.ones(n) / n
        time_points = [i for i in range(1, len(S) + 1)]
        nIter = 1000
        threshold = 10 ** -10
        alpha = 0.1

        for it in range(0, nIter):
            S_cont = self.comp_rbf_val(W, time_points)
            d_z = 2 * (-(S - S_cont) + self.lambda_m * W)
            if (abs(d_z) > threshold).sum() == 0:
                break

            W = W - d_z * alpha

        return W

    def get_features(self, time_points):
        time_points = np.asarray(time_points)
        time_points_tiled = np.tile(time_points, (len(time_points), 1))
        radial_basis = np.exp(- (time_points_tiled - np.transpose(time_points_tiled)) ** 2 / (2 * self.sigma ** 2))
        return radial_basis

    def comp_rbf_val(self, W, time_points):
        W = np.asarray(W)
        time_points = np.asarray(time_points)
        time_points_tiled = np.transpose(np.tile(time_points, (len(W), 1)))

        T = [i + 1 for i in range(0, len(W))]
        T = np.tile(T, (len(time_points), 1))
        radial_basis = np.exp(- (time_points_tiled - T) ** 2 / (2 * self.sigma ** 2))
        return np.matmul(radial_basis, np.transpose(W))

    def filter_negative(self, matrix):
        matrix = np.asarray(matrix)
        row, col = matrix.shape
        matrix = [matrix[i, j] if matrix[i, j] > 0 else 0 for i in range(0, row) for j in range(0, col)]
        return np.asarray(matrix).reshape((row, col))

    def get_result(self, S):
        t = [i + 1 for i in range(0, len(S))]
        X = self.get_features(t)
        W = np.matmul(
            np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X) +
                                    np.multiply(self.lambda_m, np.identity(len(t)))),
                      np.transpose(X)), np.transpose(S))  # Y = (X^T*X)^-1 * X^T * Y^T

        S_pred = np.matmul(X, np.transpose(W))

        return S_pred

    def fit_rbf_to_longitudinal_connectomes(self, connectome_list):
        n = len(connectome_list[0])
        S_list = []

        # Find the links that exist in every connectomes along the time
        S_ones = np.ones((n, n))
        for mat in connectome_list:
            S_ones = np.multiply(mat, S_ones)

        pos = [(i, j) if S_ones[i, j] > 0 and i > j else (-1, -1) for i in range(0, n) for j in range(0, n)]
        pos = list(filter(lambda a: a != (-1, -1), pos))

        if self.debug:
            print("Number of links: ", len(pos))

        for i in range(0, len(pos)):
            S = []
            row, col = pos[i]

            if self.debug:
                print("row, col = ", row, col, "\n")

            for mat in connectome_list:
                S.append(mat[row][col])

            S_list.append(S)

        long_link_val = []

        for S in S_list:
            S = np.asarray(S)
            val = self.get_result(S)
            if self.debug:
                print("\nactual val: ", S,
                      "\noutput : ", val)
            long_link_val.append(val)

        # Generate new set of connectomes from the interpolated values
        cont_mat_list = []
        for t in range(0, len(connectome_list)):
            cont_mat = np.zeros((n, n))
            for i in range(0, len(long_link_val)):
                row, col = pos[i]
                cont_mat[row, col] = long_link_val[i][t]
                cont_mat[col, row] = cont_mat[row, col]
            cont_mat_list.append(cont_mat)

        # Replace negative values with zero
        filtered_cont_mat_list = []
        for mat in cont_mat_list:
            filtered_cont_mat_list.append(self.filter_negative(mat))

        return filtered_cont_mat_list


if __name__ == '__main__':
    rbf = RBF(1, 0.1, True)
    S = [10, 11, 8, 5]
    S_pred = rbf.get_result(S)
    print("\nprediction: ", S_pred,
          "\nactual: ", S)


from utils.readFile import *
from utils.EProjSimplex import EProjSimplex

def rbf_optimization(S):
    n = len(S)
    W = np.ones(n) * 1/n
    iter = 100
    threshold = 10 ** -3
    alpha = 0.2

    for it in range(0, iter):
        print("Iteration : ", it)
        d_z = (1 - S) ** 2 + 2 * W
        #if (abs(d_z) > threshold).sum() == 0:
           # break

        W = W - d_z * alpha

    return W


def comp_rbf_val(W, time_points, sigma):
    W = np.asarray(W)
    time_points = np.asarray(time_points)
    time_points_tiled = np.transpose(np.tile(time_points, (len(W), 1)))
    T = [i for i in range(0, len(W))]
    T = np.tile(T, (len(time_points), 1))
    return np.matmul((np.exp(- (time_points_tiled - T) ** 2) / (2 * sigma ** 2)), np.transpose(W))

if __name__ == "__main__":
    mat_list = readMatricesFromDirectory("016_S_4121")

    max_val = []
    for mat in mat_list:
        max_val.append(np.amax(mat))

    max_val = max_val - np.mean(max_val)

    max_val = np.asarray(max_val)
    W = rbf_optimization(max_val)
    time_points = [i for i in range(1, len(max_val) + 1)]
    val = comp_rbf_val(W, time_points, 1)
    print("\nactual val: ", max_val,
          "\noutput : ", val,
          "\nW = ", W)

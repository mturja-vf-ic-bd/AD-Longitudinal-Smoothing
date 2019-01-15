from utils.readFile import *

def rbf_optimization(S, sigma, lambda_m):
    n = len(S)
    W = np.ones(n) * 1/n
    time_points = [i for i in range(1, len(max_val) + 1)]
    nIter = 1000
    threshold = 10 ** -10
    alpha = 0.2

    for it in range(0, nIter):
        print("Iteration : ", it)
        # S_cont = comp_rbf_val(W, time_points, sigma)
        S_cont = comp_avg3_val(S, W)
        # S_cont = np.exp(S_cont) / sum(S_cont)
        d_z = 2 * (-(S - S_cont) + lambda_m * W)
        if (abs(d_z) > threshold).sum() == 0:
            break

        W = W - d_z * alpha

    return W


def comp_rbf_val(W, time_points, sigma):
    W = np.asarray(W)
    time_points = np.asarray(time_points)
    time_points_tiled = np.transpose(np.tile(time_points, (len(W), 1)))
    T = [i for i in range(0, len(W))]
    T = np.tile(T, (len(time_points), 1))
    return np.matmul((np.exp(- (time_points_tiled - T) ** 2) / (2 * sigma ** 2)), np.transpose(W))


def comp_avg3_val(S, W):
    S_cont = [S[0]]
    for i in range(1, len(S) - 1):
        S_cont.append(W[i - 1] * S[i - 1] + W[i] * S[i] + W[i + 1] * S[i + 1])

    S_cont.append(S[len(S) - 1])

    return S_cont


if __name__ == "__main__":
    mat_list = readMatricesFromDirectory("016_S_4121")

    max_val = []
    for mat in mat_list:
        max_val.append(np.amax(mat))

    max_val = np.asarray(max_val)
    sigma = 10
    lambda_m = 0.4
    W = rbf_optimization(max_val, sigma, lambda_m)
    time_points = [i for i in range(1, len(max_val) + 1)]
    val = comp_avg3_val(max_val, W)
    print("\nactual val: ", max_val,
          "\noutput : ", val,
          "\nW = ", W)

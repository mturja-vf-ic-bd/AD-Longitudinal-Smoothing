from utils.readFile import *
from matplotlib import pyplot as plt

def rbf_optimization(S, sigma, lambda_m):
    S = np.insert(S, len(S), S[len(S) - 1])
    S = np.insert(S, len(S), S[len(S) - 1])
    S = np.insert(S, 0, S[0])
    S = np.insert(S, 0, S[0])
    print("\nS : ", S)
    n = len(S)
    W = np.ones(n) / n
    time_points = [i for i in range(1, len(S) + 1)]
    nIter = 1000
    threshold = 10 ** -10
    alpha = 0.2

    for it in range(0, nIter):
        print("Iteration : ", it)
        # S_cont = comp_rbf_val(W, time_points, sigma)
        S_cont = comp_rbf_val(W, time_points, sigma)
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

    T = [i+1 for i in range(0, len(W))]
    T = np.tile(T, (len(time_points), 1))
    #print('\ntime_points_tiled: ', time_points_tiled,
    #      '\nT: ', T)

    radial_basis = np.exp(- (time_points_tiled - T) ** 2 / (2 * sigma ** 2))
    return np.matmul(radial_basis, np.transpose(W))



if __name__ == "__main__":
    mat_list = readMatricesFromDirectory("016_S_4121")

    max_val = []
    for mat in mat_list:
        max_val.append(np.amax(mat))

    max_val = np.asarray(max_val)
    sigma = 2
    lambda_m = 0.1
    T = [i+1 for i in range(0, len(max_val))]
    step=3
    time_points = [i/step for i in range(step, step*(len(max_val) + 1))]

    W = rbf_optimization(max_val, sigma, lambda_m)

    val = comp_rbf_val(W, time_points, sigma)
    print("\nactual val: ", max_val,
          "\noutput : ", val,
          "\nW = ", W)

    plt.scatter(T, max_val)
    plt.plot(time_points, val)
    plt.show()

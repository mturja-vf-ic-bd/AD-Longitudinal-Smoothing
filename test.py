from utils.readFile import *
from compare_network import *
import pickle
from matplotlib import pyplot as plt
from matplotlib import pylab

def rbf_optimization(S, sigma, lambda_m):
    #S = np.insert(S, len(S), S[len(S) - 1])
    #S = np.insert(S, len(S), S[len(S) - 1])
    #S = np.insert(S, 0, S[0])
    #S = np.insert(S, 0, S[0])
    #print("\nS : ", S)
    n = len(S)
    W = np.ones(n) / n
    time_points = [i for i in range(1, len(S) + 1)]
    nIter = 1000
    threshold = 10 ** -10
    alpha = 0.1

    for it in range(0, nIter):
        #print("Iteration : ", it)
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
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized')
    sub = '094_S_4234'
    mat_list = readMatricesFromDirectory(os.path.join(data_dir, sub))
    '''
    subjects = [f for f in os.listdir(data_dir)]
    out = open('community_stat.txt', 'w+')
    for f in subjects:
        mat_list = readMatricesFromDirectory(os.path.join(data_dir, f))
        out.write(f + ' -> ' + str(getNumberOfComponents(mat_list)) + '\n')
    

    out.close()
    
    
    n = len(mat_list[0])
    S_list = []

    avg = np.ones((n, n))
    for mat in mat_list:
        avg = np.multiply(mat, avg)

    pos = [(i, j) if avg[i, j] > 0 else (-1, -1) for i in range(0, n) for j in range(0, n)]
    pos = list(filter(lambda a: a != (-1, -1), pos))

    for i in range(0, len(pos)):
        S = []
        row, col = pos[i]
        for mat in mat_list:
            S.append(mat[row][col])

        S_list.append(S)

    print(S_list)

    num_of_plot = len(pos)
    i = 1
    long_link_val = []
    for S in S_list:
        S = np.asarray(S)
        sigma = 2
        lambda_m = 0.1
        T = [i + 1 for i in range(0, len(S))]
        step = 1
        time_points = [i / step for i in range(step, step * (len(S) + 1))]

        mean_S = np.mean(S)
        S = S - mean_S

        W = rbf_optimization(S, sigma, lambda_m)

        val = comp_rbf_val(W, time_points, sigma) + mean_S
        S = S + mean_S
        long_link_val.append(val)
        #print("\nactual val: ", S,
         #     "\noutput : ", val,
         #     "\nW = ", W)

    cont_mat_list = []
    for t in range(0, len(mat_list)):
        cont_mat = np.zeros((n, n))
        for i in range(0, len(long_link_val)):
            row, col = pos[i]
            cont_mat[row, col] = long_link_val[i][t]
        cont_mat_list.append(cont_mat)

    with open('cont_mat_list.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(cont_mat_list, f)
    '''

    cont_mat_list = pickle.load(open('cont_mat_list.pkl', 'rb'))
    print(cont_mat_list)
    print('\nmat_list_len: ', len(mat_list),
          '\ncont_mat_list_len: ', len(cont_mat_list))

    n_comp, _ = get_number_of_components(mat_list)
    n_comp_smoothed, _ = get_number_of_components(cont_mat_list)

    print(n_comp,
          '\n',
          n_comp_smoothed)









import os
from utils.readFile import *
from matplotlib import pyplot as plt
from args import Args


if __name__ == '__main__':
    args = Args()
    data_dir = os.path.join(os.path.dirname(args.root_directory), 'AD-Data_Organized')
    sub = '027_S_4926'
    connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub))
    smoothed_connectomes = readMatricesFromDirectory(os.path.join(data_dir, sub+'_smoothed'), False)
    A = []
    S = []
    ind = np.argsort(smoothed_connectomes[1], axis=None)
    ind1 = ind // 148
    ind2 = ind % 148
    n_ind = 6
    offset = 0
    max_five_ind = zip(ind1[len(ind1) - n_ind - offset: len(ind1) - offset],
                       ind2[len(ind1) - n_ind - offset: len(ind1) - offset])
    count = 1
    for a, b in max_five_ind:
        S = []
        A = []
        for t in range(0, len(connectome_list)):
            A.append(connectome_list[t][a, b])
            S.append(smoothed_connectomes[t][a, b])

        plt.subplot(3, 2, count)
        plt.ylim(0, 1)
        plt.plot(A, color='red')
        plt.plot(S, color='blue')
        count = count + 1

    plt.show()

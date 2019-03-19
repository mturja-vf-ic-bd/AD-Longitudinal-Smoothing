from utils.readFile import *
from utils.helper import *
from matplotlib import pyplot as plt
from plot_functions import plot_matrix_all
from args import Args
import json

def plot_matrix(connectome, fname="connectome", vmin=0, vmax=0.25):
    #plt.set_size_inches(10, 10)
    im = plt.matshow(connectome, vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    #plt.show()
    plt.savefig(fname + '.png')


def test_result(sub, connectome_list, smoothed_connectomes, max_ind, plot_all=False):
    args = Args()
    data_dir = os.path.join(os.path.join(args.root_directory, os.pardir), 'AD-Data_Organized')
    file_parcellation_table = os.path.join(
        os.path.join(os.path.join(data_dir, sub), 'helper_files/parcellationTable.json'))
    plot_matrix_all(connectome_list, fname="raw")
    plot_matrix_all(smoothed_connectomes, fname="smoothed")


    # Reading parcellation table to get the coordinates of each region
    with open(file_parcellation_table) as f:
        table = json.load(f)

    row_sum = connectome_list[1].sum(axis=1)
    sum_t = connectome_list[1].sum()
    row_sum_percent = row_sum/sum_t * 100
    n_ind = len(max_ind)
    count = 1

    fig = plt.gcf()
    fig.set_size_inches(30, 15)
    #fig.savefig('test2png.png', dpi=100)

    for a, b in max_ind:
        print("a, b = ", a, b)
        print("a = ", table[a]["name"],
              "b = ", table[b]["name"])
        if (a // 74) == (b // 74):
            print("Connection type: Intra hemispheric")
        else:
            print("Connection type: Inter hemispheric")

        for t in range(0, len(connectome_list)):
            S = smoothed_connectomes[t][a, :]
            A = connectome_list[t][a, :]
            print("\nA sum: ", sum(A),
                  "\nS sum:", sum(S))
            plt.subplot(n_ind, len(connectome_list), count)
            if t == 1:
                plt.xlabel(row_sum_percent[a])
                plt.title(table[a]["name"] + "_" + str(a // 74) + " to \n" + table[b]["name"]
                          + "_" + str(b // 74))
            plt.ylim(0, 1)
            plt.plot(A, color='red')
            plt.plot(S, color='blue')
            count = count + 1

    '''
    for a, b in max_ind:
        print("a = ", table[a]["name"],
              "b = ", table[b]["name"])
        if (a // 74) == (b // 74):
            print("Connection type: Intra hemispheric")
        else:
            print("Connection type: Inter hemispheric")
        S = []
        A = []
        for t in range(0, len(connectome_list)):
            A.append(connectome_list[t][a, b])
            S.append(smoothed_connectomes[t][a, b])

        plt.subplot(n_ind, 1, count)
        #plt.ylim(0, 1)
        #plt.title(table[a]["name"] + "_" + str(a // 74) + " to \n" + table[b]["name"]
        #           + "_" + str(b // 74))
        plt.plot(A, color='red')
        plt.plot(S, color='blue')
        count = count + 1
    '''

    plt.show()


if __name__ == '__main__':
    sub_names = get_subject_names(3)
    sub = '027_S_5110'
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized')
    connectomes, smoothed_connectomes = readSubjectFiles(sub, method="row")
    N = connectomes[0].shape[0]
    max_ind = get_top_links(connectomes[2], count=4, offset=0)

    vmin=0
    vmax=0.005
    '''
    for t in range(0, len(connectomes)):
        connectomes[t] = sort_matrix(connectomes[t])
        smoothed_connectomes[t] = sort_matrix(smoothed_connectomes[t])

    test_result(sub, connectomes, smoothed_connectomes, max_ind, plot_all=True)
    '''
    for t in range(0, len(connectomes)):
        connectomes[t] = threshold(connectomes[t], vmin, vmax)
        smoothed_connectomes[t] = threshold(smoothed_connectomes[t], vmin, vmax)

    ind = 0
    plot_matrix(connectomes[ind], "raw_single", vmin=vmin, vmax=vmax)
    plot_matrix(smoothed_connectomes[ind], "smooth_single", vmin=vmin, vmax=vmax)




    #connectome_list_noisy = add_noise_all(connectome_list, t=0.1)
    #smoothed_connectomes_noisy, M, E = optimize_longitudinal_connectomes(connectome_list_noisy, Args.dfw, Args.sw,
     #                                                                    Args.lmw,
     #                                                                    Args.lmd)
    #max_ind = get_top_links(connectome_list_noisy[1], count=4, offset=0)
    #test_result(sub, connectome_list_noisy, smoothed_connectomes_noisy, max_ind, plot_all=True)
    '''
    # Compute noise in raw
    threshold = 0.0007  # threshold for raw connectome to have comaparable sparsity with smoothed version
    noise_rw = 0
    noise_sm = 0
    for t in range(0, len(connectome_list)):
        noise_rw = noise_rw + abs((connectome_list[t] > threshold) * connectome_list -
                                  (connectome_list_noisy[t] > threshold) * connectome_list_noisy).sum()
        noise_sm = noise_sm + abs(smoothed_connectomes[t] - smoothed_connectomes_noisy[t]).sum()

    noise_rw = noise_rw / (N * N)
    noise_sm = noise_sm / (N * N)
    print(np.log10(1/noise_rw), np.log10(1/noise_sm))
    #max_ind = get_top_links(connectome_list_noisy[1], count=4, offset=0)
    
    '''

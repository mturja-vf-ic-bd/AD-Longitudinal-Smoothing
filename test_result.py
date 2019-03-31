from utils.readFile import *
from utils.helper import *
from plot_functions import plot_matrix_all
from args import Args
import json
from Evaluation.modularity_analysis import sort_connectomes_by_modularity


def test_result(sub, connectome_list, smoothed_connectomes, max_ind, plot_all=False):
    args = Args()
    data_dir = os.path.join(os.path.join(args.root_directory, os.pardir), 'AD-Data_Organized')
    file_parcellation_table = os.path.join(
        os.path.join(os.path.join(data_dir, sub), 'helper_files/parcellationTable.json'))
    if plot_all:
        plot_matrix_all(connectome_list, fname="raw")
        plot_matrix_all(smoothed_connectomes, fname="smoothed")


    # Reading parcellation table to get the coordinates of each region
    with open(file_parcellation_table) as f:
        table = json.load(f)

    n_ind = len(max_ind)
    '''
    row_sum = connectome_list[1].sum(axis=1)
    sum_t = connectome_list[1].sum()
    row_sum_percent = row_sum/sum_t * 100
    count = 1
    
    fig = plt.gcf()
    fig.set_size_inches(30, 15)

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
    count=1
    import pylab
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

        pylab.subplot(n_ind, 1, count)
        #plt.axis('off')
        #plt.ylim(0.59, 0.70)
        #plt.title(table[a]["name"] + "_" + str(a // 74) + " to \n" + table[b]["name"]
        #           + "_" + str(b // 74))
        time = np.arange(len(A))
        pylab.plot(time, A, '-r', label='Raw')
        pylab.plot(time, S, '-b', label='Intrinsic')
        #pylab.legend(loc='lower right', prop={'size': 15})
        count = count + 1

    pylab.show()


if __name__ == '__main__':
    sub_names = get_subject_names(3)
    sub = '027_S_5110'
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized')
    connectomes, smoothed_connectomes = readSubjectFiles(sub, method="row")
    N = connectomes[0].shape[0]

    c_i = None
    s_i = None

    for t in range(0, len(connectomes)):
        connectomes[t], c_i = sort_connectomes_by_modularity(connectomes[t], c_i)
        smoothed_connectomes[t], s_i = sort_connectomes_by_modularity(smoothed_connectomes[t], s_i)
        smoothed_connectomes[t] = rescale_sm_mat_to_raw(connectomes[t], smoothed_connectomes[t])


    max_ind = get_top_links(connectomes[1], count=1, offset=0)
    test_result(sub, connectomes, smoothed_connectomes, max_ind, plot_all=True)


    '''
    vmin=0
    vmax=0.003
    for t in range(0, len(connectomes)):
        connectomes[t] = threshold(connectomes[t], vmin, vmax)
        smoothed_connectomes[t] = threshold(smoothed_connectomes[t], vmin, vmax)
    
    ind = 0
    plot_matrix_all([connectomes[ind], smoothed_connectomes[ind]], vmin=vmin, vmax=vmax, fname="noise", savefig=True)
    #plot_matrix(connectomes[ind], "raw_single", )
    #plot_matrix(smoothed_connectomes[ind], "smooth_single", vmin=vmin, vmax=vmax)
    
    #max_ind = get_top_links(connectome_list_noisy[1], count=4, offset=0)
    #test_result(sub, connectome_list_noisy, smoothed_connectomes_noisy, max_ind, plot_all=True)
    #print(compute_psnr([sub], 0.1))
    #max_ind = get_top_links(connectome_list_noisy[1], count=4, offset=0)
    '''




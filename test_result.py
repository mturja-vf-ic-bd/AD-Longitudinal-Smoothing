from utils.readFile import *
from utils.helper import *
from matplotlib import pyplot as plt
from args import Args
import json

def test_result(sub, connectome_list, smoothed_connectomes, max_ind):
    args = Args()
    data_dir = os.path.join(os.path.join(args.root_directory, os.pardir), 'AD-Data_Organized')
    file_parcellation_table = os.path.join(
        os.path.join(os.path.join(data_dir, sub), 'helper_files/parcellationTable.json'))

    vmin = 0
    vmax = 0.1
    for t in range(0, len(connectome_list)):
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(connectome_list[t], vmin=vmin, vmax=vmax, cmap='bwr')
        plt.colorbar(orientation='horizontal')
        plt.subplot(1, 2, 2)
        plt.imshow(smoothed_connectomes[t], vmin=vmin, vmax=vmax, cmap='bwr')
        plt.colorbar(orientation='horizontal')
        plt.show()
        fig.savefig("matplt" + str(t) + ".pdf", bbox_inches='tight')


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
    fig.savefig('test2png.png', dpi=100)

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
            #plt.ylim(0, 1)
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
    sub = '027_S_2336'
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized')
    connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub), False)
    max_ind = get_top_links(connectome_list[0], count=5, offset=2000)
    connectome_list = [normalize_matrix(connectome_list[t]) for t in range(0, len(connectome_list))]
    smoothed_connectome = readMatricesFromDirectory(os.path.join(data_dir, sub + '_smoothed'), False)
    test_result(sub, connectome_list, smoothed_connectome, max_ind)

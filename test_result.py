from utils.readFile import *
from matplotlib import pyplot as plt
from args import Args
import json


if __name__ == '__main__':
    args = Args()
    data_dir = os.path.join(os.path.dirname(args.root_directory), 'AD-Data_Organized')
    sub = '027_S_2336'
    connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub))
    smoothed_connectomes = readMatricesFromDirectory(os.path.join(data_dir, sub+'_smoothed'), False)
    args = Args()
    data_dir = os.path.join(os.path.join(args.root_directory, os.pardir), 'AD-Data_Organized')
    file_parcellation_table = os.path.join(
        os.path.join(os.path.join(data_dir, sub), 'helper_files/parcellationTable.json'))

    for t in range(0, len(connectome_list)):
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(connectome_list[t])
        #plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(smoothed_connectomes[t])
        #plt.colorbar()
        #plt.show()
        fig.savefig("matplt" + str(t) + ".pdf", bbox_inches='tight')
    # Reading parcellation table to get the coordinates of each region
    with open(file_parcellation_table) as f:
        table = json.load(f)
    A = []
    S = []
    ind = np.argsort(connectome_list[2], axis=None)
    ind1 = ind // 148
    ind2 = ind % 148
    n_ind = 3
    offset = 0
    max_five_ind = zip(ind1[len(ind1) - n_ind - offset: len(ind1) - offset],
                       ind2[len(ind1) - n_ind - offset: len(ind1) - offset])
    count = 1

    fig = plt.gcf()
    fig.set_size_inches(25, 15)
    fig.savefig('test2png.png', dpi=100)
    for a, b in max_five_ind:
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
            plt.subplot(3, len(connectome_list), count)
            if t == 1:
                plt.title(table[a]["name"] + "_" + str(a // 74) + " to \n" + table[b]["name"]
                      + "_" + str(b // 74))
            plt.ylim(0, 1)
            plt.plot(A, color='red')
            plt.plot(S, color='blue')
            count = count + 1

    
    '''
    for a, b in max_five_ind:
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

        plt.subplot(3, 2, count)
        plt.ylim(0, 1)
        plt.title(table[a]["name"] + "_" + str(a // 74) + " to \n" + table[b]["name"]
                   + "_" + str(b // 74))
        plt.plot(A, color='red')
        plt.plot(S, color='blue')
        count = count + 1
    '''

    plt.show()

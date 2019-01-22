from utils.readFile import *
from compare_network import get_number_of_components
import json
from pprint import pprint

if __name__ == '__main__':
    data_dir = os.path.join(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir), 'AD-Data_Organized')
    sub = '094_S_4234'
    file_parcellation_table = os.path.join(os.path.join(os.path.join(data_dir, sub), 'helper_files/parcellationTable.json'))

    # Reading parcellation table to get the coordinates of each region
    with open(file_parcellation_table) as f:
        table = json.load(f)

    connectomes = readMatricesFromDirectory(os.path.join(data_dir, sub))  # Reading connectomes
    n_components, connectome_labels = get_number_of_components(connectomes)

    T = len(connectomes)  # Number of time points
    N = len(table)  # Number of regions
    node_size = 4

    for t in range(1, T + 1):
        print("\n----", t, "----\n")
        assert (len(connectome_labels[t - 1]) == N), "Incorrect number of regions"
        with open(os.path.join(os.path.join(data_dir, sub + '/helper_files/brain_net/node_file'), sub + "_t" + str(t) + ".node"), 'w') as node_file:
            print("Writing: ", node_file.name)
            for i in range(0, N):
                node_file.write(str(table[i]["coord"][0]) + " " + str(table[i]["coord"][1]) + " " +
                                str(table[i]["coord"][2]) + " " + str(connectome_labels[t - 1][i] * 10) + " " + str(connectome_labels[t - 1][i]) + " " + table[i]["name"] + "\n")

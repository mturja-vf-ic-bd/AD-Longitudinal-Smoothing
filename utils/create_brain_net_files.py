import json
from collections import Counter
from compare_network import *
from args import Args

def replace_labels(ll1, ll2):
    """

    :param ll1: label list 1
    :param ll2: label list 2
    :return: mapping
    """

    paired_label = zip(ll1, ll2)
    label_histogram = Counter(paired_label)

    sorted_pairs = sorted(label_histogram.items(), key=operator.itemgetter(1))
    mapping = {a: b for (a, b), c in sorted_pairs}
    return np.asarray([mapping.get(n, n) for n in ll1])

def create_brain_net_node_files(sub, tentative_label=[]):
    """

    :param sub: subject name
    :param tentative_label: optional label list which can influence the label numbers
    :return:
    """
    args = Args()
    data_dir = os.path.join(os.path.join(args.root_directory, os.pardir), 'AD-Data_Organized')
    file_parcellation_table = os.path.join(
        os.path.join(os.path.join(data_dir, sub), 'helper_files/parcellationTable.json'))

    # Reading parcellation table to get the coordinates of each region
    with open(file_parcellation_table) as f:
        table = json.load(f)

    connectomes = readMatricesFromDirectory(os.path.join(data_dir, sub))  # Reading connectomes
    n_components, connectome_labels = get_number_of_components(connectomes)
    if len(tentative_label) > 0:
        print(len(tentative_label[0]))
        connectome_labels = [replace_labels(connectome_labels[t], tentative_label[0]) for t in range(0, len(connectome_labels))]


    T = len(connectomes)  # Number of time points
    N = len(table)  # Number of regions
    node_size = 4

    output_dir = os.path.join(data_dir, sub + '/helper_files/brain_net/node_file')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for t in range(1, T + 1):
        print("\n----", t, "----\n")
        print(len(connectome_labels[t - 1]), N)
        assert (len(connectome_labels[t - 1]) == N), "Incorrect number of regions"
        with open(os.path.join(output_dir,
                               sub + "_t" + str(t) + ".node"), 'w') as node_file:
            print("Writing: ", node_file.name)
            for i in range(0, N):
                node_file.write(str(table[i]["coord"][0]) + " " + str(table[i]["coord"][1]) + " " +
                                str(table[i]["coord"][2]) + " " + str(connectome_labels[t - 1][i] * 10) + " " + str(node_size) + " " + table[i]["name"] + "\n")


if __name__ == '__main__':
    # Read data
    data_dir = os.path.join(os.path.join(os.path.dirname(os.getcwd()), os.pardir), 'AD-Data_Organized')
    sub = '094_S_4234'
    connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub))
    connectome_list_smt = readMatricesFromDirectory(os.path.join(data_dir, sub + '_smoothed'))
    args = Args()
    t = 0
    A = connectome_list[t]
    S = connectome_list_smt[t]

    _, ll1 = get_number_of_components([A])
    _, ll2 = get_number_of_components([S])

    print("Before: ", (ll1 == ll2).sum())
    ll1 = replace_labels(ll1, ll2)
    print("After: ", (ll1 == ll2).sum())

    print(
          "\nll1:\n", ll1,
          "\nll2:\n", ll2)

import pickle
import numpy as np
from read_file import get_region_names
import csv
from itertools import zip_longest

def model_intersection(groups):

    pair = []
    coeff = []
    edge_set = []
    for group in groups:
        file_p = 't_pairs_' + group[0] + '_' + group[1] + '.pkl'
        file_c = 'coeff_' + group[0] + '_' + group[1] + '.pkl'

        with open(file_p, 'rb') as f:
            pair.append(pickle.load(f))
        with open(file_c, 'rb') as f:
            coeff.append(pickle.load(f))


    for cf, p in zip(coeff, pair):
        cf = cf.reshape(-1, 3)[:, 0]
        ind = np.nonzero(cf)[0]
        edge_set.append([p[x] for x in ind])

    for i in range(len(edge_set)):
        edge_set[i] = edge_set[i] + [(y, x) for (x, y) in edge_set[i]]

    return edge_set


def write_result(edge_list, groups):
    region_names = get_region_names()
    csv_list = [None] * 6
    i = 0
    node_list = [None] * 3
    for edge, group in zip(edge_list, groups):
        j = i // 2
        if csv_list[i] == None:
            csv_list[i] = []
            csv_list[i+1] = []
            node_list[j] = []

        csv_list[i].append(group[0])
        csv_list[i + 1].append(group[1])
        for elem in edge:
            csv_list[i].append(region_names[elem[0]])
            csv_list[i + 1].append(region_names[elem[1]])
            node_list[j].append(region_names[elem[0]])
            node_list[j].append(region_names[elem[1]])
        node_list[j] = list(set(node_list[j]))
        i = i + 2

    csv_list = zip_longest(*csv_list)

    with open('disc_regions.csv', 'w') as f:
        writer = csv.writer(f,  delimiter=',')
        for row in csv_list:
            writer.writerow(row)

    node_list = zip_longest(*node_list)
    with open('node_list.csv', 'w') as f:
        writer = csv.writer(f)
        for row in node_list:
            writer.writerow(row)

if __name__ == '__main__':
    groups = [['1', '2'], ['1', '3'], ['2', '3']]
    edge_set = model_intersection(groups)
    int_dict = {}
    for group in groups:
        g1 = int(group[0]) - 1
        g2 = int(group[1]) - 1
        int_dict[(g1 + 1, g2 + 1)] = set(edge_set[g1]).intersection(set(edge_set[g2]))

    print(int_dict)
    write_result(edge_set, [["CN-MCI", ""], ["CN-AD", ""], ["MCI-AD", ""]])
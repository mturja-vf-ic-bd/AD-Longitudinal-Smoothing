import pickle
import numpy as np
from read_file import get_region_names
import csv
from itertools import zip_longest
import json
from arg import Args
from utils.sortDetriuxNodes import sort_matrix
from circle_plot import plot_circle

class utils:
    def get_lasso_edge_set(self, groups):
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

    def generate_brain_net_files(self, node_set, color_highlight=2, size_highlight=5, color_other=1, size_other=2):
        # Generates brainnet file to visualize discriminative regions
        lines = []
        with open(Args.HOME_DIR + '/parcellationTable_Ordered.json', 'r') as f:
            table = json.load(f)
            for entry in table:
                coord = entry["coord"]
                name = entry["name"]
                if name in node_set:
                    color = color_highlight
                    size = size_highlight
                else:
                    color = color_other
                    size = size_other

                lines.append("{} {} {} {} {} {}".format(coord[0], coord[1], coord[2], color, size, name))

        with open("disc_node.node", "w+") as f:
            for line in lines:
                f.write(line)
                f.write('\n')
            print("Node file written")

        return lines

    def generate_circle_plot_files(self, edges_set):
        n_nodes = 148
        color_hlt = '#ff0000'
        color_nrm = '#008000'
        order = sort_matrix()
        rvrs_order = {}
        for i, o in enumerate(order):
            rvrs_order[o] = i

        node_colors = []
        mod_edge_sets = []
        for edge_set in edges_set:
            mod_edge_set = []
            node_color = [color_nrm] * n_nodes
            for (r1, r2) in edge_set:
                ro1= rvrs_order[r1]
                ro2 = rvrs_order[r2]
                node_color[ro1] = color_hlt
                node_color[ro2] = color_hlt
                mod_edge_set.append((ro1, ro2))

            node_colors.append(node_color)
            mod_edge_sets.append(mod_edge_set)

        plot_circle(node_colors, node_colors, mod_edge_set, save=False)


    def get_endpoint_set(self, edge_set):
        endpoints = [None] * len(edge_set)
        reg = get_region_names()
        for i, edge in enumerate(edge_set):
            endpoints[i] = []
            for e in edge:
                endpoints[i].append(reg[e[0]])
                endpoints[i].append(reg[e[1]])
            endpoints[i] = list(set(endpoints[i]))

        return endpoints


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
    groups = [['1', '3'], ['1', '2'], ['2', '3']]
    util = utils()
    edge_set = util.get_lasso_edge_set(groups)
    util.generate_circle_plot_files(edge_set)
    # int_dict = {}
    # for group in groups:
    #     g1 = int(group[0]) - 1
    #     g2 = int(group[1]) - 1
    #     int_dict[(g1 + 1, g2 + 1)] = set(edge_set[g1]).intersection(set(edge_set[g2]))
    #
    # print(int_dict)
    # write_result(edge_set, [["CN-MCI", ""], ["CN-AD", ""], ["MCI-AD", ""]])
    # node_set = util.get_endpoint_set(edge_set)
    # util.generate_brain_net_files(node_set[1])
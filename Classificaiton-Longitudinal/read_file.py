from typing import List

import numpy as np
import json
import os
from arg import Args
import csv
import warnings
from sklearn.mixture import GaussianMixture

def read_matrix_from_text_file(network_id, debug=False):
    file_path = os.path.join(Args.NETWORK_DIR, network_id + "_fdt_network_matrix")
    if debug:
        print("Reading File: " + file_path)
    a = []
    try:
        fin = open(file_path, 'r')
        for line in fin.readlines():
            a.append([float(x) for x in line.split()])

        a = np.asarray(a)
        return a
    except FileNotFoundError:
        if Args.DEBUG:
            print("network not found for {}. Returning None".format(network_id))
        return None


def read_parcellation_table(network_id):
    """
    Read the parcellation table containing node features for a network
    :param network_id: network id
    :return: parcellation table
    """
    try:
        with open(os.path.join(Args.PARC_DIR, network_id + "_parcellationTable.json")) as pt:
            table = json.load(pt)
    except FileNotFoundError:
        if Args.DEBUG:
            print("parcellation table not found for {}. Returning None".format(network_id))
        return None
    return table


def convert_to_feat_mat(parc):
    """
    Convert a parcellation table to feature matrix (Certain features are considered)
    :param parc: parcellation table
    :return: feature matrix
    """
    if parc is None:
        return None

    feat_name = ("SurfArea", "GrayVol", "ThickAvg", "NumVert")
    feat_mat = np.zeros((len(parc), len(feat_name)))
    for i, p in enumerate(parc):
        if all(feat in p.keys() for feat in feat_name):
            feat_mat[i][0] = p["SurfArea"]
            feat_mat[i][1] = p["GrayVol"]
            feat_mat[i][2] = p["ThickAvg"]
            feat_mat[i][3] = p["NumVert"]
        else:
            if Args.DEBUG:
                print("Parcellation table doesn't contain the required features")
            return None

    return feat_mat


def read_subject_data(subject_id, data_type='all'):
    """
    Read all the temporal adjacency matrix and their node features
    :param subject_id: subject id
    :return: node features and adjacency matrix at all time points
    """
    with open(Args.SUB_TO_NET_MAP, 'r+') as s2n:
        s2n_map = json.load(s2n)

    if subject_id not in s2n_map.keys():
        if Args.DEBUG:
            print("{} not found in temporal mapping".format(subject_id))
        return

    network_arr = s2n_map[subject_id]
    adj_mat = []
    node_feat = []
    dx_label = []
    for network in network_arr:
        parc_table = read_parcellation_table(network["network_id"])
        network_data = read_matrix_from_text_file(network["network_id"])
        features = convert_to_feat_mat(parc_table)
        if parc_table is None:
            continue
        elif network_data is None:
            continue
        elif features is None:
            continue
        else:
            if Args.DEBUG:
                print("All data found {}".format(network["network_id"]))
            adj_mat.append(network_data)
            node_feat.append(features)
            dx_label.append(network['dx_data'])

    if data_type == 'all':
        return {"node_feature": node_feat, "adjacency_matrix": adj_mat, "dx_label": dx_label}

    elif data_type == 'network':
        return adj_mat
    else:
        return node_feat


def read_full_csv_file(col=[2, 3]):
    filename = os.path.join(Args.OTHER_DIR, "data.csv")
    table = []
    with open(filename, 'r') as f:
        readCSV = csv.reader(f, delimiter=',')
        i = 0
        for row in readCSV:
            if i == 0:
                header = row
                i = i + 1
            else:
                temp_row = []
                for c in col:
                    if type(c) != int:
                        ind = header.index(c)
                    else:
                        ind = c

                    temp_row.append(row[ind])
                table.append(temp_row)

    return table

def read_temporal_mapping():
    with open(os.path.join(Args.OTHER_DIR, 'temporal_mapping.json')) as f:
        temap = json.load(f)

    return temap

def read_all_subjects(data_type='all'):
    data_set = []
    temp_map = read_temporal_mapping()
    for subject in temp_map.keys():
        data_set.append(read_subject_data(subject, data_type))

    return data_set


def get_baselines():
    data_set = read_all_subjects()
    network = []
    feature = []
    dx_label = []

    for item in data_set:
        if not item["adjacency_matrix"]:
            continue

        network.append(item["adjacency_matrix"][0])
        feature.append(item["node_feature"][0])
        dx_label.append(item["dx_label"][0])

    return {"node_feature": feature, "adjacency_matrix": network, "dx_label": dx_label}


def get_region_names():
    r_names = []
    with open(Args.HOME_DIR + '/parcellationTable_Ordered.json', 'r') as f:
        table = json.load(f)
        for i, elem in enumerate(table):
            r_names.append(elem["name"])
    return r_names


if __name__ == '__main__':
    #data_set = get_baselines()
    #print(data_set)
    print(get_region_names())

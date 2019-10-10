import numpy as np
import json
import os
from Longitudinal_Classifier.arg import Args
# from Longitudinal_Classifier.helper import getFiedlerFeature
import csv
import torch

def read_matrix_from_text_file(network_id, net_dir = Args.NETWORK_DIR):
    file_path = os.path.join(net_dir, network_id + "_fdt_network_matrix")
    if Args.DEBUG:
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


def read_subject_data(subject_id, data_type='all', net_dir=Args.NETWORK_DIR, conv_to_tensor=False):
    """
    Read all the temporal adjacency matrix and their node features
    :param data_type: 'all'/'network'/'node_feature'
    :param net_dir: network directory
    :param subject_id: subject id
    :return: node features and adjacency matrix at all time points if data_type is 'all'
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
        network_data = read_matrix_from_text_file(network["network_id"], net_dir)

        if parc_table is None:
            continue

        features = convert_to_feat_mat(parc_table)
        if features is not None:
            features = features[:, 2]
            # features = features / np.linalg.norm(features)

        if network_data is None:
            continue
        elif features is None:
            continue
        else:
            if Args.DEBUG:
                print("All data found {}".format(network["network_id"]))
            adj_mat.append(network_data)

            # features = getFiedlerFeature(np.expand_dims(network_data, axis=0))
            node_feat.append(features)
            dx_label.append(int(network['dx_data']) - 1)

    if conv_to_tensor:
        import torch
        node_feat = torch.FloatTensor(node_feat)
        adj_mat = torch.FloatTensor(adj_mat)
    if data_type == 'all':
        return {"subject": subject_id, "node_feature": node_feat, "adjacency_matrix": adj_mat, "dx_label": dx_label}

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

def read_all_subjects(data_type='all', net_dir=Args.NETWORK_DIR, classes=[0, 1, 2], conv_to_tensor=False):
    data_set = []
    temp_map = read_temporal_mapping()
    count = [0, 0, 0]
    for subject in temp_map.keys():
        data = read_subject_data(subject, data_type, net_dir, conv_to_tensor=conv_to_tensor)
        if len(data['dx_label']) > 0 and data['dx_label'][0] in classes:
            data_set.append(data)
            count[data['dx_label'][0]] = count[data['dx_label'][0]] + 1

    w = torch.FloatTensor(count)
    w = w / torch.sum(w)
    return data_set, w.to(Args.device)


def get_baselines(normalize=False, net_dir=Args.NETWORK_DIR, label=None):
    data_set = read_all_subjects(net_dir=net_dir)
    network = []
    feature = []
    dx_label = []

    for item in data_set:
        if not item["adjacency_matrix"]:
            continue

        net = item["adjacency_matrix"][0]
        if normalize:
            net = (net.T + net) / 2
            net = (net < 0.01) * net
            net /= net.sum()
        network.append(net)
        feature.append(item["node_feature"][0])
        if label is None:
            dx_label.append(int(item["dx_label"][0]))
        else:
            dx_label.append(int(label[item["subject"]]))

    return {"node_feature": feature, "adjacency_matrix": network, "dx_label": dx_label}


def get_region_names():
    r_names = []
    with open(Args.HOME_DIR + '/parcellationTable_Ordered.json', 'r') as f:
        table = json.load(f)
        for i, elem in enumerate(table):
            r_names.append(elem["name"])
    return r_names


def read_csv(fields):
    fname = os.path.join(Args.OTHER_DIR, 'data.csv')
    data = []
    with open(fname, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                header = row
                projection = [header.index(field) for field in fields]
                print(projection)
            else:
                temp = []
                for ind in projection:
                    temp.append(row[ind])
                data.append(temp)
            line_count = line_count + 1
    return np.array(data)



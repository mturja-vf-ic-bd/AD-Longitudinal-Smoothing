import numpy as np
import json
import os
from arg import Args

def read_matrix_from_text_file(network_id, debug=False):
    file_path = os.path.join(Args.NETWORK_DIR, network_id + "_fdt_network_matrix")
    if debug:
        print("Reading File: " + file_path)
    a = []
    fin = open(file_path, 'r')
    for line in fin.readlines():
        a.append([float(x) for x in line.split()])

    a = np.asarray(a)
    return a


def read_parcellation_table(network_id):
    """
    Read the parcellation table containing node features for a network
    :param network_id: network id
    :return: parcellation table
    """
    with open(os.path.join(Args.PARC_DIR, network_id + "_parcellationTable.json")) as pt:
        table = json.load(pt)
    return table


def conver_to_feat_mat(parc):
    """
    Convert a parcellation table to feature matrix (Certain features are considered)
    :param parc: parcellation table
    :return: feature matrix
    """
    n_feat = 4
    feat_mat = np.zeros((len(parc), n_feat))
    for i, p in enumerate(parc):
        feat_mat[i][0] = p["NumVert"]
        feat_mat[i][1] = p["SurfArea"]
        feat_mat[i][2] = p["GrayVol"]
        feat_mat[i][3] = p["ThickAvg"]

    return feat_mat


def read_subject_data(subject_id):
    """
    Read all the temporal adjacency matrix and their node features
    :param subject_id: subject id
    :return: node features and adjacency matrix at all time points
    """
    with open(Args.SUB_TO_NET_MAP, 'r+') as s2n:
        s2n_map = json.load(s2n)

    network_arr = s2n_map[subject_id]
    adj_mat = []
    node_feat = []
    for network in network_arr:
        adj_mat.append(read_matrix_from_text_file(network["network_id"]))
        node_feat.append(conver_to_feat_mat(read_parcellation_table(network["network_id"])))

    return {"node_feature": node_feat, "adjacency_matrix": adj_mat}
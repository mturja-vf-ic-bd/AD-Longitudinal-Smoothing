import numpy as np
import json
import os
from arg import Args
import csv
from sklearn.mixture import GaussianMixture

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


def convert_to_feat_mat(parc):
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
        node_feat.append(convert_to_feat_mat(read_parcellation_table(network["network_id"])))

    return {"node_feature": node_feat, "adjacency_matrix": adj_mat}


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


if __name__ == '__main__':
    mix_model = GaussianMixture(n_components=2, verbose=3, init_params='random'
                                , means_init=[np.mean(EMCI_MMSE, axis=0), np.mean(LMCI_MMSE, axis=0)])
    X = np.concatenate((EMCI_MMSE, LMCI_MMSE))
    labels = mix_model.fit_predict(X)
    print(labels)
    pred_EMCI = X[labels==1]
    pred_LMCI = X[labels==0]
    flip = False
    if len(pred_EMCI) < len(pred_LMCI):
        pred_EMCI, pred_LMCI = pred_LMCI, pred_EMCI
        flip = True

    print("Before: {}, After: {}".format(np.mean(EMCI_MMSE, axis=0), np.mean(pred_EMCI, axis=0)))
    print("Before: {}, After: {}".format(np.mean(LMCI_MMSE, axis=0), np.mean(pred_LMCI, axis=0)))
    print("Count Before: {}, Count After: {}".format(len(EMCI_MMSE), len(pred_EMCI)))
    print("Count Before: {}, Count After: {}".format(len(LMCI_MMSE), len(pred_LMCI)))

    if not flip:
        pred_label_emci = labels[0:len(EMCI_MMSE)]
    else:
        pred_label_emci = labels[0:len(LMCI_MMSE)]
    print("Change in EMCI: {}".format(pred_label_emci.sum()))

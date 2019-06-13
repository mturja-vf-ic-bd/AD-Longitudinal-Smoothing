import pandas as pd
import numpy as np
from read_file import read_temporal_mapping
from sklearn.model_selection import StratifiedKFold
from operator import itemgetter
import torch


def get_Kfold_multilabel(y):
    hash_map = dict()
    for i in range(len(y)):
        id = ''
        for j in range(y.shape[1]):
            id += str(y[i, j])

        hash_map[id] = i
    return hash_map

def categorize_data(data, headers):
    dx_map = {'CN': '0', 'SMC': '0', 'EMCI': '1', 'LMCI': '1', 'AD': '2'}
    sex_map = {'Female': 1, 'Male': 1}
    df = pd.DataFrame(data, columns=headers)
    df.AGE = pd.to_numeric(df.AGE)
    bin = [i for i in range(int(np.min(df.AGE)), int(np.max(df.AGE)), 10)]
    df['AGE'] = pd.cut(df.AGE,
                         bins=bin,
                         labels=[i for i in range(len(bin) -1)])
    df.DX_bl = df.DX_bl.replace(dx_map)
    df.PTGENDER = df.PTGENDER.replace(sex_map)
    df = df.assign(category=df.DX_bl.astype('str') + df.PTGENDER.astype('str') + df.AGE.astype('str'))

    # Replace category with label starting from 0
    cat_map = dict()
    i = -1
    for val in df.category:
        if val not in cat_map.keys():
            i = i + 1
            cat_map[val] = i
    df.category = df.category.replace(cat_map)
    return df

def stratified_sampling_label(data, header):
    df = categorize_data(data, header)
    temp_map = read_temporal_mapping()
    net_id = []
    for key, val in temp_map.items():
        net_id.append(val[0]["network_id"])
    strat_label = df.DX_bl[df.subject.isin(net_id)].to_numpy()
    return dict(zip(temp_map.keys(), strat_label))


def get_train_test_fold(dataset, ratio=0.2):
    data_set_train = dict()
    data_set_test = dict()
    n_fold = int(1/ratio)
    kf = StratifiedKFold(n_splits=n_fold, shuffle=True)
    train_fold = []
    test_fold = []

    for train_index, test_index in kf.split(dataset["adjacency_matrix"], dataset["dx_label"]):
        data_set_test["node_feature"] = itemgetter(*test_index)(dataset["node_feature"])
        data_set_test["adjacency_matrix"] = itemgetter(*test_index)(dataset["adjacency_matrix"])
        data_set_test["dx_label"] = itemgetter(*test_index)(dataset["dx_label"])

        data_set_train["node_feature"] = itemgetter(*train_index)(dataset["node_feature"])
        data_set_train["adjacency_matrix"] = itemgetter(*train_index)(dataset["adjacency_matrix"])
        data_set_train["dx_label"] = itemgetter(*train_index)(dataset["dx_label"])

        test_fold.append(data_set_test)
        train_fold.append(data_set_train)

    return list(zip(train_fold, test_fold))

#
# def process_data(dataset):
#     net_data = dataset["adjacency_matrix"]
#     label = torch.from_numpy(dataset["dx_label"])
#     v_len = net_data[0].shape[0] * net_data[0].shape[1]
#     net_data_flat = torch.from_numpy(np.array(net_data)).view(-1, v_len)
#
#     return net_data_flat, label

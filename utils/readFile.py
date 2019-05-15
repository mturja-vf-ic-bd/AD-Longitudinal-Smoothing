import os
from os.path import join
import numpy as np
from utils.helper import rescale_matrix, get_parcellation_table
import time
from args import Args
import json
from utils.sortDetriuxNodes import sort_matrix


def readMatrixFromTextFile(fname, debug=False):
    if debug == True:
        print("Reading File: " + fname)
    a = []
    fin = open(fname, 'r')
    for line in fin.readlines():
        a.append([float(x) for x in line.split()])

    a = np.asarray(a)
    return a


def readMatricesFromDirectory(directory, normalize=True, method="row"):
    files = [f for f in os.listdir(directory)]
    files.sort()
    mat_list = []
    for file in files:
        file = os.path.join(directory, file)
        if os.path.isdir(file) and Args.debug:
            print(file, " is a directory")
        elif os.path.isfile(file):
            a = readMatrixFromTextFile(join(directory, file))
            if normalize:
                a = normalize_matrix(a, method)
            mat_list.append(a)
        elif Args.debug:
            print(file, " is weird")

    return mat_list


def read_files_from_dir(dirname):
    files = os.listdir(dirname)
    selected_files = []
    for file_name in files:
        file = os.path.join(dirname, file_name)
        if os.path.isfile(file):
            selected_files.append(file_name)

    return selected_files


def read_csv(filename, skiphead=True, project=None):
    import csv
    from operator import itemgetter
    table = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if not skiphead:
                if project is None:
                    table.append(row)
                else:
                    table.append(itemgetter(*project)(row))
            skiphead = False

    return table

def readSubjectFiles(subject, method="whole", sort=True):
    dir_rw = os.path.join(Args.data_directory, subject)
    dir_smth = os.path.join(Args.data_directory, subject + '_smoothed')
    rw_data = readMatricesFromDirectory(dir_rw, True, method)
    smth_data = readMatricesFromDirectory(dir_smth, False)
    if sort:
        rw_data = [sort_matrix(rw)[0] for rw in rw_data]
        smth_data = [sort_matrix(sm)[0] for sm in smth_data]

    if method == "whole":
        for t in range(0, len(rw_data)):
            smth_data[t] = rescale_matrix(smth_data[t], rw_data[t].sum(axis=1))
            smth_data[t] = (smth_data[t] + smth_data[t].T) / 2
            rw_data[t] = (rw_data[t] + rw_data[t].T) / 2

    return rw_data, smth_data


def normalize_matrix(mat, method="row"):
    if method == "row":
        row_sums = mat.sum(axis=1)
        mat /= (row_sums[:, np.newaxis] + Args.eps)
    else:
        mat /= (mat.sum() + Args.eps)

    return mat


def get_subject_info(min_bound=-1):
    table = read_csv(Args.data_file, project=[0, 1, 2, 3])
    table.sort(key=lambda x: (x[1], time.mktime(time.strptime(x[2], "%m/%d/%Y"))))
    sub_info = {}
    for row in table:
        if row[1] in sub_info.keys():
            sub_info[row[1]].append(
                dict(scanId=row[0], date=row[2], DX=row[3]))
        else:
            sub_info[row[1]] = [dict(scanId=row[0], date=row[2], DX=row[3])]

    if min_bound > 0:
        for key in list(sub_info.keys()):
            if sub_info[key].__len__() < min_bound:
                del sub_info[key]
    return sub_info


def write_file(subject, data, name, data_dir=None, suffix=None):
    """
    :param subject: name of the subject
    :param data: list of matrices
    :param name: name of the matrices
    :param suffix: to be added at the end of the directory name (for example: smooth, noisy)
    """

    if data_dir is None:
        data_dir = os.path.join(os.path.join(Args.root_directory, os.pardir), 'AD-Data_Organized')

    if suffix is None:
        data_dir = os.path.join(data_dir, subject)
    else:
        data_dir = os.path.join(data_dir, subject + '_' + suffix)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    for i in range(0, len(data)):
        wfile = os.path.join(data_dir, name[i])
        np.savetxt(wfile, data[i])


def write_node_file(node_size, color, out_file):
    pt = get_parcellation_table()
    with open(out_file, 'w') as f:
        f.write('#destriux 148\n')
        for i, p in enumerate(pt):
            #Fix coordinates
            if i < 74:
                p["coord"][0] = p["coord"][0] - 30
                p["coord"][1] = p["coord"][1]*0.9 - 20
                p["coord"][2] = p["coord"][2] + 10
            else:
                p["coord"][0] = p["coord"][0] + 30
                p["coord"][1] = p["coord"][1]*0.9 - 20
                p["coord"][2] = p["coord"][2] + 10
            f.write("%f %f %f %s %f %s\n" % (p["coord"][0], p["coord"][1],
                    p["coord"][2], color[i],
                    node_size[i]*5, p["name"]))

def read_temporal_mapping():
    with open(os.path.join(Args.other_file, 'temporal_mapping.json')) as f:
        temap = json.load(f)

    return temap

if __name__ == '__main__':
    from args import Args
    import collections
    from matplotlib import pyplot as plt

    args = Args()
    data_dir = os.path.join(os.path.join(args.root_directory, os.pardir), 'AD-Data_Organized')
    rect_files = []
    for f in os.listdir(data_dir):
        if f.find('smoothed') == -1:
            rect_files.append(f)

    count = [len(os.listdir(os.path.join(data_dir, f))) for f in rect_files]
    hist = collections.Counter(count)
    print(hist)

    plt.hist(count, align='left', bins=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.xlabel('#scans')
    plt.ylabel('#subjects')
    plt.show()





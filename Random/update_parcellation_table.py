import json
import csv
from args import Args
import os

def write_table(table, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(table, f)
            print("{} found".format(filename))
    except FileNotFoundError:
        print("{} not found".format(filename))

def add_DX_label(tname, label):
    try:
        with open(tname, 'r') as f:
            table = json.load(f)
            table['DX'] = label
            return table
    except FileNotFoundError:
        print("{} not found".format(tname))


def label_map():
    label_map = {}
    with open(Args.other_file + '/DXData.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] != 'subject':
                label_map[row[0]] = row[3]

    return label_map


def read_temporal_mapping():
    with open(os.path.join(Args.other_file, 'temporal_mapping.json')) as f:
        temap = json.load(f)

    return temap

def write_temporal_mapping(tmap, filename):
    with open(os.path.join(Args.other_file, filename), 'w') as f:
        json.dump(tmap, f)


if __name__ == '__main__':
    lmap = label_map()
    tmap = read_temporal_mapping()

    for key, val in tmap.items():
        for i in range(len(val)):
            val[i]['dx_data'] = lmap[val[i]['network_id']]


    write_temporal_mapping(tmap, os.path.join(Args.other_file, 'temporal_mapping.json'))






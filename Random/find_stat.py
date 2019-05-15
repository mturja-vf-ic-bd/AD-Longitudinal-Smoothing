import os
import json
from args import Args

def read_temporal_mapping():
    with open(os.path.join(Args.other_file, 'temporal_mapping.json')) as f:
        temap = json.load(f)

    return temap

def get_dx_change_count():
    """
    Find out how many subjects change diagnosis labels
    :return: count
    """
    tmap = read_temporal_mapping()
    count = 0
    for key, val in tmap.items():
        if val[0]['dx_data'] != val[-1]['dx_data']:
            print("Label Change in {}".format(key))
            count = count + 1

    print("Total DX label change count : ", count)


if __name__ == '__main__':
    get_dx_change_count()


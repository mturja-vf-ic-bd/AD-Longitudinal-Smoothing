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
    count_nc2mci = 0
    count_mci2ad = 0
    count_nc2ad = 0
    for key, val in tmap.items():
        if val[0]['dx_data'] != val[-1]['dx_data']:
            print("Label Change in {}".format(key))
            if val[0]['dx_data'] == '1' and val[-1]['dx_data'] == '2':
                count_nc2mci = count_nc2mci + 1
            elif val[0]['dx_data'] == '2' and val[-1]['dx_data'] == '3':
                count_mci2ad = count_mci2ad + 1
            elif val[0]['dx_data'] == '1' and val[-1]['dx_data'] == '3':
                print("{} is big change".format(key))
                count_nc2ad = count_nc2ad + 1
            else:
                print("{} is abnomal".format(key))

    print("Total DX label change count : nc2mci : {}, mci2ad : {}, nc2ad : {}".format(count_nc2mci, count_mci2ad, count_nc2ad))


if __name__ == '__main__':
    get_dx_change_count()


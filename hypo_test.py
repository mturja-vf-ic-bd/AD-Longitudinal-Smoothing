from utils.cartesian_product import *
from scipy.stats import ttest_ind
import math
from utils.helper import *

def prepare_data_for_test(sub_info, testname='tst', feat='cc'):
    label_to_sample = {}

    if testname == 'tst':
        for key in sub_info.keys():
            for sub in sub_info[key]:
                if sub["DX"] in label_to_sample.keys():
                    if feat == 'cc':
                        label_to_sample[sub["DX"]].append((sub["scanId"], sub["cc_rw"], sub["cc_sm"]))
                    elif feat == 'mod':
                        label_to_sample[sub["DX"]].append((sub["scanId"], sub["mod_rw"], sub["mod_sm"]))
                    elif feat == 'deg':
                        label_to_sample[sub["DX"]].append((sub["scanId"], sub["d_rw"], sub["d_sm"]))
                    elif feat == 'bc':
                        label_to_sample[sub["DX"]].append((sub["scanId"], sub["bc_rw"], sub["bc_sm"]))

                else:
                    if feat == 'cc':
                        label_to_sample[sub["DX"]] = [(sub["scanId"], sub["cc_rw"], sub["cc_sm"])]
                    elif feat == 'mod':
                        label_to_sample[sub["DX"]] = [(sub["scanId"], sub["mod_rw"], sub["mod_sm"])]
                    elif feat == 'deg':
                        label_to_sample[sub["DX"]] = [(sub["scanId"], sub["d_rw"], sub["d_sm"])]
                    elif feat == 'bc':
                        label_to_sample[sub["DX"]] = [(sub["scanId"], sub["bc_rw"], sub["bc_sm"])]

        return label_to_sample


def two_sample_t_test(subinfo):
    label_to_sample = prepare_data_for_test(subinfo, feat='cc')  # Get data for each label
    label_set = list(label_to_sample.keys())  # set of labels
    # prepare data for two sample t test
    label_pair = []
    for l1, l2 in cartesian([label_set, label_set]):
        set1 = []
        set2 = []
        set3 = []
        set4 = []
        if l1 != l2 and (l2, l1) not in label_pair:
            label_pair.append((l1, l2))
            for _, cc_rw, cc_sm in label_to_sample[l1]:
                if not math.isnan(cc_sm):
                    set1.append(cc_rw)
                    set2.append(cc_sm)
            for _, cc_rw, cc_sm in label_to_sample[l2]:
                if not math.isnan(cc_sm):
                    set3.append(cc_rw)
                    set4.append(cc_sm)

            print(l1, l2)

            print(ttest_ind(set1, set3),
                  "\n", ttest_ind(set2, set4))


def group_consistency(sub_info):
    label_to_data_rw = {}
    label_to_data_sm = {}
    for key in list(sub_info.keys()):
        for t in range(0, len(sub_info[key])):
            rw = sub_info[key][t]["raw_data"]
            sm = sub_info[key][t]["sm_data"]
            label = sub_info[key][t]["DX"]
            if label in label_to_data_rw.keys():
                label_to_data_rw[label].append(rw)
                label_to_data_sm[label].append(sm)
            else:
                label_to_data_rw[label] = [rw]
                label_to_data_sm[label] = [sm]

    mean_list_rw = []
    mean_list_sm = []
    for label in label_to_data_rw.keys():
        rw_list = label_to_data_rw[label]
        sm_list = label_to_data_sm[label]
        mean_rw, _ = findMeanAndStd(rw_list)
        mean_sm, _ = findMeanAndStd(sm_list)
        mean_list_rw.append(mean_rw)
        mean_list_sm.append(mean_sm)
        d_rw = find_distance_between_matrices(mean_list_rw)
        d_sm = find_distance_between_matrices(mean_list_sm)

    return get_entropy_list(d_rw), get_entropy_list(d_sm)





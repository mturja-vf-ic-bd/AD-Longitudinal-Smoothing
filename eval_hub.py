import bct
import numpy as np

from utils.readFile import readSubjectFiles
from utils.helper import threshold_all, get_entropy_list, get_subject_names
from args import Args
from scipy import stats

def get_hubs(connectome, hub_count=6):
    node_betwn = bct.betweenness_wei(connectome)
    idx = node_betwn.argsort()
    n = len(node_betwn)
    node_betwn[idx[0:n-hub_count]] = 0
    node_betwn[idx[n-hub_count:]] = 1

    return node_betwn

def get_hub_match(cn_list, hub_count=6):
    ind_vec = np.array([get_hubs(cn, hub_count) for cn in cn_list])
    return np.mean(ind_vec, axis=0)


def entropy_eval():

    sub_names = get_subject_names(3)
    hub_count = 6
    rw_en_list = []
    sm_en_list = []
    for sub in sub_names:
        rw, sm = readSubjectFiles(sub, "row")
        rw = threshold_all(rw, vmin=Args.threshold)
        rw_hub_match = get_hub_match(rw, hub_count)
        rw_en = get_entropy_list(rw_hub_match)
        sm_hub_match = get_hub_match(sm, hub_count)
        sm_en = get_entropy_list(sm_hub_match)
        rw_en_list.append(rw_en)
        sm_en_list.append(sm_en)
        print("Raw: ", rw_en)
        print("Smooth: ", sm_en)
        '''
        plt.bar(np.arange(0, 148, 1), rw_hub_match)
        plt.ylim(0, 2)
        plt.show()
        plt.bar(np.arange(0, 148, 1), sm_hub_match)
        plt.ylim(0,2)
        plt.show()
        '''

    print("Mean rw: ", np.mean(rw_en),
          "\nMean sm: ", np.mean(sm_en))

    import pickle as pk
    with open('hub_eval.pkl', 'wb') as f:
        print(pk.dump([rw_en_list, sm_en_list], f))


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    #entropy_eval()

    import pickle as pk
    with open('hub_eval.pkl', 'rb') as f:
        rw_en_list, sm_en_list = pk.load(f)

    print(stats.ttest_rel(rw_en_list, sm_en_list))

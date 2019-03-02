from utils.readFile import get_subject_info, readSubjectFiles
from bct import *
from hypo_test import *
from compare_network import n_comp
from utils.helper import get_centrality_measure


if __name__ == '__main__':
    sub_info = get_subject_info()
    count = 0
    min_bound = 6

    for key in list(sub_info.keys())   :
        if sub_info[key].__len__() < min_bound:
            del sub_info[key]
            count = count + 1
        else:
            rw, sm = readSubjectFiles(key)
            for t in range(0, len(sub_info[key])):
                print(key, t)
                sub_info[key][t]["raw_data"] = rw[t]
                sub_info[key][t]["sm_data"] = sm[t]
                sub_info[key][t]["cc_rw"] = np.mean(clustering_coef_wu(rw[t]))
                sub_info[key][t]["cc_sm"] = np.mean(clustering_coef_wu(sm[t]))
                sub_info[key][t]["mod_rw"] = n_comp(rw[t])
                sub_info[key][t]["mod_sm"] = n_comp(sm[t])
                sub_info[key][t]["d_rw"] = np.max(strengths_und(rw[t]))
                sub_info[key][t]["d_sm"] = np.max(strengths_und(sm[t]))
                #_, sub_info[key][t]["bc_rw"], _ = flow_coef_bd(rw[t])
                #_, sub_info[key][t]["bc_sm"], _ = flow_coef_bd(sm[t])

    #two_sample_t_test(sub_info)
    print(group_consistency(sub_info))

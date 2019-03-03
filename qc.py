from compare_network import *
from utils.helper import *

def find_outlier(list):
    list = np.asarray(list)
    list_std = np.std(list)
    list_mean = np.mean(list)
    list = abs(list - list_mean)/list_std > 3
    return list


def modularity_qc(list_of_connectomes):
    """
    Checks if modularity is reasonable
    :return: List of faulty connectome
    """

    mod_list = []
    for cn in list_of_connectomes:
        n, _ = get_number_of_components(cn)
        mod_list = mod_list + n

    outlier = find_outlier(mod_list)
    mod_list = np.array(mod_list)
    print(mod_list[outlier])
    return outlier


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized')
    sub_names = get_subject_names()
    # sub_names = ["007_S_4516"]
    listc = []
    fname_list = []
    for sub in sub_names:
        scan_count = get_scan_count(sub)
        print("---------------\n\nRunning ", sub, " with scan count : ", scan_count)
        connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub), normalize=False)
        listc.append(connectome_list)
        fname_list = fname_list + [sub for t in range(len(connectome_list))]

    idx = modularity_qc(listc)
    print(np.array(fname_list)[idx])

from scipy.stats import ttest_ind
import numpy as np
from args import Args
import json

class stat_sig:
    def compute_sig(self, n1, n2, equal_var=True):
        return ttest_ind(n1, n2, equal_var=equal_var, axis=0)[0]


if __name__ == '__main__':
    i = 0
    j = 2
    n1 = np.loadtxt('slope_dx_' + str(i) + '.txt')
    n2 = np.loadtxt('slope_dx_' + str(j) + '.txt')

    st = stat_sig()
    count = 15
    t = st.compute_sig(n1, n2, False)
    idx1 = np.argsort(t)[-count:]
    print(idx1)

    t = st.compute_sig(n1, n2, True)
    idx2 = np.argsort(t)[-count:]
    print(idx2)

    print(set(idx1).difference(set(idx2)))
    print(set(idx2).difference(set(idx1)))

    pt_name = Args.root_directory + "/utils/parcellationTable_Ordered.json"  # parcellation table to edit VisuOrder

    # Read parcellation table to edit VisuOrder
    with open(pt_name) as f:
        pt = json.load(f)
    f.close()

    reg = []
    for i in idx1:
        print(pt[i]['name'], ' -> ', pt[i]['VisuHierarchy'])
        reg.append(pt[i]['name'])

    np.savetxt('idx.txt', idx1)
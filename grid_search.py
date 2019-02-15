from test import optimize_longitudinal_connectomes
from utils import cartesian_product as cp
import os

from utils.readFile import readMatricesFromDirectory


def grid_search(param_set):
    """

    :param param_set: List of sets with each set containing a range. Ex: param_set = [[1, 2, 3], [0.1, 0.5], [-1, 5]]
    :return:
    """

    param_set = cp.cartesian(param_set)
    # Read data
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized')
    sub = '052_S_4944'
    connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub))

    E_min = 100000
    ps_min = None
    for ps in param_set:
        dfw, sw, lmw, dd, r = ps
        _, _, E = optimize_longitudinal_connectomes(connectome_list, dfw, sw, lmw, diag_dist_factor=dd, r=r)
        if E < E_min:
            E_min = E
            ps_min = ps

        print("Loss: ", E,
              "\nParamSet: ", ps)
    return E_min, ps_min


if __name__ == '__main__':
    param_set = [[0.1, 0.3, 0.5, 0.7], [0.1, 0.3, 0.5, 0.7], [0.1, 0.3, 0.5, 0.7], [0], [-1]]
    E_min, ps_min = grid_search(param_set)
    print(ps_min)

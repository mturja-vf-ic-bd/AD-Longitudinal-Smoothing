from test import optimize_longitudinal_connectomes
from utils import cartesian_product as cp
import os

from utils.readFile import readMatricesFromDirectory


def grid_search(sub, param_set):
    """

    :param param_set: List of sets with each set containing a range. Ex: param_set = [[1, 2, 3], [0.1, 0.5], [-1, 5]]
    :return:
    """

    param_set = cp.cartesian(param_set)
    # Read data
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized')
    connectome_list = readMatricesFromDirectory(os.path.join(data_dir, sub))

    E_min = 1000000000
    ps_min = None
    for ps in param_set:
        dfw, sw, lmw, lmd, pro, sig, lam = ps
        _, _, E = optimize_longitudinal_connectomes(connectome_list, dfw=dfw, sw=sw, lmw=lmw, lmd=lmd,
                                                    pro=pro, rbf_sigma=sig, lambda_m=lam)
        if E < E_min:
            E_min = E
            ps_min = ps

        print("ParamSet: ", ps,
              "\tLoss: ", E,
              "\tMin: ", E_min)
    return E_min, ps_min


if __name__ == '__main__':
    param_set = [[1], [1], [0.1, 0.5, 1], [0.1, 0.5, 1], [0.1, 0.5, 1], [0.01, 0.1, 1], [0.001, 0.1, 1]]
    E_min, ps_min = grid_search('027_S_2219', param_set)
    print(ps_min)

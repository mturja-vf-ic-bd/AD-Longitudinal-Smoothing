from utils.helper import get_gamma, get_eigen
from utils.L2_distance import *
from utils.create_brain_net_files import *
from args import Args
from compare_network import get_number_of_components


def EProjSimplex(v, k = 1):
    """
        Problem:

        min 1/2 * || x - v || ^ 2
        s.t. x >= 0 1'x = 1
    """

    ft = 1
    n = len(v)
    v = np.asarray(v)
    v0 = v - np.mean(v) + k/n
    vmin = min(v0)
    if vmin < 0:
        sum_pos = 1
        lambda_m = 0
        while abs(sum_pos) > 10e-10:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = sum(posidx)
            sum_pos = sum(v1[posidx]) - k
            lambda_m = lambda_m + sum_pos/npos
            ft = ft + 1
            if ft > 100:
                break
        x = np.maximum(v1, 0)
    else:
        x = v0

    return x, ft


if __name__ == '__main__':
    v = np.array([1, 2, 3, 4, 6, 5, 0, 0, 0], dtype=np.float16)
    print(v)
    print("v = ", EProjSimplex(v))
    print("v/3 = ", EProjSimplex(v/3))
    print("v/10 = ", EProjSimplex(v/10))




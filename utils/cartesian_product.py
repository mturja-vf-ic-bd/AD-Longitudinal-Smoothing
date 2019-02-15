from itertools import product
import numpy as np

def cartesian(list):
    temp = product(list[0], list[1])
    for i in range(2, len(list)):
        temp = [(*a, b) for a, b in product(temp, list[i])]

    return temp


if __name__ == '__main__':
    A = np.array([1, 2, 3])
    B = np.array([4, 5])
    D = np.array([3, 3, 3])
    E = np.array([-1, -2])
    C = [A, B, D, E]
    Res = list(cartesian(C))
    print(Res,
          "\nLen: ", len(Res))

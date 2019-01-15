import numpy as np

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
    v1 = v0
    vmin = min(v0)

    if vmin < 0:
        sum_pos = 1
        lambda_m = 0
        while abs(sum_pos) > 0.00000000001:
            v1 = v0 - lambda_m
            print("iteration ", ft,
                  "v1 = ", v1)
            posidx = v1 > 0
            npos = sum(posidx)
            sum_pos = sum(v1[posidx]) - k
            lambda_m = lambda_m + sum_pos/npos
            ft = ft + 1
            if ft > 100:
                break

        x = [v1[i] if v1[i] >= 0 else 0 for i in range(0, n)]
    else:
        x = v0

    return x, ft


if __name__ == '__main__':
    v = np.array([0.1, -0.2, -0.4, 0.2, 0.5])
    x, ft = EProjSimplex(v)
    print("x = ", x,
          "ft = ", ft)

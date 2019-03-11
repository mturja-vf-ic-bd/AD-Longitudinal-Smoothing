import numpy as np
from matplotlib import pyplot as plt


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
            if npos == 0:
                break
            sum_pos = sum(v1[posidx]) - k
            lambda_m = lambda_m + sum_pos/npos
            ft = ft + 1
            if ft > 100:
                break
        x = np.maximum(v1, 0)
    else:
        x = v0

    return x, ft, sum((x - v) ** 2)

def softmax(X):
    ex = np.exp(X)
    return ex / ex.sum()


if __name__ == '__main__':
    v = np.array([1, 2, 3, 4, 6, 5, 0, 0, 0], dtype=np.float16)
    #v /= v.sum()
    print(v)
    epv, _ = EProjSimplex(v)
    print("v = ", epv,
          "\nloss: ", sum((epv - v) ** 2))
    sv = softmax(v)
    print("softmax v = ", sv,
          "\nloss: ", sum((sv - v) ** 2))
    epv3, _ = EProjSimplex(v/3)
    print("v/3 = ", epv3,
          "\nloss: ", sum((epv3 - v) ** 2))
    sv3 = softmax(v/3)
    print("softmax v/3 = ", sv3,
          "\nloss: ", sum((sv3 - v) ** 2))
    epv10, _ = EProjSimplex(v/10)
    print("v/10 = ", epv10,
          "\nloss: ", sum((epv10 - v) ** 2))
    sv10 = softmax(v/10)
    print("softmax v/10 = ", sv10,
          "\nloss: ", sum((sv10 - v) ** 2))

    plt.subplot(3, 1, 1)
    plt.plot(v, color='g')
    plt.plot(epv, color='r')
    plt.plot(sv, color='b')

    plt.subplot(3, 1, 2)
    plt.plot(v, color='g')
    plt.plot(epv3, color='r')
    plt.plot(sv3, color='b')

    plt.subplot(3, 1, 3)
    plt.plot(v, color='g')
    plt.plot(epv10, color='r')
    plt.plot(sv10, color='b')

    plt.show()




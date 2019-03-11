#from utils.readFile import readMatricesFromDirectory
#import numpy as np
#import os

def get_left_hemisphere(wb):
    n = len(wb)
    return wb[0:n//2, 0:n//2]

def get_right_hemisphere(wb):
    n = len(wb)
    return wb[n//2:n, n//2:n]

def get_hemispheres(wb):
    return get_left_hemisphere(wb), get_right_hemisphere(wb)


if __name__ == '__main__':
    data_dir = '/home/turja/AD-Data_Organized'
    sub = '094_S_4234'
    mat_list = readMatricesFromDirectory(os.path.join(data_dir, sub))

    for c in mat_list:
        print("\nleft: ", get_left_hemisphere(c).shape,
              "\nrigth: ", get_right_hemisphere(c).shape)

    testA = np.reshape([1 if 148 - i > j else 0 for i in range(0, 148) for j in range(0, 148)], (148, 148))
    print(testA)
    print("\nleft: ", get_left_hemisphere(testA).sum() == 74 * 74)
    print("\nright: ", get_right_hemisphere(testA).sum() == 0)

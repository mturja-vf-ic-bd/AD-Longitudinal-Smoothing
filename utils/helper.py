import numpy as np


def group_elements_of_matrices(input_mat_list):
    mat_list = []
    i = 1
    for mat in input_mat_list:
        mat_list.append(np.asarray(mat).flatten().tolist())
        i = i + 1

    element_list = []
    for j in range(0, len(mat_list[0])):
        el = []
        for i in range(0, len(mat_list)):
            el.append(mat_list[i][j])
        element_list.append(el)

    return element_list


def find_mean(mat_list, weights):
    weights = np.transpose(np.asarray(weights))
    row, col = mat_list[0].shape
    element_list = group_elements_of_matrices(mat_list)
    print(np.asarray(element_list[0]).shape)
    print(weights.shape)
    meanind = [np.matmul(np.asarray(element_list[i]), weights) for i in range(0, row*col)]
    return np.array(meanind).reshape((row, col))


if __name__ == '__main__':
    mat_list = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[3, 2, 1], [6, 5, 4]])]
    w = np.ones(2)/8
    print(find_mean(mat_list, w))

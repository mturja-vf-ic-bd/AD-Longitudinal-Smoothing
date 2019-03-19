import os
from test import optimize_longitudinal_connectomes
from utils.helper import get_subject_names, add_noise_all, threshold
from utils.readFile import readSubjectFiles, write_file
import numpy as np
from args import Args

def compute_psnr():
    sub_names = get_subject_names(3)
    threshold = Args.threshold  # threshold for raw connectome to have comaparable sparsity with smoothed version
    noise_rw = 0
    noise_sm = 0
    for sub in sub_names:
        connectome_list, smoothed_connectomes = readSubjectFiles(sub, method="row")
        N = connectome_list[0].shape[0]

        connectome_list_noisy = add_noise_all(connectome_list, t=0.1)
        smoothed_connectomes_noisy, M, E = optimize_longitudinal_connectomes(connectome_list_noisy, Args.dfw, Args.sw,
                                                                             Args.lmw,
                                                                             Args.lmd)
        # compute MSE
        for t in range(0, len(connectome_list)):
            noise_rw = noise_rw + (((connectome_list[t] > threshold) * connectome_list -
                                      (connectome_list_noisy[t] > threshold) * connectome_list_noisy) ** 2).sum()
            noise_sm = noise_sm + ((smoothed_connectomes[t] - smoothed_connectomes_noisy[t]) ** 2).sum()

    noise_rw = noise_rw / (N * N * len(sub_names))
    noise_sm = noise_sm / (N * N * len(sub_names))
    return np.log10(1 / noise_rw), np.log10(1 / noise_sm)


def write_noisy_file_in_brain_net_viewer():
    sub = '027_S_5110'
    brain_net_data_dir = "/home/turja/Desktop/BrainNetViewer/BrainNet-Viewer/Data/mydata"
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized/' + sub)
    scan_ids = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    scan_ids.sort()
    connectome_list, _ = readSubjectFiles(sub, method="row")
    # print((smoothed_connectome[t] > 0.001).sum())
    # test_result(sub, connectome_list, smoothed_connectome, max_ind, plot_all=True)

    connectome_list[2][0, 80] = 0.5
    connectome_list[2][3, 10] = 0.7
    connectome_list[0][10, 30] = 0.6
    connectome_list[1][5, 40] = 0.6
    connectome_list[3][40, 100] = 0.7
    write_file(sub, connectome_list, scan_ids, brain_net_data_dir, "noisy")


if __name__ == '__main__':
    noise_rw, noise_sm = compute_psnr()
    print("Noise raw: ", noise_rw,
          "\nNoise sm: ", noise_sm)

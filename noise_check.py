import os

from args import Args
from test import optimize_longitudinal_connectomes
from utils.helper import add_noise_all
from utils.readFile import readSubjectFiles

if __name__ == '__main__':
    args = Args()
    data_dir = os.path.join(os.path.join(args.root_directory, os.pardir), 'AD-Data_Organized')
    sub = '027_S_2336'
    connectome_list, smoothed_connectome = readSubjectFiles(sub)
    connectome_list_noisy = add_noise_all(connectome_list)
    smoothed_connectomes_noisy, M, E = optimize_longitudinal_connectomes(connectome_list_noisy, Args.dfw, Args.sw, Args.lmw,
                                                                   Args.lmd)

    # Compute noise in raw
    noise_rw = 0
    noise_sm = 0
    for t in range(0, len(connectome_list)):
        noise_rw = noise_rw + abs(connectome_list - connectome_list_noisy).sum()
        noise_sm = noise_sm + abs(smoothed_connectome - smoothed_connectomes_noisy).sum()

    print("Raw: ", noise_rw,
          "\nSM: ", noise_sm)

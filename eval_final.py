import os
from test import optimize_longitudinal_connectomes
from utils.helper import get_subject_names, add_noise_all, threshold
from utils.readFile import readSubjectFiles, write_file
import numpy as np
from args import Args
from scipy import stats

def calculate_psnr(A, B):
    mse = (np.power(np.subtract(A, B), 2)).sum()
    return np.log10(1/mse)

def calculate_psnr_list(A_list, B_list):
    mse = 0
    N = A_list[0].shape[0]
    for t in range(len(A_list)):
        mse = mse + calculate_psnr(A_list[t], B_list[t])

    mse = mse/(N*N*len(A_list))
    return np.log10(1/mse)

def compute_psnr(sub_names, noise, threshold=0):
    noise_rw = 0
    noise_rw_th = 0
    noise_sm = 0
    total = 0
    for sub in sub_names:
        connectome_list, smoothed_connectomes = readSubjectFiles(sub, method="row")
        N = connectome_list[0].shape[0]

        connectome_list_noisy = add_noise_all(connectome_list, noise)
        smoothed_connectomes_noisy, M, E = optimize_longitudinal_connectomes(connectome_list_noisy, Args.dfw, Args.sw,
                                                                             Args.lmw,
                                                                             Args.lmd)
        # compute MSE
        for t in range(0, len(connectome_list)):
            noise_rw = noise_rw + np.power(np.subtract(connectome_list[t],
                                      connectome_list_noisy[t]), 2).sum()
            connectome_list[t][connectome_list[t] < threshold] = 0
            connectome_list_noisy[t][connectome_list_noisy[t] < threshold] = 0
            noise_rw_th = noise_rw_th + np.power(np.subtract(connectome_list[t],
                                      connectome_list_noisy[t]), 2).sum()
            noise_sm = noise_sm + np.power(np.subtract(smoothed_connectomes[t], smoothed_connectomes_noisy[t]), 2).sum()
            total = total + 1

    noise_rw = noise_rw / (N * N * total)
    noise_rw_th = noise_rw_th / (N * N * total)
    noise_sm = noise_sm / (N * N * total)
    print("\nNoise raw: ", noise_rw,
              "\nNoise sm: ", noise_sm,
              "\nNoise th: ",  noise_rw_th)
    return np.log10(1 / noise_rw), np.log10(1 / noise_rw_th), np.log10(1 / noise_sm)

def add_spurious_connections(connectome, count=1, val=None):
    import random
    idx = np.where(connectome == 0)
    idx_list = [(x, y) for x, y in zip(idx[0], idx[1])]
    random.shuffle(idx_list)
    idx = list(zip(*idx_list))
    if val is None:
        val = random.uniform(0.2, 1)

    connectome[idx[0], idx[1]] = val
    return connectome, idx_list[0:count]

def spurious_noise(sub, count=1):
    brain_net_data_dir = "/home/turja/Desktop/BrainNetViewer/BrainNet-Viewer/Data/mydata"
    data_dir = os.path.join(os.path.dirname(os.getcwd()), 'AD-Data_Organized/' + sub)
    scan_ids = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    scan_ids.sort()
    connectome_list, _ = readSubjectFiles(sub, method="row")
    smoothed_connectomes, M, E = optimize_longitudinal_connectomes(connectome_list, Args.dfw, Args.sw, Args.lmw,
                                                                           Args.lmd)
    res = [add_spurious_connections(rw, count=count) for rw in connectome_list]
    for sm, idx in res:
        count = 0
        for x, y in idx:
            if sm[x, y] == 0:
                count = count + 1
            else:
                print(sm[x, y])
        print("Cleaned: ", count)
    smoothed_connectomes_spurious, M, E = optimize_longitudinal_connectomes(connectome_list, Args.dfw, Args.sw, Args.lmw,
                                                                           Args.lmd)

    psnr = calculate_psnr_list(smoothed_connectomes, smoothed_connectomes_spurious)
    return psnr
    # write_file(sub, connectome_list, scan_ids, brain_net_data_dir, "noisy")

def psnr_all():
    import pickle
    th = [i / 10 for i in range(1, 10)]
    threshold = Args.threshold
    sub_names = get_subject_names(3)
    print(len(sub_names))

    rw = []
    rw_th = []
    sm = []
    recalculate = False
    if recalculate:
        for t in th:
            noise_rw, noise_rw_th, noise_sm = compute_psnr(sub_names, t, threshold=threshold)
            rw.append(noise_rw)
            rw_th.append(noise_rw_th)
            sm.append(noise_sm)
            print("t =", t, "Threshold: ", threshold,
                  "\nNoise raw: ", noise_rw,
                  "\nNoise sm: ", noise_sm,
                  "\nNoise th: ", noise_rw_th)

        pickle.dump(rw, open("PSNR_rw.pkl", "wb"))
        pickle.dump(rw_th, open("PSNR_rw_th.pkl", "wb"))
        pickle.dump(sm, open("PSNR_sm.pkl", "wb"))
    else:
        rw = pickle.load(open('PSNR_rw.pkl', 'rb'))
        rw_th = pickle.load(open('PSNR_rw_th.pkl', 'rb'))
        sm = pickle.load(open('PSNR_sm.pkl', 'rb'))
        print("Raw psnr: ", np.mean(rw))
        print("Thresholded psnr: ", np.mean(rw_th))
        print("Smooth psnr: ", np.mean(sm))

    import pylab
    pylab.plot(th, rw, '-r', label="Raw")
    pylab.plot(th, rw_th, '-g', label="Thresholded")
    pylab.plot(th, sm, '-b', label="Intrinsic")
    pylab.legend(loc='upper right', prop={'size': 15})
    pylab.show()


if __name__ == '__main__':
    sub = '027_S_5110'
    print("Noise: ", spurious_noise(sub, count=1))
    #psnr_all()

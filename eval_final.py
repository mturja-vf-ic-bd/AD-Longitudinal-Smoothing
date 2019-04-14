import os
from test import optimize_longitudinal_connectomes, initialize_connectomes
from test_result import test_result
from utils.helper import get_subject_names, add_noise_all, threshold_all, threshold, add_noise, get_top_links, central_difference_of_links
from utils.readFile import readSubjectFiles, write_file
import numpy as np
from args import Args
from scipy import stats
from utils.sortDetriuxNodes import sort_matrix
import pickle

def calculate_psnr(A, B):
    mse = np.mean(np.power(np.subtract(A, B), 2), axis=None)
    return np.log10(1/mse)

def calculate_psnr_list(A_list, B_list):
    psnr = 0
    for t in range(len(A_list)):
        psnr = psnr + calculate_psnr(A_list[t], B_list[t])

    psnr = psnr/(len(A_list))
    return psnr

def calculate_psnr_all(sub_names, p=0.5):
    noise_rw = 0
    noise_th = 0
    noise_sm = 0
    noise_d = 0
    total = 0
    for sub in sub_names:
        _, connectome_list = readSubjectFiles(sub, method="row")
        # connectome_list = threshold_all(connectome_list, vmin=0.001)
        connectome_list_noisy = add_noise_all(connectome_list, p)
        smoothed_connectomes_noisy, M, E = optimize_longitudinal_connectomes(connectome_list_noisy, Args.dfw, Args.sw,
                                                                             Args.lmw,
                                                                             Args.lmd)
        degraded_connectomes, _, _, _, _, _ = initialize_connectomes(connectome_list_noisy)
        connectome_list_th = threshold_all(connectome_list_noisy, vmin=0.01)

        for t in range(0, len(connectome_list)):
            noise_rw = noise_rw + calculate_psnr(connectome_list[t], connectome_list_noisy[t])
            noise_th = noise_th + calculate_psnr(connectome_list[t], connectome_list_th[t])
            noise_d = noise_d + calculate_psnr(connectome_list[t], degraded_connectomes[t])
            noise_sm = noise_sm + calculate_psnr(connectome_list[t], smoothed_connectomes_noisy[t])
            total = total + 1

    noise_rw = noise_rw / total
    noise_th = noise_th / total
    noise_d = noise_d / total
    noise_sm = noise_sm / total
    print("\nNoise raw: ", noise_rw,
          "\nNoise th: ", noise_th,
          "\nNoise d: ", noise_d,
          "\nNoise sm: ", noise_sm)
    return noise_rw, noise_th, noise_d, noise_sm

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

def psnr_all():
    import pickle
    th = [i / 10 for i in range(9, 19)]
    #sub_names = get_subject_names(3)
    sub_names = ["027_S_5110"]
    print("File Read")
    print(len(sub_names))

    rw = []
    thr = []
    d= []
    sm = []
    recalculate = False
    if recalculate:
        for t in th:
            noise_rw, noise_th, noise_d, noise_sm = calculate_psnr_all(sub_names, t)
            rw.append(noise_rw)
            thr.append(noise_th)
            d.append(noise_d)
            sm.append(noise_sm)

        pickle.dump(rw, open("PSNR_rw.pkl", "wb"))
        pickle.dump(thr, open("PSNR_th.pkl", "wb"))
        pickle.dump(d, open("PSNR_d.pkl", "wb"))
        pickle.dump(sm, open("PSNR_sm.pkl", "wb"))
    else:
        rw = pickle.load(open('PSNR_rw.pkl', 'rb'))
        thr = pickle.load(open('PSNR_th.pkl', 'rb'))
        d = pickle.load(open('PSNR_d.pkl', 'rb'))
        sm = pickle.load(open('PSNR_sm.pkl', 'rb'))
        print("Raw psnr: ", np.mean(rw))
        print("Thresholded psnr: ", np.mean(thr))
        print("D psnr: ", np.mean(d))
        print("Smooth psnr: ", np.mean(sm))

    import pylab
    font = {'weight': 'bold',
            'size': 13}

    pylab.figure(figsize=(10, 4.2))
    pylab.tight_layout()
    pylab.rc('font', **font)
    pylab.plot(th, rw, 'r-', label=r'$PSNR_{tweaked}$', linewidth=4)
    pylab.plot(th, thr, 'g-.', label=r'$PSNR_{thresholded}$', linewidth=4)
    pylab.plot(th, d, color='#FF8C00', ls=':', label=r'$PSNR_{degraded}$', linewidth=4)
    pylab.plot(th, sm, 'bo-', label='our method', linewidth=4)
    pylab.legend(loc='lower left', prop={'size': 15})
    #pylab.xticks()
    pylab.xticks(np.arange(1, 1.81, step=0.4), ('Low', 'Medium', 'High'))
    pylab.ylim(2.5, 4)
    pylab.show()

def plot_psnr(data, th, color, labels, c=9):
    data = data[:, 0:c]
    color = color[0:c]
    th = th[0:c]
    labels = labels[0:c]
    import pylab

    pylab.rc('font', family='Times New Roman')
    font = {'weight': 'bold',
            'size': 22}

    pylab.figure(figsize=(10, 4.5))
    pylab.tight_layout()
    pylab.rc('font', **font)

    pylab.axis('off')
    for i in range(len(data)):
        pylab.plot(th, data[i], color[i], label=labels[i], linewidth=4, markersize=15)

    low = th[0]
    high = th[-1] + 0.1 * th[-1]
    step = (th[-1] - th[0]) / (3 - 1)
    #pylab.xticks(np.arange(low, high, step=step), ('Low', 'Medium', 'High'))
    #pylab.ylim(2, 5)
    #pylab.legend(loc='upper right', prop={'size': 18})
    pylab.show()

def simulate_longitudinal_network(baseline, noise_level=0.5, count=5):
    long_net = []
    for i in range(count):
        long_net.append(add_noise(baseline, t=noise_level))

    return long_net

def simulated_community_structure(ncom=4, shape=(148, 148)):
    A = np.zeros(shape)
    node_count = int(shape[0] / ncom)
    for i in range(node_count):
        A[i*node_count:(i+1)*node_count, i*node_count:(i+1)*node_count] = 1/node_count

    np.fill_diagonal(A, 0)
    return A

def binarize(A, threshold=0):
    A = 1 * (A > threshold)
    return A

def binarize_list(list, threshold=0):
    new_list = []
    for a in list:
        new_list.append(binarize(a, threshold))
    return new_list

def measure_change():
    sub = "027_S_2336"
    rw, sm = readSubjectFiles(sub, method="row")
    baseline_rw, _ = sort_matrix(rw[0], True)
    baseline_sm, _ = sort_matrix(sm[0], True)

    T = 7
    p = 1
    sim_long_net = simulate_longitudinal_network(baseline_rw, noise_level=p, count=T)
    sim_long_net_int, M, E = optimize_longitudinal_connectomes(sim_long_net, Args.dfw, Args.sw,
                                                               Args.lmw,
                                                               Args.lmd)
    sim_long_net_deg = initialize_connectomes(sim_long_net)
    sim_long_net_th = threshold_all(sim_long_net, vmin=0.0002)

    sim = central_difference_of_links(sim_long_net)
    sim_sm = central_difference_of_links(sim_long_net_int)
    sim_th = central_difference_of_links(sim_long_net_th)
    sim_deg = central_difference_of_links(sim_long_net_deg)

    print("Simulated data with noise: %f\nOur method: %f\nOur method degraded: %f\nThresholded: %f\n" % (sim, sim_sm, sim_deg, sim_th))


def main_sim_net():
    sub = "027_S_2336"
    rw, sm = readSubjectFiles(sub, method="row")
    baseline_rw, _ = sort_matrix(rw[0], True)
    baseline_sm, _ = sort_matrix(sm[0], True)
    # baseline_rw = binarize(baseline_rw, ht)
    # baseline_sm = binarize(baseline_sm, ht)
    # baseline_rw = simulated_community_structure()
    # baseline_sm = simulated_community_structure()

    th = [i / 10 for i in range(12, 25)]
    T = 7
    recalculate = False
    if recalculate:
        psnr = np.empty(shape=(4, len(th)))
        for i, p in enumerate(th):
            print("p = ", p)
            sim_long_net = simulate_longitudinal_network(baseline_rw, noise_level=p)
            for sl in sim_long_net:
                print((sl == 0).sum(axis=None))
            sim_long_net_int, M, E = optimize_longitudinal_connectomes(sim_long_net, Args.dfw, Args.sw,
                                                                                 Args.lmw,
                                                                                 Args.lmd)
            sim_long_net_deg = initialize_connectomes(sim_long_net)
            sim_long_net_th = threshold_all(sim_long_net, vmin=0.02)

            psnr[0][i] = calculate_psnr_list(sim_long_net, [baseline_rw] * T)
            psnr[1][i] = calculate_psnr_list(sim_long_net_int, [baseline_sm] * T)
            psnr[2][i] = calculate_psnr_list(sim_long_net_deg, [baseline_sm] * T)
            psnr[3][i] = calculate_psnr_list(sim_long_net_th,
                                             [baseline_rw] * T)

            with open('psnr.pkl', 'wb') as f:
                pickle.dump(psnr, f)

            print("PSNR sim = ", psnr[0][i],
                  "\nPSNR int = ", psnr[1][i],
                  "\nPSNR_th = ", psnr[3][i],
                  "\nPSNR_deg = ", psnr[2][i])
    else:
        with open('psnr.pkl', 'rb') as f:
            psnr = pickle.load(f)

    plot_psnr(psnr, th, color=['r-', 'b--*', 'y--.', 'go'], labels=['Simulated data w/ noise',
                                                                   'Our method', 'Our degraded method',
                                                                   'Thresholded method'])


def add_all(mat_list):
    M = np.zeros(mat_list[0].shape)
    for m in mat_list:
        M = M + m

    M = M / len(mat_list)
    return M

def get_sparse_mask(A):
    B = np.zeros(A.shape)
    B[A == 0] = 1
    return B

def get_sparse_hist(mat_list):
    sparse_mask = [get_sparse_mask(A) for A in mat_list]
    return add_all(sparse_mask)

def eval_low_range_consistency():
    sub = "027_S_5110"
    rw, sm = readSubjectFiles(sub, method="row")
    th = 0.0002
    rw_th = threshold_all(rw, vmin=th)
    hist_rw = get_sparse_hist(rw_th)
    hist_sm = get_sparse_hist(sm)
    print("Raw: ", (hist_rw == 1).sum(axis=None) / (hist_rw > 0).sum(axis=None))
    print("Smt: ", (hist_sm == 1).sum(axis=None) / (hist_sm > 0).sum(axis=None))
    write_low_range_connectivity(hist_rw, hist_sm)

def write_low_range_connectivity(hist_rw, hist_sm):
    A = 1 - hist_rw
    A[A == 1] = 0
    A[A != np.max(A)] = 0
    B = 1 - hist_sm
    B[B == 1] = 0

    basedir = '/home/turja/Desktop/ADNI_processing/CirclePlot/'
    with open(basedir + 'rw_low.txt', 'w') as f:
        np.savetxt(f, A)
    with open(basedir + 'sm_low.txt', 'w') as f:
        np.savetxt(f, B)
    basedir = '/home/turja/Desktop/BrainNetViewer/BrainNet-Viewer/Data/mydata/'
    with open(basedir + 'rw_low.txt', 'w') as f:
        np.savetxt(f, A)
    with open(basedir + 'sm_low.txt', 'w') as f:
        np.savetxt(f, B)


if __name__ == '__main__':
    #eval_low_range_consistency()
    #measure_change()
    main_sim_net()
    # psnr_all()
    # sub = "027_S_5110"
    # rw, sm = readSubjectFiles(sub, "row")
    # vmin=0
    # vmax=0.00005
    # rw_th = threshold_all(rw, vmin, vmax)
    # write_file(sub, rw_th, ["t1", "t2", "t3", "t4"], suffix="low_rw")


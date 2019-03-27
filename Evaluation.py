from utils.readFile import readSubjectFiles
from utils.helper import *
import numpy as np
from compare_network import get_number_of_components


class Evaluation:
    def __init__(self, subject=None):
        if subject is not None:
            self.rw_data, self.smth_data = readSubjectFiles(subject, method="row")
            self.T = len(self.rw_data)

    def evaluation_cc(self, property='clustering-coeff'):
        """
        Computes consistency of clustering coefficients for raw and smoothed data
        :return: rw_cc_ent, smth_cc_ent
        """

        if property == 'clustering-coeff':
            rw_cc = [np.mean(clustering_coef_wu(self.rw_data[t])) for t in range(0, self.T)]
            smth_cc = [np.mean(clustering_coef_wu(self.smth_data[t])) for t in range(0, self.T)]
        elif property == 'transitivity':
            rw_cc = [np.mean(transitivity_wu(self.rw_data[t])) for t in range(0, self.T)]
            smth_cc = [np.mean(transitivity_wu(self.smth_data[t])) for t in range(0, self.T)]
        elif property == 'coreness':
            rw_cc = [np.mean(core.core_periphery_dir(self.rw_data[t])) for t in range(0, self.T)]
            smth_cc = [np.mean(core.core_periphery_dir(self.smth_data[t])) for t in range(0, self.T)]
        elif property == 'assortativity':
            rw_cc = [np.mean(core.assortativity_wei(self.rw_data[t], 0)) for t in range(0, self.T)]
            smth_cc = [np.mean(core.assortativity_wei(self.smth_data[t], 0)) for t in range(0, self.T)]
        elif property == 'modularity':
            rw_cc, _ = get_number_of_components(self.rw_data)
            smth_cc, _ = get_number_of_components(self.smth_data)
        elif property == 'path_length':
            rw_cc = [charpath(rw)[0] for rw in self.rw_data]
            smth_cc = [charpath(sm)[0] for sm in self.smth_data]

        # rw_cc_ent = get_entropy_list(rw_cc)
        # smth_cc_ent = get_entropy_list(smth_cc)

        return rw_cc, smth_cc

    def change_of_network_over_time(self, connectome_list):
        change = 0
        for t in range(len(connectome_list) - 1):
            change = change + (connectome_list[t + 1] != connectome_list[t]).sum()

        return change / (len(connectome_list) - 1)

    def evaluate_binary_consistency(self):
        """
        This function binarize the raw and smoothed connectomes and checks the consistency over time
        :return:
        """

        change_rw = 0
        change_sm = 0
        th = [0.005]
        for threshold in th:
            raw_th = [self.rw_data[t] > threshold for t in range(0, self.T)]
            smooth_th = [self.smth_data[t] > 0 for t in range(0, self.T)]
            # print("Zeros rw:", get_avg_zeros_per_row(raw_th))
            # print("Zeros sm:", get_avg_zeros_per_row(self.smth_data))
            change_rw = change_rw + self.change_of_network_over_time(raw_th)
            change_sm = change_sm + self.change_of_network_over_time(smooth_th)

        change_rw = change_rw / len(th)
        change_sm = change_sm / len(th)

        return change_rw, change_sm


def plot_data(x, fig, c):
    fig.plot(np.arange(len(x)), x, c=c)
    return fig


def normalize_mean_std(x):
    x = np.array(x)
    m = np.mean(x)
    sig = np.std(x)
    x = (x - m) / sig
    return x


if __name__ == '__main__':
    sub_names = get_subject_names(5)
    for sub in sub_names:
        evl = Evaluation(sub)
        rw_cc, sm_cc = evl.evaluation_cc('modularity')
        t = [i+1 for i in range(len(rw_cc))]
        import pylab
        rw_cc = normalize_mean_std(rw_cc)
        sm_cc = normalize_mean_std(sm_cc)
        pylab.plot(t, rw_cc, '-r')
        pylab.plot(t, sm_cc, '-b')
        pylab.show()

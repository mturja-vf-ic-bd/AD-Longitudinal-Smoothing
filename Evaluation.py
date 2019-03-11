from args import Args
from utils.readFile import readSubjectFiles
from bct import *
from utils.helper import *
from matplotlib import pyplot as plt
import numpy as np
import random
from compare_network import get_number_of_components

class Evaluation:
    def __init__(self, subject=None):
        if subject is not None:
            self.rw_data, self.smth_data = readSubjectFiles(subject)
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

        #rw_cc_ent = get_entropy_list(rw_cc)
        #smth_cc_ent = get_entropy_list(smth_cc)

        return rw_cc, smth_cc


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
    rw_ent_list = {}
    sw_ent_list = {}
    for subject in sub_names:
        evl = Evaluation(subject)
        rw_ent, sw_ent = evl.evaluation_cc()
        #print(rw_ent, sw_ent)
        rw_ent_list[subject] = rw_ent
        sw_ent_list[subject] = sw_ent

    plot_count = 5
    rand_sample = random.sample(sub_names, plot_count)

    i = 0
    fig = plt.gcf()
    fig.set_size_inches(20, 15)
    for sub in rand_sample:
        ax_rw = plt.subplot(plot_count, 1, i + 1)
        #ax_sm = plt.subplot(plot_count, 1, 2*i + 2)
        plot_data(normalize_mean_std(rw_ent_list[sub]), ax_rw, c='r')
        plot_data(normalize_mean_std(sw_ent_list[sub]), ax_rw, c='b')
        i = i + 1

    plt.show()

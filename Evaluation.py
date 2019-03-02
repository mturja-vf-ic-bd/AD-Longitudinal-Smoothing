from args import Args
from utils.readFile import readSubjectFiles
from bct import *
from utils.helper import *
from matplotlib import pyplot as plt
import numpy as np

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

        rw_cc_ent = get_entropy_list(rw_cc)
        smth_cc_ent = get_entropy_list(smth_cc)

        return rw_cc_ent, smth_cc_ent


if __name__ == '__main__':
    sub_names = get_subject_names()
    rw_ent_list = []
    sw_ent_list = []
    for subject in sub_names:
        if get_scan_count(subject) > 3:
            evl = Evaluation(subject)
            rw_ent, sw_ent = evl.evaluation_cc(property='transitivity')
            print(rw_ent, sw_ent)
            rw_ent_list.append(rw_ent)
            sw_ent_list.append(sw_ent)

    hist1, bins = np.histogram(rw_ent_list)
    hist2, _ = np.histogram(sw_ent_list, bins=bins)

    plt.plot(hist1, color='r')
    plt.plot(hist2, color='b')
    plt.show()

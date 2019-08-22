import os
import utils.readFile
from args import Args
from Disease_Progression.helper import degree_distribution
from Disease_Progression.plot_helper import plot_dist
from Disease_Progression.networkx_interface import convert_to_netx_list
from matplotlib import pyplot as plt
from utils.helper import threshold_percentile_all
import numpy as np

if __name__ == '__main__':
    print("Experimenting with how the AD progresses")

    # sub_names = get_subject_names(4)
    sub_names = [ "094_S_2201", "057_S_4888", "094_S_4089"]
    nr = []
    ns = []
    for sub in sub_names:
        connectome_list = utils.readFile.readMatricesFromDirectory(os.path.join(Args.data_directory, sub), normalize=False)
        connectome_list = threshold_percentile_all(connectome_list, 95)
        connectome_list = convert_to_netx_list(connectome_list)

        for g in connectome_list:
            deg, cnt = degree_distribution(g)
            plot_dist(np.log(deg), cnt, title=sub)
            plt.show()
        # f_diff = forward_diff(connectome_list)


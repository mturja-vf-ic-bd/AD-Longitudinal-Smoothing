from utils.helper import threshold
from utils.readFile import readSubjectFiles
import numpy as np
from args import Args
from numpy import linalg as LA
from plot_functions import plot_eigen_specturm

def get_graph_specturm(G):
    G = 0.5 * (G + G.T)
    D = np.diag(G.sum(axis=1))
    L = np.subtract(D, G)
    eigenValues, eigenVectors = LA.eigh(G)
    idx = eigenValues.argsort()
    #eigenValues = np.flip(eigenValues[idx])
    #eigenValues = (eigenValues > 0) * eigenValues
    return eigenValues

# Topological smoothing evaluation
def compute_eigen_spectrum(sub):
    th = Args.threshold  # threshold for raw connectome to have comaparable sparsity with smoothed version
    connectome_list, smoothed_connectomes = readSubjectFiles(sub, method="row")
    connectome_list = [threshold(connectome, vmin=th) for connectome in connectome_list]

    raw_spectrum_list = [get_graph_specturm(connectome) for connectome in connectome_list]
    smth_spectrum_list = [get_graph_specturm(connectome) for connectome in smoothed_connectomes]

    return raw_spectrum_list, smth_spectrum_list


if __name__ == '__main__':
    sub = "027_S_2336"
    rw_spectrum, sm_spectrum = compute_eigen_spectrum(sub)
    plot_eigen_specturm(rw_spectrum, sm_spectrum, savefig=True)

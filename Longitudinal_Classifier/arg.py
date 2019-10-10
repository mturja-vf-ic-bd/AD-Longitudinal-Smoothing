import os
import torch

class Args:
    DEBUG = False
    BASE_DIR = "/home/mturja"
    NETWORK_DIR = os.path.join(BASE_DIR, "AD_network")
    NETWORK_DIR_SM = os.path.join(BASE_DIR, "AD_network_sm")
    STAT_DIR = os.path.join(BASE_DIR, "AD_stats")
    PARC_DIR = os.path.join(BASE_DIR, "AD_parc")
    OTHER_DIR = os.path.join(BASE_DIR, "AD_files")
    SUB_TO_NET_MAP = os.path.join(BASE_DIR, "AD_files/temporal_mapping.json")
    HOME_DIR = os.path.join(BASE_DIR, "AD-Longitudinal-Smoothing")

    n_class = 3
    eps = 1e-15
    cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_t = 8
    n_nodes = 148

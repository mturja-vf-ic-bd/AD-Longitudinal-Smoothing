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
    ORIG_DATA = os.path.join(OTHER_DIR, "data.csv")
    SUB_TO_NET_MAP = os.path.join(OTHER_DIR, "temporal_mapping_new.json")
    HOME_DIR = os.path.join(BASE_DIR, "AD-Longitudinal-Smoothing")
    MODEL_CP_PATH = os.path.join(HOME_DIR, "model_cp")

    n_class = 3
    eps = 1e-15
    cuda = True
    device = torch.device('cuda' if cuda else 'cpu')
    max_t = 8
    n_nodes = 148
    AGE_MEAN = 74
    AGE_STD = 6.8

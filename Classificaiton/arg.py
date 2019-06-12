import os
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

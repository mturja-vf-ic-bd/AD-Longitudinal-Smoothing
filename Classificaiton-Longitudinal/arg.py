import os
class Args:
    DEBUG = True
    BASE_DIR = "/home/mturja"
    NETWORK_DIR = os.path.join(BASE_DIR, "AD_network")
    STAT_DIR = os.path.join(BASE_DIR, "AD_stats")
    PARC_DIR = os.path.join(BASE_DIR, "AD_parc")
    OTHER_DIR = os.path.join(BASE_DIR, "AD_files")
    SUB_TO_NET_MAP = "/home/mturja/AD_files/temporal_mapping.json"

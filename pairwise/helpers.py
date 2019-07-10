import numpy as np

LABEL_SET_1 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
CLUSTER_SET_1 = np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 2, 2, 2])

LABEL_SET_2 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
CLUSTER_SET_2 = np.array([-1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 2, 2, 1, 2, 2, 2])
###
# Test data and expected solutions.
###
DATA_SET_A = {
    "Y": LABEL_SET_1,                                    # true labels
    "YP": CLUSTER_SET_1,                                 # assigned cluster ID
    "stats":{'TP': 20, 'TN': 72, 'FP': 20, 'FN': 24},   # resulting confusion stats
    "P": 0.5,                                           # pairwise precision
    "R": 0.45454545454545453,                                         # pairwise recall
    "Purity": 0.71,                                     # purity measure
    "NMI": 0.36,                                        # normalized mutual information
    "RI": 0.68,                                         # Random index
    "F5": 0.46,                                         # F-5 score
    "K": len(np.unique(CLUSTER_SET_1)),                  # Number of clusters
    "N": len(LABEL_SET_1),                               # Number of samples
    "NC": len(np.unique(LABEL_SET_1)),                   # Number of classes
}


DATA_SET_B = DATA_SET_A

DATA_SET_C = {
    "Y": np.repeat(LABEL_SET_1,2),
    "YP": np.repeat(CLUSTER_SET_1,2),
    "stats":{'TP': 20, 'TN': 72, 'FP': 20, 'FN': 24},
    "P": 0.5,
    "R": 0.455,
    "Purity": 0.71,
    "NMI": 0.36,
    "RI": 0.68,
    "F5": 0.46,
}
import numpy as np
import pairwise.helpers as helpers
from sklearn.preprocessing import LabelEncoder
# Calculate pair-wise metrics.
#
# Note, all metrics handle pairwise relationships (i.e., counting pairs)
# 
#               Predicted Classes
#                    p'    n'
#              ___|_____|_____| 
#       Actual  p |     |     |
#      Classes  n |     |     |
#
# 
# precision = TP / (TP + FP)                  for each class label
# recall = TP / (TP + FN)                     for each class label
# specificity = TN / (FP + TN)                for each class label
# fscore = 2*TP /(2*TP + FP + FN)             for each class label
# 
#
# TP: true positive, TN: true negative, 
# FP: false positive, FN: false negative
#
# A true positive (TP) when documents are in the same cluster, a true negative (TN) when two dissimilar documents are in
# different clusters. Two types of errors: A (FP) decision is when two dissimilar documents are in the same cluster. A
# (FN) decision is when two similar documents are in different clusters.
#
# @author Joseph P. Robinson
# @date 2019 July 12
from math import factorial


def nchoosek(n, k):
    """
    Determines number of combinations from expressions of form n choose k.
    n choose k = [n ; k] = n!/k!(n-k)! for 0 <= k <= n, where ! is factorial.
    :param n:  The total number of items.
    :param k:  The number to choose each time.
    :return:   n choose k (see description above)
    """
    assert 0 <= k <= n
    return factorial(n) / (factorial(k) * factorial(n - k))

def align_pseudo_labels(*labels):
    """
    Utility function that aligns variable number of label lists, i.e.,
        true_ids,cluster_ids=align_pseudo_labels(true_ids, cluster_ids)
    :param labels:
    :return:
    """
    return (label_encoder(l) for l in labels)

def calculate_tp(true_ids, cluster_ids):
    """
    Calculate the number of TP for a set of cluster assignments.
    :param sample_assignments: Key-Value corresponding to cluster ID - list of labels for samples cluster ID contains.s
    :return: Number of true positives.
    """
    # calibrate labels such to start from 0,.., M, where M is # of unique labels
    true_ids, cluster_ids = align_pseudo_labels(true_ids, cluster_ids)

    tp = 0  # TP (running sum)
    for i, cluster_id in enumerate(np.unique(cluster_ids)):
        # for each cluster, determine contents wrt to true labels
        cluster = true_ids[cluster_ids == cluster_id]
        # how many of each label type
        unique, counts = np.unique(cluster, return_counts=True)
        # count pairs for bins with more than 1 sample (i.e., 1 sample = 0 pairs, 0!)
        tp += sum(nchoosek(c, 2) for c in counts if c > 1)
    return tp

def calculate_fn(true_ids, cluster_ids):
    """
    Calculate the number of FN for a set of cluster assignments.
    :param sample_assignments: Key-Value corresponding to cluster ID - list of labels for samples cluster ID contains.s
    :return: Number of true positives.
    """
    # calibrate labels such to start from 0,.., M, where M is # of unique labels
    true_ids, cluster_ids = align_pseudo_labels(true_ids, cluster_ids)

    fn = 0  # TP (running sum)
    for i, cluster_id in enumerate(np.sort(np.unique(cluster_ids)[:-1])):
        # for each cluster, determine contents wrt to true labels
        cluster = true_ids[cluster_ids == cluster_id]
        # only look at larger values to not count same pair 2x (i.e., not look back, as prior already was calculated.
        another = true_ids[cluster_ids > cluster_id]
        # how many of each label in current cluster
        unique, counts = np.unique(cluster, return_counts=True)
        # how many of each in other clusters
        other_unique, other_counts = np.unique(another, return_counts=True)

        # make dictionaries and determine common keys to count
        lut = dict(zip(unique, counts))
        other_lut = dict(zip(other_unique, other_counts))

        common = list(set(lut.keys()).intersection(set(other_lut.keys())))

        for key in common:
            # number of elements in current * number outside
            fn += lut[key]*other_lut[key]

    return fn

def label_encoder(labels):
    le = LabelEncoder()
    le.fit(labels)
    return le.transform(labels)


def confusion_matrix_values(true_ids, cluster_ids):
    """
    Calculate TP, FP, TN, and FN and store in dictionary container.
    :param true_ids:    Ground-truth label [ Nx1 ].
    :param clabels:     Cluster assignment [ Nx1 ].
    :return: Confusion stats {TP, FP, TN, FN} (dictionary)
    """
    true_ids, cluster_ids = align_pseudo_labels(true_ids, cluster_ids)

    stats = {}
    cluster_refs = np.unique(cluster_ids)
    nclasses = len(np.unique(true_ids))

    nsamples_per_class, _ = np.histogram(true_ids, range(nclasses + 1))
    npairs_per_class = [nchoosek(val, 2) for val in nsamples_per_class]


    stats['TP'] = calculate_tp(true_ids, cluster_ids)
    nsamples_per_cluster, _ = np.histogram(cluster_ids, range(len(cluster_refs) + 1))
    total_positive = sum(nchoosek(val, 2) for val in nsamples_per_cluster)

    stats['FP'] = total_positive - stats['TP']

    stats['FN'] = calculate_fn(true_ids, cluster_ids)

    stats['TN'] = nchoosek(len(true_ids), 2) - stats['FP'] - stats['TP'] - stats['FN']

    return stats


def precision(true_ids, cluster_ids):
    """
    Calculate precision of the ith cluster w.r.t. assigned clusterins. True labels are used to determine those from same
    class and, hence, should be clustered together. It is assumed all N samples are clustered.

    Precision (P): How accurate are the positive predictions.

    Precision = TP / (TP + FP) (per class)
    :param true_ids:    Ground-truth label [ Nx1 ].
    :param clabels:     Cluster assignment [ Nx1 ].
    :return: Precision value (float)
    """

    stats = confusion_matrix_values(true_ids, cluster_ids)
    return stats['TP'] / (stats['TP'] + stats['FP'])


def recall(true_ids, cluster_ids):
    """
    Calculate recall of the ith cluster w.r.t. clabels. Ground-truth is used to determine the observations from the same
    class (identity) and, hence, should be clustered together.

    Recall (R): Coverage of actual positive sample.

    R = TP / (TP + FN)

    :param true_ids:    Ground-truth label [ Nx1 ].
    :param cluster_ids: Cluster assignment [ Nx1 ].
    :return: Recall value (float)
    """
    stats = confusion_matrix_values(true_ids, cluster_ids)

    return stats['TP'] / (stats['TP'] + stats['FN'])


def accuracy(true_ids, cluster_ids):
    """
    Calculate accuracy.

    Accuracy (Acc): Overall performance of model

    Acc = (TP + TN) / (TP + FP + FN + TN)
    :param true_ids:    Ground-truth label [ Nx1 ].
    :param cluster_ids: Cluster assignment [ Nx1 ].
    :return:
    """
    stats = confusion_matrix_values(true_ids, cluster_ids)

    return (stats['TP'] + stats['TN']) / (stats['TP'] + stats['FP'] + stats['FN'] + stats['TN'])


def specificity(true_ids, cluster_ids):
    """
    Calculate specificity: Coverage of actual negative sample.

    Recall = TN / (TN + FP)
    :param true_ids:    Ground-truth label [ Nx1 ].
    :param cluster_ids: Cluster assignment [ Nx1 ].
    :return:
    """
    stats = confusion_matrix_values(true_ids, cluster_ids)

    return stats['TN'] / (stats['TN'] + stats['FP'])


def f1score(true_ids, cluster_ids):
    """
    Calculate F1-score: Hybrid metric useful for unbalanced classes.

    Recall = 2TP / (2TP + FP + FN)
    :param true_ids:    Ground-truth label [ Nx1 ].
    :param cluster_ids: Cluster assignment [ Nx1 ].
    :return:
    """
    stats = confusion_matrix_values(true_ids, cluster_ids)

    return 2*stats['TP'] / (2*stats['TP'] + stats['FP'] + stats['FN'])


if __name__ == '__main__':
    DATA_SET_A = helpers.DATA_SET_A
    print('{} samples in {} clusters from {} classes'.format(DATA_SET_A['N'], DATA_SET_A['K'], DATA_SET_A['NC']))
    print('Precision: {} '.format(precision(DATA_SET_A['Y'], DATA_SET_A['YP'])))
    print('Recall: {} '.format(recall(DATA_SET_A['Y'], DATA_SET_A['YP'])))
    # stats = confusion_matrix_values(DATA_SET_A['Y'], DATA_SET_A['YP'])
    # print(stats)
    print('Accuracy: {} '.format(accuracy(DATA_SET_A['Y'], DATA_SET_A['YP'])))
    print('Specificity: {} '.format(specificity(DATA_SET_A['Y'], DATA_SET_A['YP'])))
    print('F1: {} '.format(f1score(DATA_SET_A['Y'], DATA_SET_A['YP'])))

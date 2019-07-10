import numpy as np
import pairwise.helpers as helpers
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


def calculate_tp(sample_assignments):
    """
    Calculate the number of TP for a set of cluster assignments.
    :param sample_assignments: Key-Value corresponding to cluster ID - list of labels for samples cluster ID contains.s
    :return: Number of true positives.
    """
    raise NotImplemented


def confusion_matrix_values(true_ids, cluster_ids):
    """
    Calculate TP, FP, TN, and FN and store in dictionary container.
    :param true_ids:    Ground-truth label [ Nx1 ].
    :param clabels:     Cluster assignment [ Nx1 ].
    :return: Confusion stats {TP, FP, TN, FN} (dictionary)
    """
    stats = {}
    stats['TP'] = 0
    classes = {}
    cluster_refs = np.unique(cluster_ids)
    nclasses = len(np.unique(true_ids))

    nsamples_per_class, _ = np.histogram(true_ids, range(nclasses + 1))
    npairs_per_class = [nchoosek(val, 2) for val in nsamples_per_class]
    for k, cluster_id in enumerate(cluster_refs):
        # for each cluster index items assigned to kth cluster
        ids = true_ids == cluster_id
        # determine true labels for items
        classes[cluster_id] = np.sort(cluster_ids[ids])

    for label, assignments in classes.items():
        categories = np.unique(assignments)
        for category in categories:
            flag_ids = category == assignments
            npresent = sum(flag_ids)
            if npresent > 1:
                # if more than single sample for given class in respective cluster
                stats['TP'] += nchoosek(npresent, 2)

    nsamples_per_cluster, _ = np.histogram(cluster_ids, range(len(cluster_refs) + 1))
    total_positive = sum(nchoosek(val, 2) for val in nsamples_per_cluster)

    stats['FP'] = total_positive - stats['TP']

    stats['FN'] = sum(npairs_per_class) - stats['TP']

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

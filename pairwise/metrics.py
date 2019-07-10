import numpy as np
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
# Confusion Matrices:
# cluster_confusion  -   confusion matrix sorted by cluster
# class_confusion   -   confusion matrix sorted by classes
# 
# Settings/ Params:
# nclasses                                    No. of true classes
# clabels                                     true class labels
# k                                           No. of clusters
# ids                                         cluster assignments
# 
# TODO
# - add additional pairwise metric
#   e.g., stats.random_index = (TP + TN)/(TP + FP + FN + TN) avg. accuracy 
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

def precision(true_ids, cluster_ids):
    """
    Calculate precision of the ith cluster w.r.t. assigned clusterins. True labels are used to determine those from same
    class and, hence, should be clustered together. It is assumed all N samples are clustered.
    Precision = TP / (TP + FP) (per class)
    :param true_ids:    Ground-truth label [ Nx1 ].
    :param clabels:     Cluster assignment [ Nx1 ].
    :return:
    """
    stats = {}
    # get list of cluster IDs and number of unique assignments (i.e., cluster count)
    clusters = np.unique(cluster_ids)
    k_predicted  = len(clusters)                  # number of clusters

    # Determine number of samples for each class, i.e., how many should be assigned to cluster for a given class labels.
    nsamples_per_cluster, _ =np.histogram(cluster_ids, range(k_predicted + 1))

    total_positive = 0
    for nsamples in nsamples_per_cluster:
        # determine the number of positive predictions (i.e., TP + FP) to later calculate FP
        total_positive += nchoosek(nsamples, 2)

    classes = {}
    stats['TP'] = 0
    for k, cluster_id in enumerate(clusters):
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
    stats['FP'] = total_positive - stats['TP']
    stats['precision'] = stats['TP']/(stats['TP'] + stats['FP'])
    return stats


if __name__ == '__main__':
    sample_clusters = np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 2, 1, 2, 2, 2])
    N = len(sample_clusters)
    k = len(np.unique(sample_clusters))

    true_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])

    classes = np.unique(true_labels)
    stats = {}
    print('Cluster {} samples into {} clusters from {} classes'.format(N, k, len(classes)))
    stats['precision'] = precision(true_labels, sample_clusters)
    # stats['recall'] = recall(true_labels, sample_clusters)
    print(stats)

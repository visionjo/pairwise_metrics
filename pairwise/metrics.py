import numpy as np
import pairwise.helpers as helpers
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from math import factorial

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


def label_encoder(labels):
    le = LabelEncoder()
    le.fit(labels)
    return le.transform(labels)


class Metrics:

    def __repr__(self):
        return "Class to evaluate pairwise measures using definitions of confusion stats."


    def calculate_tp(self, true_ids, cluster_ids):
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

    def calculate_fp(self, true_ids, cluster_ids):
        """
        Calculate the number of FP for a set of cluster assignments.
        :param sample_assignments: Key-Value corresponding to cluster ID - list of labels for samples cluster ID contains.s
        :return: Number of true positives.
        """
        # calibrate labels such to start from 0,.., M, where M is # of unique labels
        true_ids, cluster_ids = align_pseudo_labels(true_ids, cluster_ids)

        fp = 0  # FP (running sum)
        for i, cluster_id in enumerate(np.unique(cluster_ids)):
            # for each cluster, determine contents wrt to true labels
            cluster = true_ids[cluster_ids == cluster_id]
            # how many of each label type
            unique, counts = np.unique(cluster, return_counts=True)
            pairs = list(combinations(unique, 2))
            lut = dict(zip(unique, counts))
            # sum of products from each count for each class
            fp += sum(lut[pair[0]] * lut[pair[1]] for pair in pairs)
        return fp

    def calculate_fn(self, true_ids, cluster_ids):
        """
        Calculate the number of FN for a set of cluster assignments.
        :param sample_assignments: Key-Value corresponding to cluster ID - list of labels for samples cluster ID contains.s
        :return: Number of true positives.
        """
        # calibrate labels such to start from 0,.., M, where M is # of unique labels
        true_ids, cluster_ids = align_pseudo_labels(true_ids, cluster_ids)

        fn = 0  # FN (running sum)
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
                fn += lut[key] * other_lut[key]

        return fn

    def confusion_matrix_values(self, true_ids, cluster_ids):
        """
        Calculate TP, FP, TN, and FN and store in dictionary container.
        :param true_ids:    Ground-truth label [ Nx1 ].
        :param clabels:     Cluster assignment [ Nx1 ].
        :return: Confusion stats {TP, FP, TN, FN} (dictionary)
        """
        true_ids, cluster_ids = align_pseudo_labels(true_ids, cluster_ids)

        stats = {}
        stats['TP'] = self.calculate_tp(true_ids, cluster_ids)
        stats['FP'] = self.calculate_fp(true_ids, cluster_ids)
        stats['FN'] = self.calculate_fn(true_ids, cluster_ids)
        npairs = nchoosek(len(true_ids), 2)  # total number of pairs
        npositive = stats['FP'] + stats['TP']  # total number of positive pairs
        nnegative = npairs - npositive  # total number of negative pairs
        stats['TN'] = nnegative - stats['FN']

        return stats

    def precision(self, true_ids, cluster_ids):
        """
        Calculate precision of the ith cluster w.r.t. assigned clusterins. True labels are used to determine those from same
        class and, hence, should be clustered together. It is assumed all N samples are clustered.

        Precision (P): How accurate are the positive predictions.

        Precision = TP / (TP + FP) (per class)
        :param true_ids:    Ground-truth label [ Nx1 ].
        :param clabels:     Cluster assignment [ Nx1 ].
        :return: Precision value (float)
        """

        stats = self.confusion_matrix_values(true_ids, cluster_ids)
        return stats['TP'] / (stats['TP'] + stats['FP'])

    def recall(self, true_ids, cluster_ids):
        """
        Calculate recall of the ith cluster w.r.t. clabels. Ground-truth is used to determine the observations from the same
        class (identity) and, hence, should be clustered together.

        Recall (R): Coverage of actual positive sample.

        R = TP / (TP + FN)

        :param true_ids:    Ground-truth label [ Nx1 ].
        :param cluster_ids: Cluster assignment [ Nx1 ].
        :return: Recall value (float)
        """
        stats = self.confusion_matrix_values(true_ids, cluster_ids)

        return stats['TP'] / (stats['TP'] + stats['FN'])

    def accuracy(self, true_ids, cluster_ids):
        """
        Calculate accuracy.

        Accuracy (Acc): Overall performance of model

        Acc = (TP + TN) / (TP + FP + FN + TN)
        :param true_ids:    Ground-truth label [ Nx1 ].
        :param cluster_ids: Cluster assignment [ Nx1 ].
        :return:
        """
        stats = self.confusion_matrix_values(true_ids, cluster_ids)

        return (stats['TP'] + stats['TN']) / (stats['TP'] + stats['FP'] + stats['FN'] + stats['TN'])

    def specificity(self, true_ids, cluster_ids):
        """
        Calculate specificity: Coverage of actual negative sample.

        Recall = TN / (TN + FP)
        :param true_ids:    Ground-truth label [ Nx1 ].
        :param cluster_ids: Cluster assignment [ Nx1 ].
        :return:
        """
        stats = self.confusion_matrix_values(true_ids, cluster_ids)

        return stats['TN'] / (stats['TN'] + stats['FP'])

    def f1score(self, true_ids, cluster_ids):
        """
        Calculate F1-score: Hybrid metric useful for unbalanced classes.

        Recall = 2TP / (2TP + FP + FN)
        :param true_ids:    Ground-truth label [ Nx1 ].
        :param cluster_ids: Cluster assignment [ Nx1 ].
        :return:
        """
        stats = self.confusion_matrix_values(true_ids, cluster_ids)

        return 2 * stats['TP'] / (2 * stats['TP'] + stats['FP'] + stats['FN'])


if __name__ == '__main__':
    DATA_SET_A = helpers.DATA_SET_A

    print('{} samples in {} clusters from {} classes'.format(DATA_SET_A['N'], DATA_SET_A['K'], DATA_SET_A['NC']))
    mm = Metrics()
    print('Precision: {} '.format(mm.precision(DATA_SET_A['Y'], DATA_SET_A['YP'])))
    print('Recall: {} '.format(mm.recall(DATA_SET_A['Y'], DATA_SET_A['YP'])))
    # stats = confusion_matrix_values(DATA_SET_A['Y'], DATA_SET_A['YP'])
    # print(stats)
    print('Accuracy: {} '.format(mm.accuracy(DATA_SET_A['Y'], DATA_SET_A['YP'])))
    print('Specificity: {} '.format(mm.specificity(DATA_SET_A['Y'], DATA_SET_A['YP'])))
    print('F1: {} '.format(mm.f1score(DATA_SET_A['Y'], DATA_SET_A['YP'])))

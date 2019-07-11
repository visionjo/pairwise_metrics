from .context import Metrics
from pairwise.helpers import DATA_SET_A, DATA_SET_B, DATA_SET_C

mm = Metrics()
def test_precision():
    """
    Test pairwise.metrics.precision()
    :return:
    """
    print("[ CHECK ] : {}".format("Precision"))
    assert mm.precision(DATA_SET_A['Y'], DATA_SET_A['YP']) == DATA_SET_A['P']


def test_recall():
    """
    Test pairwise.metrics.recall()
    :return:
    """
    print("[ CHECK ] : {}".format("Recall"))
    assert mm.recall(DATA_SET_A['Y'], DATA_SET_A['YP']) == DATA_SET_A['R']


def test_confusion_stats():
    """
    Test pairwise.metrics.confusion_matrix_values()
    :return:
    """
    stats = mm.confusion_matrix_values(DATA_SET_A['Y'], DATA_SET_A['YP'])

    for key in stats.keys():
        print("[ CHECK ] : {}".format(key))
        assert stats[key] == DATA_SET_A['stats'][key]


def test_f1score():
    """
    Test pairwise.metrics.f1score()
    :return:
    """
    print("[ CHECK ] : {}".format("F1-Score"))
    assert mm.f1score(DATA_SET_A['Y'], DATA_SET_A['YP']) == DATA_SET_A['F1']


def test_accuracy():
    """
    Test pairwise.metrics.accuracy()
    :return:
    """
    print("[ CHECK ] : {}".format("Accuracy"))
    assert mm.accuracy(DATA_SET_A['Y'], DATA_SET_A['YP']) == DATA_SET_A['Acc']


def test_specificity():
    """
    Test pairwise.metrics.specificity()
    :return:
    """
    print("[ CHECK ] : {}".format("Specificity"))
    pass
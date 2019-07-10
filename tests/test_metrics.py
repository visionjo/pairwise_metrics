from .context import metrics, helpers


def test_precision():
    """
    Test pairwise.metrics.precision()
    :return:
    """
    print("[ CHECK ] : {}".format("Precision"))
    assert metrics.precision(helpers.DATA_SET_A['Y'], helpers.DATA_SET_A['YP']) == helpers.DATA_SET_A['P']


def test_recall():
    """
    Test pairwise.metrics.recall()
    :return:
    """
    print("[ CHECK ] : {}".format("Recall"))
    assert metrics.recall(helpers.DATA_SET_A['Y'], helpers.DATA_SET_A['YP']) == helpers.DATA_SET_A['R']


def test_confusion_stats():
    """
    Test pairwise.metrics.confusion_matrix_values()
    :return:
    """
    stats = metrics.confusion_matrix_values(helpers.DATA_SET_A['Y'], helpers.DATA_SET_A['YP'])

    for key in stats.keys():
        print("[ CHECK ] : {}".format(key))
        assert stats[key] == helpers.DATA_SET_A['stats'][key]


def test_f1score():
    """
    Test pairwise.metrics.f1score()
    :return:
    """
    print("[ CHECK ] : {}".format("F1-Score"))
    assert metrics.f1score(helpers.DATA_SET_A['Y'], helpers.DATA_SET_A['YP']) == helpers.DATA_SET_A['F1']


def test_accuracy():
    """
    Test pairwise.metrics.accuracy()
    :return:
    """
    print("[ CHECK ] : {}".format("Accuracy"))
    assert metrics.accuracy(helpers.DATA_SET_A['Y'], helpers.DATA_SET_A['YP']) == helpers.DATA_SET_A['Acc']


def test_specificity():
    """
    Test pairwise.metrics.specificity()
    :return:
    """
    print("[ CHECK ] : {}".format("Specificity"))
    pass
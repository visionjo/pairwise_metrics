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

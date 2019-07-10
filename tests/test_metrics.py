from .context import metrics, helpers

def test_example():
    """
    But really, test cases should be callables containing assertions:
    """
    print("\nRunning test_example...")
    assert helpers.DATA_SET_A == helpers.DATA_SET_B

def test_precision():
    assert metrics.precision(helpers.DATA_SET_A['Y'], helpers.DATA_SET_A['YP']) == helpers.DATA_SET_A['P']


def test_recall():
    assert metrics.recall(helpers.DATA_SET_A['Y'], helpers.DATA_SET_A['YP']) == helpers.DATA_SET_A['R']

def test_confusion_stats():
    stats = metrics.confusion_matrix_values(helpers.DATA_SET_A['Y'], helpers.DATA_SET_A['YP'])

    for key in stats.keys():
        assert stats[key] == helpers.DATA_SET_A[key]
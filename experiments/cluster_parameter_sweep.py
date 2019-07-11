"""
Sweep across values of eps threshold used to cluster. Generate relevant statistics (i.e., metrics) and output as pickle.

TODO visualizations
TODO Refactor this script
@author Joe Robinson
@date   9 July 2019
"""
from sklearn.cluster import DBSCAN
from pairwise.metrics import precision, recall, accuracy
import pandas as pd
import numpy as np
from sklearn import preprocessing
f_features = '../data/eval-features.pkl'
# load feature set
do_save = True
data = pd.read_pickle(f_features)
X = np.array(data['X'])
y = np.array(data['y'])
allpaths = data['fpaths']

# align labels to range from 0, .., M, where M is number of unique labels
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# n samples between lower-bound and upper-bound
n = 100 # number of samples to cluster 14225
lb = 1.01
ub = 1.12
# eps_array = np.arange(0.4,0.7,0.05)  # threshold for DBScan
eps_array = np.linspace(lb,ub,n)

ulabs = np.unique(y)
nunique = len(ulabs)
print("[INFO] {} unique subjects and {} faces in total".format(nunique, n))

recall_scores = []
precision_scores = []
accuracy_scores = []
for eps in eps_array:
    print('Eps: {}'.format(eps))
    # Compute DBSCAN
    db = DBSCAN(eps=eps, algorithm='kd_tree', metric='l2', n_jobs=4).fit(X)
    Y_predict = db.labels_
    recall_scores.append(recall(y, Y_predict))
    precision_scores.append(precision(y, Y_predict))
    accuracy_scores.append(accuracy(y, Y_predict))
    print('P: {}\nR: {}\nA: {}'.format(precision_scores[-1], recall_scores[-1], accuracy_scores[-1]))


stats={}
stats['eps']=eps_array
stats['p']=np.array(precision_scores)
stats['r']=np.array(recall_scores)
stats['a']=np.array(accuracy_scores)
if do_save:
    pd.to_pickle(stats, 'pr_acc_eps_stats5.pkl')
print('MAX\nP: {} at {}\nR: {} at {}\nA: {} at {}'.format(stats['p'].max(), eps_array[stats['p'].argmax()],
                                                          stats['r'].max(), eps_array[stats['r'].argmax()],
                                                          stats['a'].max(), eps_array[stats['a'].argmax()]
                                                          ))
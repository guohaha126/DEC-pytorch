import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment, linear_sum_assignment
from sklearn import metrics
nmi = normalized_mutual_info_score
ari = adjusted_rand_score

def NMI(y_true,y_pred):
    return metrics.normalized_mutual_info_score(y_true, y_pred)

def ARI(y_true,y_pred):
    return metrics.adjusted_rand_score(y_true, y_pred)
def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
#     #--------------*=-------
#     y_true = y_true.astype(np.int64)
#     assert y_pred.size == y_true.size
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#
#     ind = linear_sum_assignment(w.max() - w)
#
#     accuracy = 0
#     for idx in range(len(ind[0]) - 1):
#         i = ind[0][idx]
#         j = ind[1][idx]
#         accuracy += w[i, j]
#     accuracy = accuracy * 1.0 / y_pred.size
#     return accuracy
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size



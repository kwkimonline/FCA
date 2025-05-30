import numpy as np
from scipy.spatial import distance
from sklearn.metrics import silhouette_score

def evaluation(color_xs, color_assignments, centers, K):
    """
    Evaluates the clustering results based on the objective function, balance, and silhouette score.
    """
    # total sample size
    n0, n1 = color_xs[0].shape[0], color_xs[1].shape[0]
    n = n0 + n1
    # objectives
    cluster_cnts = []
    objective = 0.0
    for xs_i, assignments_i in zip(color_xs, color_assignments):
        distances_i = distance.cdist(xs_i, centers, metric='minkowski', p=2)
        objective += (distances_i[np.arange(distances_i.shape[0]), assignments_i]**2).sum()
        sub_cluster_cnts = []
        for k in range(K):
            sub_cluster_cnts.append((assignments_i == k).sum())
        cluster_cnts.append(sub_cluster_cnts)
    # balance
    min_ratios = []
    for k in range(np.array(cluster_cnts).shape[1]):
        a_k = np.array(cluster_cnts)[0, k]
        b_k = np.array(cluster_cnts)[1, k]
        ratio1 = a_k / b_k
        ratio2 = b_k / a_k
        min_ratios.append(min(ratio1, ratio2))
    balance = min(min_ratios)

    return objective / n, balance
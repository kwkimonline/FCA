import numpy as np
from scipy.spatial import distance
from sklearn.metrics import silhouette_score


def evaluation(color_xs, color_assignments, centers, K):
    n = sum(x.shape[0] for x in color_xs)
    
    X_all = np.concatenate(color_xs, axis=0)                       # (n, d)
    z_all = np.concatenate(color_assignments, axis=0).astype(np.int64)  # (n,)
    d = X_all.shape[1]

    cnt_all = np.bincount(z_all, minlength=K)                      # (K,)
    sum_all = np.zeros((K, d), dtype=X_all.dtype)
    np.add.at(sum_all, z_all, X_all)

    centers = np.zeros((K, d), dtype=X_all.dtype)
    nonempty = cnt_all > 0
    centers[nonempty] = sum_all[nonempty] / cnt_all[nonempty, None]
    
    cluster_cnts = []
    objective = 0.0

    for xs_i, asg_i in zip(color_xs, color_assignments):
        asg_i = asg_i.astype(np.int64)

        diff = xs_i - centers[asg_i]
        objective += np.sum(diff * diff)

        cluster_cnts.append(np.bincount(asg_i, minlength=K))

    cluster_cnts = np.asarray(cluster_cnts)  # shape (n_color, K)

    a = cluster_cnts[0].astype(np.float64)
    b = cluster_cnts[1].astype(np.float64)

    min_ratios = np.zeros(K, dtype=np.float64)
    mask = (a > 0) & (b > 0)
    r1 = np.zeros(K, dtype=np.float64)
    r2 = np.zeros(K, dtype=np.float64)
    r1[mask] = a[mask] / b[mask]
    r2[mask] = b[mask] / a[mask]
    min_ratios[mask] = np.minimum(r1[mask], r2[mask])

    balance = float(np.min(min_ratios))

    return float(objective / n), balance

import numpy as np
from scipy.spatial.distance import cdist


def random_hard_assigning(arr):
    """
    Randomly assigns indices among the maximum values in each row.
    """
    # Create a mask of maximum entries per row
    max_mask = arr == arr.max(axis=1, keepdims=True)
    # Randomly pick one of the maxima for each row
    return np.array([np.random.choice(np.flatnonzero(row)) for row in max_mask])


def assigning(xs, Ws, Taxs, gammas, centers, K):
    """
    xs: [[batch_xs[0], batch_xs[1]], [], ... , []] # (n, d) # n = n0 + n1
    Ws: (n0, n1) for each batch # 1 for Wc and 0 for W
    Taxs: (n0 * n1, d) for each batch
    gammas: (n0 * n1, ) for each batch
    n_majors: (B, ) # B = batch_size
    n_minors: (B, ) # B = batch_size
    centers: (K, d)
    K: int
    """
    color_xs = [[], []]
    color_assignments = [[], []]
    for sub_xs, sub_Ws, sub_Taxs, sub_gammas in zip(xs, Ws, Taxs, gammas):
        
        n_0, n_1 = sub_xs[0].shape[0], sub_xs[1].shape[0]
        
        sub_distances = cdist(sub_Taxs, centers, metric='minkowski', p=2)
        sub_assignments = np.argmin(sub_distances, axis=1)
        
        sub_distances_0 = cdist(sub_xs[0], centers, metric='minkowski', p=2)
        sub_assignments_0 = np.argmin(sub_distances_0, axis=1)
        
        sub_distances_1 = cdist(sub_xs[1], centers, metric='minkowski', p=2)
        sub_assignments_1 = np.argmin(sub_distances_1, axis=1)

        # shape: (n0, n1)
        sub_gammas_i_j = sub_gammas.reshape(n_0, n_1)
        sub_assignments = sub_assignments.reshape(n_0, n_1)

        # for s = 0
        prob_assignments_0 = np.zeros(shape=(n_0, K))
        for row in range(n_0):
            for k in range(K):
                prob_assignments_0[row, k] = n_0 * np.sum(
                    (sub_gammas_i_j[row][sub_assignments[row] == k]) * (sub_Ws[row][sub_assignments[row] == k] == 1).astype(float)
                    )
                prob_assignments_0[row, k] += n_0 * np.sum(
                    sub_gammas_i_j[row] * (sub_Ws[row] == 0).astype(float)
                ) * (sub_assignments_0[row] == k).astype(float)


        # for s = 1
        prob_assignments_1 = np.zeros(shape=(n_1, K))
        for col in range(n_1):
            for k in range(K):
                prob_assignments_1[col, k] = n_1 * np.sum(
                    (sub_gammas_i_j.T[col][sub_assignments.T[col] == k]) * (sub_Ws.T[col][sub_assignments.T[col] == k] == 1).astype(float)
                    )
                prob_assignments_1[col, k] += n_1 * np.sum(
                    sub_gammas_i_j.T[col] * (sub_Ws.T[col] == 0).astype(float)
                ) * (sub_assignments_1[col] == k).astype(float)

        color_xs[0].append(sub_xs[0])
        color_xs[1].append(sub_xs[1])
        color_assignments[0].append(random_hard_assigning(prob_assignments_0))
        color_assignments[1].append(random_hard_assigning(prob_assignments_1))

    
    color_xs[0] = np.concatenate(color_xs[0])
    color_xs[1] = np.concatenate(color_xs[1])
    color_assignments[0] = np.concatenate(color_assignments[0])
    color_assignments[1] = np.concatenate(color_assignments[1])

    return color_xs, color_assignments
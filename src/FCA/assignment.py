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


def assigning(xs, Taxs, gammas, centers, K):
    """
    Assigns colors to data points based on the optimal transport plan and centers.
    """
    color_xs = [[], []]
    color_assignments = [[], []]
    # Precompute identity for one-hot encoding
    eye_K = np.eye(K)

    for sub_xs, sub_Taxs, sub_gammas in zip(xs, Taxs, gammas):
        n0 = sub_xs[0].shape[0]
        n1 = sub_xs[1].shape[0]

        # Compute pairwise distances and reshape to (n0, n1)
        assignments = (
            cdist(sub_Taxs, centers, metric='minkowski', p=2)
            .argmin(axis=1)
            .reshape(n0, n1)
        )

        # Reshape gamma to matrix form
        gammas_matrix = sub_gammas.reshape(n0, n1)

        # One-hot encode assignments and weight by gammas
        one_hot = eye_K[assignments]       # shape (n0, n1, K)
        weighted = gammas_matrix[:, :, None] * one_hot

        # Sum over pairs to get probabilistic assignments
        prob0 = weighted.sum(axis=1) * n0  # shape (n0, K)
        prob1 = weighted.sum(axis=0) * n1  # shape (n1, K)

        # Collect for final concatenation
        color_xs[0].append(sub_xs[0])
        color_xs[1].append(sub_xs[1])
        color_assignments[0].append(random_hard_assigning(prob0))
        color_assignments[1].append(random_hard_assigning(prob1))

    # Concatenate across all batches
    xs0 = np.concatenate(color_xs[0]) if color_xs[0] else np.array([])
    xs1 = np.concatenate(color_xs[1]) if color_xs[1] else np.array([])
    assigns0 = np.concatenate(color_assignments[0]) if color_assignments[0] else np.array([])
    assigns1 = np.concatenate(color_assignments[1]) if color_assignments[1] else np.array([])

    return [xs0, xs1], [assigns0, assigns1]

import numpy as np
from scipy.spatial.distance import cdist


def random_hard_assigning(arr, *, rng=None, eps=1e-12):
    rng = np.random.default_rng() if rng is None else rng
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("arr must be 2D (n, K).")
    a = np.clip(a, 0.0, None)
    row_sum = a.sum(axis=1, keepdims=True)
    probs = np.divide(a, row_sum, out=np.full_like(a, 1.0 / a.shape[1]), where=row_sum > eps)
    cdf = np.cumsum(probs, axis=1)
    u = rng.random((a.shape[0], 1))
    return (u > cdf).sum(axis=1)


def assigning(xs_blocks, masks_blocks, Taxs_blocks, gammas_blocks, centers, K, *, deterministic=True, seed=0):
    rng = np.random.default_rng(seed)

    color_xs = [[], []]
    color_assignments = [[], []]

    for xs, M, TA, g_flat in zip(xs_blocks, masks_blocks, Taxs_blocks, gammas_blocks):
        X0, X1 = xs
        n0, n1 = X0.shape[0], X1.shape[0]
        if n0 == 0 or n1 == 0:
            continue

        G = g_flat.reshape(n0, n1).astype(np.float64)
        M = M.astype(np.float64)

        row_sum = G.sum(axis=1)
        col_sum = G.sum(axis=0)

        a0 = cdist(X0, centers, metric="sqeuclidean").argmin(axis=1).astype(int)
        a1 = cdist(X1, centers, metric="sqeuclidean").argmin(axis=1).astype(int)

        if np.all(row_sum == 0) or np.all(col_sum == 0):
            z0, z1 = a0, a1
            color_xs[0].append(X0); color_xs[1].append(X1)
            color_assignments[0].append(z0); color_assignments[1].append(z1)
            continue

        row_safe = np.where(row_sum > 0, row_sum, 1.0)
        col_safe = np.where(col_sum > 0, col_sum, 1.0)

        ii, jj = np.nonzero(G > 0)
        gv = G[ii, jj]
        lin = ii * n1 + jj

        kTA = cdist(TA[lin], centers, metric="sqeuclidean").argmin(axis=1).astype(int)

        is_wc = (M[ii, jj] > 0.5)
        is_w = ~is_wc

        # --- Wc term: accumulate raw gamma mass by (i,k) and (j,k) ---
        wc_wgt = gv * is_wc.astype(np.float64)

        acc0 = np.bincount(ii * K + kTA, weights=wc_wgt, minlength=n0 * K).reshape(n0, K)
        acc1 = np.bincount(jj * K + kTA, weights=wc_wgt, minlength=n1 * K).reshape(n1, K)

        P0_wc = acc0 / row_safe[:, None]
        P1_wc = acc1 / col_safe[:, None]

        # --- W term: probability mass that goes to original-space nearest center ---
        w_mass_row = np.bincount(ii, weights=gv * is_w.astype(np.float64), minlength=n0) / row_safe
        w_mass_col = np.bincount(jj, weights=gv * is_w.astype(np.float64), minlength=n1) / col_safe

        P0 = P0_wc.copy()
        P1 = P1_wc.copy()
        P0[np.arange(n0), a0] += w_mass_row
        P1[np.arange(n1), a1] += w_mass_col

        s0 = P0.sum(axis=1, keepdims=True)
        s1 = P1.sum(axis=1, keepdims=True)
        P0 = np.divide(P0, s0, out=np.full_like(P0, 1.0 / K), where=s0 > 0)
        P1 = np.divide(P1, s1, out=np.full_like(P1, 1.0 / K), where=s1 > 0)

        if deterministic:
            z0 = P0.argmax(axis=1).astype(int)
            z1 = P1.argmax(axis=1).astype(int)
        else:
            z0 = random_hard_assigning(P0, rng=rng)
            z1 = random_hard_assigning(P1, rng=rng)

        color_xs[0].append(X0); color_xs[1].append(X1)
        color_assignments[0].append(z0); color_assignments[1].append(z1)

    color_xs[0] = np.concatenate(color_xs[0], axis=0)
    color_xs[1] = np.concatenate(color_xs[1], axis=0)
    color_assignments[0] = np.concatenate(color_assignments[0], axis=0)
    color_assignments[1] = np.concatenate(color_assignments[1], axis=0)

    return color_xs, [color_assignments[0], color_assignments[1]]

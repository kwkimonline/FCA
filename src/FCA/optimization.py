import numpy as np
import ot
import torch
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def partition_L_plus_remainder(n, L, *, seed=0):
    if L is None or L <= 0:
        return [np.arange(n, dtype=int)]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    if L >= n:
        return [idx[i:i+1] for i in range(n)]

    q = n // L
    if q == 0:
        return [idx[i:i+1] for i in range(n)]

    blocks = []
    start = 0
    for _ in range(L):
        blocks.append(idx[start:start+q])
        start += q

    if start < n:
        blocks.append(idx[start:])

    return blocks


def get_TA_pairs(X0, X1, pi_0, pi_1):
    n0, n1 = X0.shape[0], X1.shape[0]
    TA = pi_0 * np.repeat(X0, n1, axis=0) + pi_1 * np.tile(X1, (n0, 1))
    return TA.astype(np.float32)


def _min_sqdist_to_centers(X, centers):
    if X.size == 0:
        return np.zeros((0,), dtype=np.float64)
    d2 = cdist(X, centers, metric="sqeuclidean")
    return d2.min(axis=1).astype(np.float64)


def compute_eta_samples(X0, X1, pi_0, pi_1, centers, *, i_idx, j_idx):
    x0 = X0[i_idx]
    x1 = X1[j_idx]
    TA = (pi_0 * x0 + pi_1 * x1).astype(np.float64)
    C = _min_sqdist_to_centers(TA, centers)
    D = 2.0 * pi_0 * pi_1 * np.sum((x0 - x1) ** 2, axis=1)
    return (C + D).astype(np.float64)


def estimate_eta_threshold_global(X0, X1, pi_0, pi_1, centers, epsilon, *, n_pairs=200000, seed=0):
    """
    We estimate threshold t = quantile(eta, 1-epsilon) by uniform pair sampling.
    """
    rng = np.random.default_rng(seed)
    n0, n1 = X0.shape[0], X1.shape[0]
    m = int(min(max(1, n_pairs), n0 * n1))

    i = rng.integers(0, n0, size=m, endpoint=False)
    j = rng.integers(0, n1, size=m, endpoint=False)

    eta = compute_eta_samples(X0, X1, pi_0, pi_1, centers, i_idx=i, j_idx=j)
    q = float(epsilon)
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(eta, q))


def optimize_W_from_threshold(xs, pi_0, pi_1, centers, eta_thr):
    """
    Build block mask M (n0,n1):
      W = {(i,j): eta(i,j) > eta_thr}
      Wc = complement.
    """
    X0, X1 = xs
    n0, n1 = X0.shape[0], X1.shape[0]

    # D = 2*pi0*pi1*||x0-x1||^2
    D = 2.0 * pi_0 * pi_1 * ot.dist(X0, X1, metric="sqeuclidean")  # (n0,n1)

    # C = min_k ||TA - mu||^2
    TA = get_TA_pairs(X0, X1, pi_0, pi_1)  # (n0*n1,d)
    C_flat = cdist(TA, centers, metric="sqeuclidean").min(axis=1).astype(np.float64)
    C = C_flat.reshape(n0, n1)

    eta = D + C
    M = (eta > eta_thr).astype(np.float32)  # 1=Wc, 0=W
    return M


# =========================================================
# Phase 1: Coupling
# =========================================================
def optimize_coupling(xs, mask, centers, *, pi_0=None, pi_1=None, numItermax=5000, numThreads=3):
    X0, X1 = xs
    n0, n1 = X0.shape[0], X1.shape[0]

    if pi_0 is None or pi_1 is None:
        pis = np.array([n0, n1], dtype=np.float64)
        pis = pis / pis.sum()
        pi_0, pi_1 = float(pis[0]), float(pis[1])

    w0 = np.ones(n0, dtype=np.float64) / max(n0, 1)
    w1 = np.ones(n1, dtype=np.float64) / max(n1, 1)

    # TA and C(TA)
    TA = get_TA_pairs(X0, X1, pi_0, pi_1)  # (n0*n1,d)
    C_flat = cdist(TA, centers, metric="sqeuclidean").min(axis=1).astype(np.float64)
    C = C_flat.reshape(n0, n1)

    # D(x0,x1)
    D = 2.0 * pi_0 * pi_1 * ot.dist(X0, X1, metric="sqeuclidean").astype(np.float64)

    CFCA = D + C

    # CKmeans
    c0 = _min_sqdist_to_centers(X0, centers)  # (n0,)
    c1 = _min_sqdist_to_centers(X1, centers)  # (n1,)
    CK = (pi_0 * c0)[:, None] + (pi_1 * c1)[None, :]  # (n0,n1)

    M = mask.astype(np.float64)
    cost = M * CFCA + (1.0 - M) * CK

    Gamma = ot.emd(w0, w1, cost, numItermax=numItermax, numThreads=numThreads)
    return TA.astype(np.float32), Gamma.flatten().astype(np.float64), n0, n1


# =========================================================
# Phase 2: Global center update AFTER all couplings are computed
# =========================================================
def optimize_centers_global(
    xs_blocks, masks_blocks, Taxs_blocks, gammas_blocks,
    *, centers, K, pi_0, pi_1,
    seed=0,
    kmeans_max_iter=50,
    gradient_descent=False,
    centers_module=None,
    centers_optimizer=None,
    use_cuda=False,
    gd_steps=20,
    ):

    X_list = []
    w_list = []

    for xs, M, TA, g_flat in zip(xs_blocks, masks_blocks, Taxs_blocks, gammas_blocks):
        X0, X1 = xs
        n0, n1 = X0.shape[0], X1.shape[0]
        if n0 == 0 or n1 == 0:
            continue

        G = g_flat.reshape(n0, n1).astype(np.float64)
        M = M.astype(np.float64)

        ii, jj = np.nonzero(G > 0)
        if ii.size == 0:
            continue

        gv = G[ii, jj]
        is_wc = (M[ii, jj] > 0.5)
        is_w = ~is_wc

        if np.any(is_wc):
            lin = ii[is_wc] * n1 + jj[is_wc]
            X_list.append(TA[lin].astype(np.float32))
            w_list.append(gv[is_wc].astype(np.float64))

        if np.any(is_w):
            ii_w = ii[is_w]
            jj_w = jj[is_w]
            gv_w = gv[is_w]

            w0 = pi_0 * np.bincount(ii_w, weights=gv_w, minlength=n0).astype(np.float64)
            w1 = pi_1 * np.bincount(jj_w, weights=gv_w, minlength=n1).astype(np.float64)

            if np.any(w0 > 0):
                X_list.append(X0.astype(np.float32))
                w_list.append(w0)
            if np.any(w1 > 0):
                X_list.append(X1.astype(np.float32))
                w_list.append(w1)

    if len(X_list) == 0:
        return centers.astype(np.float32)

    X_all = np.vstack(X_list).astype(np.float32)
    w_all = np.concatenate(w_list).astype(np.float64)
    keep = w_all > 0
    X_all = X_all[keep]
    w_all = w_all[keep]

    if X_all.shape[0] == 0:
        return centers.astype(np.float32)

    # Option 1: global weighted KMeans
    if not gradient_descent:
        km = KMeans(
            n_clusters=K,
            init=centers,
            n_init=1,
            random_state=seed,
            max_iter=int(kmeans_max_iter),
        )
        km.fit(X_all, sample_weight=w_all)
        return km.cluster_centers_.astype(np.float32)

    # Option 2: GD hard-kmeans-like
    if centers_module is None or centers_optimizer is None:
        km = KMeans(n_clusters=K, init=centers, n_init=1, random_state=seed, max_iter=int(kmeans_max_iter))
        km.fit(X_all, sample_weight=w_all)
        return km.cluster_centers_.astype(np.float32)

    device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
    X_t = torch.tensor(X_all, device=device)
    w_t = torch.tensor(w_all, device=device)

    for _ in range(int(gd_steps)):
        d = torch.cdist(X_t, centers_module.weight, p=2)
        a = torch.argmin(d, dim=1)
        e = (w_t * (d[torch.arange(d.size(0)), a] ** 2)).sum()
        centers_optimizer.zero_grad()
        e.backward()
        centers_optimizer.step()

    return centers_module.weight.detach().cpu().numpy().astype(np.float32)

import numpy as np
import ot
import torch
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def get_Taxs(xs):
    """
    xs: [xs[0], xs[1]]
    """

    # only for binary sensitive attribute
    n_colors = 2
    assert n_colors == len(xs)
    ns = [len(xs_i) for xs_i in xs]
    pis = np.array(ns) / np.sum(ns)
    pi_0, pi_1 = pis[0], pis[1]
    n_0, n_1 = ns[0], ns[1]

    Taxs_0_1 = pi_0 * np.repeat(xs[0], n_1, 0) + pi_1 * np.tile(xs[1], (n_0, 1))

    return Taxs_0_1


def optimize_coupling(xs, mask, centers, numItermax=5000, numThreads=3):
    """
    xs: [xs[0], xs[1]]
    mask: (n0, n1) matrix, 1 if Wc & 0 else
    """

    # only for binary sensitive attribute
    n_colors = 2
    assert n_colors == len(xs)
    ns = [len(xs_i) for xs_i in xs]
    pis = np.array(ns) / np.sum(ns)
    pi_0, pi_1 = pis[0], pis[1]
    n_0, n_1 = ns[0], ns[1]
    w_0, w_1 = np.ones(n_0) / n_0, np.ones(n_1) / n_1

    Taxs_0_1 = get_Taxs(xs)
    mid_distances = cdist(Taxs_0_1, centers, metric='minkowski', p=2)
    mid_assignments = np.argmin(mid_distances, axis=1)
    C_01 = mid_distances[np.arange(mid_distances.shape[0]), mid_assignments]**2
    C_01 = C_01.reshape(n_0, n_1)

    D_01 = 2 * pi_0 * pi_1 * ot.dist(xs[0], xs[1])
    Coupling_01 = ot.emd(w_0, w_1, mask * D_01 + C_01,
                         numItermax=numItermax,
                         numThreads=numThreads)

    return Taxs_0_1, Coupling_01.flatten(), n_0, n_1


def compute_eta(xs, pi_0, pi_1, centers):
    n_0, n_1 = len(xs[0]), len(xs[1])
    Taxs_0_1 = get_Taxs(xs)
    D_01 = 2 * pi_0 * pi_1 * ot.dist(xs[0], xs[1])
    mid_distances = cdist(Taxs_0_1, centers, metric='minkowski', p=2)
    mid_assignments = np.argmin(mid_distances, axis=1)
    C_01 = mid_distances[np.arange(mid_distances.shape[0]), mid_assignments]**2
    C_01 = C_01.reshape(n_0, n_1)

    return (C_01 + D_01).flatten()


def optimize_W(xs, pi_0, pi_1, centers, epsilon):
    """
    xs: [xs[0], xs[1]]
    W_sort: bool, True => sort descending
    epsilon: float, balance upper bound
    """
    eta = compute_eta(xs, pi_0, pi_1, centers)
    sort_ids = np.argsort(eta)
    portion = int(xs[0].shape[0] * xs[1].shape[0] * epsilon)

    W = np.ones(xs[0].shape[0] * xs[1].shape[0])
    W[sort_ids[:portion]] = 0.0
    return W.reshape(xs[0].shape[0], xs[1].shape[0])


def optimize_center(xs, Taxs, gammas, masks,
                    centers, centers_module, centers_optimizer,
                    K, seed=2024, gradient_descent=False, use_cuda=False):
    """
    xs: [xs[0], xs[1]]
    Taxs: (n0 * n1, d)
    gammas: (n0 * n1,)
    masks: (n0, n1)
    """
    n0, n1 = xs[0].shape[0], xs[1].shape[0]

    # Wc cost
    Wc_Taxs = Taxs[masks.flatten() > 0]
    Wc_gammas = gammas[masks.flatten() > 0]
    Wc_Taxs = Wc_Taxs[Wc_gammas > 0]
    Wc_gammas = Wc_gammas[Wc_gammas > 0]

    # W cost
    idx = np.argwhere(masks == 0)
    i_idx, j_idx = idx[:, 0], idx[:, 1]
    W_xs = np.concatenate([xs[0][i_idx], xs[1][j_idx]])
    wgt = gammas.reshape(n0, n1)[i_idx, j_idx]
    wgt = np.concatenate([wgt, wgt])
    W_xs = W_xs[wgt > 0]
    wgt = wgt[wgt > 0]

    if gradient_descent:
        device = 'cuda' if use_cuda else 'cpu'
        Wc_T = torch.tensor(Wc_Taxs, device=device)
        Wc_g = torch.tensor(Wc_gammas, device=device)
        W_T = torch.tensor(W_xs, device=device)
        W_g = torch.tensor(wgt, device=device)
        for _ in range(20):
            d0 = torch.cdist(Wc_T, centers_module.weight, p=2)
            a0 = torch.argmin(d0, dim=1)
            e = (Wc_g * d0[torch.arange(d0.size(0)), a0]**2).sum()
            d1 = torch.cdist(W_T, centers_module.weight, p=2)
            a1 = torch.argmin(d1, dim=1)
            e += (W_g * d1[torch.arange(d1.size(0)), a1]**2).sum()
            centers_optimizer.zero_grad()
            e.backward()
            centers_optimizer.step()
        return centers_module.weight.detach().cpu().numpy()
    else:
        km = KMeans(n_clusters=K, init=centers, random_state=seed)
        km.fit(np.vstack([Wc_Taxs, W_xs]), sample_weight=np.concatenate([Wc_gammas, wgt]))
        return km.cluster_centers_
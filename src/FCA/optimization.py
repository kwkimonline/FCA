import numpy as np
import ot
import torch
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def optimize_coupling(xs, centers, numItermax=5000, numThreads=3):
    """
    Compute optimal coupling between two color sets with added transport cost to centers.
    """
    # Two-color scenario
    n0, n1 = len(xs[0]), len(xs[1])
    pis = np.array([n0, n1], dtype=float) / (n0 + n1)
    pi0, pi1 = pis
    w0 = np.full(n0, 1.0 / n0)
    w1 = np.full(n1, 1.0 / n1)
    # D
    D = 2 * pi0 * pi1 * ot.dist(xs[0], xs[1])
    # C
    X0 = np.repeat(xs[0], n1, axis=0)
    X1 = np.tile(xs[1], (n0, 1))
    Taxs = pi0 * X0 + pi1 * X1
    d_mid = cdist(Taxs, centers, metric='minkowski', p=2)
    assign_mid = d_mid.argmin(axis=1)
    C = (d_mid[np.arange(d_mid.shape[0]), assign_mid] ** 2).reshape(n0, n1)
    # Solve Kantorovich problem
    M = D + C
    coupling = ot.emd(w0, w1, M, numItermax=numItermax, numThreads=numThreads)

    return Taxs, coupling.ravel(), n0, n1


def optimize_center(Taxs, gammas, centers, centers_module, centers_optimizer,
                    K, max_iter=100, seed=2024, gradient_descent=False, use_cuda=False):
    """
    Update cluster centers via weighted KMeans or gradient-based optimization.
    """
    if gradient_descent:
        device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        xs_t = torch.tensor(Taxs, dtype=torch.float32, device=device)
        gm_t = torch.tensor(gammas, dtype=torch.float32, device=device)
        for _ in range(20):
            dists = torch.cdist(xs_t, centers_module.weight, p=2)
            assign = dists.argmin(dim=1)
            energy = (gm_t * (dists[torch.arange(len(xs_t)), assign] ** 2)).sum()
            centers_optimizer.zero_grad()
            energy.backward()
            centers_optimizer.step()
        return centers_module.weight.detach().cpu().numpy()
    else:
        # Weighted KMeans
        kmeans = KMeans(
            n_clusters=K, init=centers, max_iter=max_iter,
            random_state=seed
        )
        kmeans.fit(Taxs, sample_weight=gammas)
        return kmeans.cluster_centers_


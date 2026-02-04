import os
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from .optimization import (
    partition_L_plus_remainder,
    estimate_eta_threshold_global,
    optimize_W_from_threshold,
    optimize_coupling,
    optimize_centers_global,
)
from .assignment import assigning
from ..evaluation import evaluation
from ..datasets import load_data


def _merge_last_block_if_extra(blocks_a, blocks_b):
    if len(blocks_a) == len(blocks_b):
        return blocks_a, blocks_b
    if len(blocks_a) == len(blocks_b) + 1:
        if len(blocks_a) >= 2:
            blocks_a[-2] = np.concatenate([blocks_a[-2], blocks_a[-1]])
            blocks_a = blocks_a[:-1]
        else:
            blocks_a = blocks_a[:len(blocks_b)]
        return blocks_a, blocks_b

    if len(blocks_b) == len(blocks_a) + 1:
        if len(blocks_b) >= 2:
            blocks_b[-2] = np.concatenate([blocks_b[-2], blocks_b[-1]])
            blocks_b = blocks_b[:-1]
        else:
            blocks_b = blocks_b[:len(blocks_a)]
        return blocks_a, blocks_b

    while len(blocks_a) > len(blocks_b):
        blocks_a[-2] = np.concatenate([blocks_a[-2], blocks_a[-1]])
        blocks_a = blocks_a[:-1]
    while len(blocks_b) > len(blocks_a):
        blocks_b[-2] = np.concatenate([blocks_b[-2], blocks_b[-1]])
        blocks_b = blocks_b[:-1]
    return blocks_a, blocks_b

def sum_sse(X, centers_, labels_):
    return float(np.sum(np.sum((X - centers_[labels_]) ** 2, axis=1)))

def global_recenter_from_hard_assignments(
    color_xs,
    assignments,
    centers_prev,
    K,
    *,
    reinit_empty="keep",
    seed=0,
    ):
    rng = np.random.default_rng(seed)
    X_all = np.concatenate(color_xs, axis=0)
    z_all = np.concatenate(assignments, axis=0).astype(int)

    K_prev, d = centers_prev.shape
    assert K_prev == K, f"K mismatch: centers_prev has {K_prev}, but K={K}"

    sums = np.zeros((K, d), dtype=np.float64)
    cnts = np.zeros((K,), dtype=np.int64)

    for k in range(K):
        mask = (z_all == k)
        cnt = int(mask.sum())
        cnts[k] = cnt
        if cnt > 0:
            sums[k] = X_all[mask].sum(axis=0)

    centers_new = centers_prev.astype(np.float64).copy()
    nonempty = cnts > 0
    centers_new[nonempty] = sums[nonempty] / cnts[nonempty, None]

    empty_ks = np.where(cnts == 0)[0]
    if len(empty_ks) > 0:
        if reinit_empty == "keep":
            pass
        elif reinit_empty == "random_data":
            idx = rng.integers(0, X_all.shape[0], size=len(empty_ks))
            centers_new[empty_ks] = X_all[idx]
        else:
            raise ValueError(f"Unknown reinit_empty={reinit_empty}")

    return centers_new.astype(np.float32), cnts

def runner(args):
    """
    After all couplings computed, update centers globally using ALL blocks.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load data
    data_dir = Path(f"data/{args.data_name}/")
    _, np_data, _, np_colors, K_data, d, n_color = load_data(
        name=args.data_name,
        l2_normalize=args.l2_normalize,
        data_dir=str(data_dir)
    )

    K = args.K if args.K > 0 else K_data

    X0 = np_data[np_colors == 0]
    X1 = np_data[np_colors == 1]
    xs_full = [X0, X1]
    n0, n1 = X0.shape[0], X1.shape[0]
    n = n0 + n1
    pi_0 = n0 / n
    pi_1 = n1 / n

    print(f"[Info] Data shape for color 0: {X0.shape}")
    print(f"[Info] Data shape for color 1: {X1.shape}")

    # Unfair baseline (KMeans)
    all_x = np.concatenate([X0, X1], axis=0)
    kmeans = KMeans(n_clusters=K, random_state=args.seed)
    unfair_labels = kmeans.fit_predict(all_x)
    unfair_assignments = [unfair_labels[:n0], unfair_labels[n0:]]
    centers_base = kmeans.cluster_centers_.astype(np.float32)

    obj_base, bal_base = evaluation(xs_full, unfair_assignments, centers_base, K)
    print(f"[Unfair] Cost / Balance: {obj_base:.3f} / {bal_base:.3f}")

    # Initialize centers from baseline
    centers = centers_base.copy()

    # (optional) gradient descent
    centers_module = None
    centers_optimizer = None
    if getattr(args, "gradient_descent", False):
        centers_module = nn.Linear(d, K, bias=False)
        centers_module.weight.data = torch.from_numpy(centers).float()
        centers_optimizer = torch.optim.Adam(centers_module.parameters(), lr=getattr(args, "lr", 1e-2))
        if getattr(args, "use_cuda", False) and torch.cuda.is_available():
            centers_module = centers_module.cuda()

    deterministic_assign = getattr(args, "deterministic_assign", True)
    use_global_recenter = getattr(args, "global_recenter", False)
    eta_pairs = int(getattr(args, "eta_pairs", 200000))

    # Choose L from target block size
    block_size = max(1, int(args.batch_size))
    L = max(1, int(min(n0, n1) // block_size))

    t0 = time.time()
    best = {'balance': 0.0, 'energy': float('inf'), 'it': 0}
    subbest = {'balance': 0.0, 'energy': float('inf'), 'it': 0}

    for it in range(1, args.iters + 1):
        if centers_module is not None:
            centers = centers_module.weight.detach().cpu().numpy().astype(np.float32)
        eta_thr = estimate_eta_threshold_global(
            X0, X1, pi_0, pi_1, centers,
            epsilon=float(args.epsilon),
            n_pairs=eta_pairs,
            seed=args.seed + it
        )
        blocks0 = partition_L_plus_remainder(n0, L, seed=args.seed + it)
        blocks1 = partition_L_plus_remainder(n1, L, seed=args.seed + it)
        blocks0, blocks1 = _merge_last_block_if_extra(blocks0, blocks1)

        B = len(blocks0)
        assert B == len(blocks1), "Block count mismatch after merge; should not happen."

        # Debug: ensure full coverage
        cov0 = sum(len(b) for b in blocks0)
        cov1 = sum(len(b) for b in blocks1)
        if cov0 != n0 or cov1 != n1:
            raise RuntimeError(f"[BUG] Coverage mismatch: cov0={cov0},n0={n0} cov1={cov1},n1={n1}")

        batches = []
        for b in range(B):
            I0 = blocks0[b]
            I1 = blocks1[b]
            X0b = X0[I0]
            X1b = X1[I1]
            if X0b.shape[0] == 0 or X1b.shape[0] == 0:
                continue
            group_xs = [X0b, X1b]

            W = optimize_W_from_threshold(group_xs, pi_0, pi_1, centers, eta_thr)
            Taxs, gammas, _, _ = optimize_coupling(
                group_xs, W, centers,
                pi_0=pi_0, pi_1=pi_1,
                numItermax=args.numItermax,
                numThreads=args.numThreads
            )
            gammas = gammas / max(B, 1)

            batches.append((group_xs, W, Taxs, gammas))

        if len(batches) == 0:
            print(f"[{it}/{args.iters}] No valid blocks (empty). Skipping.")
            continue

        xs_batches, Ws, Taxs_b, gammas_b = zip(*batches)

        centers = optimize_centers_global(
            xs_batches, Ws, Taxs_b, gammas_b,
            centers=centers,
            K=K,
            pi_0=pi_0, pi_1=pi_1,
            seed=args.seed + it,
            kmeans_max_iter=int(getattr(args, "center_kmeans_iter", 50)),
            gradient_descent=getattr(args, "gradient_descent", False),
            centers_module=centers_module,
            centers_optimizer=centers_optimizer,
            use_cuda=getattr(args, "use_cuda", False),
            gd_steps=int(getattr(args, "center_gd_steps", 20)),
        )

        if centers_module is not None and getattr(args, "gradient_descent", False):
            with torch.no_grad():
                w = torch.from_numpy(centers).to(centers_module.weight.device)
                centers_module.weight.copy_(w)

        color_xs, assignments = assigning(
            xs_batches, Ws, Taxs_b, gammas_b,
            centers=centers, K=K,
            deterministic=deterministic_assign,
            seed=args.seed + it
        )

        if len(color_xs[0]) != n0 or len(color_xs[1]) != n1:
            raise RuntimeError(
                f"[ERROR] assigning() did not return full assignments: "
                f"got (n0'={len(color_xs[0])}, n1'={len(color_xs[1])}) "
                f"expected (n0={n0}, n1={n1})."
            )

        obj, bal = evaluation(color_xs, assignments, centers, K)
        print(f"[{it}/{args.iters}] Cost / Balance: {obj:.3f} / {bal:.3f}")

        if use_global_recenter:
            centers_recentered, cnts = global_recenter_from_hard_assignments(
                color_xs, assignments, centers, K,
                reinit_empty="keep",
                seed=args.seed + it
            )
            obj2, bal2 = evaluation(color_xs, assignments, centers_recentered, K)
            if obj2 <= obj + 1e-12:
                centers = centers_recentered
                obj, bal = obj2, bal2
                print(f"[{it}/{args.iters}] Recentered -> Cost / Balance: {obj:.3f} / {bal:.3f}")

        if bal > best['balance']:
            best.update(balance=bal, energy=obj, it=it)
        if obj < subbest['energy']:
            subbest.update(balance=bal, energy=obj, it=it)

    elapsed = time.time() - t0

    # Save results
    stats = {
        'seed': args.seed,
        'iters': args.iters,
        'epsilon': args.epsilon,
        'target_block_size': block_size,
        'main_blocks_L': L,
        'l2_normalize': args.l2_normalize,
        'eta_pairs': eta_pairs,
        'deterministic_assign': deterministic_assign,
        'global_recenter': use_global_recenter,
        'time_sec': elapsed,
        'best_it': best['it'],
        'best_balance': best['balance'],
        'best_energy': best['energy'],
        'subbest_it': subbest['it'],
        'subbest_balance': subbest['balance'],
        'subbest_energy': subbest['energy'],
    }
    df = pd.DataFrame([stats])
    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"{args.data_name}.csv"
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', index=False, header=False)
    else:
        df.to_csv(csv_path, mode='w', index=False, header=True)

    print(f"[BEST balance] It {best['it']}: {best['energy']:.3f} / {best['balance']:.3f}")
    print(f"[BEST energy] It {subbest['it']}: {subbest['energy']:.3f} / {subbest['balance']:.3f}")

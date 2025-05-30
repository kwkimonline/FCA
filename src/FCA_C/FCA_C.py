import os
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

from .optimization import optimize_coupling, optimize_center, optimize_W
from .assignment import assigning
from ..evaluation import evaluation
from ..datasets import load_data, NumpyDataset, NumpyDataLoader


def runner(args):
    """
    FCA-C runner
    """
    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load and split data by color
    data_dir = Path(f"data/{args.data_name}/")
    _, np_data, _, np_colors, K, d, n_color = load_data(
        name=args.data_name,
        l2_normalize=args.l2_normalize,
        data_dir=str(data_dir)
    )
    K = args.K if args.K > 0 else K
    xs = [np_data[np_colors == c] for c in range(n_color)]
    for idx, arr in enumerate(xs):
        print(f"[Info] Data shape for color {idx}: {arr.shape}")

    # DataLoader setup
    counts = np.array([len(x) for x in xs], dtype=float)
    pis = counts / counts.sum()
    pi_0, pi_1 = pis if n_color == 2 else (None, None)

    all_x = np.concatenate(xs)
    labels = np.repeat(np.arange(n_color), counts.astype(int))
    dataset = NumpyDataset(all_x, labels)
    batch_size = args.batch_size
    loader = NumpyDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        output_ids=True
    )

    # Unfair baseline
    kmeans = KMeans(n_clusters=K, random_state=args.seed)
    unfair_labels = kmeans.fit_predict(all_x)
    split = len(xs[0])
    unfair_assignments = [unfair_labels[:split], unfair_labels[split:]]
    centers = kmeans.cluster_centers_
    obj, bal = evaluation(xs, unfair_assignments, centers, K)
    print(f"[Unfair] Cost / Balance: {obj:.3f} / {bal:.3f}")

    # Setup for center updates
    centers_module = None
    centers_optimizer = None
    if args.gradient_descent:
        centers_module = nn.Linear(d, K, bias=False)
        centers_module.weight.data = torch.from_numpy(centers).float()
        centers_optimizer = torch.optim.Adam(centers_module.parameters(), lr=args.lr)
        if args.use_cuda and torch.cuda.is_available():
            centers_module = centers_module.cuda()

    # Tracking bests and timings
    best = {'balance': 0.0, 'energy': float('inf'), 'it': 0}
    subbest = {'balance': 0.0, 'energy': float('inf'), 'it': 0}

    # Iterative optimization
    for it in range(1, args.iters + 1):
        if args.gradient_descent:
            centers = centers_module.weight.detach().cpu().numpy()

        batches = []
        for _, batch_xs, batch_cols in loader:
            group_xs = [batch_xs[batch_cols == c] for c in range(n_color)]
            # inner loop for optimization
            for _ in range(args.iters_inner):
                W = optimize_W(group_xs, pi_0, pi_1, centers, args.epsilon)
                Taxs, gammas, _, _ = optimize_coupling(
                    group_xs, W, centers,
                    numItermax=args.numItermax,
                    numThreads=args.numThreads
                )
                # Update centers
                centers = optimize_center(
                    group_xs,
                    Taxs,
                    gammas / gammas.size,
                    W,
                    centers,
                    centers_module,
                    centers_optimizer,
                    K,
                    seed=args.seed,
                    gradient_descent=args.gradient_descent,
                    use_cuda=args.use_cuda
                )
            batches.append((group_xs, W, Taxs, gammas))

        # Assign and evaluate
        xs_batches, Ws, Taxs_b, gammas_b = zip(*batches)
        color_xs, assignments = assigning(xs_batches, Ws, Taxs_b, gammas_b, centers, K)
        obj, bal = evaluation(color_xs, assignments, centers, K)
        print(f"[{it}/{args.iters}] Cost / Balance: {obj:.3f} / {bal:.3f}")

        # Track best/subbest
        if bal > best['balance']:
            best.update(balance=bal, energy=obj, it=it)
        if obj < subbest['energy']:
            subbest.update(balance=bal, energy=obj, it=it)

    # Save results
    stats = {
        'seed': args.seed,
        'gradient_descent': args.gradient_descent,
        'iters': args.iters,
        'epsilon': args.epsilon,
        'batch_size': args.batch_size,
        'l2_normalize': args.l2_normalize,
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

    # Final summary
    print(f"[BEST balance] It {best['it']}: {best['energy']:.3f} / {best['balance']:.3f}")
    print(f"[BEST energy] It {subbest['it']}: {subbest['energy']:.3f} / {subbest['balance']:.3f}")

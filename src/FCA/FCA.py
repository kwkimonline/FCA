import os
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

from .optimization import optimize_coupling, optimize_center
from .assignment import assigning
from ..evaluation import evaluation
from ..datasets import load_data, NumpyDataset, NumpyDataLoader


def runner(args):
    """
    FCA runner
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
    labels = np.concatenate([np.full(x.shape[0], i) for i, x in enumerate(xs)])
    dataset = NumpyDataset(np.concatenate(xs), labels)
    loader = NumpyDataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=False)

    # Unfair baseline
    flat_xs = np.concatenate(xs)
    kmeans = KMeans(n_clusters=K, random_state=args.seed)
    unfair_labels = kmeans.fit_predict(flat_xs)
    split_idx = xs[0].shape[0]
    unfair_assignments = [unfair_labels[:split_idx], unfair_labels[split_idx:]]
    centers = kmeans.cluster_centers_
    obj, bal = evaluation(xs, unfair_assignments, centers, K)
    print(f"[Unfair] Cost / Balance: {obj:.3f} / {bal:.3f}")

    # Setup for center updates
    centers_module = None
    centers_optimizer = None
    if args.gradient_descent:
        centers_module = nn.Linear(d, K, bias=False)
        centers_optimizer = torch.optim.Adam(centers_module.parameters(), lr=args.lr)
        if args.use_cuda and torch.cuda.is_available():
            centers_module.cuda()

    # Tracking bests and timings
    best = {'balance': 0.0, 'energy': np.inf, 'it': 0}
    subbest = {'balance': 0.0, 'energy': np.inf, 'it': 0}

    # Iterative optimization
    for it in range(1, args.iters + 1):
        if args.gradient_descent:
            centers = centers_module.weight.detach().cpu().numpy()

        batch_results = []
        for batch_xs, batch_colors in loader:
            sub_xs = [batch_xs[batch_colors == c] for c in range(n_color)]
            Taxs, gammas, *_ = optimize_coupling(
                sub_xs,
                centers,
                numItermax=args.numItermax,
                numThreads=args.numThreads
            )
            batch_results.append((sub_xs, Taxs, gammas))

        # Aggregate gamma normalization
        all_Taxs = [res[1] for res in batch_results]
        all_gammas = [res[2] / res[2].size for res in batch_results]

        # Update centers
        flat_Taxs = np.concatenate(all_Taxs)
        flat_gammas = np.concatenate(all_gammas)
        centers = optimize_center(
            flat_Taxs,
            flat_gammas,
            centers,
            centers_module,
            centers_optimizer,
            K,
            max_iter=args.max_iter,
            seed=args.seed,
            gradient_descent=args.gradient_descent,
            use_cuda=args.use_cuda
        )

        # Assign and evaluate
        xs_batches = [res[0] for res in batch_results]
        color_xs, color_assignments = assigning(xs_batches, all_Taxs, all_gammas, centers, K)
        obj, bal = evaluation(color_xs, color_assignments, centers, K)
        print(f"[{it}/{args.iters}] Cost / Balance: {obj:.3f} / {bal:.3f}")

        # Track best/subbest
        if bal > best['balance']:
            best.update(balance=bal, energy=obj, it=it)
        if obj < subbest['energy']:
            subbest.update(energy=obj, balance=bal, it=it)

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


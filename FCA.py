import os
import ot
import numpy as np
import pandas as pd
import argparse
import time
from scipy.spatial import distance
from sklearn.cluster import KMeans
from base.datasets import load_data, NumpyDataset, NumpyDataLoader



def optimize_coupling(xs, centers,
                      numItermax=5000,
                      stopThr=1e-7, verbose=False, numThreads=3):

    # only for binary sensitive attribute
    n_colors = 2
    assert n_colors == len(xs)
    ns = [len(xs_i) for xs_i in xs]
    pis = np.array(ns) / np.sum(ns)

    major_id, minor_id = np.argmax(ns), np.argmin(ns)
    pi_major, pi_minor = pis[major_id], pis[minor_id]
    
    all_mid_xs = [np.zeros(xs_i.shape) for xs_i in xs]
    
    xs_ma, xs_mi = xs[major_id], xs[minor_id]

    n_major, n_minor = ns[major_id], ns[minor_id]
    w_major, w_minor = np.ones(n_major) / n_major, np.ones(n_minor) / n_minor
    
    M_mami = 2 * pi_major * pi_minor * ot.dist(xs_ma, xs_mi)
    mid_xs_mami = pi_major * np.repeat(xs_ma, n_minor, 0) + pi_minor * np.tile(xs_mi, (n_major, 1))
    mid_distances = distance.cdist(mid_xs_mami, centers, metric='minkowski', p=2)
    mid_assignments = np.argmin(mid_distances, axis=1)
    C_mami = (mid_distances[np.arange(mid_distances.shape[0]), mid_assignments]**2)
    C_mami = C_mami.reshape(n_minor, n_major).T
    
    Coupling_mami = ot.emd(w_major, w_minor, M_mami+C_mami, numItermax=numItermax, numThreads=numThreads)
    
    Txs_ma = n_major * Coupling_mami @ xs_mi
    all_mid_xs[major_id] += pi_major * xs_ma + pi_minor * Txs_ma
    NN_ids = np.argmin(distance.cdist(xs_mi, Txs_ma, metric='minkowski', p=2), axis=1)
    NN_Txs_mi, NN_xs_mi = xs_ma[NN_ids], Txs_ma[NN_ids]
    all_mid_xs[minor_id] += pi_minor * NN_xs_mi + pi_major * NN_Txs_mi

    concat_mid_xs = np.concatenate(all_mid_xs)
    train_concat_mid_xs = all_mid_xs[major_id]

    return train_concat_mid_xs, concat_mid_xs, all_mid_xs



def optimize_center(all_mid_xs, K, seed=2024):
    kmeans = KMeans(n_clusters=K, random_state=seed)
    kmeans.fit(all_mid_xs)
    new_centers = kmeans.cluster_centers_
    return new_centers



def assigning(original_xs, assign_xs, centers, colors, K):
    n_color = len(original_xs)
    
    # objectives
    color_original_cluster_cnts, color_assign_cluster_cnts = [], []
    original_objective, assign_objective, original_assign_objective = 0.0, 0.0, 0.0
    for original_xs_i, assign_xs_i in zip(original_xs, assign_xs):
        original_distances_i = distance.cdist(original_xs_i, centers, metric='minkowski', p=2)
        original_assignments_i = np.argmin(original_distances_i, axis=1)
        original_objective += (original_distances_i[np.arange(original_distances_i.shape[0]), original_assignments_i]**2).sum()

        assign_distances_i = distance.cdist(assign_xs_i, centers, metric='minkowski', p=2)
        assign_assignments_i = np.argmin(assign_distances_i, axis=1)
        assign_objective += (assign_distances_i[np.arange(assign_distances_i.shape[0]), assign_assignments_i]**2).sum()
        
        original_assign_objective += (original_distances_i[np.arange(original_distances_i.shape[0]), assign_assignments_i]**2).sum()
        
        sub_color_original_cluster_cnts, sub_color_assign_cluster_cnts = [], []
        for k in range(K):
            sub_color_original_cluster_cnts.append((original_assignments_i == k).sum())
            sub_color_assign_cluster_cnts.append((assign_assignments_i == k).sum())
        color_original_cluster_cnts.append(sub_color_original_cluster_cnts)
        color_assign_cluster_cnts.append(sub_color_assign_cluster_cnts)
    
    # balance
    original_cluster_cnts, assign_cluster_cnts = np.array(color_original_cluster_cnts).sum(axis=0), np.array(color_assign_cluster_cnts).sum(axis=0)
    original_k_ratio, assign_k_ratio = original_cluster_cnts / original_cluster_cnts.sum(), assign_cluster_cnts / assign_cluster_cnts.sum()
    original_s_balances, assign_s_balances = [], []
    for color in range(n_color):
        original_s_balances.append((np.array(color_original_cluster_cnts[color]) / np.sum(color_original_cluster_cnts[color])) / original_k_ratio)
        assign_s_balances.append((np.array(color_assign_cluster_cnts[color]) / np.sum(color_assign_cluster_cnts[color])) / assign_k_ratio)
    original_balance = np.array(original_s_balances).min(axis=1).min()
    assign_balance = np.array(assign_s_balances).min(axis=1).min()
    
    objectives = (original_objective, assign_objective, original_assign_objective)
    balances = (original_balance, assign_balance)
    return objectives, balances



def runner(args):
    np.random.seed(args.seed)

    data_dir = 'data/'
    np_data, np_colors, K, d, n_color = load_data(name=args.data_name, l2_normalize=args.l2_normalize, data_dir=data_dir)
    
    xs = []
    for color in range(n_color):
        xs.append(np_data[np_colors == color])
        print(f'[Info] Data shape for {color}th color: {np_data[np_colors == color].shape}')
    colors = [i*np.ones(xs_i.shape[0]) for i, xs_i in enumerate(xs)]

    dset = NumpyDataset(np.concatenate(xs), np.concatenate(colors))
    dloader = NumpyDataLoader(dset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # initial centers
    centers = np.random.normal(0, 1, (K, d))

    # optimizing
    best_it, best_original_energy, best_energy, best_balance = 0, 1e+10, 1e+10, 0.0
    subbest_it, subbest_original_energy, subbest_energy, subbest_balance = 0, 1e+10, 1e+10, 0.0
    for it in range(args.iters):
        start_time = time.time()
        it += 1
        
        all_xs, all_train_concat_mid_xs, all_concat_mid_xs, all_concat_colors = [], [], [], []
        for batch_xs, batch_colors in dloader:
            sub_batch_xs = []
            for color in range(n_color):
                sub_batch_xs.append(batch_xs[batch_colors == color])
            sub_batch_colors = np.concatenate([i*np.ones(sub_batch_xs_i.shape[0]) for i, sub_batch_xs_i in enumerate(sub_batch_xs)])
            train_concat_mid_xs, concat_mid_xs, all_mid_xs = optimize_coupling(sub_batch_xs, centers, numItermax=args.numItermax)
            
            all_xs.append(np.concatenate(sub_batch_xs))
            all_train_concat_mid_xs.append(train_concat_mid_xs)
            all_concat_mid_xs.append(concat_mid_xs)
            all_concat_colors.append(sub_batch_colors)
        all_xs, all_train_concat_mid_xs = np.concatenate(all_xs), np.concatenate(all_train_concat_mid_xs)
        all_concat_mid_xs, all_concat_colors = np.concatenate(all_concat_mid_xs), np.concatenate(all_concat_colors)
        
        centers = optimize_center(all_train_concat_mid_xs, K, seed=args.seed)

        color_xs, color_mid_xs, color_colors = [], [], []
        for color in range(n_color):
            color_xs.append(all_xs[all_concat_colors == color])
            color_mid_xs.append(all_concat_mid_xs[all_concat_colors == color])
            color_colors.append(all_concat_colors[all_concat_colors == color])
        objectives, balances = assigning(color_xs, color_mid_xs, centers, color_colors, K)
        
        print(f'[{it}/{args.iters}] Cost / Balance: {objectives[2]:.3f} / {balances[1]:.3f}')
        
        if balances[1] > best_balance:
            best_it = it
            best_balance = balances[1]
            best_energy = objectives[2]
            best_original_energy = objectives[0]
        if objectives[2] < subbest_energy:
            subbest_it = it
            subbest_balance = balances[1]
            subbest_energy = objectives[2]
            subbest_original_energy = objectives[0]

    # results
    results = {'seed':[args.seed],
                'iters': [args.iters],
                'batch_size': [args.batch_size],
                'l2_normalize': [args.l2_normalize],
                'best_it': [best_it],
                'best_original_energy': [best_original_energy],
                'best_energy': [best_energy],
                'best_balance': [best_balance],
                'subbest_it': [subbest_it],
                'subbest_original_energy': [subbest_original_energy],
                'subbest_energy': [subbest_energy],
                'subbest_balance': [subbest_balance]
                }

    columns = list(results.keys())
    df_results = pd.DataFrame(results, columns=columns)
    os.makedirs('results/', exist_ok=True)
    result_name = f'results/{args.data_name}_FCA.csv'
    if os.path.exists(result_name):
        df_results.to_csv(result_name, mode='a', index=False, header=False)
    else:
        df_results.to_csv(result_name, mode='w', index=False, header=True)

    print(f'[BEST balance @ iter={args.iters}] Cost / Balance: {best_energy:.3f} / {best_balance:.3f}')
    print(f'[BEST cost @ iter={args.iters}] Cost / Balance: {subbest_energy:.3f} / {subbest_balance:.3f}')



""" MAIN """
parser = argparse.ArgumentParser(description='FCA')
parser.add_argument('--seed', default=2024, type=int)
parser.add_argument('--iters', default=100, type=int)
parser.add_argument('--numItermax', default=1000000, type=int)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--data_name', default='Adult', type=str)
parser.add_argument('--l2_normalize', action='store_true')

args = parser.parse_args()
print('='*20)
for key, value in vars(args).items():
    print(f'\t [{key}]: {value}')

if __name__ == "__main__":
    runner(args)
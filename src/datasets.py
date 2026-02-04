import numpy as np
import torch
import os
from sklearn.datasets import make_blobs
import sys
import requests, zipfile, io
import pandas as pd
from sklearn.preprocessing import StandardScaler


def L2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-7)
    return X / norms


def load_data(name, l2_normalize=False, data_dir=None, balancing=False):

    data = None
    colors = None
    K = None
    d = None

    if name == 'Adult':        
        _path = 'adult.data'
        data_path = os.path.join(data_dir,_path)
        race_is_sensitive_attribute = 0        
        if race_is_sensitive_attribute==1:
            m = 5
        else:
            m = 2
        K = 10
        if (not os.path.exists(data_path)): 
            print('Adult data set does not exist in current folder --- Have to download it')
            r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', allow_redirects=True)
            if r.status_code == requests.codes.ok:
                print('Download successful')
            else:
                print('Could not download Adult data set - please download it manually')
                sys.exit()
            open(data_path, 'wb').write(r.content)        
        df = pd.read_csv(data_path, sep=',',header=None)
        n = df.shape[0]        
        sens_attr = 9
        sex = df[sens_attr]
        sens_attributes = list(set(sex.astype(str).values))
        df = df.drop(columns=[sens_attr])
        colors = np.zeros(n, dtype=int)
        colors[sex.astype(str).values == ' Male'] = 1
        cont_types = np.where(df.dtypes=='int')[0]
        df = df.iloc[:,cont_types]
        data = np.array(df.values, dtype=float)
        data = data[:,[0,1,2,3,5]]
        d = data.shape[1]
        n_color = 2

    else:
        NotImplementedError('You should align with this function if using a customized dataset.')

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    if l2_normalize:
        data = L2_normalize(data)
    np_data, np_colors = data.copy(), colors.copy()
    data, colors = torch.from_numpy(data).float(), torch.from_numpy(colors).float()

    return data, np_data, colors, np_colors, K, d, n_color



# numpy datasets
class NumpyDataset:
    def __init__(self, data, colors):
        self.data = data
        self.colors = colors

    def __getitem__(self, index):
        return self.data[index], self.colors[index]

    def __len__(self):
        return len(self.data)


class NumpyDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, output_ids=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.output_ids = output_ids

    def __iter__(self):
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)
        if self.drop_last:
            self.indices = self.indices[:len(self.indices) - len(self.indices) % self.batch_size]
        return self

    def __next__(self):
        if len(self.indices) == 0:
            raise StopIteration
        
        indices = self.indices[:self.batch_size]
        self.indices = self.indices[self.batch_size:]
        
        if self.output_ids:
            batch = []
            for i in indices:
                batch.append((i, self.dataset[i][0], self.dataset[i][1]))
            ids, data, labels = zip(*batch)
            return np.array(ids), np.array(data), np.array(labels)
        else:
            batch = [self.dataset[i] for i in indices]
            data, labels = zip(*batch)
            return np.array(data), np.array(labels)

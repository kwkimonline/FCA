import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from base.utils import normalizefea



def load_data(name, l2_normalize=False, data_dir=None, balancing=False):
    # from VFC: https://github.com/imtiazziko/Variational-Fair-Clustering

    data = None
    colors = None
    K = None
    d = None

    if name == 'Adult':
        
        _path = 'adult.data'
        data_path = os.path.join(data_dir,_path)

        K = 10        
        df = pd.read_csv(data_path, sep=',',header=None)
        n = df.shape[0]
        
        sens_attr = 9
        sex = df[sens_attr]
        sens_attributes = list(set(sex.astype(str).values))
        df = df.drop(columns=[sens_attr])
        colors = np.zeros(n, dtype=int)
        colors[sex.astype(str).values == sens_attributes[1]] = 1

        cont_types = np.array([0, 2, 4, 9, 10, 11]) # np.where(df.dtypes=='int')[0]
        df = df.iloc[:,cont_types]
        data = np.array(df.values, dtype=float)
        
        data = data[:,[0,1,2,3,5]]

        d = data.shape[1]
        n_color = 2

    # return
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    if l2_normalize:
        data = normalizefea(data)
    np_data, np_colors = data.copy(), colors.copy()

    return np_data, np_colors, K, d, n_color


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


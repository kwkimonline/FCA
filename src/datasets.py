import numpy as np
import torch
import os
from sklearn.datasets import make_blobs
import sys
import requests, zipfile, io
import pandas as pd
from sklearn.preprocessing import StandardScaler


def L2_normalize(X):
    """
    L2-normalize each row of X, with stability threshold.
    """
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

    elif name == 'Bank':
        K = 10
        _path = 'bank-additional-full.csv'
        data_path = os.path.join(data_dir, _path)
        if (not os.path.exists(data_path)): 
            print('Bank dataset does not exist in current folder --- Have to download it')
            r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip', allow_redirects=True)
            if r.status_code == requests.codes.ok:
                print('Download successful')
            else:
                print('Could not download - please download')
                sys.exit()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            open(data_path, 'wb').write(z.read('bank-additional/bank-additional-full.csv'))
        df = pd.read_csv(data_path,sep=';')
        sex = df['marital'].astype(str).values
        sens_attributes = list(set(sex))
        if 'unknown' in sens_attributes:
            sens_attributes.remove('unknown')
        df1 = df.loc[(df['marital'] == 'single') + (df['marital'] == 'divorced')]
        df2 = df.loc[df['marital'] == 'married']
        df = [df1, df2]
        df = pd.concat(df)
        sex = df['marital'].astype(str).values
        df = df[['age','duration','euribor3m', 'nr.employed', 'cons.price.idx', 'campaign']].values
        colors = np.zeros(df.shape[0], dtype=int)
        colors[sex == 'married'] = 1
        data = np.array(df, dtype=float)
        d = data.shape[1]
        n_color = 2

    elif name == 'Census':
        csv_file = 'subsampled_census1990.csv'
        data_path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(data_path)
        df = df.sample(n=20000).reset_index(drop=True)
        selected_columns = ['dAncstry1','dAncstry2','iAvail','iCitizen','iClass','dDepart',
                            'iDisabl1','iDisabl2','iEnglish','iFeb55','iFertil',
                            'dHispanic','dHour89','dHours','iImmigr',
                            'dIncome1','dIncome2','dIncome3','dIncome4','dIncome5','dIncome6','dIncome7','dIncome8',
                            'dIndustry','iKorean','iLang1','iLooking','iMarital','iMay75880','iMeans','iMilitary','iMobility','iMobillim',
                            'dOccup','iOthrserv','iPerscare','dPOB','dPoverty','dPwgt1','iRagechld','dRearning',
                            'iRelat1','iRelat2','iRemplpar','iRiders','iRlabor','iRownchld','dRpincome','iRPOB','iRrelchld','iRspouse','iRvetserv',
                            'iSchool','iSept80','iSubfam1','iSubfam2','iTmpabsnt','dTravtime','iVietnam','dWeek89','iWork89','iWorklwk','iWWII',
                            'iYearsch','iYearwrk','dYrsserv']
        variables_of_interest = ['dAge', 'iSex']
        text_columns = []
        for col in text_columns:
            df[col] = df[col].astype('category').cat.codes
        variable_columns = [df[var] for var in variables_of_interest]
        for col in df:
            if col in text_columns or col not in selected_columns: continue
            df[col] = df[col].astype(float)
        data = np.array(df[[col for col in selected_columns]], dtype=float)
        colors = np.array(df[['iSex']], dtype=float).flatten()
        K = 20
        d = data.shape[1]
        n_color = 2

    else:
        NotImplementedError('Are you trying to use a customized dataset?')

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

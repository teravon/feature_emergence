import numpy as np
from datetime import datetime
import os
from src.datasets.paths import *
from src.datasets.load_ascadr import *
from src.datasets.load_eshard import *
from src.datasets.load_chesctf import *
from os.path import exists

from sklearn.preprocessing import MinMaxScaler

def snr_fast(x, y):
    ns = x.shape[1]
    unique = np.unique(y)
    means = np.zeros((len(unique), ns))
    variances = np.zeros((len(unique), ns))

    for i, u in enumerate(unique):
        new_x = x[np.argwhere(y == int(u))]
        means[i] = np.mean(new_x, axis=0)
        variances[i] = np.var(new_x, axis=0)
    return np.var(means, axis=0) / np.mean(variances, axis=0)



def load_dataset(identifier: str, path: str, target_byte: int, traces_dim: int, leakage_model="ID", n_prof=None):
    
    dataset_file = get_dataset_filepath(path, identifier, traces_dim, leakage_model=leakage_model)
    
    if identifier == "eshard":
        dataset = ReadEshard(70000 if n_prof is None else n_prof, 0, 10000, target_byte, leakage_model, dataset_file, number_of_samples=traces_dim)
    if identifier == "ascad-variable":
        dataset = ReadASCADr(200000 if n_prof is None else n_prof, 0, 10000, target_byte, leakage_model,
                                                dataset_file,
                                                number_of_samples=traces_dim)
    if identifier == "ches_ctf":
        dataset = ReadCHESCTF(30000 if n_prof is None else n_prof, 0, 10000, target_byte, leakage_model,
                                                         dataset_file,
                                                         number_of_samples=traces_dim)
        
    return dataset


def desynch_scale(prof_set, attack_set):
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler = StandardScaler()
        if prof_set.shape[0] > 20000:
            scaler.fit(prof_set.T)
            prof_new = scaler.transform(prof_set.T).T
        else:
            prof_new = scaler.fit_transform(prof_set.T).T

        if attack_set is not None:
            attack_new = scaler.fit_transform(attack_set.T).T
        else:
            attack_new = None
        return prof_new, attack_new

def scale_dataset(prof_set, attack_set, scaler):
        if prof_set.shape[0] > 20000:
            scaler.fit(prof_set[:20000])
            prof_new = scaler.transform(prof_set)
        else:
            prof_new = scaler.fit_transform(prof_set)

        if attack_set is not None:
            attack_new = scaler.transform(attack_set)
        else:
            attack_new = None
        return prof_new, attack_new

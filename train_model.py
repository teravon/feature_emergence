import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
from random_class_mlp_cnn import *
from profiling_and_attack import mlp, attack, mlp_eshard, cnn_ASCAD, cnn
lm = "ID"
if len(sys.argv) > 2:
    lm = sys.argv[2] 

dataset_id = 'eshard' #ascad-variable
traces_dim = 1400    #2000

class HackyDatasetObject(object):
    pass

n_prof= int(sys.argv[1])

target_byte = 2
#Base path for datasets
path = "/base/path/Datasets"
dataset = load_dataset(dataset_id, path, target_byte, traces_dim, leakage_model=lm)
dataset.x_profiling, dataset.x_attack = scale_dataset(dataset.x_profiling[:n_prof], dataset.x_attack, StandardScaler())
dataset.profiling_labels = dataset.profiling_labels[:n_prof]

# Update Attack Model here to other options
attack_mod=  cnn(256 if lm=="ID" else 9, traces_dim)
temp = "cnn_eshard_"
#attack_mod=  mlp_eshard(256 if lm=="ID" else 9, traces_dim)
#temp = "mlp_eshard_"
# attack_mod=  mlp(256 if lm=="ID" else 9, traces_dim)
#temp = "mlp_ascadr_"
# attack_mod=  cnn_ASCAD(256 if lm=="ID" else 9, traces_dim)
#temp = "cnn_ascadr_"

broad_dataset = HackyDatasetObject()
broad_dataset.dataset_target = dataset

#Update path for checkpoint saving here
ge, nt, pi, ge_v = attack(broad_dataset, None, traces_dim, attack_model=attack_mod, batch_size=400, 
                    original_traces=True, synthetic_traces=False, save_path=f"model_checkpoints/{temp}")
attack_mod.save(f"model_{dataset_id}.keras")

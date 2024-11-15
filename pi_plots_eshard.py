from profiling_and_attack import mlp_eshard, information, pi_mlp_probe, mlp, cnn

from sklearn.preprocessing import StandardScaler
from utils import *
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA

from pi_networks import pi_layers_mlp
import sys

def hw(input):

    result = np.zeros_like(input)
    for i in range(8):
        result += (input >> i) & 0x1
    return result

class HackyDatasetObject(object):
    pass
dataset_id = 'eshard'
resample_window=80
traces_dim = 1400
model = cnn(9,traces_dim)
n_prof= 50000

lm = "HW"
target_byte = 2
path = "/base/path/Datasets"
dataset = load_dataset(dataset_id, path, target_byte, traces_dim, leakage_model=lm, n_prof=n_prof)
pi_hw_s1, pi_hw_s2, pi_label = [], [],[]
irrelevant_features= np.zeros((15, 100))

dataset.x_profiling, dataset.x_attack = scale_dataset(dataset.x_profiling[:n_prof], dataset.x_attack, StandardScaler())
layer = int(sys.argv[1])
N_COMP = 5
byte_list = [2, 0, 1, 3]
pi_s1, pi_s2, pi_label_tmp = np.zeros(16), np.zeros(16), np.zeros(1)
irrelevant_features = []
layers = ["conv_1", "conv_2", "fc_1", "fc_2"]
for i in range(1, 100, 1):
    print(i)
    model.load_weights(f"model_checkpoints/cnn_eshard_{i:02d}.weights.h5")
    out_layer = model.get_layer(layers[layer]).output
    new_model = Model(model.inputs, out_layer)
    tmp =  new_model.predict(dataset.x_profiling)
    tmp_att = new_model.predict(dataset.x_attack).copy()

    s1, s2_l, label = pi_layers_mlp(None, dataset, tmp, tmp_att, layer, np.zeros(16), np.zeros(16), np.zeros(1))
    for j in range(len(byte_list)):
        if byte_list[j] == target_byte:
            pi_hw_s2.append(s2_l[j])
        else:
            irrelevant_features.append(s2_l[j])
    pi_hw_s1.append(s1)
    pi_label.append(label)
    
np.savez(f"pi_eshard_layer_cnn_{layer}.npz", pi_hw_s1=np.array(pi_hw_s1), pi_hw_s2=np.array(pi_hw_s2), pi_label=np.array(pi_label), irrelevant_features=irrelevant_features)
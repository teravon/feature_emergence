from profiling_and_attack import mlp_eshard, information, pi_mlp_probe, mlp, cnn_ASCAD

from sklearn.preprocessing import StandardScaler
from utils import *
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA
import sys

from pi_vary_layer_ASCAD import pi_layers_mlp

def hw(input):

    result = np.zeros_like(input)
    for i in range(8):
        result += (input >> i) & 0x1
    return result

class HackyDatasetObject(object):
    pass
dataset_id = 'ascad-variable'
resample_window=80
traces_dim = 2000
model = cnn_ASCAD(256,traces_dim)
n_prof= 50000

lm = "ID"
target_byte = 2
path = "/mnt/d/Datasets"

dataset = load_dataset(dataset_id, path, target_byte, traces_dim, leakage_model=lm, n_prof=n_prof)
pi_hw_s1, pi_hw_s2, pi_label, pi_s5_l, pi_s6_l = [], [],[], [], []
irrelevant_features= []
byte_list = [2, 4, 5, 11]
dataset.x_profiling, dataset.x_attack = scale_dataset(dataset.x_profiling[:n_prof], dataset.x_attack, StandardScaler())
layer = 1 if len(sys.argv) < 2 else int(sys.argv[1])
N_COMP = 5
irrelevant_features = []
layers = ["conv_1", "conv_2", "conv_3", "conv_4", "fc_1", "fc_2"]
for i in range(1, 100):
    print(i)
    model.load_weights(f"model_checkpoints/cnn_ascadr_{i:02d}.weights.h5")
    outputs = [model.get_layer(layers[layer]).output]
    new_model = Model(model.inputs, outputs)
    tmp =  new_model.predict(dataset.x_profiling)#.reshape(tmp.shape[0], tmp.shape[1]*tmp.shape[2])
    print(tmp.shape)
    tmp_att = new_model.predict(dataset.x_attack).copy()#.reshape(tmp_att.shape[0], tmp.shape[1]*tmp.shape[2])
    pi_s1, pi_s2, pi_s6, pi_s5, pi_label_tmp = pi_layers_mlp(None, dataset, tmp, tmp_att, layer, np.zeros(16), np.zeros(16), np.zeros(1))
    k = 0
    for j in range(len(byte_list)):
        print(len(pi_s1))
        if byte_list[j] == target_byte:
            pi_hw_s1.append(pi_s1[j])
            pi_hw_s2.append(pi_s2[j])
            pi_s6_l.append(pi_s6[j])
            
        else:
            irrelevant_features.append(pi_s1[j])
            irrelevant_features.append(pi_s2[j])
            irrelevant_features.append(pi_s6[j])
            k = k + 1

    pi_label.append(pi_label_tmp)
    pi_s5_l.append(pi_s5)

np.savez(f"data/pi_ascad_layer_cnn_{layer}.npz", pi_hw_s1=np.array(pi_hw_s1), pi_hw_s2=np.array(pi_hw_s2), pi_label=np.array(pi_label), pi_s6=np.array(pi_s6_l), pi_s5 = pi_s5_l, irrelevant_features=np.array(irrelevant_features))
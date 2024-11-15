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
#layer = int(sys.argv[1])
N_COMP = 5
byte_list = [2, 0, 1, 3]
byte_list = [2, 0, 1, 3, 4, 5,6, 7, 8,9, 10, 11, 12,13,14,15]
pi_s1, pi_s2, pi_label_tmp = np.zeros(16), np.zeros(16), np.zeros(1)
irrelevant_features = []
layers = ["conv_1", "conv_2", "fc_1", "fc_2"]
og = []
easy = []
hard = []
for i in range(1, 100, 1):
    print(i)
    model.load_weights(f"model_checkpoints/cnn_eshard_{i:02d}.weights.h5")
    tmp_att = model.predict(dataset.x_attack).copy()

    easy_list = np.where(hw(dataset.share1_attack[target_byte].astype(np.uint8))==0)[0]
    easy_list = np.append(easy_list, np.where(hw(dataset.share1_attack[target_byte].astype(np.uint8))==1)[0])
    easy_list = np.append(easy_list, np.where(hw(dataset.share1_attack[target_byte].astype(np.uint8))==7)[0])
    easy_list = np.append(easy_list, np.where(hw(dataset.share1_attack[target_byte].astype(np.uint8))==8)[0])

    hard_list = np.where(hw(dataset.share1_attack[target_byte].astype(np.uint8))==2)[0]
    hard_list = np.append(hard_list, np.where(hw(dataset.share1_attack[target_byte].astype(np.uint8))==3)[0])
    hard_list = np.append(hard_list, np.where(hw(dataset.share1_attack[target_byte].astype(np.uint8))==4)[0])
    hard_list = np.append(hard_list, np.where(hw(dataset.share1_attack[target_byte].astype(np.uint8))==5)[0])
    hard_list = np.append(hard_list, np.where(hw(dataset.share1_attack[target_byte].astype(np.uint8))==6)[0])
    #easy_list = np.append(easy_list, np.where(hw(dataset.share1_attack[target_byte].astype(np.uint8))==1)[0])
    #easy_list = np.append(easy_list, np.where(hw(dataset.share2_attack[target_byte].astype(np.uint8))==1)[0])
    #easy_list = np.append(easy_list, np.where(hw(dataset.share2_attack[target_byte].astype(np.uint8))==7)[0])
    # easy_list = np.append(easy_list, np.where(hw(dataset.share2_attack[target_byte].astype(np.uint8))==8)[0])
    # easy_list = np.append(easy_list, np.where(hw(dataset.share2_attack[target_byte].astype(np.uint8))==0)[0])
    og.append(information(tmp_att, dataset.attack_labels, 9))
    easy.append(information(tmp_att[easy_list], dataset.attack_labels[easy_list], 9))
    hard.append(information(tmp_att[hard_list], dataset.attack_labels[hard_list], 9))

np.savez(f"pi_information_cnn.npz", original=np.array(og), easy=easy, hard=hard)
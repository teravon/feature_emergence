from profiling_and_attack import mlp_eshard, information, pi_mlp_probe, cnn_ASCAD

from sklearn.preprocessing import StandardScaler
from utils import *
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA
import sys
from matplotlib.pyplot import plot



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
path = "/base/path"
dataset = load_dataset(dataset_id, path, target_byte, traces_dim, leakage_model=lm, n_prof=n_prof)
pi_og, pi_patch_new, pi_patch_og = [], [], []
dataset.x_profiling, dataset.x_attack = scale_dataset(dataset.x_profiling, dataset.x_attack, StandardScaler())
layer = 1 if len(sys.argv) < 2 else int(sys.argv[1])

# Number of Components varies per layer as the size of X_p^l is not consistent as for MLPs
N_COMP = 40 if layer == 0 else 16 if layer ==1 else 5
layers = ["conv_1", "conv_2", "conv_3", "conv_4", "fc_1", "fc_2"]
input_layers = ["bn_1", "bn_2", "bn_3", "bn_4", "fc_2", "output"]
for i in range(1, 101, 1):
    print(i)
    model.load_weights(f"model_checkpoints/cnn_ascadr_{i:02d}.weights.h5")
    #new_model = Model(model.inputs, [l.output for l in model.layers])
    outputs = [model.get_layer(layers[layer]).output]
    new_model = Model(model.inputs, outputs)
    tmp =  new_model.predict(dataset.x_profiling)
    tmp_att = new_model.predict(dataset.x_attack).copy()
    
    if layers[layer].__contains__("conv"):
        og_1, og_2 = tmp.shape[1], tmp.shape[2]
        tmp = tmp.reshape(tmp.shape[0], og_1*og_2)
        tmp_att = tmp_att.reshape(tmp_att.shape[0], og_1*og_2)

    if layer >= 0:
        pca = PCA(n_components=N_COMP)
        pca.fit(tmp)
        s1_prof = pca.transform(tmp)
        s1_att = pca.transform(tmp_att)
        snrs1 = snr_fast(s1_prof, (dataset.share1_profiling[target_byte].astype(np.uint8)))
        indices_share_1 = np.argsort(snrs1*-1)[:1]
        indices_share_1 = np.where(snrs1 > np.mean(snrs1))[0]
        snrs5 = snr_fast(s1_prof, (dataset.share5_profiling[target_byte].astype(np.uint8)))
        indices_share_5 = np.where(snrs5 > np.mean(snrs5))[0]
        #indices_share_5 = np.argsort(snrs5*-1)[:3]
        print(indices_share_5)
        #print(dataset.share1_profiling[target_byte][indices_0_share_1[0]])
    else: 
        s1_att = dataset.x_attack.copy()
    shuffle_indices = np.random.permutation(tmp_att.shape[0])
    new_model_out = Model(model.get_layer(input_layers[layer]).input, model.outputs)
    indices_0_share_5 = np.where(dataset.share5_profiling[target_byte].astype(np.uint8)==0)
    indices_0_share_1 = np.where(dataset.share1_profiling[target_byte].astype(np.uint8)==0)

    pi_og.append(information(new_model_out.predict(tmp_att.reshape((tmp_att.shape[0], og_1, og_2)) if layers[layer].__contains__("conv") else tmp_att), dataset.attack_labels, 256))
    
    for j in indices_share_5:
        # if j in indices_share_6:
        #     continue
        s1_att[:, j] = np.average(s1_prof[indices_0_share_5, j]) 
    
    pathced_labs = aes_sbox[dataset.share6_attack[target_byte].astype(np.uint8)].astype(np.uint8)
    #pathced_labs = dataset.share2_attack[target_byte].astype(np.uint8)
    if layer >= 0:
        hihi= pca.inverse_transform(s1_att)
        if layers[layer].__contains__("conv"):

            hihi= hihi.reshape((tmp_att.shape[0], og_1, og_2))
        preds = new_model_out.predict(hihi)
    else:
        preds = new_model_out.predict(s1_att)
    
    pi_patch_new.append( information(preds, pathced_labs,256))

    pi_patch_og.append(information(preds, dataset.attack_labels, 256))
    print(f"PI OG = {pi_og[-1]}, PI Patch = {pi_patch_new[-1]}, PI Patch OG = {pi_patch_og[-1]}")

np.savez(f"data/pi_ASCAD_cnn_patch_layer_{layer}_s5_no_ablate.npz",pi_og = np.array(pi_og),pi_patch_new = np.array(pi_patch_new), pi_patch_og = np.array(pi_patch_og))
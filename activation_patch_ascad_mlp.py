from profiling_and_attack import mlp_eshard, information, pi_mlp_probe, mlp

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
model = mlp(256,traces_dim)
n_prof= 50000

lm = "ID"
target_byte = 2
path = "/base/path/Datasets"
dataset = load_dataset(dataset_id, path, target_byte, traces_dim, leakage_model=lm, n_prof=n_prof)
pi_og, pi_patch_new, pi_patch_og = [], [], []
dataset.x_profiling, dataset.x_attack = scale_dataset(dataset.x_profiling, dataset.x_attack, StandardScaler())
layer = 1 if len(sys.argv) < 2 else int(sys.argv[1])
print(layer)
N_COMP = 8
for i in range(1, 101, 1):
    print(i)
    model.load_weights(f"model_checkpoints/mlp_ascadr_{i:02d}.weights.h5")
    new_model = Model(model.inputs, [l.output for l in model.layers])
    tmp =  new_model.predict(dataset.x_profiling)
    tmp_att = new_model.predict(dataset.x_attack).copy()

    if layer > 0:
        pca = PCA(n_components=N_COMP)
        pca.fit(tmp[layer])
        s1_prof = pca.transform(tmp[layer])
        s1_att = pca.transform(tmp_att[layer])
        snrs1 = snr_fast(s1_prof, (dataset.share1_profiling[target_byte].astype(np.uint8)))
        indices_share_1 = np.argsort(snrs1*-1)[:1]
        indices_share_1 = np.where(snrs1 > np.mean(snrs1))[0]
        snrs5 = snr_fast(s1_prof, (dataset.share5_profiling[target_byte].astype(np.uint8)))
        indices_share_5 = np.where(snrs5 > np.mean(snrs5))[0]
        #indices_share_5 = np.argsort(snrs5*-1)[:3]
        snrs6 = snr_fast(s1_prof, (dataset.share6_profiling[target_byte].astype(np.uint8)))
        #indices_share_6 = np.argsort(snrs6*-1)[:3]
        indices_share_6 = np.where(snrs6 > np.mean(snrs6))[0]
        print(indices_share_6)
        snrs2 = snr_fast(s1_prof, (dataset.share2_profiling[target_byte].astype(np.uint8)))
        indices_share_2 = np.argsort(snrs2*-1)[:1]
        indices_share_2 = np.where(snrs2 > np.mean(snrs2))[0]
 
        indices_0_share_2 = np.where(dataset.share2_profiling[target_byte].astype(np.uint8)==0)
    else: 
        s1_att = dataset.x_attack.copy()
    shuffle_indices = np.random.permutation(tmp_att[layer].shape[0])
    new_model_out = Model(model.layers[layer+1].input, model.outputs)
    #print(tmp[layer][shuffle_indices][:, indices_share_1] )
 
    #INdices for patching either r_in share 5 or r_i share 1
    indices_0_share_5 = np.where(dataset.share5_profiling[target_byte].astype(np.uint8)==0)
    indices_0_share_1 = np.where(dataset.share1_profiling[target_byte].astype(np.uint8)==0)
    pi_og.append(information(new_model_out.predict(tmp_att[layer]), dataset.attack_labels, 256))
    #s1_att[:, indices_share_5] = s1_att[shuffle_indices][:, indices_share_5] 

    #Replace share 5 with share 1 to replace patch
    for j in indices_share_5:
        s1_att[:, j] = np.average(s1_prof[indices_0_share_5, j])
    
    pathced_labs = aes_sbox[dataset.share6_attack[target_byte].astype(np.uint8)].astype(np.uint8)
    #pathced_labs = dataset.share2_attack[target_byte].astype(np.uint8)
    if layer > 0:
        preds = new_model_out.predict(pca.inverse_transform(s1_att))
    else:
        preds = new_model_out.predict(s1_att)
    pi_patch_new.append( information(preds, pathced_labs,256))
    pi_patch_og.append(information(preds, dataset.attack_labels, 256))
    
    print(f"PI OG = {pi_og[-1]}, PI Patch = {pi_patch_new[-1]}, PI Patch OG = {pi_patch_og[-1]}")
    
np.savez(f"data/pi_ASCAD_patch_layer_{layer}_s5_no_ablate.npz",pi_og = np.array(pi_og),pi_patch_new = np.array(pi_patch_new), pi_patch_og = np.array(pi_patch_og))
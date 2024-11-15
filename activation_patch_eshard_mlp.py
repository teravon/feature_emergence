from profiling_and_attack import mlp_eshard, information, mlp

from sklearn.preprocessing import StandardScaler
from utils import *
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA

import sys

def hw(input):

    result = np.zeros_like(input)
    for i in range(8):
        result += (input >> i) & 0x1
    return result

dataset_id = 'eshard'
resample_window=80
traces_dim = 1400
model = mlp_eshard(9,traces_dim)
n_prof= 70000

lm = "HW"
target_byte = 2
path = "/base/path/Datasets"
dataset = load_dataset(dataset_id, path, target_byte, traces_dim, leakage_model=lm)

dataset.x_profiling, dataset.x_attack = scale_dataset(dataset.x_profiling[:n_prof], dataset.x_attack, StandardScaler())
layer = int(sys.argv[1])
N_COMP = 5
pi_og, pi_patch_new, pi_patch_og = [], [], []
for i in range(1, 101, 1):
    model.load_weights(f"model_checkpoints/mlp_eshard_{i:02d}.weights.h5")
    new_model = Model(model.inputs, [l.output for l in model.layers])
    tmp =  new_model.predict(dataset.x_profiling)
    tmp_att = new_model.predict(dataset.x_attack).copy()

    # Do the PCA
    pca = PCA(n_components=N_COMP)
    pca.fit(tmp[layer])
    s1_prof = pca.transform(tmp[layer])
    s1_att = pca.transform(tmp_att[layer])

    #Find components for shares
    snrs1 = snr_fast(s1_prof, hw(dataset.share1_profiling[target_byte].astype(np.uint8)))
    #indices_share_1 = np.argsort(snrs1*-1)[:1]
    indices_share_1 = np.where(snrs1 > np.mean(snrs1))[0]

    snrs2 = snr_fast(s1_prof, hw(dataset.share2_profiling[target_byte].astype(np.uint8)))
    #indices_share_2 = np.argsort(snrs2*-1)[:1]
    indices_share_2 = np.where(snrs2 > np.mean(snrs2))[0]

    #Find traces with r_out = 0
    indices_0_share_1 = np.where(dataset.share1_profiling[target_byte].astype(np.uint8)==0)
    
    #Construct new model to continue forward pass
    new_model_out = Model(model.layers[layer+1].input, model.outputs)

    #Original PI calculation
    pi_og.append(information(new_model_out.predict(tmp_att[layer]), dataset.attack_labels, 9))

    #Replace Selected components with average of said components with r_m = 0
    for j in indices_share_1:
        s1_att[:, j ] = np.average(s1_prof[indices_0_share_1][:, j])

    #Invert PCA and finish forward pass with patched activations
    preds = new_model_out.predict(pca.inverse_transform(s1_att))

    #Patched Activations/Patched Label
    pi_patch_new.append(information(preds, hw((dataset.share2_attack[target_byte].astype(np.uint8))),9))
    
    #Patched Activations/Original Label
    pi_patch_og.append(information(preds, dataset.attack_labels, 9))


np.savez(f"data/pi_eshard_mlp_patch_layer_{layer}_no_ablate.npz",pi_og = np.array(pi_og),pi_patch_new = np.array(pi_patch_new), pi_patch_og = np.array(pi_patch_og))
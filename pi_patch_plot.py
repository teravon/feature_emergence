import matplotlib.pyplot as plt
import numpy as np
import os


figure = plt.gcf()
figure.set_size_inches(10, 5)

layer_names_title = ["fc 1", "fc 2", "fc 3"]
layer_names = ["fc 1", "fc 2", "fc 3", "fc 4", "fc 5", "fc 6"]
#layer_names = ["conv 1", "conv 2", "conv 3", "conv 4", "fc 1", "fc 2"]
#layer_names = ["conv 1", "conv 2", "fc 1", "fc 2"]
for l_i, layer_name in enumerate(layer_names):
    pi_s = np.load(f"pi_ASCAD_patch_layer_{l_i+1}_s5_no_ablate.npz", allow_pickle=True)
    pi_s1 = np.load(f"pi_ASCAD_patch_layer_{l_i+1}_s1_no_ablate.npz", allow_pickle=True)
    # plt.subplot(1, len(layer_names), l_i + 1)
    if len(layer_names) % 2 == 0:
        plt.subplot(2, int(len(layer_names)/2), l_i + 1)
    else:
        plt.subplot(1, len(layer_names), l_i + 1)
    # plt.title(layer_names_title[l_i], fontsize=10)
    plt.title(layer_name, fontsize=10)
    plt.plot(pi_s["pi_og"], color="tab:green", label="Original PI/Original Label")
    plt.plot(pi_s["pi_patch_new"], color="tab:orange", label="Patched Activation $m_{2}$/Patched Label")
    plt.plot(pi_s["pi_patch_og"], color="tab:blue", label="Patched Activation $m_{2}$/Original Label")
    plt.plot(pi_s1["pi_patch_new"], color="tab:orange",linestyle=":", label="Patched Activation $m_{2}$/Patched Label")
    plt.plot(pi_s1["pi_patch_og"], color="tab:blue",linestyle=":", label="Patched Activation $m_{2}$/Original Label")
    
    if l_i == 4:
        plt.legend(fontsize=8)
    plt.grid(color="lightgrey")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Perceived Information", fontsize=10)
    plt.xlim([1, 100])
    plt.yscale('log')
    #splt.ylim([-0.5, 3])
plt.tight_layout()
plt.savefig("pi_patching_mlp_no_ablate_ASCAD.png", dpi=1000)
plt.show()


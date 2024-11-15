import matplotlib.pyplot as plt
import numpy as np
import os


figure = plt.gcf()
figure.set_size_inches(10, 5)

layer_names_title = ["fc 1", "fc 2", "fc 3", "fc 4", "fc 5", "fc 6"]
layer_names = ["fc 1", "fc 2", "fc 3", "fc 4" ]
layer_names = ["conv 1", "conv 2", "fc 1", "fc 2"]
for l_i, layer_name in enumerate(layer_names):
    pi_s = np.load(f"pi_eshard_layer_cnn_{l_i}.npz", allow_pickle=True)
    # plt.subplot(1, len(layer_names), l_i + 1)
    if len(layer_names) % 2 == 0:
        plt.subplot(2, int(len(layer_names)/2), l_i + 1)
    else:
        plt.subplot(1, len(layer_names), l_i + 1)
    # plt.title(layer_names_title[l_i], fontsize=10)
    plt.title(layer_name, fontsize=10)
    plt.plot(pi_s["pi_hw_s1"], color="tab:blue", label="$m_{out}$")
    plt.plot(pi_s["pi_hw_s2"], color="tab:orange", label="$SBox[k_2 \oplus d_2] \oplus m_{out}$")
    plt.plot(pi_s["pi_label"], color="tab:green", label="$SBox[k_2 \oplus d_2]$")
    # plt.plot(pi_s["pi_s5"], color="c", label="$m_{in}$")
    # plt.plot(pi_s["pi_s6"], color="m", label="$k_2 \oplus d_2 \oplus m_{in}$")
    
    num_irrevalent_bytes = 3
    for j in range(num_irrevalent_bytes):
        
        plt.plot(pi_s["irrelevant_features"][[_*num_irrevalent_bytes + j for _ in range (99)]], color="tab:grey", alpha=0.4, label=None if j > 0 else "Irrelevant Features")
    
    if l_i == 0:
        plt.legend(fontsize=8)
    plt.grid(color="lightgrey")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Perceived Information", fontsize=10)
    plt.xlim([1, 100])
    if l_i > 1:
        plt.yscale('log')
    plt.yscale('log')
plt.tight_layout()
plt.savefig("pi_eshard_inmasks_cnn.png", dpi=1000)
plt.show()
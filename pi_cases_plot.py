import matplotlib.pyplot as plt
import numpy as np
import os


figure = plt.gcf()
figure.set_size_inches(10, 5)

layer_names_title = ["fc 1", "fc 2", "fc 3"]
layer_names = ["fc 1", "fc 2", "fc 3", "fc 4", "fc 5", "fc 6"]
#layer_names = ["conv 1", "conv 2", "conv 3", "conv 4", "fc 1", "fc 2"]
#layer_names = ["conv 1", "conv 2", "fc 1", "fc 2"]
layer_names = ["MLP","CNN"]
for l_i, layer_name in enumerate(layer_names):
    pi_s = np.load(f"pi_information_{layer_name.lower()}.npz", allow_pickle=True)
    # plt.subplot(1, len(layer_names), l_i + 1)
    if len(layer_names) % 2 == 0:
        plt.subplot(2, int(len(layer_names)/2), l_i + 1)
    else:
        plt.subplot(1, len(layer_names), l_i + 1)
    # plt.title(layer_names_title[l_i], fontsize=10)
    plt.title(layer_name, fontsize=10)
    plt.plot(pi_s["original"], color="tab:green", label="Original")
    plt.plot(pi_s["easy"], color="tab:orange", label="Easy Cases")
    plt.plot(pi_s["hard"], color="tab:blue", label="Hard Cases")
    
    if l_i == 0:
        plt.legend(fontsize=12)
    plt.grid(color="lightgrey")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("Perceived Information", fontsize=10)
    plt.xlim([1, 100])
    plt.yscale('log')
    #splt.ylim([-0.5, 3])
plt.tight_layout()
plt.savefig("pi_easy_hard.png", dpi=1000)
plt.show()


import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.losses import *
from tensorflow.keras import *
import tensorflow.keras.backend as K
import numpy as np
import tensorflow.keras as tk
import os
from pi_callback import *
from tensorflow.keras.utils import *
import gc

def get_optimizer(optimizer, lr):
    if optimizer == "Adam":
        return Adam(lr=lr)
    else:
        return RMSprop(lr=lr)


def cnn(classes, number_of_samples, hp):
    tf.random.set_seed(hp["seed"])

    input_shape = (number_of_samples, 1)
    input_layer = Input(shape=input_shape, name="input_layer")

    if hp["kernel_regularizer"] == "l1":
        regularizer = l1(l=hp["kernel_regularizer_value"])
    elif hp["kernel_regularizer"] == "l2":
        regularizer = l2(l=hp["kernel_regularizer_value"])
    else:
        regularizer = None

    x = None
    x = Conv1D(hp['filters'], hp['kernel_size'], strides=hp['strides'], activation=hp['activation'],
               kernel_initializer=hp['kernel_initializer'], kernel_regularizer=regularizer, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = AveragePooling1D(hp['pool_size'], strides=hp['pool_strides'], padding='same')(x)
    for l_i in range(1, hp["conv_layers"]):
        x = Conv1D(hp['filters'] * (l_i + 1), hp['kernel_size'], strides=hp['strides'], activation=hp['activation'],
                   kernel_initializer=hp['kernel_initializer'], kernel_regularizer=regularizer, padding='same')(x)
        x = BatchNormalization()(x)
        x = AveragePooling1D(hp['pool_size'], strides=hp['pool_strides'], padding='same')(x)
    x = Flatten()(x)

    x = Dense(hp['neurons'], activation=hp['activation'], kernel_initializer=hp['kernel_initializer'],
              kernel_regularizer=regularizer, name=f'fc_1')(x)
    for l_i in range(1, hp["layers"]):
        x = Dense(hp['neurons'], activation=hp['activation'], kernel_initializer=hp['kernel_initializer'],
                  kernel_regularizer=regularizer, name=f'fc_{l_i + 1}')(x)
        if hp["dropout"]:
            x = Dropout(rate=hp["dropout_rate"])(x)

    output_layer = Dense(classes, activation='softmax', name=f'output')(x)

    m_model = Model(input_layer, output_layer, name='cnn_search')
    m_model.compile(loss="categorical_crossentropy", optimizer=get_optimizer(hp["optimizer"], hp["learning_rate"]), metrics=["accuracy"])
    m_model.summary()
    return m_model


def mlp(classes, number_of_samples, hp):
    tf.random.set_seed(hp["seed"])

    input_shape = (number_of_samples)
    input_layer = Input(shape=input_shape, name="input_layer")

    if hp["kernel_regularizer"] == "l1":
        regularizer = l1(l=hp["kernel_regularizer_value"])
    elif hp["kernel_regularizer"] == "l2":
        regularizer = l2(l=hp["kernel_regularizer_value"])
    else:
        regularizer = None

    x = None
    for l_i in range(hp["layers"]):
        x = Dense(hp["neurons"],
                  activation=hp["activation"],
                  kernel_initializer=hp["kernel_initializer"],
                  kernel_regularizer=regularizer,
                  name=f"layer_{l_i}")(
            input_layer if l_i == 0 else x)
        if hp["dropout"]:
            x = Dropout(rate=hp["dropout_rate"])(x)
    output = Dense(classes, activation="softmax", name='output')(x)

    m_model = Model(input_layer, output, name='mlp_search')
    m_model.compile(loss="categorical_crossentropy", optimizer=get_optimizer(hp["optimizer"], hp["learning_rate"]), metrics=["accuracy"])
    m_model.summary()
    return m_model


def mlp_encoded_multiple_softmax(classes, number_of_samples, flatten=False, shape=None, hp_values=None, num_outputs=3):
    tf_random_seed = np.random.randint(1048576)
    tf.random.set_seed(tf_random_seed)

    outputs = []
    if flatten:
        input_shape = (shape)
        input_layer = Input(shape=input_shape)
        x = Flatten()(input_layer)
        for n_out in range(num_outputs):
            outputs.append(Dense(classes, activation='softmax', name=f'output_m{n_out}')(x))
    else:
        input_shape = (number_of_samples)
        input_layer = Input(shape=input_shape)
        for n_out in range(num_outputs):
            outputs.append(Dense(classes, activation='softmax', name=f'output_m{n_out}')(input_layer))

    m_model = Model(input_layer, outputs, name='mlp_softmax')
    optimizer = Adam(lr=0.005)

    m_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    m_model.summary()
    return m_model

def hw(input):

    result = np.zeros_like(input)
    for i in range(8):
        result += (input >> i) & 0x1
    return result

def pi_layers_mlp(args, dataset, layer_activations_profiling, layer_activations_attack, layer_name,
                  pi_epochs_share_1, pi_epochs_share_2, pi_epochs_true_label):
    print(f"Layer: {layer_name}")

    p_outputs = []
    a_outputs = []

    if dataset.name == "ascad-variable":
        byte_list = [2, 4, 5, 11]  # ascadr
    elif dataset.name == "dpa_v42":
        byte_list = [0, 4, 5, 9]  # dpav42
    else:
        byte_list = [2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # aes_sim_mask

    #This for eshard with HW and only r (share 1) and SBox r(share 2)
    for byte in byte_list:
        p_outputs.append(to_categorical(hw(dataset.share2_profiling[byte, :].astype(np.uint8)), num_classes=dataset.classes))
        a_outputs.append(hw(dataset.share2_attack[byte].astype(np.uint8)))
    p_outputs.append(to_categorical(hw(dataset.share1_profiling[dataset.target_byte, :].astype(np.uint8)), num_classes=dataset.classes))
    a_outputs.append(hw(dataset.share1_attack[dataset.target_byte].astype(np.uint8)))
    p_outputs.append(to_categorical(dataset.profiling_labels, num_classes=dataset.classes))
    a_outputs.append(dataset.attack_labels)

    if len(np.shape(layer_activations_profiling)) > 2:
        layer_activations_profiling = layer_activations_profiling.transpose(0, 2, 1)
        layer_activations_profiling = layer_activations_profiling.reshape(layer_activations_profiling.shape[0],
                                                                          layer_activations_profiling.shape[1] *
                                                                          layer_activations_profiling.shape[2])
    if len(np.shape(layer_activations_attack)) > 2:
        layer_activations_attack = layer_activations_attack.transpose(0, 2, 1)
        layer_activations_attack = layer_activations_attack.reshape(layer_activations_attack.shape[0],
                                                                    layer_activations_attack.shape[1] *
                                                                    layer_activations_attack.shape[2])

    batch_size = 400
    number_of_epochs = 20
    callback_pi_encoded = GetPerceivedInformation(layer_activations_attack, a_outputs, dataset.classes, number_of_epochs,
                                                  len(p_outputs))

    model_encoded = mlp_encoded_multiple_softmax_tuning(dataset.classes, layer_activations_profiling.shape[1],
                                                        hp_values={"layers": 1, "neurons": 20, "activation": "elu",
                                                                   "kernel_initializer": "he_uniform", "optimizer": "Adam",
                                                                   "learning_rate": 0.001}, num_outputs=len(p_outputs))
    model_encoded.fit(
        x=layer_activations_profiling,
        y=p_outputs,
        batch_size=batch_size,
        verbose=2,
        epochs=number_of_epochs,
        shuffle=True,
        callbacks=[callback_pi_encoded])
    pi_epochs_share_1, pi_epochs_share_2, pi_epochs_share_6 = [], [],[]
    for byte in range(len(byte_list)):
        #print(callback_pi_encoded.get_pi_multiple())
        #pi_epochs_share_1.append(np.max(callback_pi_encoded.get_pi_multiple()[:, byte * 3]))
        pi_epochs_share_2.append(np.max(callback_pi_encoded.get_pi_multiple()[:, byte]))
        #pi_epochs_share_6.append(np.max(callback_pi_encoded.get_pi_multiple()[:, byte * 3 + 2]))
   
    pi_epochs_share_1= np.max(callback_pi_encoded.get_pi_multiple()[:, -2])
    pi_epochs_true_label = np.max(callback_pi_encoded.get_pi_multiple()[:, -1])
    del callback_pi_encoded
    del model_encoded
    gc.collect()
    return pi_epochs_share_1, pi_epochs_share_2, pi_epochs_true_label

def mlp_encoded_multiple_softmax_tuning(classes, number_of_samples, hp_values=None, num_outputs=3):
    tf_random_seed = np.random.randint(1048576)
    tf.random.set_seed(tf_random_seed)

    outputs = []
    input_shape = (number_of_samples)
    input_layer = Input(shape=input_shape)

    x = None
    for n_layers in range(hp_values["layers"]):
        x = Dense(hp_values["neurons"], hp_values["activation"], kernel_initializer=hp_values["kernel_initializer"])(
            input_layer if n_layers == 0 else x)
    for n_out in range(num_outputs):
        outputs.append(Dense(classes, activation='softmax', name=f'output_m{n_out}')(x))

    m_model = Model(input_layer, outputs, name='mlp_softmax')

    m_model.compile(loss='categorical_crossentropy', optimizer=get_optimizer(hp_values["optimizer"], hp_values["learning_rate"]),
                    metrics=[])
    m_model.summary()
    return m_model

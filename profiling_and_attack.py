import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from scipy.stats import entropy
from tensorflow.keras.regularizers import *
import tensorflow as tf

from utils import *

def cnn(classes, number_of_samples):
    input_shape = (number_of_samples, 1)
    input_layer = Input(shape=input_shape, name="input_layer")


    regularizer = l1(l=0.000075)

    x = None
    kernel_size=10
    stride=5
    activation='selu'
    init='he_uniform'
    x = Conv1D(16, kernel_size, strides=stride, activation=activation,
               kernel_initializer=init, kernel_regularizer=regularizer, padding='same',name=f'conv_1')(input_layer)
    x = BatchNormalization(name="bn_1")(x)
    x = AveragePooling1D(2, strides=2, padding='same')(x)
    for l_i in range(1, 2):
        x = Conv1D(16 * (l_i + 1), kernel_size, strides=stride, activation=activation,
                   kernel_initializer=init, kernel_regularizer=regularizer, padding='same', name=f'conv_{l_i+ 1}')(x)
        x = BatchNormalization(name=f"bn_{l_i+1}")(x)
        x = AveragePooling1D(2, strides=2, padding='same')(x)
    x = Flatten()(x)

    x = Dense(100, activation=activation, kernel_initializer=init,
              kernel_regularizer=regularizer, name=f'fc_1')(x)
    for l_i in range(1, 2):
        x = Dense(100, activation=activation, kernel_initializer=init,
                  kernel_regularizer=regularizer, name=f'fc_{l_i + 1}')(x)
    output_layer = Dense(classes, activation='softmax', name=f'output')(x)

    m_model = Model(input_layer, output_layer, name='cnn_search')
    m_model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0025), metrics=["accuracy"])
    m_model.summary()
    return m_model

def cnn_ASCAD(classes, number_of_samples):
    
    input_shape = (number_of_samples, 1)
    input_layer = Input(shape=input_shape, name="input_layer")

    regularizer = None

    x = None
    kernel_size=40
    stride=15
    activation='selu'
    init='glorot_normal'
    x = Conv1D(12, kernel_size, strides=stride, activation=activation,
               kernel_initializer=init, kernel_regularizer=regularizer, padding='same',name=f'conv_1')(input_layer)
    x = BatchNormalization(name="bn_1")(x)
    x = AveragePooling1D(2, strides=2, padding='same')(x)
    for l_i in range(1, 4):
        x = Conv1D(12 * (l_i + 1), kernel_size, strides=stride, activation=activation,
                   kernel_initializer=init, kernel_regularizer=regularizer, padding='same', name=f'conv_{l_i+ 1}')(x)
        x = BatchNormalization(name=f"bn_{l_i+1}")(x)
        x = AveragePooling1D(2, strides=2, padding='same')(x)
    x = Flatten()(x)

    x = Dense(20, activation=activation, kernel_initializer=init,
              kernel_regularizer=regularizer, name=f'fc_1')(x)
    for l_i in range(1, 2):
        x = Dense(20, activation=activation, kernel_initializer=init,
                  kernel_regularizer=regularizer, name=f'fc_{l_i + 1}')(x)
    output_layer = Dense(classes, activation='softmax', name=f'output')(x)

    m_model = Model(input_layer, output_layer, name='cnn_search')
    m_model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=1e-3), metrics=["accuracy"])
    m_model.summary()
    return m_model


def mlp(classes, number_of_samples, learning_rate=0.0005):
    input_shape = (number_of_samples)
    input_layer = Input(shape=input_shape, name="input_layer")

    x = Dense(100, kernel_initializer="random_uniform", activation="elu")(input_layer)
    x = Dense(100, kernel_initializer="random_uniform", activation="elu")(x)
    x = Dense(100, kernel_initializer="random_uniform", activation="elu")(x)
    x = Dense(100, kernel_initializer="random_uniform", activation="elu")(x)
    x = Dense(100, kernel_initializer="random_uniform", activation="elu")(x)
    x = Dense(100, kernel_initializer="random_uniform", activation="elu")(x)
    output_layer = Dense(classes, activation='softmax', name=f'output')(x)

    m_model = Model(input_layer, output_layer, name='mlp_softmax')
    optimizer = Adam(learning_rate=learning_rate)
    m_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    m_model.summary()
    return m_model

def mlp_eshard(classes, number_of_samples, learning_rate=0.0025):
    input_shape = (number_of_samples)
    input_layer = Input(shape=input_shape, name="input_layer")

    x = Dense(40, kernel_initializer="he_uniform",kernel_regularizer=regularizers.L1(0.000075), activation="relu")(input_layer)
    x = Dense(40, kernel_initializer="he_uniform", activation="relu",kernel_regularizer=regularizers.L1(0.000075))(x)
    x = Dense(40, kernel_initializer="he_uniform", activation="relu",kernel_regularizer=regularizers.L1(0.000075))(x)
    x = Dense(40, kernel_initializer="he_uniform", activation="relu",kernel_regularizer=regularizers.L1(0.000075))(x)
    output_layer = Dense(classes, activation='softmax', name=f'output')(x)

    m_model = Model(input_layer, output_layer, name='mlp_softmax')
    optimizer = Adam(learning_rate=learning_rate)
    m_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    m_model.summary()
    return m_model


def cnn_eshard(classes, number_of_samples, learning_rate=0.0025):
    input_shape = (number_of_samples,1)
    input_layer = Input(shape=input_shape, name="input_layer")

    x = Conv1D(16, 10, strides=5, kernel_initializer="he_uniform",kernel_regularizer=regularizers.L1(0.000075), activation="selu")(input_layer)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, 2)(x)
    x = Conv1D(32, 10, strides=5, kernel_initializer="he_uniform",kernel_regularizer=regularizers.L1(0.000075), activation="selu")(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, 2)(x)
    x = Flatten()(x)
    x = Dense(100, kernel_initializer="he_uniform", activation="selu",kernel_regularizer=regularizers.L1(0.000075))(x)
    x = Dense(100, kernel_initializer="he_uniform", activation="selu",kernel_regularizer=regularizers.L1(0.000075))(x)
    #x = Dense(40, kernel_initializer="he_uniform", activation="relu",kernel_regularizer=regularizers.L1(0.000075))(x)
    output_layer = Dense(classes, activation='softmax', name=f'output')(x)

    m_model = Model(input_layer, output_layer, name='mlp_softmax')
    optimizer = Adam(learning_rate=learning_rate)
    m_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    m_model.summary()
    return m_model

def guessing_entropy(predictions, labels_guess, good_key, key_rank_attack_traces, key_rank_report_interval=1):
    """
    Function to compute Guessing Entropy
    - this function computes a list of key candidates, ordered by their probability of being the correct key
    - if this function returns final_ge=1, it means that the correct key is actually indicated as the most likely one.
    - if this function returns final_ge=256, it means that the correct key is actually indicated as the least likely one.
    - if this function returns final_ge close to 128, it means that the attack is wrong and the model is simply returing a random key.

    :return
    - final_ge: the guessing entropy of the correct key
    - guessing_entropy: a vector indicating the value 'final_ge' with respect to the number of processed attack measurements
    - number_of_measurements_for_ge_1: the number of processed attack measurements necessary to reach final_ge = 1
    """

    nt = len(predictions)

    key_rank_executions = 40

    # key_ranking_sum = np.zeros(key_rank_attack_traces)
    key_ranking_sum = np.zeros(
        int(key_rank_attack_traces / key_rank_report_interval))

    predictions = np.log(predictions + 1e-36)

    probabilities_kg_all_traces = np.zeros((nt, 256))
    for index in range(nt):
        probabilities_kg_all_traces[index] = predictions[index][
            np.asarray([int(leakage[index])
                        for leakage in labels_guess[:]])
        ]

    for run in range(key_rank_executions):
        r = np.random.choice(
            range(nt), key_rank_attack_traces, replace=False)
        probabilities_kg_all_traces_shuffled = probabilities_kg_all_traces[r]
        key_probabilities = np.zeros(256)

        kr_count = 0
        for index in range(key_rank_attack_traces):

            key_probabilities += probabilities_kg_all_traces_shuffled[index]
            key_probabilities_sorted = np.argsort(key_probabilities)[::-1]

            if (index + 1) % key_rank_report_interval == 0:
                key_ranking_good_key = list(
                    key_probabilities_sorted).index(good_key) + 1
                key_ranking_sum[kr_count] += key_ranking_good_key
                kr_count += 1

    guessing_entropy = key_ranking_sum / key_rank_executions

    number_of_measurements_for_ge_1 = key_rank_attack_traces
    if guessing_entropy[int(key_rank_attack_traces / key_rank_report_interval) - 1] < 2:
        for index in range(int(key_rank_attack_traces / key_rank_report_interval) - 1, -1, -1):
            if guessing_entropy[index] > 2:
                number_of_measurements_for_ge_1 = (
                                                          index + 1) * key_rank_report_interval
                break

    final_ge = guessing_entropy[int(
        key_rank_attack_traces / key_rank_report_interval) - 1]
    print("GE = {}".format(final_ge))
    print("Number of traces to reach GE = 1: {}".format(
        number_of_measurements_for_ge_1))

    return final_ge, guessing_entropy, number_of_measurements_for_ge_1


def information(model, labels, num_classes):
    """
    implements I(K;L)

    p_k = the distribution of the sensitive variable K
    data = the samples we 'measured'. It its the n^k_p samples from p(l|k)
    model = the estimated model \hat{p}(l|k).

    returns an estimated of mutual information
    """

    labels = np.array(labels, dtype=np.uint8)
    p_k = np.ones(num_classes, dtype=np.float64)
    for k in range(num_classes):
        p_k[k] = np.count_nonzero(labels == k)
    p_k /= len(labels)

    acc = entropy(p_k, base=2)  # we initialize the value with H(K)

    y_pred = np.array(model + 1e-36)

    for k in range(num_classes):
        trace_index_with_label_k = np.where(labels == k)[0]
        y_pred_k = y_pred[trace_index_with_label_k, k]

        y_pred_k = np.array(y_pred_k)
        if len(y_pred_k) > 0:
            p_k_l = np.sum(np.log2(y_pred_k)) / len(y_pred_k)
            acc += p_k[k] * p_k_l

    #print(f"PI: {acc}")

    return acc

def attack(dataset, generator, features_dim: int, attack_model=None, original_traces=False, batch_size=400, save_path=None):
    
    features_target_attack = np.array(dataset.dataset_target.x_attack)
    features_target_profiling = np.array(dataset.dataset_target.x_profiling)


    #Callback for saving model after each epoch
    pre_fix = save_path if not save_path is None else 'model_checkpoints/cnn_eshard_'
    checkpoint = ModelCheckpoint(pre_fix + '{epoch:02d}.weights.h5', save_weights_only=True, save_best_only=False)
    if attack_model is None:
        #Default model
        model = mlp(dataset.classes, features_target_profiling.shape[0])
    else:
        model = attack_model
    model.fit(
        x=features_target_profiling,
        y=to_categorical(dataset.dataset_target.profiling_labels, num_classes=dataset.dataset_target.classes),
        batch_size=batch_size,
        verbose=2,
        epochs=100,
        shuffle=True,
        validation_data=(
            features_target_attack, to_categorical(dataset.dataset_target.attack_labels, num_classes=dataset.dataset_target.classes)),
        callbacks=[checkpoint])

    """ Predict the trained model with target/attack measurements """
    predictions = model.predict(features_target_attack)
    """ Check if we are able to recover the key from the target/attack measurements """
    ge, ge_vector, nt = guessing_entropy(predictions, dataset.dataset_target.labels_key_hypothesis_attack,
                                         dataset.dataset_target.correct_key_attack, 4000)
    pi = information(predictions, dataset.dataset_target.attack_labels, dataset.dataset_target.classes)
    return ge, nt, pi, ge_vector



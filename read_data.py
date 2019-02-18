import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

APN_PRESENT = '1011101110000000000000000000011101100000000000000010000000000'
APN_ABSENT  = '1011101110000000000000000000011101100000000000010000000000000'

SEQUENCE_SIZE = 6000


def load_ecg(filename:str):
    ecg = np.fromfile(filename, dtype=np.int16)
    ecg = ecg[:(ecg.size - ecg.size%SEQUENCE_SIZE)]
    ecg = np.reshape(ecg, (-1, SEQUENCE_SIZE))
    return ecg


def load_apn(filename:str):
    apns = np.fromfile(filename, dtype=np.uint64)
    apns = (apns & int('10000000000000', 2) > 0).astype(np.int8)
    # for i in range(1,len(apns)):
        # if apns[i] == 1:
            # apns[i-1] = 1
    return apns


def load_data():
    all_ecgs = []
    all_apns = []
    ecg_paths = sorted(glob.glob(r"./data/[abc][0-9][0-9].dat"))
    apn_paths = sorted(glob.glob(r"./data/[abc][0-9][0-9].apn"))
    for (ecg_path, apn_path) in zip(ecg_paths, apn_paths):
        ecgs = load_ecg(ecg_path)
        apns = load_apn(apn_path)
        max_len = min(ecgs.shape[0], apns.shape[0])
        ecgs = ecgs[:max_len]
        apns = apns[:max_len]
        all_ecgs.append(ecgs)
        all_apns.append(apns)

    return np.concatenate(all_ecgs), np.concatenate(all_apns)

data, labels = load_data()
p = np.random.permutation(len(labels))
data = data[p]
labels = labels[p]
data_length = data.shape[0]
split_index = int(data_length*0.75)
print(split_index)
x_train, y_train = np.expand_dims(data[:split_index], axis=2), labels[:split_index]
x_test, y_test = np.expand_dims(data[split_index:], axis=2), labels[split_index:]
print('Train: ', x_train.shape, y_train.shape)
print('Test : ', x_test.shape, y_test.shape)
data = np.expand_dims(data, axis=2)
print(data.shape, data.shape[-1], labels.shape)

model = tf.keras.Sequential()
model.add(layers.Conv1D(filters=16, kernel_size=7, activation='relu', input_shape=(SEQUENCE_SIZE, 1), kernel_initializer='glorot_uniform'))
model.add(layers.MaxPool1D(2))
model.add(layers.Conv1D(filters=32, kernel_size=5, activation='relu', kernel_initializer='glorot_uniform'))
model.add(layers.MaxPool1D(2))
model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu', kernel_initializer='glorot_uniform'))
model.add(layers.MaxPool1D(2))
model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
model.add(layers.MaxPool1D(2))
model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
model.add(layers.MaxPool1D(2))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu', kernel_regularizer=l2(0.1), kernel_initializer='glorot_uniform'))
model.add(layers.Dense(128, activation='relu', kernel_regularizer=l2(0.1), kernel_initializer='glorot_uniform'))
model.add(layers.Dense(1, kernel_regularizer=l2(0.01), activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Nadam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(data, labels, epochs=50, batch_size=128, validation_split=0.25)

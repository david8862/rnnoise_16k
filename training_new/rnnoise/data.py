#!/usr/bin/env python3
# -*- coding=utf-8 -*-
"""
Train RNNoise model
"""
import numpy as np
import h5py


def get_train_data(train_data_file, window_size=2000):
    # load h5 dataset file
    with h5py.File(train_data_file, 'r') as hf:
        all_data = hf['data'][:]

    nb_sequences = len(all_data) // window_size

    # split & reshape input data
    x_train = all_data[:nb_sequences*window_size, :38]
    x_train = np.reshape(x_train, (nb_sequences, window_size, 38))

    # split & reshape denoise output data
    y_train = np.copy(all_data[:nb_sequences*window_size, 38:56])
    y_train = np.reshape(y_train, (nb_sequences, window_size, 18))

    # split & reshape noise output data (abandoned)
    #noise_train = np.copy(all_data[:nb_sequences*window_size, 56:74])
    #noise_train = np.reshape(noise_train, (nb_sequences, window_size, 18))

    # split & reshape vad output data
    vad_train = np.copy(all_data[:nb_sequences*window_size, 74:75])
    vad_train = np.reshape(vad_train, (nb_sequences, window_size, 1))

    return x_train, y_train, vad_train


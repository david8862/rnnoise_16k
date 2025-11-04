#!/usr/bin/env python3
# -*- coding=utf-8 -*-
"""
Train RNNoise model
"""
import numpy as np
import h5py

def get_train_data(train_data_file, bands_num=18, delta_ceps_num=6, sequence_length=2000):
    # load h5 dataset file
    with h5py.File(train_data_file, 'r') as hf:
        all_data = hf['data'][:]

    nb_sequences = len(all_data) // sequence_length

    input_feature_num = bands_num + 3*delta_ceps_num + 2
    data_dim = input_feature_num + bands_num*2 + 1

    # check train data file dimension
    assert (all_data.shape[-1] == data_dim), 'train data dimension does not match with model config'

    # split & reshape input data
    x_train = all_data[:nb_sequences*sequence_length, :input_feature_num]
    x_train = np.reshape(x_train, (nb_sequences, sequence_length, input_feature_num))

    # split & reshape denoise output data
    y_train = np.copy(all_data[:nb_sequences*sequence_length, input_feature_num:input_feature_num+bands_num])
    y_train = np.reshape(y_train, (nb_sequences, sequence_length, bands_num))

    # split & reshape noise output data (abandoned)
    #noise_train = np.copy(all_data[:nb_sequences*sequence_length, input_feature_num+bands_num:input_feature_num+2*bands_num])
    #noise_train = np.reshape(noise_train, (nb_sequences, sequence_length, 18))

    # split & reshape vad output data
    vad_train = np.copy(all_data[:nb_sequences*sequence_length, -1:])
    vad_train = np.reshape(vad_train, (nb_sequences, sequence_length, 1))

    return x_train, y_train, vad_train


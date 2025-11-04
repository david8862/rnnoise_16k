#!/usr/bin/env python3
# -*- coding=utf-8 -*-
"""
Train RNNoise model
"""
import numpy as np
import torch


def mask(g):
    return torch.clamp(g+1, max=1)


class RNNoiseDataset(torch.utils.data.Dataset):
    def __init__(self,
                feature_file,
                bands_num=18,
                delta_ceps_num=6,
                sequence_length=2000):

        self.sequence_length = sequence_length
        self.data = np.memmap(feature_file, dtype='float32', mode='r')

        self.bands_num = bands_num
        self.input_feature_dim = bands_num + 3*delta_ceps_num + 2
        data_dim = self.input_feature_dim + self.bands_num*2 + 1

        self.nb_sequences = self.data.shape[0]//self.sequence_length//data_dim
        self.data = self.data[:self.nb_sequences*self.sequence_length*data_dim]

        self.data = np.reshape(self.data, (self.nb_sequences, self.sequence_length, data_dim))


    def __len__(self):
        return self.nb_sequences

    def __getitem__(self, index):
        return self.data[index, :, :self.input_feature_dim].copy(), self.data[index, :, self.input_feature_dim:self.input_feature_dim+self.bands_num].copy(), self.data[index, :, -1:].copy()



def get_dataloader(feature_file, bands_num, delta_ceps_num, sequence_length, batch_size, use_cuda):
    # prepare dataset loader
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset = RNNoiseDataset(feature_file, bands_num=bands_num, delta_ceps_num=delta_ceps_num, sequence_length=sequence_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    return dataloader


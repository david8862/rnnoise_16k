#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.keras.backend as K


def groundtruth_mask(y_true):
    return K.minimum(y_true+1., 1.)

def msse(y_true, y_pred):
    return K.mean(groundtruth_mask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)

def denoise_loss(y_true, y_pred):
    return K.mean(groundtruth_mask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01*K.binary_crossentropy(y_pred, y_true)), axis=-1)

def vad_loss(y_true, y_pred):
    return K.mean(2*K.abs(y_true - 0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)

def my_accuracy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)


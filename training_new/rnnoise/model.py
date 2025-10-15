#!/usr/bin/env python3
# -*- coding=utf-8 -*-
"""
Train RNNoise model
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU#, LSTM, SimpleRNN, Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import Constraint#, min_max_norm
from tensorflow.keras import backend as K


class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        #return {'name': self.__class__.__name__,
        #    'c': self.c}
        return {'c': self.c}



def get_model(batch_size=None, weights_path=None):
    reg = 0.000001
    constraint = WeightClip(0.499)

    input_tensor = Input(shape=(None, 38), batch_size=batch_size, name='feature_input')
    input_dense = Dense(24, activation='tanh', name='input_dense', kernel_constraint=constraint, bias_constraint=constraint)(input_tensor)

    vad_gru = GRU(24, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='vad_gru',
                  kernel_regularizer=regularizers.l2(reg),
                  recurrent_regularizer=regularizers.l2(reg),
                  kernel_constraint=constraint,
                  recurrent_constraint=constraint,
                  bias_constraint=constraint,
                  reset_after=False)(input_dense)
    vad_output = Dense(1, activation='sigmoid', name='vad_output',
                       kernel_constraint=constraint,
                       bias_constraint=constraint)(vad_gru)

    noise_input = concatenate([input_dense, vad_gru, input_tensor])
    noise_gru = GRU(48, activation='relu', recurrent_activation='sigmoid', return_sequences=True, name='noise_gru',
                    kernel_regularizer=regularizers.l2(reg),
                    recurrent_regularizer=regularizers.l2(reg),
                    kernel_constraint=constraint,
                    recurrent_constraint=constraint,
                    bias_constraint=constraint,
                    reset_after=False)(noise_input)

    denoise_input = concatenate([vad_gru, noise_gru, input_tensor])
    denoise_gru = GRU(96, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='denoise_gru',
                      kernel_regularizer=regularizers.l2(reg),
                      recurrent_regularizer=regularizers.l2(reg),
                      kernel_constraint=constraint,
                      recurrent_constraint=constraint,
                      bias_constraint=constraint,
                      reset_after=False)(denoise_input)
    denoise_output = Dense(18, activation='sigmoid', name='denoise_output',
                           kernel_constraint=constraint,
                           bias_constraint=constraint)(denoise_gru)

    model = Model(inputs=input_tensor, outputs=[denoise_output, vad_output])

    if weights_path:
        model.load_weights(weights_path, by_name=False)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    return model


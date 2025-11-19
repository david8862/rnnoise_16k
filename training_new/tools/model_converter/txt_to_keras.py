#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to fulfill RNNoise/RNNVad model weights from txt model weights file, e.g. weights.txt.
The txt model weights file is dumped from "rnn_data.o" with "tools/rnnoise_weight_to_txt.c"
"""
import os, sys, argparse
from tqdm import tqdm
import numpy as np
import re

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from rnnoise.model import get_model


def txt_to_keras(txt_file, bands_num, delta_ceps_num, keras_model_file):
    # get rnnoise model with random weights
    model = get_model(bands_num=bands_num, delta_ceps_num=delta_ceps_num)
    #model.summary()

    # load model weights from txt file
    with open(txt_file, 'r', encoding='utf-8') as f:
        txt_data = f.readlines()

    # txt model weights file would be like:
    # $ cat weights.txt
    # input_dense_weights 912
    #  -6, 2, -10, -10, 8, 6, -15, -10, -12, ...
    #
    # input_dense_bias 24
    #  -28, -106, -45, 11, 84, 74, -81, -46, 116, ...
    #
    # vad_gru_weights 1728
    #  66, 77, -55, 120, 121, 15, 30, 127, 82, ...
    #
    # vad_gru_recurrent_weights 1728
    #  127, 75, -9, 90, 127, 33, 45, 13, 38, ...
    #
    # vad_gru_bias 72
    #  45, -91, -35, -6, 82, -109, 114, -58, -76, ...
    #
    # noise_gru_weights 12384
    #  2, -79, 72, 125, 19, 89, 21, 26, -3, ...
    #
    # noise_gru_recurrent_weights 6912
    #  15, -6, 8, 12, 24, 3, -49, -39, -60, ...
    #
    # noise_gru_bias 144
    #  45, -22, 42, 11, -77, 52, 1, 30, 31, ...
    #
    # denoise_gru_weights 31680
    #  -18, -13, 30, 10, -6, 2, -51, -16, -76, ...
    #
    # denoise_gru_recurrent_weights 27648
    #  40, -49, 32, -48, -9, 20, 73, -27, -121, ...
    #
    # denoise_gru_bias 288
    #  61, 47, 40, -6, 46, -13, 32, -98, -30, ...
    #
    # denoise_output_weights 1728
    #  -8, 21, 84, 43, -14, -3, -31, -12, -28, ...
    #
    # denoise_output_bias 18
    #  2, -7, -5, 9, -21, -39, -25, -11, -13, ...
    #
    # vad_output_weights 24
    #  127, -128, 127, 127, -128, 127, -122, -128, 127, ...
    #
    # vad_output_bias 1
    #  -93,
    #
    pbar = tqdm(total=len(model.layers), desc='txt to keras model')
    for i, layer in enumerate(model.layers):
        # only need to handle layer with weights
        if len(layer.get_weights()) > 0:
            weights = layer.get_weights()
            update_count = 0

            # search txt data for the layer weights
            for j, txt_line in enumerate(txt_data):
                if len(txt_line) > 0 and txt_line.startswith(layer.name):
                    # found weights name in txt file which matches model layer
                    weights_name, weights_num = txt_line.split(' ')
                    weights_num = int(weights_num)

                    # the next line (j+1) would be weights data. parse it to int list
                    weights_data = [int(weight.strip()) for weight in txt_data[j+1].strip().split(',') if len(weight) > 0]
                    # check if weights number matches with config
                    assert (len(weights_data) == weights_num), 'weights number of %s does not match with config' % (weights_name)

                    # convert int weights list to float32 weights array, here
                    # we do "weights_array /= 256.0" to align with the weights
                    # dump action in "keras_to_c.py"
                    weights_array = np.asarray(weights_data, dtype=np.float32)
                    weights_array /= 256.0

                    if len(weights) == 3:
                        # GRU layer has 3 weights array (weights/recurrent_weights/bias)
                        if weights_name.endswith('gru_weights'):
                            weights_array = weights_array.reshape(weights[0].shape)
                            weights[0] = weights_array
                        elif weights_name.endswith('gru_recurrent_weights'):
                            weights_array = weights_array.reshape(weights[1].shape)
                            weights[1] = weights_array
                        elif weights_name.endswith('gru_bias'):
                            weights_array = weights_array.reshape(weights[2].shape)
                            weights[2] = weights_array
                        else:
                            raise ValueError('Unsupported gru weights name:', weights_name)
                    elif len(weights) == 2:
                        # Dense layer has 2 weights array (weights/bias)
                        if weights_name.endswith('_weights'):
                            weights_array = weights_array.reshape(weights[0].shape)
                            weights[0] = weights_array
                        elif weights_name.endswith('_bias'):
                            weights_array = weights_array.reshape(weights[1].shape)
                            weights[1] = weights_array
                        else:
                            raise ValueError('Unsupported dense weights name:', weights_name)
                    else:
                        raise ValueError('Unsupported weights array number {} for layer {}'.format(len(weights), layer.name))
                    update_count += 1

            # double check if all the layer weights have been updated
            assert (update_count == len(weights)), 'some of the weights does not update for layer %s' % (layer.name)
            # update model weights after going through all the txt weights data
            model.layers[i].set_weights(weights)
        pbar.update(1)
    pbar.close()
    # save the updated model weights to .h5 model file
    model.save(keras_model_file)




def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Fulfill RNNoise/RNNVad model weights from txt model weights file')
    parser.add_argument('--txt_file', required=False, type=str, default='weights.txt',
            help='input txt weights file, default=%(default)s')
    parser.add_argument('--bands_num', type=int, required=False, default=18,
            help="number of bands, default=%(default)s")
    parser.add_argument('--delta_ceps_num', type=int, required=False, default=6,
            help="number of delta ceps, default=%(default)s")
    parser.add_argument('--keras_model_file', type=str, required=False, default='pretrained_weights.h5',
            help='output .h5 keras model file, default=%(default)s')

    args = parser.parse_args()

    txt_to_keras(args.txt_file, args.bands_num, args.delta_ceps_num, args.keras_model_file)

    print('\nDone. model weights has been saved to pre-trained model file {}'.format(args.keras_model_file))


if __name__ == '__main__':
    main()

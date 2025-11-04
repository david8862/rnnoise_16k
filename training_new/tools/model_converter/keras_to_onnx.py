#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert RNNoise keras model to ONNX model
"""
import os, sys, argparse
import shutil
import subprocess
import tensorflow as tf
from tensorflow.keras.models import load_model
import onnx

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_custom_objects

os.environ['TF_KERAS'] = '1'


def onnx_convert(keras_model_file, output_file, op_set, batch_size, sequence_length):
    import tf2onnx
    custom_object_dict = get_custom_objects()
    model = load_model(keras_model_file, compile=False, custom_objects=custom_object_dict)

    # assume only 1 input tensor for feature vector
    assert len(model.inputs) == 1, 'invalid input tensor number.'

    # assign batch size if not specified in keras model
    input_shape = list(model.inputs[0].shape)
    if input_shape[0] == 0 or input_shape[0] is None:
       input_shape[0] = batch_size

    # assign sequence length if not specified in keras model
    if input_shape[1] == 0 or input_shape[1] is None:
       input_shape[1] = sequence_length

    spec = (tf.TensorSpec(shape=input_shape, dtype=tf.float32, name="feature_input"),)

    # Reference:
    # https://github.com/onnx/tensorflow-onnx#python-api-reference
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, custom_ops=None, opset=op_set, output_path=output_file)


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='Convert RNNoise tf.keras model to ONNX model')
    parser.add_argument('--keras_model_file', required=True, type=str, help='path to keras model file')
    parser.add_argument('--output_file', required=True, type=str, help='output onnx model file')
    parser.add_argument('--op_set', required=False, type=int, help='onnx op set, default=%(default)s', default=14)
    parser.add_argument('--batch_size', required=False, type=int, help='assign batch size if not specified in keras model, default=%(default)s', default=1)
    parser.add_argument('--sequence_length', required=False, type=int, help='assign sequence length if not specified in keras model, default=%(default)s', default=2000)

    args = parser.parse_args()

    onnx_convert(args.keras_model_file, args.output_file, args.op_set, args.batch_size, args.sequence_length)

    print('\nDone. ONNX model has been saved to', args.output_file)


if __name__ == '__main__':
    main()


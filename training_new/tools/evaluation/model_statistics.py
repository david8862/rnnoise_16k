#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to calculate FLOPs & PARAMs of a tf.keras model.
Compatible with TF 1.x and TF 2.x
"""
import os, sys, argparse
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_custom_objects, optimize_tf_gpu

# check tf version to be compatible with TF 2.x
if tf.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

optimize_tf_gpu(tf, K)


def clever_format(value, format_string="%.2f"):
    """
    Convert statistic value to clever format string
    """
    if value <= 0:
        raise ValueError('invalid statistic value {}'.format(value))

    # get friendly statistic value string
    if value > 0 and value <= 1e3:
        value_string = format_string%(value)
    elif value > 1e3 and value <= 1e6:
        value_string = format_string%(value/1e3)+'K'
    elif value > 1e6 and value <= 1e9:
        value_string = format_string%(value/1e6)+'M'
    elif value > 1e9 and value <= 1e12:
        # here we can use either "GFLOPS" or "BFLOPS"
        value_string = format_string%(value/1e9)+'G'
    elif value > 1e12 and value <= 1e15:
        value_string = format_string%(value/1e12)+'T'
    elif value > 1e15 and value <= 1e18:
        value_string = format_string%(value/1e15)+'P'
    elif value > 1e18:
        value_string = format_string%(value/1e18)+'E'

    return value_string


def get_flops(model):
    run_meta = tf.RunMetadata()
    graph = tf.get_default_graph()

    # We use the Keras session graph in the call to the profiler.
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    params = tf.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)

    flops_value = flops.total_float_ops
    param_value = params.total_parameters

    # get friendly FLOPs & PARAMs value string
    flops_result_string = clever_format(flops_value, '%.4f')
    param_result_string = clever_format(param_value, '%.4f')

    print('Total FLOPs: {} float_ops'.format(flops_result_string))
    print('Total PARAMs: {}'.format(param_result_string))


def main():
    parser = argparse.ArgumentParser(description='tf.keras model FLOPs & PARAMs checking tool')
    parser.add_argument('--model_path', type=str, required=True, help='model file to evaluate')
    parser.add_argument('--batch_size', required=False, type=int, help='assign batch size if not specified in keras model, default=%(default)s', default=1)
    parser.add_argument('--sequence_length', required=False, type=int, help='assign sequence length if not specified in keras model, default=%(default)s', default=2000)
    args = parser.parse_args()

    custom_object_dict = get_custom_objects()
    model = load_model(args.model_path, compile=False, custom_objects=custom_object_dict)

    batch_size, sequence_length, feature_size = model.input.shape.as_list()

    # assign batch size if not specified in keras model
    if batch_size == 0 or batch_size is None:
       batch_size = args.batch_size

    # assign sequence length if not specified in keras model
    if sequence_length or sequence_length is None:
       sequence_length = args.sequence_length

    input_tensor = Input(shape=(sequence_length, feature_size), batch_size=1)
    output_tensor = model(input_tensor)
    model = Model(input_tensor, output_tensor)

    K.set_learning_phase(0)
    get_flops(model)


if __name__ == '__main__':
    main()

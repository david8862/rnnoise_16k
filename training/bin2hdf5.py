#!/usr/bin/env python3
"""
Tool to convert audio feature matrix from bin file generate by 'denoise_train_data_creator'
to python h5 format, for further model training
"""
import os, sys, argparse
from tqdm import tqdm
import numpy as np
import h5py


def bin2hdf5(input_bin_file, matrix_shape, output_h5_file):
    feature_data = np.fromfile(input_bin_file, dtype='float32')
    feature_data = np.reshape(feature_data, matrix_shape)

    h5f = h5py.File(output_h5_file, 'w')
    h5f.create_dataset('data', data=feature_data)


def main():
    parser = argparse.ArgumentParser(description='tool to convert audio feature matrix from bin file to h5')
    parser.add_argument('--input_bin_file', type=str, required=True,
                        help='input audio feature matrix bin file, default=%(default)s')
    parser.add_argument('--matrix_shape', type=str, required=False, default='10000x75',
                        help="feature matrix shape as <feature_number>x<feature_length>, default=%(default)s")
    parser.add_argument('--output_h5_file', type=str, required=False, default='train_data.h5',
                        help='output h5 format feature matrix file. default=%(default)s')

    args = parser.parse_args()
    feature_number, feature_length = args.matrix_shape.split('x')
    args.matrix_shape = (int(feature_number), int(feature_length))

    bin2hdf5(args.input_bin_file, args.matrix_shape, args.output_h5_file)
    print('\nDone. Audio feature matrix has been saved to', args.output_h5_file)

if __name__ == "__main__":
    main()

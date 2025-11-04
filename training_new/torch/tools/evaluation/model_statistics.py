#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to calculate MACs & PARAMs of a PyTorch model.
"""
import os, sys, argparse
import torch
from thop import profile, profile_origin, clever_format

# add root path of model definition here,
# to make sure that we can load .pth model file with torch.load()
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
#from common.utils import get_custom_ops

def get_macs(model, input_tensor):
    #custom_ops_dict = get_custom_ops()
    custom_ops_dict = None

    macs, params = profile(model, inputs=(input_tensor, ), custom_ops=custom_ops_dict, verbose=True)
    macs, params = clever_format([macs, params], "%.3f")

    print('Total MACs: {}'.format(macs))
    print('Total PARAMs: {}'.format(params))


def main():
    parser = argparse.ArgumentParser(description='PyTorch model MACs & PARAMs checking tool')
    parser.add_argument('--model_path', type=str, required=True, help='model file to evaluate')
    parser.add_argument('--bands_num', type=int, required=False, default=18, help="number of bands, default=%(default)s")
    parser.add_argument('--delta_ceps_num', type=int, required=False, default=6, help="number of delta ceps, default=%(default)s")
    parser.add_argument('--sequence_length', type=int, required=False, default=2000, help="input sequence length, default=%(default)s")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.model_path, map_location=device, weights_only=False)

    input_feature_dim = args.bands_num + 3*args.delta_ceps_num + 2
    input_tensor = torch.randn(1, args.sequence_length, input_feature_dim).to(device)

    get_macs(model, input_tensor)


if __name__ == '__main__':
    main()

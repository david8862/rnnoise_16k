#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import os, sys, argparse
import torch
#from torchsummary import summary

# add root path of model definition here,
# to make sure that we can load .pth model file with torch.load()
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))


def model_export(model_path, bands_num, delta_ceps_num, sequence_length, output_path, batch_size):
    # Input
    input_feature_dim = bands_num + 3*delta_ceps_num + 2

    if batch_size == -1:
        input_tensor = torch.zeros(1, sequence_length, input_feature_dim)

    else:
        input_tensor = torch.zeros(batch_size, sequence_length, input_feature_dim)

    # Load PyTorch model
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False).float()
    model.eval()
    y = model(input_tensor)  # dry run

    # Strip model file name
    model_basename = os.path.basename(model_path).split('.')
    model_basename = '.'.join(model_basename[:-1])
    os.makedirs(output_path, exist_ok=True)

    # TorchScript export
    try:
        print('\nStarting TorchScript export with torch %s...' % torch.__version__)
        export_file = os.path.join(output_path, model_basename+'.torchscript.pt')

        ts = torch.jit.trace(model, input_tensor)
        ts.save(export_file)
        print('TorchScript export success, saved as %s' % export_file)
    except Exception as e:
        print('TorchScript export failure: %s' % e)

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        export_file = os.path.join(output_path, model_basename+'.onnx')

        if batch_size == -1:
            # dump dynamic batch-size onnx model
            torch.onnx.export(model, input_tensor, export_file, verbose=False, opset_version=12, input_names=['feature_input'], output_names=['denoise_output', 'vad_output', 'gru1_state', 'gru2_state', 'gru3_state'],
                              dynamic_axes={"feature_input": {0: "batch_size"},
                                            "denoise_output": {0: "batch_size"},
                                            "vad_output": {0: "batch_size"},
                                            "gru1_state": {0: "batch_size"},
                                            "gru2_state": {0: "batch_size"},
                                            "gru3_state": {0: "batch_size"},
                                           })

        else:
            # dump fix batch-size onnx model
            torch.onnx.export(model, input_tensor, export_file, verbose=False, opset_version=12, input_names=['feature_input'], output_names=['denoise_output', 'vad_output', 'gru1_state', 'gru2_state', 'gru3_state'])

        # Checks
        onnx_model = onnx.load(export_file)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % export_file)
    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='model file to export')
    parser.add_argument('--bands_num', type=int, required=False, default=18, help="number of bands. default=%(default)s")
    parser.add_argument('--delta_ceps_num', type=int, required=False, default=6, help="number of delta ceps. default=%(default)s")
    parser.add_argument('--sequence_length', type=int, required=False, default=2000, help="input sequence length. default=%(default)s")
    parser.add_argument('--batch_size', type=int, required=False, help="batch size for inference, -1 for dynamic batch. default=%(default)s", default=-1)
    parser.add_argument('--output_path', type=str, required=True, help='output path for exported model')

    args = parser.parse_args()

    model_export(args.model_path, args.bands_num, args.delta_ceps_num, args.sequence_length, args.output_path, args.batch_size)


if __name__ == "__main__":
    main()




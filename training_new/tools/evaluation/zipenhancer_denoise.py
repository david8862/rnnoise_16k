#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo of ZipEnhancer Speech Denoise model

Reference from:
https://www.modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base
https://modelscope.cn/models/manyeyes/ZipEnhancer-se-16k-base-onnx/summary
https://arxiv.org/abs/2501.05183
https://mp.weixin.qq.com/s/61o2a8lewpIMMkQhuJQvSA
"""
import os, sys, argparse
import glob
from tqdm import tqdm
import numpy as np
import soundfile as sf

import torch
import onnxruntime


# from modelscope.models.audio.ans.zipenhancer
def mag_pha_stft(y,
                 n_fft,
                 hop_size,
                 win_size,
                 compress_factor=1.0,
                 center=True):
    hann_window = torch.hann_window(win_size, device=y.device)
    stft_spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode='reflect',
        normalized=False,
        return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1) + (1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1], stft_spec[:, :, :, 0] + (1e-5))
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)

    return mag, pha, com


# from modelscope.models.audio.ans.zipenhancer
def mag_pha_istft(mag,
                  pha,
                  n_fft,
                  hop_size,
                  win_size,
                  compress_factor=1.0,
                  center=True):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0 / compress_factor))
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann_window = torch.hann_window(win_size, device=com.device)

    wav = torch.istft(
        com,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center)
    return wav


# from modelscope.utils.audio.audio_utils
def audio_norm(x):
    rms = (x**2).mean()**0.5
    scalar = 10**(-25 / 20) / rms
    x = x * scalar
    pow_x = x**2
    avg_pow_x = pow_x.mean()
    rmsx = pow_x[pow_x > avg_pow_x].mean()**0.5
    scalarx = 10**(-25 / 20) / rmsx
    x = x * scalarx
    return x



class OnnxModel:
    def __init__(self, onnx_filepath, providers=None):
        self.onnx_model = onnxruntime.InferenceSession(onnx_filepath, providers=['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CPUExecutionProvider'], provider_options=None)

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def __call__(self, noisy_wav):
        n_fft = 400
        hop_size = 100
        win_size = 400

        norm_factor = torch.sqrt(noisy_wav.shape[1] / torch.sum(noisy_wav ** 2.0))
        #if is_verbose:
        #    print(f"norm_factor {norm_factor}" )

        noisy_audio = (noisy_wav * norm_factor)

        noisy_amp, noisy_pha, _ = mag_pha_stft(
            noisy_audio,
            n_fft,
            hop_size,
            win_size,
            compress_factor=0.3,
            center=True)

        ort_inputs = {self.onnx_model.get_inputs()[0].name: self.to_numpy(noisy_amp),
                    self.onnx_model.get_inputs()[1].name: self.to_numpy(noisy_pha),
                    }
        ort_outs = self.onnx_model.run(None, ort_inputs)

        amp_g = torch.from_numpy(ort_outs[0])
        pha_g = torch.from_numpy(ort_outs[1])

        #if is_verbose:
        #    print(f"Enhanced amplitude mean and std: {torch.mean(amp_g)} {torch.std(amp_g)}")
        #    print(f"Enhanced phase mean and std: {torch.mean(pha_g)} {torch.std(pha_g)}")

        wav = mag_pha_istft(
            amp_g,
            pha_g,
            n_fft,
            hop_size,
            win_size,
            compress_factor=0.3,
            center=True)

        wav = wav / norm_factor

        wav = self.to_numpy(wav)

        return wav


def zipenhancer_denoise_onnx(audio_path, output_path, onnx_model_path='zipenhancer_model.onnx'):

    os.makedirs(output_path, exist_ok=True)

    # get audio file list or single audio file
    if os.path.isfile(audio_path):
        audio_list = [audio_path]
    else:
        audio_list = glob.glob(os.path.join(audio_path, '*.wav'))
        audio_list.sort()

    pbar = tqdm(total=len(audio_list), desc='ZipEnhancer Speech Denoise')
    for audio_file in audio_list:
        # due to the onnx inference cost plenty of memory (>20GB, not sure why...), we have to create
        # zipenhancer onnx denoise model object for each audio file and delete it after inference, otherwise
        # the script will crash by OOM
        onnx_model = OnnxModel(onnx_model_path)

        # load & normalize audio file
        wav, fs = sf.read(audio_file)
        wav = audio_norm(wav).astype(np.float32)
        noisy_wav = torch.from_numpy(np.reshape(wav, [1, wav.shape[0]]))

        # run inference
        enhanced_wav = onnx_model(noisy_wav)

        # save denoised audio to output file
        output_file = os.path.join(output_path, os.path.basename(audio_file))
        sf.write(output_file, (enhanced_wav[0] * 32768).astype(np.int16), fs)

        del onnx_model
        pbar.update(1)
    pbar.close()
    print('\nDone, Denoised audio has been saved to', output_path)



def zipenhancer_denoise(audio_path, output_path):
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    os.makedirs(output_path, exist_ok=True)

    # get audio file list or single audio file
    if os.path.isfile(audio_path):
        audio_list = [audio_path]
    else:
        audio_list = glob.glob(os.path.join(audio_path, '*.wav'))
        #audio_list.sort()

    # create zipenhancer denoise pipeline
    ans = pipeline(Tasks.acoustic_noise_suppression, model='iic/speech_zipenhancer_ans_multiloss_16k_base', disable_update=True, disable_log=True)

    pbar = tqdm(total=len(audio_list), desc='ZipEnhancer Speech Denoise')
    for audio_file in audio_list:

        output_file = os.path.join(output_path, os.path.basename(audio_file))
        ans(audio_file, output_path=output_file)
        pbar.update(1)
    pbar.close()
    print('\nDone, Denoised audio has been saved to', output_path)


def main():
    parser = argparse.ArgumentParser(description='demo of zipenhancer speech denoise model')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='input audio file or directory to denoise')
    parser.add_argument('--onnx', default=False, action="store_true",
                        help='whether to use local onnx model for inference')
    parser.add_argument('--output_path', type=str, required=False, default='output',
                        help='output path to save chart, default=%(default)s')

    args = parser.parse_args()

    print("NOTE: ZipEnhancer model only support single channel, 16k sample rate, 16-bit audio data!");
    if args.onnx:
        zipenhancer_denoise_onnx(args.audio_path, args.output_path)
    else:
        zipenhancer_denoise(args.audio_path, args.output_path)


if __name__ == "__main__":
    main()


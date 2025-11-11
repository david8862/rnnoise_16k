#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference from:
https://www.cnblogs.com/LXP-Never/p/11071911.html

pypesq & pystoi could be installed with following cmd:
pip install pypesq
pip install pystoi
"""
import os, sys, argparse
import glob
from tqdm import tqdm
import numpy as np
import librosa
import matplotlib.pyplot as plt

# official ITU-T P.862 PESQ eval tool:
# https://www.itu.int/rec/T-REC-P.862-200102-I/en
# $ cd Software/source
# $ gcc -o pesq *.c -lm
# $ ./pesq +16000 voice.wav noisy.wav
#
# can also build & use with current dir version:
# $ cd ITU_T_pesq/
# $ mkdir build && cd build
# $ cmake -DCMAKE_BUILD_TYPE=Release [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] ..
# $ make
#
#
# install pypesq with following command if you meet error with "pip install pypesq"
# python -m pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
#import pypesq
#import pesq

#from pystoi import stoi

# install pysepm with
# pip install https://github.com/schmiph2/pysepm/archive/master.zip
#import pysepm


def SNR(labels, logits):
    """
    Calculate SNR (Signal-Noise Ratio) between clean voice and noisy voice

    # Arguments
        labels: clean voice data
            numpy array of shape (seq_len, )
        logits: noisy voice data
            numpy array of shape (seq_len, )

    # Returns
        snr: SNR value in dB
    """
    # np.sum: actural power
    # np.mean: average power
    signal = np.sum(labels ** 2)
    noise = np.sum((labels - logits) ** 2)

    snr = 10 * np.log10(signal / noise)
    return snr


def SI_SNR(clean, enhanced):
    """
    计算尺度不变信噪比 (SI-SNR)

    Args:
        clean: 干净语音信号
        enhanced: 增强后的语音信号

    Returns:
        si_snr_value: SI-SNR值 (dB)
    """
    eps=1e-8

    # 确保长度一致
    #min_len = min(len(clean), len(enhanced))
    #clean = clean[:min_len]
    #enhanced = enhanced[:min_len]

    # 去除直流分量
    clean = clean - np.mean(clean)
    enhanced = enhanced - np.mean(enhanced)

    # 计算目标信号在干净信号上的投影
    # s_target = (<s_hat, s> / ||s||^2) * s
    dot_product = np.dot(enhanced, clean)
    clean_norm_sq = np.dot(clean, clean) + eps

    scale = dot_product / clean_norm_sq
    s_target = scale * clean

    # 计算噪声成分
    e_noise = enhanced - s_target

    # 计算SI-SNR
    target_power = np.dot(s_target, s_target) + eps
    noise_power = np.dot(e_noise, e_noise) + eps

    si_snr_value = 10 * np.log10(target_power / noise_power)

    return si_snr_value



def SI_SDR(reference, estimate):
    """
    Scale-Invariant Signal to Distortion Ratio (SI-SDR)

    # Arguments
        reference: original clean voice data
            numpy array of shape (seq_len, )
        estimate: estimated noisy voice data
            numpy array of shape (seq_len, )

    # Returns
        si_sdr: SI_SDR value
    """
    eps = np.finfo(np.float32).eps
    alpha = np.dot(estimate.T, reference) / (np.dot(estimate.T, estimate) + eps)
    #print(alpha)

    molecular = ((alpha * reference) ** 2).sum()  # 分子
    denominator = ((alpha * reference - estimate) ** 2).sum()  # 分母

    si_sdr = 10 * np.log10((molecular) / (denominator+eps))

    return si_sdr


def visualize_metric_result(metric_type, noisy_metric_array, denoised_metric_array, metric_improvement_array):
    # create 3 sub-plot for noisy metric, denoised metric & metric improvement
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    # plot noisy & denoised metric bar chart
    axes[0].bar(range(len(noisy_metric_array)), noisy_metric_array, fc='b')
    axes[0].set_title('Noisy Speech ' + metric_type)
    if metric_type in ['SNR', 'SI-SNR', 'SI-SDR']:
        axes[0].set_ylabel('dB')

    axes[1].bar(range(len(denoised_metric_array)), denoised_metric_array, fc='b')
    axes[1].set_title('Denoised Speech ' + metric_type)
    if metric_type in ['SNR', 'SI-SNR', 'SI-SDR']:
        axes[1].set_ylabel('dB')

    # plot metric improvement bar chart
    axes[2].bar(range(len(metric_improvement_array)), metric_improvement_array, fc='b')
    axes[2].set_title(metric_type + ' improvement')
    if metric_type in ['SNR', 'SI-SNR', 'SI-SDR']:
        axes[2].set_ylabel('dB')

    plt.tight_layout()
    plt.show()


def denoise_eval(clean_voice_path, noisy_voice_path, denoised_voice_path, metric_type, sample_rate, visualize):
    # get clean voice audio file list or single clean voice audio file
    if os.path.isfile(clean_voice_path):
        clean_voice_list = [clean_voice_path]
    else:
        clean_voice_list = glob.glob(os.path.join(clean_voice_path, '*.wav'))
        clean_voice_list.sort()

    # get noisy voice audio file list or single noisy voice audio file
    if os.path.isfile(noisy_voice_path):
        noisy_voice_list = [noisy_voice_path]
    else:
        noisy_voice_list = glob.glob(os.path.join(noisy_voice_path, '*.wav'))
        noisy_voice_list.sort()

    # get denoised voice audio file list or single denoised voice audio file
    if os.path.isfile(denoised_voice_path):
        denoised_voice_list = [denoised_voice_path]
    else:
        denoised_voice_list = glob.glob(os.path.join(denoised_voice_path, '*.wav'))
        denoised_voice_list.sort()

    # file number should match
    assert (len(clean_voice_list) == len(noisy_voice_list) == len(denoised_voice_list)), 'voice audio file number mismatch'

    noisy_metric_list = list()
    denoised_metric_list = list()

    # check every clean voice audio file
    pbar = tqdm(total=len(clean_voice_list), desc='Denoise evaluation')
    for clean_voice_file in clean_voice_list:
        # search for corresponding noisy voice audio file
        if len(noisy_voice_list) == 1:
            noisy_voice_file = noisy_voice_list[0]
        else:
            clean_voice_basename = os.path.basename(clean_voice_file)
            noisy_voice_filenames = [item for item in noisy_voice_list if clean_voice_basename in item]
            # check if there is dumplate noisy voice audio file
            assert (len(noisy_voice_filenames) == 1), 'dumplate noisy voice audio file:{}'.format(noisy_voice_filenames)
            noisy_voice_file = noisy_voice_filenames[0]

        # search for corresponding denoised voice audio file
        if len(denoised_voice_list) == 1:
            denoised_voice_file = denoised_voice_list[0]
        else:
            clean_voice_basename = os.path.basename(clean_voice_file)
            denoised_voice_filenames = [item for item in denoised_voice_list if clean_voice_basename in item]
            # check if there is dumplate denoised voice audio file
            assert (len(denoised_voice_filenames) == 1), 'dumplate denoised voice audio file:{}'.format(denoised_voice_filenames)
            denoised_voice_file = denoised_voice_filenames[0]

        # load clean voice, noisy voice & denoised voice audio file
        clean_voice, clean_sr = librosa.load(clean_voice_file, sr=sample_rate)
        noisy_voice, noisy_sr = librosa.load(noisy_voice_file, sr=sample_rate)
        denoised_voice, denoised_sr = librosa.load(denoised_voice_file, sr=sample_rate)

        # check audio channel number
        assert (clean_voice.ndim == noisy_voice.ndim == denoised_voice.ndim == 1), 'only support single channel audio'

        # align audio length
        voice_len = min(min(len(clean_voice), len(noisy_voice)), len(denoised_voice))
        clean_voice = clean_voice[:voice_len]
        noisy_voice = noisy_voice[:voice_len]
        denoised_voice = denoised_voice[:voice_len]

        # check sample rate
        assert clean_sr == noisy_sr == denoised_sr, 'audio sample rate mismatch'

        if metric_type == 'SNR':
            noisy_snr = SNR(clean_voice, noisy_voice)
            denoised_snr = SNR(clean_voice, denoised_voice)
            noisy_metric_list.append(noisy_snr)
            denoised_metric_list.append(denoised_snr)
            #print('noisy SNR: {} dB'.format(noisy_snr))
            #print('denoised SNR: {} dB'.format(denoised_snr))
            #print('SNR improvement: {} dB'.format(denoised_snr - noisy_snr))

        elif metric_type == 'SI-SNR':
            noisy_si_snr = SI_SNR(clean_voice, noisy_voice)
            denoised_si_snr = SI_SNR(clean_voice, denoised_voice)
            noisy_metric_list.append(noisy_si_snr)
            denoised_metric_list.append(denoised_si_snr)

        elif metric_type == 'SI-SDR':
            noisy_si_sdr = SI_SDR(clean_voice, noisy_voice)
            denoised_si_sdr = SI_SDR(clean_voice, denoised_voice)
            noisy_metric_list.append(noisy_si_sdr)
            denoised_metric_list.append(denoised_si_sdr)

        elif metric_type == 'PESQ':
            import pypesq
            noisy_pesq = pypesq.pesq(clean_voice, noisy_voice, fs=clean_sr)
            denoised_pesq = pypesq.pesq(clean_voice, denoised_voice, fs=clean_sr)
            noisy_metric_list.append(noisy_pesq)
            denoised_metric_list.append(denoised_pesq)
            #print('noisy PESQ (-0.5~4.5, higher the better):', noisy_pesq)
            #print('denoised PESQ (-0.5~4.5, higher the better):', denoised_pesq)
            #print('PESQ improvement:', denoised_pesq - noisy_pesq)

        elif metric_type == 'STOI':
            from pystoi import stoi
            noisy_stoi = stoi(clean_voice, noisy_voice, fs_sig=clean_sr)
            denoised_stoi = stoi(clean_voice, denoised_voice, fs_sig=clean_sr)
            noisy_metric_list.append(noisy_stoi)
            denoised_metric_list.append(denoised_stoi)
            #print('noisy STOI (0~1, higher the better):', noisy_stoi)
            #print('denoised STOI (0~1, higher the better):', denoised_stoi)
            #print('STOI improvement:', denoised_stoi - noisy_stoi)

        else:
            raise ValueError('invalid metric type:', metric_type)
        pbar.update(1)
    pbar.close()

    noisy_metric_array = np.array(noisy_metric_list, dtype=np.float32)
    denoised_metric_array = np.array(denoised_metric_list, dtype=np.float32)
    metric_improvement_array = denoised_metric_array - noisy_metric_array

    # calculate average metric
    print('Metric type:', metric_type)
    if metric_type == 'SNR':
        print('average noisy SNR: {} dB'.format(np.mean(noisy_metric_array)))
        print('average denoised SNR: {} dB'.format(np.mean(denoised_metric_array)))
        print('average SNR improvement: {} dB'.format(np.mean(metric_improvement_array)))
    elif metric_type == 'SI-SNR':
        print('average noisy SI-SNR: {} dB'.format(np.mean(noisy_metric_array)))
        print('average denoised SI-SNR: {} dB'.format(np.mean(denoised_metric_array)))
        print('average SI-SNR improvement: {} dB'.format(np.mean(metric_improvement_array)))
    elif metric_type == 'SI-SDR':
        print('average noisy SI-SDR: {} dB'.format(np.mean(noisy_metric_array)))
        print('average denoised SI-SDR: {} dB'.format(np.mean(denoised_metric_array)))
        print('average SI-SDR improvement: {} dB'.format(np.mean(metric_improvement_array)))
    elif metric_type == 'PESQ':
        print('average noisy PESQ (-0.5~4.5, higher the better): {}'.format(np.mean(noisy_metric_array)))
        print('average denoised PESQ (-0.5~4.5, higher the better): {}'.format(np.mean(denoised_metric_array)))
        print('average PESQ improvement: {}'.format(np.mean(metric_improvement_array)))
    elif metric_type == 'STOI':
        print('average noisy STOI (0~1, higher the better): {}'.format(np.mean(noisy_metric_array)))
        print('average denoised STOI (0~1, higher the better): {}'.format(np.mean(denoised_metric_array)))
        print('average STOI improvement: {}'.format(np.mean(metric_improvement_array)))
    else:
        raise ValueError('invalid metric type:', metric_type)

    if visualize:
        # only plot calculate average metric
        if len(noisy_metric_array) <= 1:
            print('evaluation for single audio file, will not show metric plot')
        else:
            visualize_metric_result(metric_type, noisy_metric_array, denoised_metric_array, metric_improvement_array)
    return


def main():
    parser = argparse.ArgumentParser(description='tool to evaluate Denoise metrics')
    parser.add_argument('--clean_voice_path', type=str, required=True,
                        help='file or directory for original clean voice audio')
    parser.add_argument('--noisy_voice_path', type=str, required=True,
                        help='file or directory for noisy voice audio')
    parser.add_argument('--denoised_voice_path', type=str, required=True,
                        help='file or directory for denoised voice audio')
    parser.add_argument('--metric_type', type=str, required=False, default='SNR', choices=['SNR', 'SI-SNR', 'SI-SDR', 'PESQ', 'STOI'],
                        help='voice quality metric type, default=%(default)s')
    parser.add_argument('--sample_rate', type=int, required=False, default=None, choices=[None, 8000, 16000, 22050, 44100, 48000],
                        help='(optional) target sample rate, None is unchange. default=%(default)s')
    parser.add_argument('--visualize', default=False, action="store_true",
                        help='Whether to visualize denoise metric result')

    args = parser.parse_args()

    denoise_eval(args.clean_voice_path, args.noisy_voice_path, args.denoised_voice_path, args.metric_type, args.sample_rate, args.visualize)


if __name__ == "__main__":
    main()

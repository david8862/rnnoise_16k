#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Strip speech segment in audio files with Silero VAD.
Reference from:
https://github.com/snakers4/silero-vad
https://github.com/david8862/silero-vad/blob/master/tuning/README_cn.md
https://blog.csdn.net/gitblog_07753/article/details/142222485

Silero VAD could be installed with following cmd:
pip install silero-vad
"""
import os, sys, argparse
import glob
from tqdm import tqdm
import numpy as np
import soundfile as sf
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


def speech_strip(model, input_audio_file, sample_rate, threshold, neg_threshold, min_speech_duration_ms, output_path):
    # load audio data
    audio_data = read_audio(input_audio_file)

    # Silero VAD timestamps (in seconds) would be a list of dict, like:
    # [
    #   {'start': 7.1, 'end': 7.8},
    #   {'start': 9.9, 'end': 11.3},
    #   {'start': 21.7, 'end': 22.9}
    # ]
    speech_timestamps = get_speech_timestamps(audio_data,
                                             model,
                                             sampling_rate=sample_rate,
                                             threshold=threshold,
                                             neg_threshold=neg_threshold,
                                             min_speech_duration_ms=min_speech_duration_ms,
                                             return_seconds=True #return speech timestamps in seconds (default is samples)
                                            )

    # convert audio data to numpy array for strip
    audio_data = audio_data.detach().cpu().numpy()

    # clip out non-speech audio according to speech timestamps
    non_speech_audio_data = np.empty(0, dtype=audio_data.dtype)
    end_sample = 0
    for speech_timestamp in speech_timestamps:
        start_sample = int(sample_rate * (speech_timestamp['start'] - min_speech_duration_ms/1000.0))
        non_speech_audio_data = np.concatenate((non_speech_audio_data, audio_data[end_sample:start_sample]), axis=0)
        end_sample = int(sample_rate * speech_timestamp['end'])

    # clip out tail part of non-speech audio
    non_speech_audio_data = np.concatenate((non_speech_audio_data, audio_data[end_sample:]), axis=0)

    # save non-speech audio
    output_file = os.path.join(output_path, os.path.basename(input_audio_file))
    sf.write(output_file, non_speech_audio_data, sample_rate)



def main():
    parser = argparse.ArgumentParser(description='Strip speech segment in audio files with Silero VAD')
    parser.add_argument('--input_audio_path', type=str, required=True,
                        help='file or directory for input single-channel wav audio')
    parser.add_argument('--sample_rate', type=int, required=False, default=16000, choices=[8000, 16000],
                        help='audio sample rate. default=%(default)s')
    parser.add_argument('--threshold', type=float, required=False, default=0.5,
                        help='speech threshold. default=%(default)s')
    parser.add_argument('--neg_threshold', type=float, required=False, default=0.15,
                        help='negative threshold. default=%(default)s')
    parser.add_argument('--min_speech_duration_ms', type=int, required=False, default=50,
                        help='speech chunks shorter than it would be drop. default=%(default)s')
    parser.add_argument('--output_path', type=str, required=False, default='output',
                        help='output path to save speech stripped audio file. default=%(default)s')

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # get input audio file list or single input audio
    if os.path.isfile(args.input_audio_path):
        input_audio_list = [args.input_audio_path]
    else:
        input_audio_list = glob.glob(os.path.join(args.input_audio_path, '*.wav'))

    # load Silero VAD model
    silero_model = load_silero_vad(onnx=True, opset_version=16) # opset could be 15 or 16

    pbar = tqdm(total=len(input_audio_list), desc='Speech Strip')
    for input_audio_file in input_audio_list:
        speech_strip(silero_model, input_audio_file, args.sample_rate, args.threshold, args.neg_threshold, args.min_speech_duration_ms, args.output_path)
        pbar.update(1)
    pbar.close()
    print('\nDone. speech stripped audio have been saved to: ' + args.output_path)


if __name__ == "__main__":
    main()

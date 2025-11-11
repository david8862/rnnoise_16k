#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" a simple tool to generate .srt subtitle file with OpenAI Whisper model
Reference from:
https://github.com/openai/whisper
https://blog.csdn.net/hhy321/article/details/134897967
https://github.com/openai/whisper/discussions/1576
https://github.com/ggml-org/whisper.cpp
https://huggingface.co/mpoyraz/wav2vec2-xls-r-300m-cv7-turkish

openai-whisper could be installed with following cmd:
apt install ffmpeg
pip install openai-whisper
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
pip install ffmpeg zhconv wheel

install PyTorch if needed:
pip install torch torchvision torchaudio

NOTE: to have a better inference performance with large model (like 'large-v3-turbo'), you'd
      better to have a Nvidia GPU with >= 8GB Memory
"""
import os, sys, argparse
import glob
from tqdm import tqdm
import whisper
import zhconv


def whisper_to_srt(input_audio_path, model_type, language, no_speech_threshold, fp16, output_path):
    # get .wav audio file list or single .wav audio file
    if os.path.isfile(input_audio_path):
        audio_list = [input_audio_path]
    else:
        audio_list = glob.glob(os.path.join(input_audio_path, '*.wav'))

    os.makedirs(output_path, exist_ok=True)

    model = whisper.load_model(model_type)
    print('model "%s" support %d types of language' % (model_type, model.num_languages))

    pbar = tqdm(total=len(audio_list), desc='Whisper to subtitle')
    for audio_file in audio_list:
        result = model.transcribe(audio_file,
                                  language=language,
                                  task='transcribe',
                                  verbose=None,  # None/True/False
                                  compression_ratio_threshold=2.4,  # if gzip compression ratio above this value, treat as failed
                                  logprob_threshold=-1.0,  # if avg_logprob below this value, treat as failed
                                  # if no_speech_prob is higher than this value AND avg_logprob below
                                  # `logprob_threshold`, consider the segment as silent
                                  no_speech_threshold=no_speech_threshold,
                                  word_timestamps=False,
                                  clip_timestamps='0',
                                  fp16=fp16
                                 )

        # check if language is correct
        if language is not None:
            assert (result['language'] == language), 'language mismatch: %s' % result['language']
        else:
            print('Language:', str(result['language']))

        # save ASR result to srt file
        # .srt subtitle file would be like:
        # cat subtitle.srt
        # 1
        # 00:03:17,949 --> 00:03:19,200
        # Hello.
        #
        # 2
        # 00:03:20,159 --> 00:03:22,829
        # I'm from China,
        # and now live here.
        #
        # ...
        srt_file_basename = os.path.splitext(os.path.split(audio_file)[-1])[0]
        srt_file_name = os.path.join(output_path, srt_file_basename + '.srt')
        srt_file = open(srt_file_name, 'w', encoding='utf-8')

        # Whisper ASR segments format:
        # [{
        #    'id': 0,
        #    'seek': 0,
        #    'start': 0.0,
        #    'end': 1.0,
        #    'text': '清洁完成',
        #    'tokens': [50364, 21784, 35622, 242, 41509, 50414],
        #    'temperature': 0.0,
        #    'avg_logprob': -0.46264615740094867,
        #    'compression_ratio': 0.8873239436619719,
        #    'no_speech_prob': 0.01184895820915699
        # }]
        asr_segments = result['segments']

        # save start/stop time and text for each segment
        if len(asr_segments) > 0:
            for i, asr_segment in enumerate(asr_segments):
                # for Chinese, make sure it is simplified format
                if language == 'zh':
                    text = zhconv.convert(asr_segment['text'], 'zh-cn')
                else:
                    text = str(asr_segment['text'])

                # write speech segment id
                srt_file.write("%d\n"%(i+1))

                # parse start & stop time from asr segment to .srt format ("00:03:17,949 --> 00:03:19,200")
                start_hour = int(asr_segment['start'] // 3600)
                start_minute = int((asr_segment['start'] - start_hour*3600) // 60)
                start_second = int(asr_segment['start'] - start_hour*3600 - start_minute*60)
                start_second_decimal_part = int((asr_segment['start'] % 1)*1000.0)
                start_time = "%02d"%start_hour + ':' + "%02d"%start_minute + ':' + "%02d"%start_second + ',' + "%03d"%start_second_decimal_part

                stop_hour = int(asr_segment['end'] // 3600)
                stop_minute = int((asr_segment['end'] - stop_hour*3600) // 60)
                stop_second = int(asr_segment['end'] - stop_hour*3600 - stop_minute*60)
                stop_second_decimal_part = int((asr_segment['end'] % 1)*1000.0)
                stop_time = "%02d"%stop_hour + ':' + "%02d"%stop_minute + ':' + "%02d"%stop_second + ',' + "%03d"%stop_second_decimal_part

                # save start/stop time & text to srt file
                srt_file.write(start_time + ' ---> ' + stop_time + '\n')
                srt_file.write(asr_segment['text'] + '\n')
                srt_file.write('\n')
        else:
            print('Not detect any speech segment in', audio_file)

        srt_file.close()
        pbar.update(1)
    pbar.close()
    print('\nDone. Subtitle file has been generated to %s' % output_path)

    return



def main():
    parser = argparse.ArgumentParser(description='tool to generate .srt subtitle file with OpenAI Whisper model')
    parser.add_argument('--input_audio_path', type=str, required=True,
                        help='file or directory for input .wav audio files')
    parser.add_argument('--model_type', type=str, required=False, default='large-v3-turbo',
                        choices=['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo'],
                        help="Whisper model type to use. default=%(default)s")
    parser.add_argument('--language', type=str, required=False, default=None,
                        choices=[None, 'zh', 'en', 'fr', 'de', 'it', 'es', 'ja', 'ko', 'ru', 'tr', 'th'],
                        help = "Target language to transcribe, None for auto-detect. default=%(default)s")
    parser.add_argument('--no_speech_threshold', type=float, required=False, default=0.8,
                        help="threshold to judge if an audio segment contains speech. default=%(default)s")
    parser.add_argument('--fp16', default=False, action="store_true",
                        help='Whether to use fp16 inference. default=%(default)s')
    parser.add_argument('--output_path', type=str, required=False, default='output',
                        help='output path to save .srt subtitle files. default=%(default)s')
    args = parser.parse_args()

    whisper_to_srt(args.input_audio_path, args.model_type, args.language, args.no_speech_threshold, args.fp16, args.output_path)


if __name__ == "__main__":
    main()

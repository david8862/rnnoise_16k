#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Tool to convert .srt audio subtitle text file to voice timestamp txt file
"""
import os, sys, argparse
import glob
from tqdm import tqdm
import soundfile as sf


def get_audio_info(audio_file):
    audio_data, sample_rate = sf.read(audio_file)

    # only support single channel audio, with shape (sample_num,)
    assert audio_data.ndim == 1, 'only support single channel audio'

    # get total audio duration time (in seconds)
    audio_duration = float(audio_data.shape[0]) / sample_rate

    return audio_duration, sample_rate


def srt_to_txt(input_audio_file, input_srt_file, output_path):
    # get total audio duration time (in seconds) & sample rate from wav file
    audio_duration, sample_rate = get_audio_info(input_audio_file)

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
    srt_file = open(input_srt_file, "r", encoding='utf-8')
    srt_lines = srt_file.readlines()

    # output voice timestamps txt file would be like:
    # cat output/subtitle.txt
    # 16000               // sample rate
    # 8525.995            // total duration time (in seconds)
    # 197.949,199.200     // voice start/stop time (in seconds)
    # 200.159,202.829
    # ...
    output_txt_filename = os.path.splitext(os.path.split(input_srt_file)[-1])[0] + '.txt'
    output_txt_filename = os.path.join(output_path, output_txt_filename)
    output_txt_file = open(output_txt_filename, "w", encoding='utf-8')

    # write sample rate & total audio duration time into output file
    output_txt_file.write(str(sample_rate))
    output_txt_file.write('\n')
    output_txt_file.write(format(audio_duration, "0.3f"))
    output_txt_file.write('\n')

    # parse subtitle segments
    i = 0
    while i < len(srt_lines):
        # subtitle index, e.g. '1'
        subtitle_index = int(srt_lines[i].strip().replace('\ufeff', ''))
        i += 1

        # subtitle timestamp, e.g. '00:03:17,949 --> 00:03:19,200'
        start_time, stop_time = srt_lines[i].split(' --> ')
        # convert start time string to seconds, e.g. 197.949
        start_time_split = start_time.split(',')
        start_time_seg = start_time_split[0].split(':')
        start_time = float(start_time_seg[0])*3600.0 + float(start_time_seg[1])*60.0 + float(start_time_seg[2]) + float(start_time_split[1])*0.001
        # convert stop time string to seconds, e.g. 199.200
        stop_time_split = stop_time.split(',')
        stop_time_seg = stop_time_split[0].split(':')
        stop_time = float(stop_time_seg[0])*3600.0 + float(stop_time_seg[1])*60.0 + float(stop_time_seg[2]) + float(stop_time_split[1])*0.001
        # write start/stop time into output file
        voice_segment_string = format(start_time, "0.3f") + ',' + format(stop_time, "0.3f")
        output_txt_file.write(voice_segment_string)
        output_txt_file.write('\n')
        i += 1

        # bypass the following text in this subtitle segment
        while len(srt_lines[i]) > 1:
            i += 1

        # check & bypass the empty line between subtitle segments, whose length is 1 ('\n')
        assert len(srt_lines[i]) == 1, 'expect an empty line here'
        i += 1

    srt_file.close()
    output_txt_file.close()



def main():
    parser = argparse.ArgumentParser(description='Tool to convert .srt audio subtitle text file to voice timestamp txt file')
    parser.add_argument('--input_audio_path', type=str, required=True,
                        help='file or directory for input wav audio file')
    parser.add_argument('--input_srt_path', type=str, required=True,
                        help='file or directory for input .srt subtitle text file')
    parser.add_argument('--output_path', type=str, required=False, default='output',
                        help='output path to save voice timestamp txt file. default=%(default)s')

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # get input audio file list or single input audio
    if os.path.isfile(args.input_audio_path):
        input_audio_list = [args.input_audio_path]
    else:
        input_audio_list = glob.glob(os.path.join(args.input_audio_path, '*.wav'))

    # get input srt file list or single srt file
    if os.path.isfile(args.input_srt_path):
        input_srt_list = [args.input_srt_path]
    else:
        input_srt_list = glob.glob(os.path.join(args.input_srt_path, '*.srt'))


    pbar = tqdm(total=len(input_audio_list), desc='srt to txt convert')
    for input_audio_file in input_audio_list:
        input_srt_filename = os.path.splitext(os.path.split(input_audio_file)[-1])[0] + '.srt'
        input_srt_files = [item for item in input_srt_list if input_srt_filename in item]

        # check if there is dumplate srt file
        assert (len(input_srt_files) == 1), 'dumplate srt file:{}'.format(input_srt_files)
        input_srt_file = input_srt_files[0]

        srt_to_txt(input_audio_file, input_srt_file, args.output_path)
        pbar.update(1)
    pbar.close()
    print('\nDone. voice timestamp files have been saved to: ' + args.output_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''Evaluate VAD metrics
Reference from:
https://blog.csdn.net/matrix_space/article/details/50384518
https://blog.csdn.net/zjn295771349/article/details/84961596
'''
import os, sys, argparse
import glob
from tqdm import tqdm
import numpy as np


def get_metric(y_pred, y_true, beta=3.0):
    """
    计算二分类问题的各项性能指标

    参数:
    y_pred: 预测标签，形状为(n_samples,)的numpy数组
    y_true: 真实标签，形状为(n_samples,)的numpy数组
    beta:   计算f_beta_score时的权重系数

    返回:
    accuracy:     正确率
    precision:    精确率
    recall:       召回率
    f1_score:     F1-score值
    f_beta_score: F-beta-score值
    """
    # 计算正确率
    accuracy = np.mean(y_pred == y_true)

    # 计算真阳性、假阳性、假阴性
    true_pos = np.sum((y_true == 1) & (y_pred == 1))  # 真阳性
    false_pos = np.sum((y_true == 0) & (y_pred == 1))  # 假阳性
    false_neg = np.sum((y_true == 1) & (y_pred == 0))  # 假阴性

    # 计算精确率和召回率
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

    # 计算F1-score和F-beta-score
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f_beta_score = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall) if (beta * beta * precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score, f_beta_score


def get_metric_bk(y_pred, y_true, beta=3.0):
    # add epsilon to avoid divided by 0
    epsilon = 1e-7

    accuracy = (y_pred == y_true)

    true_pos = np.sum(y_pred * y_true, axis=0)
    false_pos = np.sum(y_pred * (1 - y_true), axis=0)
    false_neg = np.sum((1 - y_pred) * y_true, axis=0)

    precision = true_pos / (true_pos + false_pos + epsilon)
    recall = true_pos / (true_pos + false_neg + epsilon)

    f1_score = 2 * precision * recall / (precision + recall + epsilon)
    f1_score = np.where(np.isnan(f1_score), np.zeros_like(f1_score), f1_score)

    f_beta_score = (1 + beta * beta) * precision * recall / (beta * beta *precision + recall + epsilon)
    f_beta_score = np.where(np.isnan(f_beta_score), np.zeros_like(f_beta_score), f_beta_score)

    return np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f1_score), np.mean(f_beta_score)



def vad_eval(annotation_txt_path, result_txt_path, metric_type, beta):
    # get annotation txt file list or single annotation txt file
    if os.path.isfile(annotation_txt_path):
        annotation_txt_list = [annotation_txt_path]
    else:
        annotation_txt_list = glob.glob(os.path.join(annotation_txt_path, '*.txt'))
        annotation_txt_list.sort()

    # get result txt file list or single result txt file
    if os.path.isfile(result_txt_path):
        result_txt_list = [result_txt_path]
    else:
        result_txt_list = glob.glob(os.path.join(result_txt_path, '*.txt'))
        result_txt_list.sort()

    # file number should match
    assert (len(annotation_txt_list) == len(result_txt_list)), 'annotation file number mismatch with result file'

    total_annotation_vad = np.empty(0, dtype=int)
    total_result_vad = np.empty(0, dtype=int)
    #total_sample_num = 0

    # check every annotation txt file
    pbar = tqdm(total=len(annotation_txt_list), desc='VAD evaluation')
    for annotation_txt_filename in annotation_txt_list:
        # search for corresponding result txt file
        if len(result_txt_list) == 1:
            result_txt_filename = result_txt_list[0]
        else:
            annotation_txt_basename = os.path.basename(annotation_txt_filename)
            result_txt_filenames = [item for item in result_txt_list if annotation_txt_basename in item]
            # check if there is dumplate result txt file
            assert (len(result_txt_filenames) == 1), 'dumplate result txt file:{}'.format(result_txt_filenames)
            result_txt_filename = result_txt_filenames[0]

        # voice timestamps txt file would be like:
        # cat voice_timestamp.txt
        # 16000               // sample rate
        # 8525.995            // total duration time (in seconds)
        # 197.949,199.200     // voice start/stop time (in seconds)
        # 200.159,202.829
        # ...
        annotation_txt_file = open(annotation_txt_filename, "r", encoding='utf-8')
        annotation_lines = annotation_txt_file.readlines()
        annotation_sample_rate = int(annotation_lines[0].strip())
        annotation_audio_duration = float(annotation_lines[1].strip())

        # create annotation vad array for the whole audio
        annotation_sample_num = int(annotation_audio_duration * annotation_sample_rate)
        annotation_vad = np.zeros(annotation_sample_num, dtype=int)

        # assign annotation voice samples to 1 according to voice start/stop time
        for annotation_segment in annotation_lines[2:]:
            start_time, stop_time = annotation_segment.split(',')
            start_sample = int(float(start_time) * annotation_sample_rate)
            stop_sample = int(float(stop_time) * annotation_sample_rate)
            annotation_vad[start_sample:(stop_sample+1)] = 1

        # parse result txt file with the same format
        result_txt_file = open(result_txt_filename, "r", encoding='utf-8')
        result_lines = result_txt_file.readlines()
        result_sample_rate = int(result_lines[0].strip())
        result_audio_duration = float(result_lines[1].strip())

        # check if sample rate & audio duration match annotation
        assert (result_sample_rate == annotation_sample_rate), 'sample rate mismatch between annotation & result for {}'.format(annotation_txt_filename)
        assert (abs(result_audio_duration - annotation_audio_duration) <= 0.01), 'audio duration mismatch between annotation & result for {}'.format(annotation_txt_filename)
        # ignore minor duration mismatch (maybe due to rounding error), for further array alignment
        result_audio_duration = annotation_audio_duration

        # create result vad array for the whole audio
        result_sample_num = int(result_audio_duration * result_sample_rate)
        result_vad = np.zeros(result_sample_num, dtype=int)

        # assign result voice samples to 1 according to voice start/stop time
        for result_segment in result_lines[2:]:
            start_time, stop_time = result_segment.split(',')
            start_sample = int(float(start_time) * result_sample_rate)
            stop_sample = int(float(stop_time) * result_sample_rate)
            result_vad[start_sample:(stop_sample+1)] = 1

        # merge single audio vad result into total result
        total_annotation_vad = np.concatenate((total_annotation_vad, annotation_vad), axis=0)
        total_result_vad = np.concatenate((total_result_vad, result_vad), axis=0)
        #total_sample_num += annotation_sample_num
        pbar.update(1)
    pbar.close()

    # calculate total VAD metric
    accuracy, precision, recall, f1_score, f_beta_score = get_metric(total_result_vad, total_annotation_vad, beta)
    #print('VAD metrics:', accuracy, precision, recall, f1_score, f_beta_score)

    if metric_type == 'Accuracy':
        print('VAD Accuracy:', accuracy)
    elif metric_type == 'Precision':
        print('VAD Precision:', precision)
    elif metric_type == 'Recall':
        print('VAD Recall:', recall)
    elif metric_type == 'F1-score':
        print('VAD F1-score:', f1_score)
    elif metric_type == 'F-beta-score':
        print('VAD F-beta-score:', f_beta_score)
    else:
        raise ValueError('invalid metric type:', metric_type)



def main():
    parser = argparse.ArgumentParser(description='evaluate VAD metrics')
    parser.add_argument('--annotation_txt_path', type=str, required=True,
                        help='file or directory for voice timestamp annotation txt file')
    parser.add_argument('--result_txt_path', type=str, required=True,
                        help='file or directory for voice timestamp detect result txt file')
    parser.add_argument('--metric_type', type=str, required=False, default='Accuracy', choices=['Accuracy', 'Precision', 'Recall', 'F1-score', 'F-beta-score'],
                        help='metric type. default=%(default)s')
    parser.add_argument('--beta', type=float, required=False, default=3.0,
                        help='weight coef for precision in F-beta score. Higher for more focus on precision. default=%(default)s')

    args = parser.parse_args()

    vad_eval(args.annotation_txt_path, args.result_txt_path, args.metric_type, args.beta)


if __name__ == "__main__":
    main()

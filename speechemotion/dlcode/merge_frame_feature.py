#!/usr/bin/env python
# -*- coding: utf-8 -*-
# hellolzc 20200200
""" 用于读取切割音频片段，库里的片段几乎都是4s
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
import h5py


def load_features_opensmile(file_path):
    """ 读取opensmile特有的CSV格式 """
    df = pd.read_csv(open(file_path, encoding='utf-8'), sep=';')
    df.drop(columns=['name','frameTime'], inplace=True)
    return df


def merge_LLDs(file_path):
    """ 将特征文件切片，返回一个字典，字典key是“uuid-开始时间”，value是一个dataframe
    TODO：文件结束部分还需仔细处理不要浪费
    """
    # sampling_frequency = 100
    df = load_features_opensmile(file_path)
    array = df.values
    return array


def merge_LLDs_main(input_dir, output_file):
    """ LLDs（low level descriptors）LLDs指的是手工设计的一些低水平特征，一般是在一帧语音上进行的计算，是用来表示一帧语音的特征。
        HSFs（high level statistics functions）是在LLDs的基础上做一些统计而得到的特征，是用来表示一个utterance的特征。
    将保存到HDF5文件中"""

    # print('seg_time %f hop_time %f' % (feature_type, start_time, end_time))
    print('inputdir:', input_dir)
    print('outfile:', output_file)

    file_list = os.listdir(input_dir)
    file_list = [fp for fp in file_list if fp[-4:]=='.csv']
    file_list.sort()

    with h5py.File(output_file,"w") as h5file:
        for line in file_list:
            uuid = line[:-4]
            print(uuid, end=' ', flush=True)
            feature_path = os.path.join(input_dir, line)

            merged_LLD = merge_LLDs(feature_path)
            h5file.create_dataset(uuid, data=merged_LLD)
    print()


if __name__ == '__main__':
    # e.g.
    # ./slice_wav_feature.py -m 0 -s 10 -j 2 -f CPE16_keep_b
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_dir',
        type=str,
        default='../../opensmile/audio_features_egemaps_',
        help='The path of opensmile output LLDs(Local Level Descriptors)'
    )
    parser.add_argument(
        '-o', '--output_file',
        type=str,
        default='../../data/emodb/acoustic_egemaps.hdf5',
        help='The name of the out put hdf5 file'
    )

    FLAGS, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print('Unknown arguments: ', unparsed)
    args = sys.argv

    merge_LLDs_main(FLAGS.input_dir, FLAGS.output_file)


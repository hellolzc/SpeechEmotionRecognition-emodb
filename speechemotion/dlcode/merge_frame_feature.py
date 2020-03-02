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
import json
import h5py


def load_features_opensmile(file_path: str) -> pd.DataFrame:
    """ 读取opensmile特有的CSV格式 """
    df = pd.read_csv(open(file_path, encoding='utf-8'), sep=';')
    df.drop(columns=['name','frameTime'], inplace=True)
    return df


def single_LLDs(file_path):
    """ 载入一个时序特征文件，返回一个numpy array
    """
    # sampling_frequency = 100
    df = load_features_opensmile(file_path)
    array = df.values
    return array


def single_LLDs_main(input_dir, output_file):
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

            merged_LLD = single_LLDs(feature_path)
            h5file.create_dataset(uuid, data=merged_LLD)
    print()


def merge_LLDs(input_config, uuid, project_root):
    """ 载入多个时序特征文件，并拼接，返回一个numpy array
        input_config: 一个列表，元素是一个dict, 格式如下:
        {
            "name":"ComParE",
            "prefix":null,
            "dir":"opensmile/audio_features_CPE16_/",
            "selected":"pcm_RMSenergy_sma,pcm_zcr_sma,pcm_fftMag_spectralCentroid_sma,F0final_sma"
        }
    """
    # sampling_frequency = 100
    df_merged = None
    for indx, config in enumerate(input_config):
        file_path = os.path.join(project_root, input_config[indx]["dir"], "%s.csv" % uuid)
        df = load_features_opensmile(file_path)

        select_feature = [name.strip() for name in config["selected"].split(',')]
        df = df.filter(items=select_feature)

        prefix = config["prefix"]
        if prefix is not None:
            df.columns = [prefix+x for x in df.columns]

        if df_merged is None:
            df_merged = df
        else:
            assert len(df_merged) == len(df)
            df_merged = pd.concat((df_merged, df), axis=1)

    array = df_merged.values
    return array

def merge_LLDs_main(config_file_path):
    """ LLDs（low level descriptors）LLDs指的是手工设计的一些低水平特征，一般是在一帧语音上进行的计算，是用来表示一帧语音的特征。
        HSFs（high level statistics functions）是在LLDs的基础上做一些统计而得到的特征，是用来表示一个utterance的特征。
    将保存到HDF5文件中"""
    with open(config_file_path, 'rt') as json_file:
        config = json.load(json_file)
    project_root = config["project_root"]
    output_file = os.path.join(project_root, config["output_fp"])

    input_config_list = config["input"]
    first_input_dir = os.path.join(project_root, input_config_list[0]["dir"])

    print("InputConfig:")
    feat_no=0
    for item in input_config_list:
        print(item['name'], item['prefix'], item['dir'])
        for name in item['selected'].split(','):
            print("%d\t%s"%(feat_no, name))
            feat_no+=1
    print("Output:", output_file)
    print()

    file_list = os.listdir(first_input_dir)
    file_list = [fp for fp in file_list if fp[-4:]=='.csv']
    file_list.sort()

    with h5py.File(output_file,"w") as h5file:
        for line in file_list:
            uuid = line[:-4]
            print(uuid, end=' ', flush=True)
            merged_LLD = merge_LLDs(input_config_list, uuid, project_root)
            h5file.create_dataset(uuid, data=merged_LLD)
    print("\nOutput sample shape:", merged_LLD.shape)



if __name__ == '__main__':
    # e.g.
    # ./merge_frame_feature.py -c feature_selected_config_1.json
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config_file',
        type=str,
        default='',
        help='The name of the config file. Ignore other arguments if this one is set.'
    )
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

    if FLAGS.config_file == '':
        single_LLDs_main(FLAGS.input_dir, FLAGS.output_file)
    else:
        merge_LLDs_main(FLAGS.config_file)



#!/usr/bin/env python
# python3 script
import sys
import os
import time
import argparse

exe_opensmile = '/home/zhaoci/toolkit/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract'
path_config   = '/home/zhaoci/toolkit/opensmile-2.3.0/config/'
config_filenames = {
    'IS09':'IS09_emotion.conf',
    'IS10':'IS10_paraling.conf',
    'IS11':'IS11_speaker_state.conf',
    'IS12':'IS12_speaker_trait.conf',
    'IS13':'IS13_ComParE.conf',
    'CPE16':'ComParE_2016.conf',
    'egemaps': 'gemaps/eGeMAPSv01a.conf',  # eGeMAPS feature set
    'mfcc': 'MFCC12_0_D_A.conf',  # MFCCs 0-12 with delta and acceleration coefficients
}


def extract_one_audio(infilename, outfilename, instname, out_summary, feature_type):
    ''' 提取特征统计量
    Parameters: infilename：音频文件名, 绝对路径
                instname：实例名，一般是音频名(不含格式后缀)
                out_summary：存放统计量的文件名
    Return: None
    '''
    config_fp = path_config + config_filenames[feature_type]
    # options from standard_data_output.conf.inc
    # -timestampcsvlld 1 -headercsvlld 1  -output ?
    cmd = exe_opensmile + ' -C '+ config_fp + ' -I '+ infilename + \
        ' -nologfile ' + ' -instname ' + instname + \
        ' -appendcsvlld 0 -lldcsvoutput '+ outfilename + \
        ' -csvoutput ' + out_summary
    print('\n' + cmd)
    os.system(cmd)
    time.sleep(0.01)


def extract_one_audio_lldonly(infilename, outfilename, instname, feature_type):
    ''' 提取特征 MFCCs 0-12 with delta and acceleration coefficients
    Parameters: infilename：音频文件名, 绝对路径
                instname：实例名，一般是音频名(不含格式后缀)
    Return: None
    '''
    assert feature_type == 'mfcc'
    config_fp = path_config + config_filenames[feature_type]
    # options from standard_data_output_lldonly.conf.inc
    cmd = exe_opensmile + ' -configfile ' + config_fp + \
        ' -nologfile -appendcsv 0 -timestampcsv 1 -headercsv 1' + \
        ' -inputfile ' + infilename + ' -csvoutput ' + outfilename + \
        ' -instname ' + instname + ' -output ?'  # (disabling htk output)

    print('\n' + cmd)
    os.system(cmd)
    time.sleep(0.01)


def main(feature_type, audio_path, name_suffix):
    assert feature_type in config_filenames, 'Unsupport feature_type: %s' % feature_type
    output_path = '../audio_features_%s_%s/' % (feature_type, name_suffix)
    out_summary = '../func_%s_%s.csv' % (feature_type, name_suffix)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # 删除之前的文件
    if os.path.exists(out_summary):
        os.remove(out_summary)
    audio_path = os.path.abspath(audio_path)  # 转换成绝对路径, 避免OpenSMILE工具找不到
    audio_list = os.listdir(audio_path)
    audio_list.sort()

    for audio in audio_list:
        if audio[-4:]=='.wav':
            this_path_input = os.path.join(audio_path, audio)
            instname = audio[:-4]
            this_path_output = os.path.join(output_path, instname+'.csv')

            if feature_type == 'mfcc':
                extract_one_audio_lldonly(this_path_input, this_path_output, instname, feature_type)
            else:
                extract_one_audio(this_path_input, this_path_output, instname, out_summary, feature_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = 'Extract audio features using OpenSMILE. Output files are placed in the upper folder'

    support_feature_type = ','.join(list(config_filenames.keys()))
    parser.add_argument('feature_type',
                        help='Supported feature_type: %s' % support_feature_type,
                        type=str)
    parser.add_argument(
        '-i', '--audio_path',
        default='../../data/emodb/wav/',
        type=str,
        help='dir path of input audio'
    )
    parser.add_argument(
        '-n', '--name_suffix',
        default='',
        type=str,
        help='string added after name of output dir and summary'
    )
    args = parser.parse_args()

    main(args.feature_type, args.audio_path, args.name_suffix)

#!/bin/bash
# hellolzc 2020/02/11
project_dir=`pwd`

# 1. 复制EmoDB数据
# put all wav files in ./data/emodb/wav
# ./data/emodb/datalist.csv show the categories of all wav files.

# 2. 特征
# 使用OpenSMILE工具提取eGeMAPs特征
cd ./opensmile/scripts
rm ../audio_features_egemaps/*.csv

extract_all.sh


# 时长特征 （可跟上一步同时进行）
# cd ${project_dir}/code
# # use praat to calculate speech rate
# ~/toolkit/praat_nogui --run ./praat_speech_rate.txt -25 2 0.3 no ../data/slice/ > ../data/speechrate.csv


# 合并特征
# cd ${project_dir}/speechemotion/mlcode
# python merge_feature.py 'all'

# 3. 打开Jupyter, 划分数据，训练ML模型
# notebook/ML*.ipynb
# 训练深度学习模型
# notebook/DL*.ipynb
#!/bin/bash
# hellolzc 2020/02/11
project_dir=`pwd`

# 1. 复制数据
# put all wav files in ./data/wav
# ./data/datalist.csv show the categories of all wav files.

# 2. 特征
# 使用OpenSMILE工具提取eGeMAPs特征
cd ../opensmile/scripts
rm ../audio_features_egemaps/*.csv

extract_all.sh


# 时长特征 （可跟上一步同时进行）
cd ${project_dir}/code
# use praat to calculate speech rate
~/toolkit/praat_nogui --run ./praat_speech_rate.txt -25 2 0.3 no ../data/slice/ > ../data/speechrate.csv


# 合并特征
cd ${project_dir}/code
python merge_feature.py 'all'

# 3. 训练模型
# code/ML*.ipynb
cd ../network
./slice_wav_feature.py 5
# code/DL*.ipynb
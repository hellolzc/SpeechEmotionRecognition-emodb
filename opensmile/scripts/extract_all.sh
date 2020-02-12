#!/bin/bash

# 提取eGeMAPs特征统计量 提取其他音频特征

./extract_is09etc_features.py CPE16   -i /home/zhaoci/Emotion/emodb/data/wav/ # IS13
./extract_is09etc_features.py egemaps -i /home/zhaoci/Emotion/emodb/data/wav/
./extract_is09etc_features.py IS09    -i /home/zhaoci/Emotion/emodb/data/wav/
./extract_is09etc_features.py IS10    -i /home/zhaoci/Emotion/emodb/data/wav/
./extract_is09etc_features.py IS11    -i /home/zhaoci/Emotion/emodb/data/wav/
./extract_is09etc_features.py IS12    -i /home/zhaoci/Emotion/emodb/data/wav/



./convert_stat_format.py     '../func_IS09_.csv'     '../../fusion/acoustic_IS09.csv'
./convert_stat_format.py     '../func_IS10_.csv'     '../../fusion/acoustic_IS10.csv'
./convert_stat_format.py     '../func_IS11_.csv'     '../../fusion/acoustic_IS11.csv'
./convert_stat_format.py     '../func_IS12_.csv'     '../../fusion/acoustic_IS12.csv'
./convert_stat_format.py    '../func_CPE16_.csv'    '../../fusion/acoustic_CPE16.csv'
./convert_stat_format.py  '../func_egemaps_.csv'  '../../fusion/acoustic_egemaps.csv'


:<<BLOCK

rm ../audio_features_CPE16_extract_b/*.csv
./extract_is09etc_features.py CPE16   -i /home/zhaoci/ADisease/shanghai_pictalking/data/speech_extract_b/ -n extract_b # IS13
./convert_stat_format.py  '../func_CPE16_extract_b.csv'  '../../fusion/acoustic_CPE16_extract_b.csv'

BLOCK

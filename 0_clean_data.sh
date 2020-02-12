#!/bin/bash
# hellolzc 2020/02/11
project_dir=`pwd`

cd ${project_dir}

rm ./data/high-pass -r

rm ./data/png -r
rm ./data/*.hdf5

rm ./data/speechrate.csv

rm ./fusion/*.csv

rm ./list/result/*
rm ./list/split/*
rm ./opensmile/audio_features_*/ -r

rm ./opensmile/func_*.csv

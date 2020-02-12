#!/usr/bin/env python
# -*- coding: utf-8 -*-
# hellolzc 20180823
""" 将opensmile输出格式转换成项目中特征通用格式
"""

import numpy as np
import pandas as pd
import os
import re
import argparse


def convert_file_format(infile, out_file):
    data_frame = pd.read_csv(open(infile, encoding='utf-8'), sep=';')
    df_droped = data_frame.drop(columns=['frameTime'])
    for line_no in range(len(df_droped)):
        # remove ' '
        name = df_droped.iloc[line_no, 0]
        df_droped.iloc[line_no, 0] = name[1:-1]

    df_droped.rename(columns={'name': 'uuid'}, inplace=True)
    df_droped.to_csv(out_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = 'Convert openSMILE format to ordinary csv format'
    parser.add_argument('in_file', type=str,
                        help="eg. '../func_egemaps.csv'")
    parser.add_argument('out_file', type=str,
                        help="eg. '../../fusion/egemaps.csv'")
    # in_file = '../func_egemaps.csv'
    # out_file = '../../fusion/egemaps.csv'
    args = parser.parse_args()
    convert_file_format(args.in_file, args.out_file)
    #main()

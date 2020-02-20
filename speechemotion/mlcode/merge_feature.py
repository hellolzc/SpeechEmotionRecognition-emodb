#!/usr/bin/env python
import pandas as pd  # 数据分析
import numpy as np  # 科学计算
import sys
import os


####################  用于合并数据的部分  ########################

DROP_COLUMNS = []  # 'no', 

def check_filepath_list(filepath_list):
    new_filepath_list = []
    for fp in filepath_list:
        if os.path.exists(fp):
            new_filepath_list.append(fp)
        else:
            print('[WARN] %s does not exist!' % fp)
    return new_filepath_list

def merge_all(filepath_list, out_file_path, prefix_list=None):
    ''' 合并特征数据，根据uuid（是文件名也是被试的标识名）将多个特征文件合并
    类似于数据库表的JOIN操作
    参数:
        filepath_list : 所有需要合并的特征文件路径列表
        out_file_path : 合并后的特征文件路径
        prefix_list : 用于给一群特征加前缀
    '''
    filepath_list = check_filepath_list(filepath_list)
    print(filepath_list)

    df0 = pd.read_csv(filepath_list[0], encoding='utf-8')
    if prefix_list is not None:
        prefix_i = prefix_list[0]
        if prefix_i is not None:
            df0.columns = [prefix_i+x if x!='uuid' else x for x in df0.columns]

    for indx in range(1, len(filepath_list)):
        df1 = pd.read_csv(filepath_list[indx], encoding='utf-8')
        if prefix_list is not None:
            prefix_i = prefix_list[indx]
            if prefix_i is not None:
                df1.columns = [prefix_i+x if x!='uuid' else x for x in df1.columns]

        if indx == 1:
            df0 = pd.merge(df0, df1, how='inner', on='uuid')
        else:
            df0 = pd.merge(df0, df1, how='left', on='uuid')

    print(df0.head())
    print('Shape:', df0.shape)
    # 保存合并的数据
    # df0 = df0.drop(columns=DROP_COLUMNS)
    df0.set_index(keys='uuid', drop=True, inplace=True, verify_integrity=True)
    df0.to_csv(out_file_path)
    print('记得检查数据！')



if __name__ == '__main__':
    USAGE_STR = "usage: merge_duration list/duration/egemaps/linguistic/all"
    args = sys.argv
    if len(args) != 2:
        print(USAGE_STR)
        exit(0)
    choose = args[1]
    # path
    proj_root_path = '../../'

    csv_label_path = proj_root_path + 'data/datalist.csv'
    duration_fp = proj_root_path + 'fusion/duration.csv'
    egemaps_fp = proj_root_path + 'fusion/acoustic_egemaps_extract_b_norm.csv'
    linguistic_fp = proj_root_path + 'fusion/linguistic.csv'


    annatation_list_file = proj_root_path + 'list/namelist.txt'
    list_dir = proj_root_path + 'list'

    if choose == 'all':
        # 合并不同的特征集
        allfeature_out_fp = proj_root_path + 'fusion/merged.csv'
        merge_all([csv_label_path, duration_fp, linguistic_fp, egemaps_fp],
                  allfeature_out_fp)
    else:
        print(USAGE_STR)

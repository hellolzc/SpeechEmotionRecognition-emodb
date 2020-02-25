#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 刘朝辞 20190924

import sys
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from speechemotion.mlcode.data_manager import DataSet

DEBUG = False
def set_debug_print(debug_print):
    """set DEBUG
    """
    global DEBUG
    DEBUG = debug_print


class DLDataSet(DataSet):
    """ 管理深度学习数据集
    self.data_file_path : 原始数据文件的位置
    self.label_se : 保存标签信息
    """
    def __init__(self, hdf5_file_path, label_file_path, class_num):
        """ hdf5_file_path: 对应的hdf5文件提供每个sample的特征矩阵
            label_file_path: csv文件，第一列必须为index_col, 必须有名为label列, 
                            且索引和标签与hdf5文件中数据一一对应
        """
        super(DLDataSet, self).__init__()
        self.data_file_path = hdf5_file_path  # 数据文件的位置
        print('HDF5 File: %s' % hdf5_file_path)
        label_df = pd.read_csv(label_file_path, encoding='utf-8', index_col=0)
        self.label_se = label_df['label'].copy()  # 标签信息
        self.data_length = None  # 一个utterace的长度
        self.class_num = class_num

    def set_data_length(self, max_length):
        """设置截断或补齐的长度位置"""
        self.data_length = max_length

    def describe_data(self):
        """用来检查hdf5文件中存放的数据的shape信息
        返回一个numpy array, 包含所有数据的shape
        """
        with h5py.File(self.data_file_path, "r") as feat_clps:
            shape_list = []
            for key in feat_clps:
                data_shape_i = feat_clps[key].shape  # np.array(feat_clps[key]).shape
                shape_list.append(data_shape_i)
            print('data num:', len(shape_list))
            shape_mat = np.array(shape_list)
            for dim_no in range(shape_mat.shape[1]):
                col_i = shape_mat[:,dim_no]
                print('Dim %d mean:%d, min:%d, max:%d, std:%f' % \
                    (dim_no, col_i.mean(), col_i.min(), col_i.max(), col_i.std()))
            return shape_mat

    def get_input_shape(self):
        """返回要训练的数据样本维度信息"""
        with h5py.File(self.data_file_path, "r") as feat_clps:
            # firstkey = list(feat_clps.keys())[0]
            for key in feat_clps:
                firstkey = key
                break
            firstdata_shape = feat_clps[firstkey].shape # np.array(feat_clps[firstkey]).shape
            print('First data shape:', firstdata_shape)

        data_shape = list(firstdata_shape)
        if self.data_length is None:
            print('Use First data shape')
            return data_shape
        else:
            data_shape[0] = self.data_length
            print('Length is set to %d, so shape is' % self.data_length, data_shape)
            return data_shape

    @staticmethod
    def _fit_X_scaler(data_shape, feat_clps, index_clp_id):
        """处理X的截断和标准化问题
        计算X的均值和标准差以便normlize, 返回StandardScaler对象，
        """
        max_length = data_shape[0]
        # 截断
        sc = StandardScaler()
        for _, clp_id in enumerate(index_clp_id):
            data_i_truncated = feat_clps[clp_id][:max_length, :]
            # 拟合数据集X_train，返回存下来拟合参数
            sc.partial_fit(data_i_truncated)
        return sc

    @staticmethod
    def _get_X_scaled(data_shape, feat_clps, index_clp_id, scaler=None):
        """处理X的截断和标准化问题
        使用计算好的X的均值和标准差进行normlize
        scaler = None 则不标准化
        """
        max_length = data_shape[0]
        # 标准化之后进行 补0或截断
        # X_train = np.row_stack([np.array(feat_clps[clp_id])[None] for clp_id in train_index_clp_id])
        X_train = np.zeros([len(index_clp_id), *data_shape])
        for indx, clp_id in enumerate(index_clp_id):
            data_i_truncated = feat_clps[clp_id][:max_length, :]  # 截断到最大长度
            actual_length = data_i_truncated.shape[0]
            if scaler is not None:  # normlize
                data_i_truncated = scaler.transform(data_i_truncated)

            if actual_length < max_length:
                X_train[indx, :actual_length, :] = data_i_truncated  # 补0到最大长度
            else:
                X_train[indx, :, :] = data_i_truncated
        return X_train


    def get_data_scaled(self, seed, ith, normlize=True, data_splitter=None):
        """ 将X,Y划分成训练集和测试集并在训练集上做标准化，将参数应用到测试集上
        eg.
            X_train, X_test, Y_train, Y_test = get_data_scaled(1998, 2)
        description:
            X_train.shape = (n_samples, x_len, x_dim)
            X_test.shape = (n_samples, )
        """
        data_splitter = data_splitter or self.data_splitter
        if data_splitter is None:
            raise Exception("Provide a data_splitter to split data\n")

        clps_df = self.label_se
        data_shape = self.get_input_shape()  # Note: data_shape[0] == self.data_length

        train_index, test_index = data_splitter.read_split_file(seed, ith)
        # print("TRAIN:", train_index, "TEST:", test_index)

        # 从文件取数据
        with h5py.File(self.data_file_path, "r") as feat_clps:
            if normlize:
                sc = self._fit_X_scaler(data_shape, feat_clps, train_index)
            else:
                sc = None
            X_train = self._get_X_scaled(data_shape, feat_clps, train_index, scaler=sc)
            X_test = self._get_X_scaled(data_shape, feat_clps, test_index, scaler=sc)
            Y_train = clps_df.loc[train_index].values.squeeze()
            Y_test = clps_df.loc[test_index].values.squeeze()

        # print('X shape:', X_train.shape, X_test.shape)
        # print('Y shape:', Y_train.shape, Y_test.shape)
        info_dict = {
            'train_index': train_index,
            'test_index': test_index,
        }

        return X_train, X_test, Y_train, Y_test, info_dict





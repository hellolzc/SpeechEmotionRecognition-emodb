import pandas as pd
import numpy as np
import random
import json
import os
import sys

from sklearn.preprocessing import StandardScaler

DEBUG = False
def set_debug_print(debug_print):
    """set DEBUG
    """
    global DEBUG
    DEBUG = debug_print


class DataSet():
    """
    DataSet is the abstract class which determines how a dataset should be.
    Any dataset inheriting this class should do the following.

    1.  Should implement the following abstract methods `get_data_scaled`,
        `assign_data_splitter`. These methods are used by PipelineCV.

    Attributes:
        class_num (str): path to save the model.
    """
    def __init__(self):
        self.class_num = None       # 这个和class_namelist选择相关
        self.data_splitter = None

    def assign_data_splitter(self, data_splitter):
        self.data_splitter = data_splitter

    def get_data_scaled(self, seed, ith, scale=True, data_splitter=None):
        raise NotImplementedError()

class MLDataSet(DataSet):
    """ 集中处理数据集筛选和特征工程问题
    self.data_ori : 保存从文件读取的原始数据
    self.df : 保存处理后的数据集
    """
    def __init__(self, file_path):
        super(MLDataSet, self).__init__()
        self.file_path = file_path  # 数据文件的位置

        print('Read File: %s' % file_path)
        # 这里索引列用第一列，第一列应该是uuid
        data_ori = pd.read_csv(self.file_path, encoding='utf-8', index_col=0)

        self.data_ori = data_ori     # 从文件读取的原始数据
        self.df = self.data_ori.copy()  # 处理后的数据集
        self.X_df = None  # X即特征属性值
        self.Y_se = None  # y即label结果

    def reset_df(self):
        """重置 self.df, 取消所有 feature_engineering 步骤"""
        self.df = self.data_ori.copy()

    def feature_engineering(self, class_col_name='emotion_en', class_namelist=None,
                            drop_cols=None):
        """
        class_col_name : csv文本中用来存储label的列名
        class_namelist : label列中要保留的类别列表, None表示保留所有类别
        self.data_ori : 保存从文件读取的原始数据
        self.df : 保存处理后的数据集
        """
        print('\nFeature Engineering...')
        df = self.df
        # 筛选
        if class_namelist is not None:
            df = self.filter_by_class(df, class_col_name, class_namelist)
        if class_namelist is None:
            class_namelist = list(df[class_col_name].unique())
        # add label col
        df = self.map_class_to_label(df, class_col_name, class_namelist)
        self.class_num = len(class_namelist)  # 自动计算class_num

        print('\nAfter selection:')
        self.report_value_num(df, class_col_name)

        if drop_cols is not None:
            df.drop(drop_cols, axis=1, inplace=True)

        # Drop nan data in label_group
        df = self.drop_nan_row(df, 'label')

        self.find_duplicate_value(df.index)
        self.check_nan_value(df)

        # print('Attention: nan <= 0')
        # df = df.fillna(0)

        print('\nAfter FE:')
        self.report_value_num(df, 'label')
        self.df = df

    def write_tmp_df(self, filename):
        """特征工程后将处理后的数据存储下来"""
        self.df.to_csv(filename)

    # 用正则取出我们要的属性值
    # 'label.*|age|total_duration|sex_.*|.*_speak_num'
    # train_df.drop(['age','education', 'sex_F', 'sex_M',], axis=1, inplace=True)
    def feature_filter(self, feature_items=None, feature_regex=None, label_col_name='label'):
        """ 取出我们要的属性值
        feature_items : 选择的特征集list
        feature_regex : 选择的特征集用的正则, 和feature_items至少一个不是None
        label_group   : 要预测的标签列名
        """
        print('Prepare X y:')
        # 取出需要的属性值
        if feature_items is not None:
            train_df = self.df.filter(items=feature_items)
        else:  # 使用正则表达式 eg. '^IS09_*'
            assert feature_regex is not None
            train_df = self.df.filter(regex=feature_regex)

        label_se = self.df[label_col_name]

        self.X_df = train_df
        self.Y_se = label_se
        print('X.shape: ', train_df.shape, 'y.shape: ', label_se.shape)
        print(train_df.columns)

    def get_XY(self, return_matrix=False):
        """ 取出我们要的属性值
        return_matrix : 为真则返回矩阵，否则返回DataFrame
        return: X, y
        """
        X = self.X_df
        y = self.Y_se
        if return_matrix:
            X = X.values
            y = y.values.squeeze()
        return X, y


    @staticmethod
    def filter_by_class(df, col_name, class_namelist):
        """ 对数据进行筛选,保留要分的类 """
        # 宽松限制
        print('Kepted class names:', class_namelist)
        mask = df[col_name].isin(class_namelist)
        df = df[mask].copy()
        return df

    @staticmethod
    def map_class_to_label(df, class_col_name, class_namelist, label_col_name='label'):
        map_dict = {}
        for indx, item in enumerate(class_namelist):
            map_dict[item] = indx
        print('Label Map:', map_dict)
        df[label_col_name] = df[class_col_name].map(map_dict)
        return df

    @staticmethod
    def report_value_num(df, col_name):
        """
        检查指定列各个值得分布
        :param df: 要检查的DataFrame
        :param col_name: 要检查的列
        :return: 不同值得数量
        """
        print('总数：%s' % len(df))
        unique_values = df.loc[:, col_name].unique()
        unique_values.sort()
        for label_value in unique_values:
            print(label_value, '数目：%d' % (df.loc[:, col_name] == label_value).sum())
        return len(unique_values)

    @staticmethod
    def drop_nan_row(df, ref_col):
        """ 丢弃ref_col列有nan值的数据行 """
        print('Drop nan data in %s' % ref_col)
        label_df = df.loc[:, ref_col].copy()
        mask = (label_df.isna() | label_df.isnull())
        print('Row index:', df.index[mask])
        mask = ~mask
        return df[mask]

    @staticmethod
    def check_nan_value(df):
        """ 报告nan值出现的位置 """
        print('\nInfo:These data contain NaN ')
        for col_name in df.columns[df.isna().any()]:
            print(col_name, df.index[df[col_name].isna()].values)

    @staticmethod
    def fill_nan_value(df, col_name_list):
        """ 填充nan值为均值 """
        print('\nInfo:fill nan with mean value:')
        for col_name in col_name_list:
            print('Col: %s, Mean: %f' % (col_name, df[col_name].mean()))
            df[col_name] = df[col_name].transform(lambda x: x.fillna(x.mean()))

    @staticmethod
    def find_duplicate_value(series):
        """ 检查是否有重复值 """
        val_cnts = series.value_counts()
        if (val_cnts > 1).any():
            print('\nInfo:These data contain duplicate value')
            print(val_cnts[val_cnts > 1])


    
    def get_data_scaled(self, seed, ith, scale=True, data_splitter=None):
        """ 将X,Y划分成训练集和测试集并在训练集上做标准化，将参数应用到测试集上
        eg.
            X_train, X_test, Y_train, Y_test = get_data_scaled(data_splitter, 1998, 2)
        description:
            X_train.shape = (n_samples, x_dim)
            X_test.shape = (n_samples, )
        """
        data_splitter = data_splitter or self.data_splitter
        if data_splitter is None:
            sys.stderr.write(
                "Provide a data_splitter to split data\n")
            sys.exit(-1)
        X_df, Y_se = self.get_XY()  # Y_se: Series
        assert X_df.shape[0] == Y_se.shape[0]
        train_index, test_index = data_splitter.read_split_file(seed, ith)
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_df.loc[train_index].values, X_df.loc[test_index].values
        Y_train, Y_test = Y_se.loc[train_index].values.squeeze(), Y_se.loc[test_index].values.squeeze()

        if scale == True:
            sc = StandardScaler()   # 初始化一个对象sc去对数据集作变换
            sc.fit(X_train)   # 用对象去拟合数据集X_train，并且存下来拟合参数
            X_train_std = sc.transform(X_train)
            X_test_std = sc.transform(X_test)
            X_train = X_train_std
            X_test = X_test_std
        # print(X_train.shape)
        info_dict = {
            'train_index': train_index,
            'test_index': test_index,
        }
        return X_train, X_test, Y_train, Y_test, info_dict

if __name__ == '__main__':
    pass

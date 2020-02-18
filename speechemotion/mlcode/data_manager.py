import pandas as pd
import numpy as np
import random
import json
import os

from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold

DEBUG = False
def set_debug_print(debug_print):
    """set DEBUG
    """
    global DEBUG
    DEBUG = debug_print


class DataSets():
    """ 集中处理数据集筛选和特征工程问题
    self.data_ori : 保存从文件读取的原始数据
    self.df : 保存处理后的数据集
    """
    def __init__(self, file_path):
        self.file_path = file_path  # 数据文件的位置
        self.class_num = None       # 这个和class_namelist选择相关

        print('Read File: %s' % file_path)
        # 这里索引列用第一列，第一列应该是uuid
        data_ori = pd.read_csv(self.file_path, encoding='utf-8', index_col=0)

        self.data_ori = data_ori     # 从文件读取的原始数据
        self.df = self.data_ori.copy()  # 处理后的数据集

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
    # 'label.*|age|total_duration|sex_.*|.*_speak_num|.*_speak_duration_sum|.*_speak_duration_mean|.*_speak_duration_std'
    # train_df.drop(['age','education', 'sex_F', 'sex_M',], axis=1, inplace=True)
    def get_XY(self, feature_items=None, feature_regex=None, label_col_name='label', return_matrix=False):
        """ 取出我们要的属性值
        feature_items : 选择的特征集list
        feature_regex : 选择的特征集用的正则, 和feature_items至少一个不是None
        label_group   : 要预测的标签列名
        return_matrix : 为真则返回矩阵，否则返回DataFrame
        return: X, Y
        """
        # 取出需要的属性值
        if feature_items is not None:
            train_df = self.df.filter(items=feature_items)
        else:  # 使用正则表达式 eg. '^IS09_*'
            assert feature_regex is not None
            train_df = self.df.filter(regex=feature_regex)

        if DEBUG:
            print(train_df.columns)

        label_df = self.df.loc[:, label_col_name]

        if return_matrix:
            # X即特征属性值
            X = train_df.values
            # y即label结果
            Y = label_df.values.squeeze()
            print(X.shape, Y.shape)
        else:
            X = train_df
            Y = label_df
        print('X.shape: ', X.shape, end='  ')
        return X, Y


    @staticmethod
    def filter_by_class(df, col_name, class_namelist):
        """ 对数据进行筛选,保留要分的类 """
        # 宽松限制
        print('Kepted class names:', class_namelist)
        mask = df[col_name].isin(class_namelist)
        df = df[mask]
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

#####################################################################################
SPLIT_FILE_HOME = os.path.join(os.path.dirname(__file__), '../../list/split/')
class DataLoader(object):
    """处理关于划分数据集的类"""
    def __init__(self, n_splits=10, label_name='label'):
        self.n_splits = n_splits
        self.label_name = label_name

    def split(self, df, seeds):
        # 从这里开始 df里的数据顺序不能改变，否则会对不上号
        # TODO: 支持上采样
        print('shape of data_matrix', df.shape)
        for seed in seeds:
            self._splitCV(df, seed)

    def _splitCV(self, df, seed):
        """ 对df做K折交叉验证，生成分割文件，保存为 ${SPLIT_FILE_HOME}/split_%d.json
        为了防止混乱，TXT中保存的是训练和测试对应的UUID，不是df序号
        TODO: 做GroupKFold， group信息来源于df_sampled[group_col_name], group_col_name='participant_id'
        """
        n_splits = self.n_splits
        df_sampled = df

        X = df_sampled.values
        y = df_sampled[self.label_name].values  # .squeeze()

        kf = StratifiedKFold(n_splits=n_splits, random_state=seed)  # shuffle=True, 
        print(kf)

        ith = 0
        file_lines_dict = {}
        for _, test_index in kf.split(X, y=y):  # , groups=groups
            test_index = [df_sampled.index[val] for val in list(test_index)]
            file_lines_dict[ith] = ','.join(test_index)
            ith += 1

        filename = SPLIT_FILE_HOME + 'split_%d.json' % (seed)
        with open(filename, 'w') as f:
            json.dump(file_lines_dict, f, indent=2, ensure_ascii=False)


    @staticmethod
    def read_split_file(seed, ith):
        """ 指定种子和折编号，读取已保存的划分文件
        """
        filename = SPLIT_FILE_HOME + 'split_%d.json' % (seed)
        with open(filename) as f:
            data_dict = json.load(f, encoding='utf-8')
            # lines = [line.strip() for line in f]
        train_index = []
        for key in data_dict:
            if int(key) == ith:
                test_index = data_dict[key].split(',')
            else:
                train_index.extend(data_dict[key].split(','))
        return train_index, test_index

    # 参考：https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
    @staticmethod
    def read_split_file_innerCV(seed, outer_cv_ith, inner_cv_ith):
        """ 指定种子和折编号，读取已保存的划分文件
        nested cross-validation 比外层少一折
        """
        real_inner_cv_ith = inner_cv_ith
        if inner_cv_ith >= outer_cv_ith:
            real_inner_cv_ith += 1

        filename = SPLIT_FILE_HOME + 'split_%d.json' % (seed)
        with open(filename) as f:
            data_dict = json.load(f, encoding='utf-8')

        train_index = []
        for key in data_dict:
            if int(key) == outer_cv_ith:
                continue
            elif int(key) == real_inner_cv_ith:
                test_index = data_dict[key].split(',')
            else:
                train_index.extend(data_dict[key].split(','))
        return train_index, test_index

    @staticmethod
    def array2CSstr(result_array):
        """convert result 1-d array to comma separated string"""
        result_list = [str(val) for val in list(result_array)]
        return ','.join(result_list)

    @staticmethod
    def CSstr2array(result_str):
        """convert comma separated string to 1-d array"""
        result_list = result_str.split(',')
        return np.array([float(val) for val in result_list])

    @staticmethod
    def save_result(data_dict, seed, suffix):
        """保存预测结果"""
        filename = '../list/result/split_%d_%s.txt' % (seed, suffix)
        with open(filename, 'w') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)

    @staticmethod
    def read_result(seed, suffix):
        """读取预测结果"""
        filename = '../list/result/split_%d_%s.txt' % (seed, suffix)
        with open(filename) as f:
            data_dict = json.load(f, encoding='utf-8')
        return data_dict


if __name__ == '__main__':
    pass

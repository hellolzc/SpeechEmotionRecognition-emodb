import pandas as pd
import numpy as np
import random
import json
import os

from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold

_DEFAULT_SPLIT_FILE_HOME_ = os.path.join(os.path.dirname(__file__), '../../list/split/')
_DEFAULT_RESULT_FILE_HOME_ = os.path.join(os.path.dirname(__file__), '../../list/result/')


class DataSplitter(object):
    """抽象基类, 用于确定接口"""

    def split(self, df, seeds):
        """对df做数据划分，生成分割文件"""
        raise NotImplementedError

    def read_split_file(self, seed, ith):
        """ 指定种子和折编号，读取已保存的划分文件 """
        raise NotImplementedError

    def save_result(self, data_dict, seed, suffix):
        """保存预测结果"""
        raise NotImplementedError

    def read_result(self, seed, suffix):
        """读取预测结果"""
        raise NotImplementedError

    def clean(self, split=True, result=False):
        """清理已存在的分割文件, 方便重新开始"""
        raise NotImplementedError

    @staticmethod
    def array2CSstr(result_array: np.ndarray) -> str:
        """convert result 1-d array to comma separated string"""
        result_list = [str(val) for val in list(result_array)]
        return ','.join(result_list)

    @staticmethod
    def CSstr2array(result_str: str) -> np.ndarray:
        """convert comma separated string to 1-d array"""
        result_list = result_str.split(',')
        return np.array([float(val) for val in result_list])

    @staticmethod
    def _check_directory(dir_path: str):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def _delete_files(dir_path: str, file_ext: str):
        for files in os.listdir(dir_path):
            if files.endswith(file_ext):  # ".json"
                os.remove(os.path.join(dir_path, files))


class KFoldSplitter(DataSplitter):
    """处理关于划分数据集的类"""
    def __init__(self, n_splits=10, label_name='label', split_file_dir=None, result_file_dir=None):
        self.n_splits = n_splits
        self.label_name = label_name
        self.seeds = None
        if split_file_dir is None:
            self.split_file_dir = _DEFAULT_SPLIT_FILE_HOME_
        else:
            self.split_file_dir = split_file_dir
        if result_file_dir is None:
            self.result_file_dir = _DEFAULT_RESULT_FILE_HOME_
        else:
            self.result_file_dir = result_file_dir

    def split(self, df, seeds):
        # 从这里开始 df里的数据顺序不能改变，否则会对不上号
        # TODO: 支持上采样
        print('shape of data_matrix', df.shape)
        self.seeds = seeds
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

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        print(kf)

        ith = 0
        file_lines_dict = {}
        for _, test_index in kf.split(X, y=y):  # , groups=groups
            test_index = [df_sampled.index[val] for val in list(test_index)]
            file_lines_dict[ith] = ','.join(test_index)
            ith += 1

        filename = self.split_file_dir + 'split_%d.json' % (seed)
        with open(filename, 'w') as f:
            json.dump(file_lines_dict, f, indent=2, ensure_ascii=False)

    def read_split_file(self, seed, ith):
        """ 指定种子和折编号，读取已保存的划分文件
        """
        filename = self.split_file_dir + 'split_%d.json' % (seed)
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
    def read_split_file_innerCV(self, seed, outer_cv_ith, inner_cv_ith):
        """ 指定种子和折编号，读取已保存的划分文件
        nested cross-validation 比外层少一折
        """
        real_inner_cv_ith = inner_cv_ith
        if inner_cv_ith >= outer_cv_ith:
            real_inner_cv_ith += 1

        filename = self.split_file_dir + 'split_%d.json' % (seed)
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

    def save_result(self, data_dict, seed, suffix):
        """保存预测结果"""
        filename = os.path.join(self.result_file_dir, 'split_%d_%s.txt' % (seed, suffix))
        with open(filename, 'w') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)

    def read_result(self, seed, suffix):
        """读取预测结果"""
        filename = os.path.join(self.result_file_dir, 'split_%d_%s.txt' % (seed, suffix))
        with open(filename) as f:
            data_dict = json.load(f, encoding='utf-8')
        return data_dict

    def clean(self, split=True, result=False):
        """清理已存在的分割文件, 方便重新开始"""
        self._check_directory(self.result_file_dir)
        self._check_directory(self.split_file_dir)
        # 删除所有的json文件
        if split:
            self._delete_files(self.split_file_dir, '.json')
        if result:
            self._delete_files(self.result_file_dir, '.json')



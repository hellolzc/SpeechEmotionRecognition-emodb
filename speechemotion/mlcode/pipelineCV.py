""" PipelineCV Miscellaneous
"""
import os
import numpy as np
import pandas as pd
import time

import sklearn
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

from speechemotion.mlcode.helper_functions import accuracy, precision_recall_f1score
from speechemotion.mlcode.data_manager import DataSets, DataLoader
from speechemotion.dlcode.dl_data_manager import DLDataSets
from speechemotion.dlcode.nn_model import NN_MODEL


class PipelineCV():
    def __init__(self, model_base, dataset, n_splits=10, exp_name='untitled'):
        self.model_base = model_base  # sklearn classifier
        self.n_splits = n_splits
        self.exp_name = exp_name
        self.class_num = dataset.class_num
        self.dataset = dataset  # DataSets or DLDataSets

    def get_data_scaled(self, seed, ith, scale=True):
        """ 将X,Y划分成训练集和测试集并在训练集上做标准化，将参数应用到测试集上
        """
        dataset = self.dataset
        if isinstance(dataset, DataSets):
            X_df, Y_se = dataset.get_XY()  # Y_se: Series
            assert X_df.shape[0] == Y_se.shape[0]
            train_index, test_index = DataLoader.read_split_file(seed, ith)
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
        elif isinstance(dataset, DLDataSets):
            return dataset.get_data_scaled(seed, ith)


    def get_a_clone_model(self):
        """在model_base基础上重建一个模型"""
        model_base = self.model_base
        # 重置模型
        if isinstance(model_base, NN_MODEL):
            model = model_base.clone()
        else:
            model = sklearn.clone(model_base)
        return model

    def evaluate(self, train_pred, train_true, test_pred, test_true):
        """返回预测结果统计量
        """
        class_num = self.class_num
        train_accuracy = accuracy(train_true, train_pred)
        train_precision, train_recall, train_f1score = \
            precision_recall_f1score(train_true, train_pred, class_num=class_num)

        test_accuracy = accuracy(test_true, test_pred)
        test_precision, test_recall, test_f1score = \
            precision_recall_f1score(test_true, test_pred, class_num=class_num)
        conf_mx = confusion_matrix(test_true, test_pred)
        # 记录结果
        fold_i_stat = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1score': train_f1score,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1score': test_f1score,
            'conf_mx': conf_mx
        }
        return fold_i_stat

    def log_result(self, train_pred, train_true, test_pred, test_true, model=None):
        """save result for future analysising"""
        fold_i_result = {
            'train_pred': DataLoader.array2CSstr(train_pred),
            'train_true': DataLoader.array2CSstr(train_true),
            'test_pred': DataLoader.array2CSstr(test_pred),
            'test_true': DataLoader.array2CSstr(test_true),
        }
        # 记录模型参数
        if isinstance(model, linear_model.LogisticRegression):
            fold_i_result['lr_coef'] = model.coef_.tolist()
        if isinstance(model, linear_model.LogisticRegressionCV):
            fold_i_result['lr_C'] = list(model.C_)
        if isinstance(model, GridSearchCV):
            fold_i_result['grid_best_params'] = model.best_params_
        return fold_i_result


    def one_fold_in_CV(self, X_train, X_test, Y_train, Y_test):
        """ 十折交叉验证中的一折
        返回：fold_i_result, fold_i_stat, conf_mx
        """
        model = self.get_a_clone_model()
        # 训练 预测
        if isinstance(model, NN_MODEL):
            model.fit(X_train, Y_train, validation_data=(X_test, Y_test))
            model.show_history()
        else:
            model.fit(X_train, Y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        # 记录结果
        fold_i_result = self.log_result(train_pred, Y_train, test_pred, Y_test, model)
        # 记录评估
        fold_i_stat = self.evaluate(train_pred, Y_train, test_pred, Y_test)

        return fold_i_result, fold_i_stat


    def run_pipeline(self, seed):
        """ 一次交叉验证，对X，Y训练模型，返回结果的字典，包含DataFrame
        预测的标签由数据集中的label_group指定
        """
        fold_metrics = pd.DataFrame(columns=['train_acc', 'test_acc',
                                        'train_precision', 'train_recall', 'train_f1score',
                                        'test_precision', 'test_recall', 'test_f1score'])

        k_fold_results = {}
        for ith in range(self.n_splits):

            X_train, X_test, Y_train, Y_test, _ = self.get_data_scaled(seed, ith)
            fold_i_result, fold_i_stat = self.one_fold_in_CV(X_train, X_test, Y_train, Y_test)
            print('Seed: %d, Fold: %d' % (seed, ith), end='\t')
            print('Acc: Train %f, Test %f' % (fold_i_stat['train_accuracy'], fold_i_stat['test_accuracy']))

            conf_mx_i = fold_i_stat['conf_mx']
            if ith == 0:
                conf_mx = conf_mx_i
            else:
                conf_mx += conf_mx_i

            k_fold_results[ith] = fold_i_result
            fold_metrics.loc[ith] = [
                fold_i_stat['train_accuracy'],
                fold_i_stat['test_accuracy'],
                fold_i_stat['train_precision'],
                fold_i_stat['train_recall'],
                fold_i_stat['train_f1score'],
                fold_i_stat['test_precision'],
                fold_i_stat['test_recall'],
                fold_i_stat['test_f1score']
            ]

        print('Seed:%d'%seed, X_train.shape, X_test.shape, end='\t')
        print(fold_metrics['test_acc'].mean())
        DataLoader.save_result(k_fold_results, seed, self.exp_name)

        result = {
            'fold_metrics': fold_metrics,
            'conf_mx': conf_mx
        }
        return result





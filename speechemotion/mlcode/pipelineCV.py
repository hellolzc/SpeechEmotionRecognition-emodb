""" PipelineCV Miscellaneous
"""
import os
import numpy as np
import pandas as pd
import time

import sklearn
from sklearn.metrics import confusion_matrix
from speechemotion.mlcode.model_base_class import Model
from speechemotion.mlcode.helper_functions import accuracy, precision_recall_f1score
from speechemotion.mlcode.data_manager import DataSet
from speechemotion.mlcode.data_splitter import KFoldSplitter


class PipelineCV():
    def __init__(self, model_base : Model, dataset : DataSet, data_splitter, n_splits=10, exp_name='untitled'):
        self.model_base = model_base  # Model
        self.n_splits = n_splits
        self.exp_name = exp_name
        self.class_num = dataset.class_num
        self.dataset = dataset  # DataSets or DLDataSets
        self.data_splitter = data_splitter

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

    def log_result(self, train_pred, train_true, test_pred, test_true, model : Model =None):
        """save result for future analysising"""
        fold_i_result = {
            'train_pred': KFoldSplitter.array2CSstr(train_pred),
            'train_true': KFoldSplitter.array2CSstr(train_true),
            'test_pred': KFoldSplitter.array2CSstr(test_pred),
            'test_true': KFoldSplitter.array2CSstr(test_true),
        }
        # 记录模型参数
        model_params = model.log_parameters()
        fold_i_result = dict(fold_i_result, **model_params)
        return fold_i_result


    def one_fold_in_CV(self, seed, ith):
        """ 十折交叉验证中的一折
        返回：fold_i_result, fold_i_stat, conf_mx
        """
        X_train, X_test, Y_train, Y_test, _ = self.dataset.get_data_scaled(seed, ith, data_splitter=self.data_splitter)

        model = self.model_base.clone_model()
        # 训练 预测
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test))

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

            fold_i_result, fold_i_stat = self.one_fold_in_CV(seed, ith)
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

        print('Seed:%d'%seed, '\tTest Acc:', fold_metrics['test_acc'].mean())
        self.data_splitter.save_result(k_fold_results, seed, self.exp_name)

        result = {
            'fold_metrics': fold_metrics,
            'conf_mx': conf_mx
        }
        return result





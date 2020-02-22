#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from speechemotion.mlcode.helper_functions import *


# # Prepare Features

# In[ ]:


from speechemotion.mlcode.merge_feature import merge_all

proj_root_path = '../'

csv_label_path = proj_root_path + 'data/emodb/datalist.csv'

# ['acoustic_CPE16.csv', 
# 'acoustic_IS09.csv',
# 'acoustic_IS10.csv',
# 'acoustic_IS11.csv', 
# 'acoustic_IS12.csv',
# 'acoustic_egemaps.csv',
# ]

feature_choose = 'CPE16'
acoustic_fp = proj_root_path + 'fusion/acoustic_%s.csv' % feature_choose

list_dir = proj_root_path + 'list'

# 合并不同的特征集
allfeature_out_fp = proj_root_path + 'fusion/tmp_merged.csv'

merge_all([csv_label_path, acoustic_fp], allfeature_out_fp,
         [None, feature_choose+'_'])


# In[ ]:


from speechemotion.mlcode.data_manager import DataSets

CLASS_COL_NAME = 'emotion_en'
CLASS_NAMES=("neutral", "angry", "happy", "sad", "afraid", "boring", "disgust")

file_path = '../fusion/tmp_merged.csv'
ser_datasets = DataSets(file_path)


# In[ ]:


ser_datasets.feature_engineering(class_col_name=CLASS_COL_NAME, class_namelist=CLASS_NAMES, drop_cols=None)
ser_datasets.write_tmp_df('../fusion/temp_data_after_FE.csv')

ser_datasets.feature_filter(feature_regex='^%s_*' % feature_choose)

print()
ser_datasets.df.iloc[:, 0:16].describe()


# # Dataset Split

# In[ ]:


import os
from speechemotion.mlcode.data_manager import DataLoader


# In[ ]:


# 重新划分数据
# 从这里开始 df里的数据顺序不能改变，否则会对不上号
data_loader = DataLoader(n_splits=10, label_name='label')  # 

os.system('rm ../list/split/*.json')
os.system('rm ../list/result/*.json')
data_loader.split(ser_datasets.df, seeds=list(range(1998, 2018)))


# # Train & Test

# In[ ]:


import sklearn

from sklearn import linear_model, decomposition, datasets
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif


# from speechemotion.mlcode.roc import auc_classif
from speechemotion.mlcode.main_exp import main_experiment


# In[ ]:


# %pdb
#################################################

# gamma: 当kernel为‘rbf’, ‘poly’或‘sigmoid’时的kernel系数。
# 如果不设置，默认为 ‘auto’ ，此时，kernel系数设置为：1/n_features
# C: 误差项的惩罚参数，一般取值为10的n次幂，如10的-5次幂，10的-4次幂。。。。10的0次幂，10，1000,1000，在python中可以使用pow（10，n） n=-5~inf
#     C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，这样会出现训练集测试时准确率很高，但泛化能力弱。
#     C值小，对误分类的惩罚减小，容错能力增强，泛化能力较强。

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]}]

svc_model =sklearn.svm.SVC()

# # parameters = {'kernel':('linear', 'rbf'), 'C':[1, 2, 4], 'gamma':[0.125, 0.25, 0.5 ,1, 2, 4]}
# model = GridSearchCV(svc_model, tuned_parameters[0], scoring='f1', cv=4)


# fit到RandomForestRegressor之中
# C = [10, 5, 1, 0.7, 0.5, 0.2, 0.1, 0.05, 0.01]
# model = linear_model.LogisticRegression(C=1.0, penalty='l1', solver='liblinear')  # C=0.1, tol=1e-6 lbfgs
# model = sklearn.svm.SVC(kernel='linear', C=0.1)
# model = RandomForestClassifier(n_estimators=50, max_leaf_nodes=5) # , max_features=5, max_features=10, max_depth=None
# model = linear_model.LogisticRegression(C=0.1, penalty='l1', solver='liblinear')
# model = KNeighborsClassifier(n_neighbors=10)
# model = linear_model.LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], penalty='l1', solver='liblinear', cv=4, max_iter=1000)  # , 100

# model = MLPClassifier(solver='sgd', hidden_layer_sizes = (100,30), random_state = 1, max_iter=500)

model_list = {
#     'svm1':sklearn.svm.SVC(kernel='rbf', gamma=1e-4, C=10),
    'svm2':sklearn.svm.SVC(kernel='linear', C=0.1),
#     'svm1':GridSearchCV(svc_model, tuned_parameters[0], scoring='recall_macro', cv=4, n_jobs=10),
#     'svm2':GridSearchCV(svc_model, tuned_parameters[1], scoring='recall_macro', cv=4, n_jobs=10),
#     'lr1':linear_model.LogisticRegressionCV(Cs=[0.01, 0.1, 1], penalty='l1', solver='liblinear', cv=4),
#     'rf1':RandomForestClassifier(n_estimators=50, max_leaf_nodes=5)
}

# lsvc = # LinearSVC(C=0.01, penalty="l1", dual=False)
# l1norm_model = linear_model.LogisticRegression(C=0.1, penalty='l1')
# pca = decomposition.PCA(10)

# model = Pipeline([
#     # SBS(model, 15)
#     # SelectKBest(mutual_info_classif, k=5)   # auc_classif
#     # SelectKBest(auc_classif, k=10)
#     # SelectFromModel(model, threshold="0.1*mean")
#   ('feature_selection', SelectKBest(auc_classif, k=3) ),
#   ('classification', model)
# ])
# model = Pipeline(steps=[('pca', pca), ('clf', model)])

# duration egemaps linguistic score demographics    doctor all propose select test


# In[ ]:


get_ipython().system('pwd')
get_ipython().system('mkdir -p log')


# In[ ]:


from speechemotion.mlcode.main_exp import save_exp_log
for key  in model_list:
    model = model_list[key]
    result = main_experiment(ser_datasets, model)

    conf_mx = result['conf_mx_sum']
    report = result['report']

    show_confusion_matrix(conf_mx, save_pic_path='./log/cconf_mx.png')

    # result_df_stat # .describe()
    # UAR
    print(report)
    save_exp_log({
        'Memo': '|'.join(CLASS_NAMES),
        'Data': 'File: %s, Shape:%s\n' % (acoustic_fp, str(ser_datasets.df.shape)) + \
                '     feature_group: %s' % (feature_choose),
        'Model': '\n%s\n' % str(model),
        'Report': report,
        'Confusion Matrix': '\n%s\n' % repr(result['conf_mx_sum']),
        'CV_result_detail': result['cv_metrics_stat'].describe()
    }, name_str=feature_choose )


# # Analysis

# In[ ]:


# ## Plot learning curve
from speechemotion.mlcode.pipelineCV import PipelineCV

#     'svm1':sklearn.svm.SVC(kernel='rbf', gamma=1e-4, C=10),
#     'svm2':sklearn.svm.SVC(kernel='linear', C=0.1),
# model = linear_model.LogisticRegression(C=0.1, penalty='l1')
model = sklearn.svm.SVC(kernel='rbf', gamma=1e-4, C=10)
import warnings
warnings.filterwarnings("ignore", message="The default value of cv will change from 3 to 5 in version 0.22.")
print(feature_group)
X, Y = ser_datasets.get_XY(feature_regex=feature_group) # , return_matrix=True
print(X.columns)
# randomforest和logisticRegression已知对变量数量级和变化范围不敏感

X_train, X_test, Y_train, Y_test = PipelineCV.get_data_scaled(X, Y, 2000, 3)
print(model)
print(X.shape)  # _train
plot_learning_curve(model, "Learning Curve", X_train, Y_train, train_sizes=np.linspace(0.2, 1.0, 5), ylim=(0.5,1.0))
plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot') # ggplot  seaborn-poster
# basic handling
import os
import glob
import pickle
import h5py
import numpy as np
import sklearn
# audio
import librosa
import librosa.display
import IPython.display

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

print(os.getcwd())


# In[ ]:


import sys
sys.path.append('..')
from speechemotion.mlcode.helper_functions import *


# In[ ]:


from speechemotion.mlcode.merge_feature import merge_all

proj_root_path = '../'

csv_label_path = proj_root_path + 'data/emodb/datalist.csv'

# ['acoustic_CPE16.csv', 'acoustic_CPE16_lsa.csv',
# 'acoustic_IS09.csv',  'acoustic_IS09_lsa.csv',
# 'acoustic_IS10.csv',  'acoustic_IS10_lsa.csv',
# 'acoustic_IS11.csv',  'acoustic_IS11_lsa.csv', 
# 'acoustic_IS12.csv', , 'acoustic_IS12_lsa.csv',
# 'acoustic_egemaps.csv',  'acoustic_egemaps_lsa.csv',
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

# ad_datasets.df = ad_datasets.drop_nan_row(ad_datasets.df, 'mmse')


# In[ ]:


ser_datasets.feature_engineering(class_col_name=CLASS_COL_NAME, class_namelist=CLASS_NAMES, drop_cols=None)

ser_datasets.feature_filter(feature_regex='^%s_*' % feature_choose)

print()
ser_datasets.df.iloc[:, 0:16].describe()


# #

# In[ ]:


import os
from speechemotion.mlcode.data_manager import DataLoader


# # 

# In[ ]:


get_ipython().system('ls ../data/emodb/')


# In[ ]:


from speechemotion.dlcode.dl_data_manager import DLDataSets


# In[ ]:


_, Y = ser_datasets.get_XY()
Y


# In[ ]:


DL_FILE_PATH = '../data/emodb/acoustic_egemaps.hdf5'
dl_dataset = DLDataSets(Y, DL_FILE_PATH, ser_datasets.class_num)


# In[ ]:


dl_dataset.get_input_shape()


# In[ ]:


shape_stat = dl_dataset.describe_data()


# In[ ]:


import h5py
def length_of_sentences(shape_stat):
    global dl_dataset
    dl_dataset.get_input_shape()
    counts, bins, patch = plt.hist(shape_stat[:, 0])  # , bins=[50 * i for i in range(10)]
    for indx in range(len(counts)):
        plt.text(bins[indx], counts[indx], '%d'%counts[indx])
    plt.title('Length of Sentence')
    plt.show()
    # shape_stat[-40:, 0]

    max_length_value = np.max(shape_stat[:, 0])
    max_value_indexs = np.where(shape_stat[:, 0] == max_length_value)
    print('Max length is:', max_length_value, '\tCorresponding indexs:', max_value_indexs)
#     with h5py.File(DL_FILE_PATH, "r") as feat_clps:
#         print('Clip_id:', list(feat_clps.keys())[max_value_indexs[0][0]])
#         for indx in range(shape_stat.shape[0]):
#             if shape_stat[indx, 0] >= 3000:
#                 print(list(feat_clps.keys())[indx], end=' ')
        
length_of_sentences(shape_stat)


# In[ ]:


X_train, X_test, Y_train, Y_test, info_dict = dl_dataset.get_data_scaled(1998, 2, normlize=True)
# 计算方差和均值不会消耗太多内存，载入数据集X到内存约花费14G空间
print('->  X shape:', X_train.shape, X_test.shape)
print('->  Y shape:', Y_train.shape, Y_test.shape)
print(info_dict.keys())


# In[ ]:


# 测试extract_feature是否正常

def display_feature(x_i, figsize=(10,6), vmin=-10, vmax=10):
    print('x_i shape:', x_i.shape)
    plt.figure(figsize=figsize)
    librosa.display.specshow(x_i[:,:].T, sr=100, hop_length=1, x_axis='time', 
                             cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('Feature')
    plt.show()
    plt.tight_layout()

test_no = 31
x_i = X_train[test_no]
print(info_dict['train_index'][test_no])
display_feature(x_i)
print ('x_i var:', x_i.var(axis=0))
print ('x_i mean:', x_i.mean(axis=0))


# In[ ]:


del X_train, X_test


# In[ ]:


import keras
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
print(keras.__version__)


# In[ ]:


# 没有问题的话就开始搭建模型
from speechemotion.dlcode.nn_model import NN_MODEL, model_factory
import functools

UTT_LENGTH = 500
# dl_dataset.describe_data()
dl_dataset.set_data_length(UTT_LENGTH)


# In[ ]:


# model_creator = functools.partial(model_factory, model_choose='cnn_1')

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution1D, MaxPooling1D

def model_creator(input_shape):
    model = Sequential()
    # default "image_data_format": "channels_last"

    model.add(Convolution1D(128, 3, strides=2, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))

    for filter_num in [128, 128]:
        model.add(Convolution1D(filter_num, 3, strides=2, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(7, activation='softmax'))
    return model


model = NN_MODEL(dl_dataset.get_input_shape(), model_creator=model_creator)
print(model)
# visualize model layout with pydot_ng
model.plot_model()


# In[ ]:


from speechemotion.mlcode.pipelineCV import PipelineCV

pipelineCV = PipelineCV(model, dl_dataset, n_splits=10)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
result = pipelineCV.run_pipeline(2000)


# In[ ]:





# In[ ]:





# In[ ]:





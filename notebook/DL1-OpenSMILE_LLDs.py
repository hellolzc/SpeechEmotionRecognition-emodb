#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib notebook
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


# # Prepare

# In[ ]:


import sys
sys.path.append('..')
from speechemotion.mlcode.helper_functions import *


# In[ ]:


from speechemotion.mlcode.merge_feature import merge_all

proj_root_path = '../'

csv_label_path = proj_root_path + 'data/emodb/datalist.csv'

from speechemotion.mlcode.data_manager import MLDataSet

CLASS_COL_NAME = 'emotion_en'
CLASS_NAMES=("neutral", "angry", "happy", "sad", "afraid", "boring", "disgust")

file_path = '../fusion/tmp_merged.csv'
FE_file_path = '../fusion/temp_data_after_FE.csv'
ser_datasets = MLDataSet(file_path)
ser_datasets.feature_engineering(class_col_name=CLASS_COL_NAME, class_namelist=CLASS_NAMES, drop_cols=None)
ser_datasets.feature_filter(feature_regex='^%s_*' % 'CPE16')
ser_datasets.write_tmp_df(FE_file_path)
print()
ser_datasets.df.iloc[:, 0:16].describe()


# In[ ]:


import os
from speechemotion.mlcode.data_splitter import KFoldSplitter
data_splitter = KFoldSplitter()


# In[ ]:





# # Deep Learning Dataset

# In[ ]:


get_ipython().system('ls ../data/emodb/')
get_ipython().system('ls ../fusion/')


# In[ ]:


from speechemotion.dlcode.dl_data_manager import DLDataSet
DL_FILE_PATH = '../data/emodb/acoustic_emnet.hdf5'
dl_dataset = DLDataSet(DL_FILE_PATH, FE_file_path,len(CLASS_NAMES))


# In[ ]:


dl_dataset.get_input_shape()


# In[ ]:


shape_stat = dl_dataset.describe_data()


# # 

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





# In[ ]:


X_train, X_test, Y_train, Y_test, info_dict = dl_dataset.get_data_scaled(1998, 2, normlize=True, data_splitter=data_splitter)
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


# x_i_x = np.linspace(0, len(x_i)/100.0, num=len(x_i))
# plt.plot(x_i_x, x_i[:, 0], x_i_x, x_i[:, 28]+5)
# plt.show()


# In[ ]:





# In[ ]:


del X_train, X_test


# # DeepLearning Models

# In[ ]:


import keras
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(keras.__version__)


# In[ ]:


# 没有问题的话就开始搭建模型
from speechemotion.dlcode.nn_model import KerasModelAdapter
import functools

UTT_LENGTH = 512
# dl_dataset.describe_data()
dl_dataset.set_data_length(UTT_LENGTH)
print(dl_dataset.get_input_shape())


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LeakyReLU
# from keras.layers import LSTM
from keras.layers import CuDNNLSTM as LSTM
from keras.layers import CuDNNGRU as GRU
from keras.layers import Bidirectional


from keras.regularizers import l1, l2


# In[ ]:


# model_creator = functools.partial(model_factory, model_choose='cnn_1')

# def model_creator(input_shape):
#     model = Sequential()
#     # default "image_data_format": "channels_last"

#     model.add(Conv1D(64, 3, strides=1, input_shape=input_shape))
#     model.add(Activation('relu'))
#     model.add(MaxPooling1D(2))
#     model.add(Dropout(0.5))

#     for filter_num in [64, 128]:
#         model.add(Conv1D(filter_num, 3, strides=1, padding='same'))
#         model.add(Activation('relu'))
#         model.add(MaxPooling1D(2))
#         model.add(Dropout(0.5))

#     model.add(Flatten())
#     model.add(Dense(7, activation='softmax'))
#     return model

# def model_creator2(input_shape):
#     model = Sequential()
#     # default "image_data_format": "channels_last"
#     # assert K.image_data_format() == 'channels_last':

#     model.add(Reshape((*input_shape, 1), input_shape=input_shape))
#     model.add(Conv2D(64, (6,1), strides=1))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(4, 1)))
#     model.add(Dropout(0.5))
#     model.add(Reshape((-1,64*input_shape[1])))

#     for filter_num in [64, 128]:
#         model.add(Conv1D(filter_num, 3, strides=1, padding='same'))
#         model.add(Activation('relu'))
#         model.add(MaxPooling1D(2))
#         model.add(Dropout(0.5))

#     model.add(Flatten())
#     model.add(Dense(7, activation='softmax'))
#     return model


def model_creator3(input_shape):
    model = Sequential()
    # default "image_data_format": "channels_last"
    # assert K.image_data_format() == 'channels_last':

    model.add(Reshape((*input_shape, 1), input_shape=input_shape))
    model.add(Conv2D(8, (6,1), strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Dropout(0.5))
#     model.add(Conv2D(8, (6,1), strides=1, padding='same', activation='relu'))
#     model.add(MaxPooling2D(pool_size=(4, 1)))
#     model.add(Dropout(0.5))
    model.add(Reshape((-1,8*input_shape[1])))


    model.add(Conv1D(64, 3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(7, activation='softmax'))
    return model
    
def EmNet_creator(input_shape):
    model = Sequential()
    # default "image_data_format": "channels_last"
    # assert K.image_data_format() == 'channels_last':

    model.add(Reshape((*input_shape, 1), input_shape=input_shape))
    model.add(Conv2D(64, (6,1), strides=1, padding='same'))  # , kernel_regularizer=l1(0.001)
    model.add(Activation('relu'))
#     model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(4, 1)))

    model.add(Reshape((-1,64*input_shape[1])))
    model.add(Conv1D(128, 3, strides=1, padding='same', kernel_regularizer=l1(0.00001)))  # , kernel_regularizer=l1(0.00001)
    model.add(Activation('relu'))
#     model.add(LeakyReLU())
    model.add(MaxPooling1D(2))

    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(48, return_sequences=True), merge_mode='concat'))  # returns a sequence of vectors  , dropout=0.25
    model.add(Dropout(0.5))  # Attention
    model.add(Bidirectional(LSTM(48), merge_mode='concat'))  # return a single vector  , dropout=0.25
    
#     model.add(Flatten())
    model.add(Dense(7, activation='softmax'))
    return model

hyper_params = {
    'lr':0.001,
    'epochs':150,
    'lr_decay':0.01
#     'gpus':2
}
model = KerasModelAdapter(dl_dataset.get_input_shape(), model_creator=EmNet_creator, **hyper_params)
print(model)
# visualize model layout with pydot_ng
model.plot_model()


# In[ ]:


# from speechemotion.mlcode.pipelineCV import PipelineCV

# pipelineCV = PipelineCV(model, dl_dataset, data_splitter, n_splits=10)
# result = pipelineCV.run_pipeline(2000)
# from speechemotion.mlcode.main_exp import gen_report, save_exp_log
# print(result['conf_mx'])
# gen_report(result['fold_metrics'])


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from speechemotion.mlcode.main_exp import main_experiment

result = main_experiment(dl_dataset, data_splitter, model)

conf_mx = result['conf_mx_sum']
report = result['report']


# result_df_stat # .describe()
# UAR
display(report)


# In[ ]:


from speechemotion.mlcode.main_exp import save_exp_log, gen_report
plt.style.use('ggplot')

# show_confusion_matrix(conf_mx, save_pic_path='./log/cconf_mx.png')
plot_confusion_matrix(conf_mx, classes=CLASS_NAMES, figsize=(7,7))

save_exp_log({
    'Memo': '|'.join(CLASS_NAMES),
    'Data': 'File: %s\n' % (DL_FILE_PATH),
    'Model': '\n%s\n' % str(model),
    'Report': report, # gen_report(result['fold_metrics']),
    'Confusion Matrix': '\n%s\n' % repr(result['conf_mx_sum']),
    'CV_result_detail': result['cv_metrics_stat'].describe()  # fold_metrics
}, name_str='DeepLearning' )


# In[ ]:





# In[ ]:


exit(0)


# In[ ]:





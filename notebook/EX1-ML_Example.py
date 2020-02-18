#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# All code of this notebook are come form <https://github.com/harry-7/speech-emotion-recognition>

# # Common

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import matplotlib.pyplot as plt

import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')

from speechemotion.harry7.utilities import get_data, get_feature_vector_from_mfcc

_DATA_PATH = '../data/emodb/'
_CLASS_LABELS = ("neutral", "angry", "happy", "sad")


def extract_data(flatten):
    data, labels = get_data(_DATA_PATH, class_labels=_CLASS_LABELS,
                            flatten=flatten)
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(
        y_test), len(_CLASS_LABELS)


def get_feature_vector(file_path, flatten):
    return get_feature_vector_from_mfcc(file_path, flatten, mfcc_len=39)


# In[ ]:


get_ipython().system('pwd')


# # ml_example

# In[ ]:


"""
This example demonstrates how to use `NN` model ( any ML model in general) from
`speechemotionrecognition` package
"""

from speechemotion.harry7.mlmodel import NN
from speechemotion.harry7.utilities import get_feature_vector_from_mfcc


def ml_example():
    to_flatten = True
    x_train, x_test, y_train, y_test, _ = extract_data(flatten=to_flatten)
    model = NN()
    print('Starting', model.name)
    model.train(x_train, y_train)
    model.evaluate(x_test, y_test)
    filename = '../data/emodb/wav/09b03Ta.wav'
    print('prediction', model.predict_one(
        get_feature_vector_from_mfcc(filename, flatten=to_flatten)),
          'Actual 3')


if __name__ == "__main__":
    ml_example()


# # cnn_example

# In[ ]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# In[ ]:


"""
This example demonstrates how to use `CNN` model from
`speechemotionrecognition` package
"""

from keras.utils import np_utils

from speechemotion.harry7.dnn import CNN
from speechemotion.harry7.utilities import get_feature_vector_from_mfcc


def cnn_example():
    to_flatten = False
    x_train, x_test, y_train, y_test, num_labels = extract_data(
        flatten=to_flatten)
    y_train = np_utils.to_categorical(y_train)
    y_test_train = np_utils.to_categorical(y_test)
    in_shape = x_train[0].shape
    x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
    model = CNN(input_shape=x_train[0].shape,
                num_classes=num_labels)
    model.train(x_train, y_train, x_test, y_test_train)
    model.evaluate(x_test, y_test)
    filename = '../data/emodb/wav/09b03Ta.wav'
    mfcc = get_feature_vector_from_mfcc(filename, flatten=to_flatten)
    mfcc = mfcc.reshape(mfcc.shape[0], mfcc.shape[1], 1)
    print('prediction', model.predict_one(mfcc), 'Actual 3')
    # print('prediction', model.predict_one(get_feature_vector_from_mfcc(filename, flatten=to_flatten)), 'Actual 3')
    print('CNN Done')


if __name__ == "__main__":
    cnn_example()


# # lstm_example

# In[ ]:


"""
This example demonstrates how to use `LSTM` model from
`speechemotionrecognition` package
"""

from keras.utils import np_utils

from speechemotion.harry7.dnn import LSTM
from speechemotion.harry7.utilities import get_feature_vector_from_mfcc


def lstm_example():
    to_flatten = False
    x_train, x_test, y_train, y_test, num_labels = extract_data(
        flatten=to_flatten)
    y_train = np_utils.to_categorical(y_train)
    y_test_train = np_utils.to_categorical(y_test)
    print('Starting LSTM')
    model = LSTM(input_shape=x_train[0].shape,
                 num_classes=num_labels)
    model.train(x_train, y_train, x_test, y_test_train, n_epochs=50)
    model.evaluate(x_test, y_test)
    filename = '../data/emodb/wav/09b03Ta.wav'
    print('prediction', model.predict_one(
        get_feature_vector_from_mfcc(filename, flatten=to_flatten)),
          'Actual 3')


if __name__ == '__main__':
    lstm_example()


# In[ ]:





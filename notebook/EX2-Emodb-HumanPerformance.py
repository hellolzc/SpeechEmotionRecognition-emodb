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


# In[ ]:


get_ipython().system('ls ../data/emodb')


# In[ ]:


proj_root_path = '../'

csv_label_path = proj_root_path + 'data/emodb/datalist.csv'
acoustic_fp = proj_root_path + 'data/emodb/erkennung-recognition.txt'

list_dir = proj_root_path + 'list'

# 合并不同的特征集
allfeature_out_fp = proj_root_path + 'fusion/human_performance.csv'


# In[ ]:


df0 = pd.read_csv(csv_label_path, encoding='utf-8')
df0.head()


# In[ ]:


df1 = pd.read_csv(acoustic_fp, encoding='utf-8', sep='\t')
df1.head()


# In[ ]:


df1['uuid'] = df1['Satz'].apply(lambda x : x[:-4])
df1['acc1'] = df1['erkannt'].apply(lambda x : int(x.split(',')[0]))
df1['acc2'] = df1['erkannt'].apply(lambda x : int(x.split(',')[1]))
display(df1.head())
df1.describe()


# In[ ]:


df_m = pd.merge(df0, df1, how='inner', on='uuid')

display(df_m.head())
print('Shape:', df0.shape, df1.shape, df_m.shape)
df_m.describe()


# In[ ]:


df_m.groupby('emotion_en')['acc1'].mean()


# The mean accuracy of human to recognize a emotion is 93.5%

# In[ ]:


df3 = df1.loc[df1['acc2']!=0, 'acc2']


# In[ ]:


df3.describe()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib notebook
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from speechemotion.mlcode.helper_functions import *


# In[ ]:


data = pd.read_csv('../data/emodb/datalist.csv')
data


# In[ ]:


#看看各类的情况

def plot_stack_bar(data, col1, col2):
    #  col1 = 'speaker'
    #  col2 = 'emotion_en'
    df_dict = {}
    for key in data[col1].unique():
        df_dict[str(key)] = data.loc[data[col1] == key, col2].value_counts()

    # print(df_dict)
    df = pd.DataFrame(df_dict)
    display(df)

    fig = plt.figure()
    df.plot(kind='bar', stacked=True)
    plt.title(col1 + ' and ' + col2 + " Distribution")
    plt.xlabel(col2) 
    plt.ylabel("Number")

plot_stack_bar(data, 'speaker', 'emotion_en')
plot_stack_bar(data, 'emotion_en', 'speaker')


# In[ ]:





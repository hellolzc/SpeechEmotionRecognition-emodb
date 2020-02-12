#!/usr/bin/env python
# coding: utf-8

# # 测试特征提取代码

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


# %%time
# import smile_example
# infilename  = './in/20180615CJD.wav'
# df = smile_example.get_smile_summary(infilename)
# df


# In[ ]:





# In[ ]:


# import extract_is09etc_features
# from extract_is09etc_features import extract_one_audio

# extract_one_audio()


# In[ ]:


get_ipython().system('/home/zhaoci/toolkit/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract -C /home/zhaoci/toolkit/opensmile-2.3.0/config/ComParE_2016.conf -I /home/zhaoci/ADisease/dementia_bank/data/speech_keep_b/001-0c.wav -nologfile  -instname 001-0c -appendcsvlld 0 -lldcsvoutput ./tmp-0c.csv -csvoutput ./func_tmp.csv')


# # 特征查看器

# In[ ]:


import os
import sklearn
from sklearn.preprocessing import StandardScaler
# audio
import librosa
import librosa.display


# In[ ]:


get_ipython().system("ls '../audio_features_CPE16_extract_b_norm'")


# In[ ]:


filepath = '../audio_features_CPE16_extract_b_norm/20180109邢正金.csv'
# filepath = './tmp-0c.csv'
feature_df = pd.read_csv(open(filepath, encoding='utf-8'), sep=';')
print(feature_df.shape)
print(feature_df.columns)


# In[ ]:


feature_df


# In[ ]:


x_i = feature_df.values[:,2:]
print ('x_i var:', x_i.var(axis=0))
print ('x_i mean:', x_i.mean(axis=0))
sc = StandardScaler()
x_i_sd =  sc.fit_transform(x_i)

print ('x_i var:', x_i_sd.var(axis=0))
print ('x_i mean:', x_i_sd.mean(axis=0))


# In[ ]:


# 测试extract_feature是否正常
get_ipython().run_line_magic('matplotlib', 'notebook')

def display_feature(x_i, figsize=(12,8), vmin=-10, vmax=10):
    print('x_i shape:', x_i.shape)
    fig = plt.figure(figsize=figsize)
    librosa.display.specshow(x_i[:,:].T, sr=100, hop_length=1, x_axis='time', 
                             cmap='viridis', vmin=vmin, vmax=vmax)  # 
    plt.colorbar()
    plt.title('Feature')
    fig.tight_layout()
    plt.show()


display_feature(x_i_sd[:1000,:], vmin=None, vmax=None)


# In[ ]:


check_point_x = int(2.6 *100)
check_point_y = 20+2
print(x_i.shape)
feature_df.iloc[check_point_x-5:check_point_x+5, check_point_y-5:check_point_y+5]


# In[ ]:


pd.DataFrame(x_i_sd[check_point_x-5:check_point_x+5, check_point_y-7:check_point_y+3])


# In[ ]:


import seaborn as sns
fig = plt.figure(figsize=(8,6))

# plt.imshow(x_i_sd[check_point_x-5:check_point_x+5, check_point_y-7:check_point_y+3], cmap='hot')
sns.heatmap(x_i_sd[check_point_x-5-2:check_point_x+5+2, check_point_y-7-2:check_point_y+3+2],
           annot=True)
fig.tight_layout()


# In[ ]:





# In[ ]:





# In[ ]:


from IPython.display import display, HTML
df1 = feature_df.iloc[:100,:100]
# Assuming that dataframes df1 and df2 are already defined:
print ("Dataframe 1:")
display(df1)
# print ("Dataframe 2:")
# display(HTML(df1.to_html()))


# In[ ]:





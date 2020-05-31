#!/usr/bin/env python
# coding: utf-8

# # <center>Audio Emotion Recognition</center>
# ## <center>Part 5 - Data augmentation</center>
# #### <center> 7th September 2019 </center> 
# #####  <center> Eu Jin Lok </center> 

# ## Introduction 
# Continuing where we left off in [Part 3](https://www.kaggle.com/ejlok1/audio-emotion-part-3-baseline-model) where we built a simple baseline model, now we're going to take the next level up and build in some data augmentation methods. I did some reading and desk reseach and there's a few good articles around on this:
# 
# - [Edward Ma](https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6)
# - [Qishen Ha](https://www.kaggle.com/haqishen/augmentation-methods-for-audio)
# - [Reza Chu](https://towardsdatascience.com/speech-emotion-recognition-with-convolution-neural-network-1e6bb7130ce3)
# 
# Thanks to the various authors above, I've incorporated their methods into this notebook here. We'll go ahead and test a few methods for the authors, then implement some using the same 1D CNN model we used before so we can compare apples with apples exactly how much the accuracy contribution came from the data augmentation methods. 
# 
# 1. [Explore augmentation methods](#explore)
#     - [Static noise](#static)
#     - [Shift](#shift)
#     - [Stretch](#stretch)
#     - [Pitch](#pitch)
#     - [Dynamic change](#dynamic)
#     - [Speed and pitch](#speed)
# 2. [Data preparation and processing](#data)
# 3. [Modelling](#modelling)
# 4. [Model serialisation](#serialise)
# 5. [Model validation](#validation)
# 6. [Final thoughts](#final)
# 
# Most importantly, I want to thank the 4 authors for their excellent dataset, without it, writing this notebook could not have been possible. The original source of the dataset links are below:
# - [TESS](https://tspace.library.utoronto.ca/handle/1807/24487)
# - [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
# - [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/Database.html)
# - [RAVDESS](https://zenodo.org/record/1188976#.XYP8CSgzaUk)
# - [RAVDESS_Kaggle](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)

# In[ ]:


# Importing required libraries 
# Keras
import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint

# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Other  
import librosa
import librosa.display
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import seaborn as sns
import glob 
import os
from tqdm import tqdm
import pickle
import IPython.display as ipd  # To play sound in the notebook

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# In[ ]:


#########################
# Augmentation methods
#########################
def noise(data):
    """
    Adding White Noise.
    """
    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    noise_amp = 0.05*np.random.uniform()*np.amax(data)   # more noise reduce the value to 0.5
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data
    
def shift(data):
    """
    Random Shifting.
    """
    s_range = int(np.random.uniform(low=-5, high = 5)*1000)  #default at 500
    return np.roll(data, s_range)
    
def stretch(data, rate=0.8):
    """
    Streching the Sound. Note that this expands the dataset slightly
    """
    data = librosa.effects.time_stretch(data, rate)
    return data
    
def pitch(data, sample_rate):
    """
    Pitch Tuning.
    """
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform())   
    data = librosa.effects.pitch_shift(data.astype('float64'), 
                                      sample_rate, n_steps=pitch_change, 
                                      bins_per_octave=bins_per_octave)
    return data
    
def dyn_change(data):
    """
    Random Value Change.
    """
    dyn_change = np.random.uniform(low=-0.5 ,high=7)  # default low = 1.5, high = 3
    return (data * dyn_change)
    
def speedNpitch(data):
    """
    peed and Pitch Tuning.
    """
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high = 1)
    speed_fac = 1.1  / length_change # try changing 1.0 to 2.0 ... =D
    tmp = np.interp(np.arange(0,len(data),speed_fac),np.arange(0,len(data)),data)
    minlen = min(data.shape[0], tmp.shape[0])
    data  = np.zeros(data.shape)
    data[0:minlen] = tmp[0:minlen]
    return data


# <a id="explore"></a>
# ## 1. Explore augmentation method
# So before we go full scale application of the augmentation methods, lets take one audio file and run it through all the different types to get a feel for how they work. From there we'll then take a few forward for our model training and hopefully it improves our accuracy.
# 
# We'll start with the orginal audio file untouched...

# In[ ]:


# Use one audio file in previous parts again
fname = '../../data/surrey-audiovisual-expressed-emotion-savee/ALL/JK_f11.wav'  
data, sampling_rate = librosa.load(fname)
plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)

# Paly it again to refresh our memory
ipd.Audio(data, rate=sampling_rate)


# <a id="static"></a>
# ### Static noise 
# The first augmentation method we'll do is to add static noise in the background. Here's how it sounds...

# In[ ]:


x = noise(data)
plt.figure(figsize=(15, 5))
librosa.display.waveplot(x, sr=sampling_rate)
ipd.Audio(x, rate=sampling_rate)


# -------------------------
# That's sounds like a good augmentation, not too much static such that it doesn't obfuscate the signal too much. 
# 
# <a id="shift"></a>
# ### Shift
# Next method is shift...

# In[ ]:


x = shift(data)
plt.figure(figsize=(15, 5))
librosa.display.waveplot(x, sr=sampling_rate)
ipd.Audio(x, rate=sampling_rate)


# ----------------------------
# So its not very noticable but what I've done there is move the audio randomly to either the left or right direction, within the fix audio duration. So if you compare this to the original plot, you can see the same audio wave pattern, except there's a tiny bit of delay before the speaker starts speaking. 
# 
# <a id="stretch"></a>
# ### Stretch 
# Now we go to stretch, my most favourite augmentation method 

# In[ ]:


x = stretch(data)
plt.figure(figsize=(15, 5))
librosa.display.waveplot(x, sr=sampling_rate)
ipd.Audio(x, rate=sampling_rate)


# This one is one of the more dramatic augmentation methods. The method literally stretches the audio. So the duration is longer, but the audio wave gets strecthed too. Thus introducing and effect that sounds like a slow motion sound. If you look at the audio wave itself, you'll notice that compared to the orginal audio, the strected audio seems to hit a higher frequency note. Thus creating a more diverse data for augmentation. Pretty nifty eh? It does introduce abit of a challenge in the data prep stage cause it lengthens the audio duration. Something to consider especially when doing a 2D CNN. 
#  
#  <a id="pitch"></a>
# ### Pitch
# I believe, this method accentuates the high pitch notes, by... normalising it sort of. 

# In[ ]:


x = pitch(data, sampling_rate)
plt.figure(figsize=(15, 5))
librosa.display.waveplot(x, sr=sampling_rate)
ipd.Audio(x, rate=sampling_rate)


# Honestly, I'm not exactly sure how this pitch augmentation works. so I'll need to do more reading on this. But safe to say it's another way of augmenting the data. If you listen to it, you can hear the difference. 
# 
# <a id="dynamic"></a>
# ### Dynamic change
# Dynamic change.... 

# In[ ]:


x = dyn_change(data)
plt.figure(figsize=(15, 5))
librosa.display.waveplot(x, sr=sampling_rate)
ipd.Audio(x, rate=sampling_rate)


# ----------------------------
# Yes I know what you are thinking. It's exactly the same as the original. Yes true, but if you look at the frequency, the wave hits higher frequency notes compared to the original where the min is around -1 and the max is around 1. The min and max of this audio is -6 and 6 respestively. Not exactly sure how useful this is, I'll need to do some more reading.
# 
# <a id="speed"></a>
# ### Speed and pitch
# Last but not least, speed and pitch...

# In[ ]:


x = speedNpitch(data)
plt.figure(figsize=(15, 5))
librosa.display.waveplot(x, sr=sampling_rate)
ipd.Audio(x, rate=sampling_rate)


# I really like this augmentation method. It dramatically alters the audio in many ways. It compresses the audio wave but keeping the audio duration the same. If you listen to it, the effect is opposite of the stretch augmentation method. An angry person when applied this augmentation method, to the human ear, will really alter the emotion interpretation of this audio. Not sure if this is counter productive to the algorithm, but lets try it. Another potential, downside is that there will be silence in the later part of the audio.  

# # my expriment

# In[ ]:


def speedNpitch_interp(data, speed_fac=None):
    """
    peed and Pitch Tuning.
    """
    if speed_fac is None:
        # you can change low and high here
        length_change = np.random.uniform(low=0.8, high = 1)
        speed_fac = 1.1  / length_change # try changing 1.0 to 2.0 ... =D
    tmp = np.interp(np.arange(0, len(data), speed_fac),np.arange(0,len(data)),data)
    minlen = min(data.shape[0], tmp.shape[0])
    res = np.zeros(data.shape)
    res[0:minlen] = tmp[0:minlen]
    return res


# In[ ]:


x = speedNpitch_interp(data, 1.0)
plt.figure(figsize=(15, 5))
librosa.display.waveplot(x, sr=sampling_rate)
ipd.Audio(x, rate=sampling_rate)


# # pyrubberband

# In[ ]:


from pyrubberband import time_stretch, pitch_shift


# In[ ]:


def rb_stretch(data, rate=0.8):
    """
    Streching the Sound. Note that this expands the dataset slightly
    """
    data = time_stretch(data, rate)
    return data
    
def rb_pitch(data, sample_rate):
    """
    Pitch Tuning.
    """
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform())   
    data = pitch_shift(data.astype('float64'), sample_rate, n_steps=pitch_change)  # , bins_per_octave=bins_per_octave


# In[ ]:


x = pitch(data, sampling_rate)
plt.figure(figsize=(15, 5))
librosa.display.waveplot(x, sr=sampling_rate)
ipd.Audio(x, rate=sampling_rate)


# In[ ]:


x = rb_pitch(data, sampling_rate)
plt.figure(figsize=(15, 5))
librosa.display.waveplot(x, sr=sampling_rate)
ipd.Audio(x, rate=sampling_rate)


# In[ ]:


x = stretch(data)
plt.figure(figsize=(15, 5))
librosa.display.waveplot(x, sr=sampling_rate)
ipd.Audio(x, rate=sampling_rate)


# In[ ]:


x = rb_stretch(data)
plt.figure(figsize=(15, 5))
librosa.display.waveplot(x, sr=sampling_rate)
ipd.Audio(x, rate=sampling_rate)


# In[ ]:





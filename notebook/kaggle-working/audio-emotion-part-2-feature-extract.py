#!/usr/bin/env python
# coding: utf-8

# # <center>Audio Emotion Recognition</center>
# ## <center>Part 2 - Feature Extraction </center>
# #### <center> 21st August 2019 </center> 
# #####  <center> Eu Jin Lok </center> 
# 
# ## Introduction 
# Following on from [Part 1](https://www.kaggle.com/ejlok1/audio-emotion-recognition-part-1-explore-data), we are now going to check out the various techniques for extracting useful features from audio for our classifier. I've done this before in another different kernel over [here](https://www.kaggle.com/ejlok1/part-2-extracting-audio-features/notebook#Part-2---Extracting-Audio-Features). I haven't covered the entireity of the various audio features, just ones which I'm familiar with. Any suggestions or advice please drop me a note. For a more complete coverage of the various features, I suggest checking the [pyaudio journal](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0144610&type=printable)
# 
# Broadly speaking there are two category of features:
# - Time domain features<br/> 
# These are simpler to extract and understand, like the energy of signal, zero crossing rate, maximum amplitude, minimum energy, etc.
# - Frequency based features<br/>
# are obtained by converting the time based signal into the frequency domain. Whilst they are harder to comprehend, it provides extra information that can be really handy such as pitch, rhythms, melody etc. Check this infographic below:

# ![Audio_wave](https://www.nti-audio.com/portals/0/pic/news/FFT-Time-Frequency-View-540.png)
# The time vs frequency domain image sourced from __[here](https://www.nti-audio.com/en/support/know-how/fast-fourier-transform-fft)__ 
# 
# 

# Since I've already outlined the various types of features [here](https://www.kaggle.com/ejlok1/part-2-extracting-audio-features/notebook#Part-2---Extracting-Audio-Features), I'll just simplify things here and just use MFCC, because its the best feature for this particular problem and we're trying to get to a quick working baseline. Later on, during the accuracy improvement phase, we may expand our feature set to include Mel-Spectogram, Chroma, HPSS and etc... and not just a simple mean 
# 
# 1. [MFCC quick intro](#mfcc)
# 2. [Deepdive](#deep)
# 3. [Statistical features](#stats)
# 4. [Final thoughts](#final)
# 
# Upvote this notebook if you like, and be sure to check out the other parts which are now available:
# * [Part 3 | Baseline model](https://www.kaggle.com/ejlok1/audio-emotion-part-3-baseline-model)
# * [Part 4 | Apply to new audio data](https://www.kaggle.com/ejlok1/audio-emotion-part-4-apply-to-new-audio-data)
# * [Part 5 | Data augmentation](https://www.kaggle.com/ejlok1/audio-emotion-part-5-data-augmentation)
# 
# Most importantly, I want to thank the 4 authors for their excellent dataset, without it, writing this notebook could not have been possible. The original source of the dataset links are below:
# 
# - [TESS](https://tspace.library.utoronto.ca/handle/1807/24487)
# - [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
# - [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/Database.html)
# - [RAVDESS](https://zenodo.org/record/1188976#.XYP8CSgzaUk)
# - [RAVDESS_Kaggle](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)

# <a id="mfcc"></a>
# ## 1. MFCC quick intro 
# MFCC is well known to be a good feature. And there's many ways you can slice and dice this one feature. But what is MFCC? It stands for Mel-frequency cepstral coefficient, and it is a good "representation" of the vocal tract that produces the sound. Think of it like an x-ray of your mouth
# 
# This post has a good deep dive into the [MFCC](https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd) should you wish to. The most common machine learning application treats the MFCC itself as an 'image' and becomes a feature. The benefit of treating it as an image is that it provides more information, and gives one the ability to draw on transfer learning. This is certainly legit and yields good accuracy. However, research has also shown that statistics relating to MFCCs (or any other time or frequency domain) can carry good amount of information as well. We'll be investigating both of this methods

# In[ ]:


# Import our libraries
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import pandas as pd
import os
import IPython.display as ipd  # To play sound in the notebook


# <a id="deep"></a>
# ## 2. Deepdive
# We can select a few examples and visualise the MFCC. lets take 2 different emotions and 2 different genders, and play it just to get a feel for what we are dealing with. Ie. whether the data (audio) quality is good. It gives us an early insight as to how likely our classifier is going to be successful.   

# In[ ]:


# Source - RAVDESS; Gender - Female; Emotion - Angry 
path = "../../data/ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_08/03-01-05-02-01-01-08.wav"
X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  
mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)

# audio wave
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.waveplot(X, sr=sample_rate)
plt.title('Audio sampled at 44100 hrz')

# MFCC
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.specshow(mfcc, x_axis='time')
plt.ylabel('MFCC')
plt.colorbar()

ipd.Audio(path)


# In[ ]:


# Source - RAVDESS; Gender - Male; Emotion - Angry 
path = "../../data/ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_09/03-01-05-01-01-01-09.wav"
X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  
mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)

# audio wave
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.waveplot(X, sr=sample_rate)
plt.title('Audio sampled at 44100 hrz')

# MFCC
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.specshow(mfcc, x_axis='time')
plt.ylabel('MFCC')
plt.colorbar()

ipd.Audio(path)


# Very placid response from the male counter part...

# In[ ]:


# Source - RAVDESS; Gender - Female; Emotion - Happy 
path = "../../data/ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_12/03-01-03-01-02-01-12.wav"
X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  
mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)

# audio wave
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.waveplot(X, sr=sample_rate)
plt.title('Audio sampled at 44100 hrz')

# MFCC
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.specshow(mfcc, x_axis='time')
plt.ylabel('MFCC')
plt.colorbar()

ipd.Audio(path)


# In[ ]:


# Source - RAVDESS; Gender - Male; Emotion - Happy 
path = "../../data/ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_11/03-01-03-01-02-02-11.wav"
X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  
mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)

# audio wave
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.waveplot(X, sr=sample_rate)
plt.title('Audio sampled at 44100 hrz')

# MFCC
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
librosa.display.specshow(mfcc, x_axis='time')
plt.ylabel('MFCC')
plt.colorbar()

ipd.Audio(path)


# <a id="stats"></a>
# ## 3. Statistical features 
# Now we've seen the shape of an MFCC output for each file, and it's a 2D matrix format with MFCC bands on the y-axis and time on the x-axis, representing the MFCC bands over time. To simplify things, what we're going to do is take the mean across each band over time. In other words, row means. But how does it present as a distinctive feature? 
# 
# So if you look at the above MFCC plot, the first band at the bottom is the most distinctive band over the other bands. Since the time window is a short one, the changes observed overtime does not vary greatly. The key feature is capturing the information contained in the various bands. Lets plot the mean of each of the band and display it as a time series plot to illustrate the point. 
# 
# We'll compare the Angry female and Angry male for the same sentence uttered. 

# In[ ]:


# Source - RAVDESS; Gender - Female; Emotion - Angry 
path = "../../data/ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_08/03-01-05-02-01-01-08.wav"
X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  
female = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
female = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
print(len(female))

# Source - RAVDESS; Gender - Male; Emotion - Angry 
path = "../../data/ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_09/03-01-05-01-01-01-09.wav"
X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  
male = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
male = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
print(len(male))

# audio wave
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
plt.plot(female, label='female')
plt.plot(male, label='male')
plt.legend()


# So for the same sentence being uttered, there is a clear distint difference between male and female in that females tends to have a higher pitch. Lets look at a few others. Lets compare a Happy Female and a Happy Male

# In[ ]:


# Source - RAVDESS; Gender - Female; Emotion - happy 
path = "../../data/ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_12/03-01-03-01-02-01-12.wav"
X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  
female = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
female = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
print(len(female))

# Source - RAVDESS; Gender - Male; Emotion - happy 
path = "../../data/ravdess-emotional-speech-audio/audio_speech_actors_01-24/Actor_11/03-01-03-01-02-02-11.wav"
X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)  
male = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
male = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
print(len(male))

# Plot the two audio waves together
plt.figure(figsize=(20, 15))
plt.subplot(3,1,1)
plt.plot(female, label='female')
plt.plot(male, label='male')
plt.legend()


# <a id="final"></a>
# ## 4. Final thoughts 
# Using MFCC is a good feature to differentiate the gender and emotions as demonstrated above. Even thou we've ommited alot of good information by just taking the mean, it seems we still capture enough to be able to see some difference. Whether this difference is significant for distinguishing the variou emotions, we'll find out in the next part where we will create a baseline emotion classifier
# 
# Upvote this notebook if you like, and be sure to check out the other parts which are now available:
# * [Part 3 | Baseline model](https://www.kaggle.com/ejlok1/audio-emotion-part-3-baseline-model)
# * [Part 4 | Apply to new audio data](https://www.kaggle.com/ejlok1/audio-emotion-part-4-apply-to-new-audio-data)

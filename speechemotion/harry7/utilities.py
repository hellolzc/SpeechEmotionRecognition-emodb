"""

author: harry-7

This file contains functions to read the data files from the given folders and
generate Mel Frequency Cepestral Coefficients features for the given audio
files as training samples.
"""
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from speechpy.feature import mfcc

mean_signal_length = 32000  # Empirically calculated for the given data set


def get_feature_vector_from_mfcc(file_path: str, flatten: bool,
                                 mfcc_len: int = 39) -> np.ndarray:
    """
    Make feature vector from MFCC for the given wav file.

    Args:
        file_path (str): path to the .wav file that needs to be read.
        flatten (bool) : Boolean indicating whether to flatten mfcc obtained.
        mfcc_len (int): Number of cepestral co efficients to be consider.

    Returns:
        numpy.ndarray: feature vector of the wav file made from mfcc.
    """
    fs, signal = wav.read(file_path)
    s_len = len(signal)
    # pad the signals to have same size if lesser than required
    # else slice them
    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem),
                        'constant', constant_values=0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mel_coefficients = mfcc(signal, fs, num_cepstral=mfcc_len)
    if flatten:
        # Flatten the data
        mel_coefficients = np.ravel(mel_coefficients)
    return mel_coefficients


def get_data(data_path: str, flatten: bool = True, mfcc_len: int = 39,
             class_labels: Tuple = ("neutral", "angry", "happy", "sad")) -> \
        Tuple[np.ndarray, np.ndarray]:
    """Extract data for training and testing.

    1. Iterate through all the folders.
    2. Read the audio files in each folder.
    3. Extract Mel frequency cepestral coefficients for each file.
    4. Generate feature vector for the audio files as required.

    Args:
        data_path (str): path to the data set folder
        flatten (bool): Boolean specifying whether to flatten the data or not.
        mfcc_len (int): Number of mfcc features to take for each frame.
        class_labels (tuple): class labels that we care about.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Two numpy arrays, one with mfcc and
        other with labels.


    """
    data = []
    labels = []
    names = []

    df = pd.read_csv(data_path + '/datalist.csv')
    df = df.loc[df['emotion_en'].isin(class_labels)]

    for indx, label in enumerate(class_labels):
        sys.stderr.write("Started reading label %s\n" % label)
        for _, row in df.loc[df['emotion_en'] == label].iterrows():
            filename = row['uuid']
            label_num = indx
            sys.stderr.write("%s  " % filename)
            filepath = data_path + '/wav/' + filename + '.wav'
            feature_vector = get_feature_vector_from_mfcc(file_path=filepath,
                                                        mfcc_len=mfcc_len,
                                                        flatten=flatten)
            data.append(feature_vector)
            labels.append(label_num)
            names.append(filename)
    return np.array(data), np.array(labels)


    # for indx, item in enumerate(class_labels):
    #     label_map[item] = indx
    # sys.stderr.write("Reading: ")
    # for indx, row in df.iterrows():
    #     filename = row['uuid']
    #     label = row['emotion_en']
    #     label_num = label_map[label]
    #     sys.stderr.write("%s  " % filename)
    #     filepath = data_path + '/wav/' + filename + '.wav'
    #     feature_vector = get_feature_vector_from_mfcc(file_path=filepath,
    #                                                 mfcc_len=mfcc_len,
    #                                                 flatten=flatten)
    #     data.append(feature_vector)
    #     labels.append(label_num)
    #     names.append(filename)
    # return np.array(data), np.array(labels)

    # for i, directory in enumerate(class_labels):
    #     sys.stderr.write("started reading folder %s\n" % directory)
    #     os.chdir(directory)
    #     for filename in os.listdir('.'):
    #         filepath = os.getcwd() + '/' + filename
    #         feature_vector = get_feature_vector_from_mfcc(file_path=filepath,
    #                                                       mfcc_len=mfcc_len,
    #                                                       flatten=flatten)
    #         data.append(feature_vector)
    #         labels.append(i)
    #         names.append(filename)
    #     sys.stderr.write("ended reading folder %s\n" % directory)
    #     os.chdir('..')
    # os.chdir(cur_dir)
    # return np.array(data), np.array(labels)

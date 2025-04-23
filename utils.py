import numpy as np
import tensorflow as tf
import librosa
from keras import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, Reshape, TimeDistributed
from keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import mir_eval
import warnings
import os

warnings.filterwarnings("ignore")

# Constants
Nfft = 1024
win_len = 512
hop_len = 80
win_size = 500
bpo = 96

# Set up TensorFlow environment
tf.keras.backend.set_floatx('float32')

# Mean and standard deviation for normalization
mean = -18.800999505542258
std = 12.473455860620371

# Get center frequencies for pitch range
def get_cenfreq(StartFreq, StopFreq, NumPerOct):
    """Generate center frequencies within the given range and octave division."""
    Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
    central_freq = []
    for i in range(0, Nest):
        CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break
    return central_freq

# Initialize pitch range
pitch_range = get_cenfreq(51.91, 1975.53, bpo)
pitch_range = np.concatenate([np.zeros(1), pitch_range])

# Create one-hot labels
def onehotlabel(pitch_range):
    """Convert pitch range to one-hot encoded vectors."""
    values = np.asarray(pitch_range)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

# Initialize one-hot encoded pitch range
onehot_pitch_range = onehotlabel(pitch_range)

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

# Model definitions
class melody_extraction(Model):
    """CNN model for melody extraction."""
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(filters=64, name='conv1', kernel_size=(5, 5), input_shape=(win_size, 513, 1), 
                           padding='same', kernel_initializer='he_normal', 
                           kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5))
        self.bn1 = BatchNormalization(name='bn1')
        self.conv2 = Conv2D(filters=128, name='conv2', kernel_size=(5, 5), padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5))
        self.bn2 = BatchNormalization(name='bn2')
        self.conv3 = Conv2D(filters=192, name='conv3', kernel_size=(5, 5), padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5))
        self.bn3 = BatchNormalization(name='bn3')
        self.conv4 = Conv2D(filters=256, name='conv4', kernel_size=(5, 5), padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5))
        self.bn4 = BatchNormalization(name='bn4')
        self.linear1 = Dense(512, name='dense1', activation='relu', kernel_initializer='he_normal',
                             kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5))
        self.final = TimeDistributed(
            Dense(len(pitch_range), name='dense2', activation='softmax', kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5)))

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.keras.activations.relu(x)
        x = MaxPooling2D((1, 4))(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.keras.activations.relu(x)
        x = MaxPooling2D((1, 4))(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.keras.activations.relu(x)
        x = MaxPooling2D((1, 4))(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = tf.keras.activations.relu(x)
        x = MaxPooling2D((1, 4))(x)

        x = Reshape((-1, x.shape[2] * x.shape[3]))(x)
        int_output = self.linear1(x)
        x = self.final(int_output)
        return x, int_output

    def build_graph(self, raw_shape):
        x = Input(shape=raw_shape)
        return Model(inputs=[x], outputs=self.call(x))

class ConfidenceModel(Model):
    """Model for confidence estimation."""
    def __init__(self, pretrain_model=None):
        super().__init__()
        self.pretrain = pretrain_model
        self.dense2 = Dense(256, activation='relu', kernel_initializer='he_normal',
                            kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5))
        self.final = Dense(1, activation='sigmoid', kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5))

    def call(self, x):
        if self.pretrain:
            _, x = self.pretrain(x)
        else:
            _, x = melody_extraction()(x)
        
        x = self.dense2(x)
        x = Dropout(0.2)(x)
        x = self.final(x)
        return x

    def build_graph(self, raw_shape):
        x = Input(shape=raw_shape)
        return Model(inputs=[x], outputs=self.call(x))

# Pitch calculation utilities
def M(f):
    """Convert frequency to cents relative to reference frequency."""
    f_ref = 100
    return bpo * np.log2((f + 1.e-9) / f_ref)

def Thres(a):
    """Threshold function for pitch comparison."""
    if -0.5 <= a and a <= 0.5:
        t = 1
    else:
        t = 0
    return t

def RawPitch(F_predicted, F_ground):
    """Calculate raw pitch accuracy."""
    if F_predicted == float(0) and F_ground == float(0):
        return 1
    else:
        return Thres(M(F_predicted) - M(F_ground))

def closest(arr, K):
    """Find closest pitch in array using raw pitch accuracy."""
    idx = np.array([RawPitch(val, K) for val in arr])
    idx = idx.argmax()
    return idx

def get_onehot(f):
    """Convert frequency values to one-hot encoded vectors."""
    yhot = np.zeros((len(f), len(pitch_range)))
    for i in range(len(f)):
        idx = closest(pitch_range, f[i])
        yhot[i, :] = onehot_pitch_range[idx, :]
    return yhot

def calc_spec(x, fs):
    """Calculate spectrogram from audio data."""
    X = np.abs(librosa.stft(x, n_fft=Nfft, hop_length=hop_len, win_length=win_len, window='hann'))
    X = librosa.power_to_db(X, ref=np.max)
    X = X[:, :win_size]
    return np.array(X)

def copy_model(model, x):
    """Create a copy of a model with the same weights."""
    copied_model = melody_extraction()
    copied_model.call(tf.convert_to_tensor(x))
    
    copied_model.set_weights(model.get_weights())
    return copied_model

# Loss functions
loss_fn = tf.keras.losses.CategoricalCrossentropy()
mse = tf.keras.losses.MeanSquaredError()

def custom_loss(yt, yp, ts=None):
    """Custom loss function for model training."""
    if ts is None:
        # Create weights based on class frequencies
        classes = tf.reduce_sum(tf.cast(yt, tf.float32), axis=(0, 1))
        cratio = 1.0 / (classes / tf.reduce_max(classes))
        weights = tf.where(tf.math.is_finite(cratio), cratio, tf.constant(0.0))
        w = tf.gather(weights, tf.argmax(yt, axis=-1))
        loss = loss_fn(yt, yp, sample_weight=w)
    else:
        # Apply weights from support indices
        w_mask = tf.convert_to_tensor(ts)
        w_mask = tf.cast(w_mask, dtype=tf.float32)
        if len(w_mask.shape) == 1:
            w_mask = tf.expand_dims(w_mask, 0)
        classes = tf.reduce_sum(tf.cast(yt, tf.float32), axis=(0, 1))
        cratio = 1.0 / (classes / tf.reduce_max(classes))
        weights = tf.where(tf.math.is_finite(cratio), cratio, tf.constant(0.0))
        w = tf.gather(weights, tf.argmax(yt, axis=-1))
        if len(w_mask.shape) > 1:
            w = tf.multiply(w, w_mask)
        loss = loss_fn(yt, yp, sample_weight=w_mask)
    return loss

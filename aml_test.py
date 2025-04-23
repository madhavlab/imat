import os
import time
import random
import librosa
import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, Reshape, TimeDistributed
from keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import mir_eval
from glob import glob

## testing the performance of meta-learning model 


filepath_pre = '../weights/meta_model/pre/meta_weights-{epoch:02d}'
filepath_conf = '../weights/meta_model/conf/meta_weights-{epoch:02d}'

tf.keras.backend.set_floatx('float32')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpus = tf.config.list_physical_devices('GPU')

gpu_number = 0

if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[gpu_number], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
###################################################

def get_cenfreq(StartFreq, StopFreq, NumPerOct):
    Nest = int(np.ceil(np.log2(StopFreq / StartFreq)) * NumPerOct)
    central_freq = []
    for i in range(0, Nest):
        CenFreq = StartFreq * pow(2, float(i) / NumPerOct)
        if CenFreq < StopFreq:
            central_freq.append(CenFreq)
        else:
            break
    return central_freq

def onehotlabel(pitch_range):
    values = np.asarray(pitch_range)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

Nfft = 1024
win_len = 512
hop_len = 80
win_size = 500
bpo = 96
pitch_range = get_cenfreq(51.91, 1975.53, bpo)
pitch_range = np.concatenate([np.zeros(1), pitch_range])
onehot_pitch_range = onehotlabel(pitch_range)  # one hot vector
#################################################
class melody_extraction(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(filters=64, name='conv1', kernel_size=(5, 5), input_shape=(win_size, 513), padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=l2(1e-5), bias_regularizer=l2(1e-5))
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


meta_trained_classifier = melody_extraction()
meta_trained_classifier.build_graph([win_size, 513, 1])  # .summary()

for i in range(len(meta_trained_classifier.layers) - 1):
    meta_trained_classifier.layers[i].trainable = False
fe_model = melody_extraction()
fe_model.build_graph([win_size, 513, 1])
fe_model.load_weights('./models/adaptive_weights/weights/meta_model/pre/meta_weights-500')  # load the pre-train weights

for i in range(len(meta_trained_classifier.layers) - 1):
    fe_model.layers[i].trainable = False

class ConfidenceModel(Model):
    def __init__(self):
        super().__init__()
        self.pretrain = fe_model
        self.dense2 = Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5),
                            bias_regularizer=l2(1e-5))
        self.final = Dense(1, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=l2(1e-5),
                           bias_regularizer=l2(1e-5))

    def call(self, x):
        _, x = self.pretrain(x)

        x = self.dense2(x)
        x = Dropout(0.2)(x)
        x = self.final(x)
        return x

    def build_graph(self, raw_shape):
        x = Input(shape=raw_shape)
        return Model(inputs=[x], outputs=self.call(x))


meta_trained_conf = ConfidenceModel()
meta_trained_conf.build_graph([win_size, 513, 1])  # .summary()
meta_trained_conf.layers[0].trainable = False
conf_model = ConfidenceModel()
conf_model.build_graph([win_size, 513, 1])  # .summary()
conf_model.layers[0].trainable = False
conf_model.load_weights('./models/adaptive_weights/weights/meta_model/conf/meta_weights-500')

mean = -18.800999505542258
std = 12.473455860620371

alpha = 1.e-4  # 5.e-5  --> only 3 times retraining
beta = 1.e-6  # 5.e-5
loss_fn = tf.keras.losses.CategoricalCrossentropy()
mse = tf.keras.losses.MeanSquaredError()
inner_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
conf_inner_optimizer = tf.keras.optimizers.Adam(learning_rate=beta)
inner_step = 10

def conf_loss(yt, yp, yc, ts):
    yc_star = tf.zeros((yt.shape[0], yt.shape[1]), dtype=tf.float32)
    ts = tf.cast(ts, dtype=tf.int32)
    w_mask = tf.zeros(yt.shape[1], dtype=tf.float32)
    w_mask = tf.tensor_scatter_nd_update(w_mask, tf.expand_dims(ts, 1), tf.ones_like(ts, dtype=tf.float32))
    w_mask = tf.expand_dims(w_mask, 0)
    true_indx = tf.argmax(yt, axis=-1)
    true_indx = tf.reduce_max(true_indx, axis=0)
    row_no = tf.range(0, yt.shape[1], 1)
    row_no = tf.cast(row_no, tf.int32)
    true_indx = tf.cast(true_indx, tf.int32)
    indexing = tf.concat([row_no[:, tf.newaxis], true_indx[:, tf.newaxis]], axis=1) 
    y_p = tf.ones(tf.shape(yp[0]))
    c_star = tf.gather_nd(y_p, indexing)
    c_star = c_star[tf.newaxis, :, tf.newaxis]
    loss = mse(c_star, yc, sample_weight=tf.constant(w_mask))
    return loss


def support_pretrain_step(x, y, ts):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x = x[tf.newaxis, :, :, :]
    y = y[tf.newaxis, :, :]
    with tf.device('/gpu:0'):
        for _ in range(inner_step):
            with tf.GradientTape(persistent=True) as tape:
                ys_hat, _ = meta_trained_classifier.call(x)
                loss = custom_loss(y, ys_hat, ts)
            grads = tape.gradient(loss, meta_trained_classifier.trainable_variables)
            inner_optimizer.apply_gradients(zip(grads, meta_trained_classifier.trainable_variables))
    return loss

def support_conftrain_step(x, y, ts):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x = x[tf.newaxis, :, :, :]
    y = y[tf.newaxis, :, :]
    with tf.device('/gpu:0'):
        for _ in range(inner_step):
            with tf.GradientTape(persistent=True) as tape:
                ys_hat, _ = meta_trained_classifier.call(x)
                yc_hat = meta_trained_conf.call(x)
                loss = conf_loss(y, ys_hat, yc_hat, ts)
            grads = tape.gradient(loss, meta_trained_conf.trainable_variables)
            conf_inner_optimizer.apply_gradients(zip(grads, meta_trained_conf.trainable_variables))
    return loss

def verify_conf(x, y, indx):
    yc = meta_trained_conf.call(tf.convert_to_tensor(x))
    yc = tf.reshape(yc, -1)
    yp, _ = meta_trained_classifier.call(x)
    indx = indx[0]
    for i in range(tf.shape(y)[0]):
        for j in indx:
            print(f'True indx..{tf.argmax(y[i][j], axis=-1)} Pred indx..{tf.argmax(yp[i][j], axis=-1)} Conf..{yc[j]}')

def get_groundtruth_freq(Y):
    true_indx = tf.argmax(Y[0], axis=-1)
    gfv = pitch_range[true_indx]
    return gfv

rpa_tot = np.array([])

def calc_spec(x, fs):
    X = np.abs(librosa.stft(x, n_fft=Nfft, hop_length=hop_len, win_length=win_len, window='hann'))
    X = librosa.power_to_db(X, ref=np.max)
    X = X[:, :win_size]
    return np.array(X)

# for i in range(len(audio_files)):
#     pitchfile = os.path.basename(os.path.splitext(audio_files[i])[0])


def plot_melody(fe_model,S):
    efv = []
    t = []

    S = S.T
    S = tf.convert_to_tensor(S)
    S = (S - mean) / std
    S = S[tf.newaxis, :, :, tf.newaxis]

    y_p, _ = fe_model.call(S)  # this is the one-hot vector of the prediction
    # converting one hot vector to the continuous frequency values in Hz
    for j in range(y_p.shape[0]):
        for i in range(y_p.shape[1]):
            indx = np.argmax(y_p[j, i, :])
            efv.append(pitch_range[indx])
            t.append(i * 0.01)

    return y_p, efv, t

def M(f):
    f_ref = 100
    return bpo * np.log2((f + 1.e-9) / f_ref)

def Thres(a):
    if -0.5 <= a and a <= 0.5:
        t = 1
    else:
        t = 0
    return t

def RawPitch(F_predicted, F_ground):
    if F_predicted == float(0) and F_ground == float(0):
        return 1
    else:
        return Thres(M(F_predicted) - M(F_ground))

def closest(arr, K):
    idx = np.array([RawPitch(val, K) for val in arr])
    idx = idx.argmax()
    return idx

def get_onehot(f):
    yhot = np.zeros((len(f), len(pitch_range)))
    for i in range(len(f)):
        idx = closest(pitch_range, f[i])
        yhot[i, :] = onehot_pitch_range[idx, :]
    return yhot

def calc_rpa(gfv,efv):
    t = [0.01*i for i in range(len(efv))]
    t = np.array(t)
    gfv = np.array(gfv)
    efv = np.array(efv)
    (ref_v, ref_c,est_v, est_c) = mir_eval.melody.to_cent_voicing(t,gfv,t,efv)

    RPA = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c,est_v, est_c)
    RCA = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c,est_v, est_c)
    OA = mir_eval.melody.overall_accuracy(ref_v, ref_c,est_v, est_c)

def custom_loss(yt, yp, ts):
    w_mask = tf.convert_to_tensor(ts)
    w_mask = tf.cast(w_mask, dtype=tf.float32)
    w_mask = tf.expand_dims(w_mask, 0)
    classes = tf.reduce_sum(tf.cast(yt, tf.float32), axis=(0, 1))
    cratio = 1.0 / (classes / tf.reduce_max(classes))
    weights = tf.where(tf.math.is_finite(cratio), cratio, tf.constant(0.0))
    w = tf.gather(weights, tf.argmax(yt, axis=-1))
    w = tf.multiply(w, w_mask)
    loss = loss_fn(yt, yp, sample_weight=w_mask)
    return loss


def aml_test(S, active_frames, gfv,fileid):
    alpha = 1.e-4  # 5.e-5  --> only 3 times retraining
    inner_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
    S = S.T
    S = tf.convert_to_tensor(S)
    S = (S-mean)/std
    S = S[tf.newaxis, :, :, tf.newaxis]    
    efv = []
    # for every batch, start from the active-meta-trained models
    if (fileid==0):
        meta_pre_weights = fe_model.get_weights()
        meta_conf_weights = conf_model.get_weights()
    else:
        fe_model.load_weights('models/updated_weights/weights')  # load the pre-train weights
        meta_pre_weights = fe_model.get_weights()
        meta_conf_weights = conf_model.get_weights()
        alpha = alpha* 0.9 * fileid
        inner_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
        print(f'alpha used: {alpha}')  
    meta_trained_conf.set_weights(meta_conf_weights)
    meta_trained_classifier.set_weights(meta_pre_weights)    

    Y_true = get_onehot(gfv)  # get the onehot vector of the annotated frequencies - this will act as ground truth
    Y_true = Y_true[tf.newaxis, :, :]       
    num_batches = int(np.ceil(S.shape[1] / win_size))

    with tf.device('/gpu:0'):
        for epoch in range(inner_step):
            tot_loss = 0
            print(f'Epoch...{epoch}')
            for i in range(num_batches):
                start_indx = i * win_size
                end_indx = min((i+1) * win_size, S.shape[1])
                X = S[:,start_indx:end_indx,:,:]
                
                frames = active_frames[start_indx:end_indx]
                
                y_true = Y_true[:,start_indx:end_indx,:]
                
                if sum(frames)>0:
                    with tf.GradientTape() as tape:
                        ys_hat, _ = meta_trained_classifier.call(X)
                        loss = custom_loss(y_true, ys_hat, frames)
                        tot_loss+=loss
            grads = tape.gradient(tot_loss, meta_trained_classifier.trainable_variables)
            inner_optimizer.apply_gradients(zip(grads, meta_trained_classifier.trainable_variables))
        ## melody estimation after adaptation
        yq_hat, _ = meta_trained_classifier.call(S)
        meta_trained_classifier.save_weights('models/updated_weights/weights')
        # converting one hot vector to the continuous frequency values in Hz
        for j in range(yq_hat.shape[0]):
            for i in range(yq_hat.shape[1]):
                indx = np.argmax(yq_hat[j, i, :])
                efv.append(pitch_range[indx])         
    return efv


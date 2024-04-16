import json
import os
import random

import keras
import librosa
import mir_eval
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, ReLU, Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization, LSTM, Dropout, Reshape, TimeDistributed, add, Bidirectional
from tensorflow.keras import Model, Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.regularizers import l2


filepath = '../weights/conf/meta-tcp-weights-{epoch:02d}'
filepath1 = '../weights/conf/meta-conf-weights-{epoch:02d}'

tf.keras.backend.set_floatx('float32')

os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
        
##############################################################################
# meta_audio_files = sorted(glob('/hdd_storage/data/kavyars/Datasets/spec_data/Mir1K/audio/'+'*.npy')[1800:])
# pitch_files = sorted(glob('/hdd_storage/data/kavyars/Datasets/spec_data/Mir1K/pitch/'+'*.csv'))

# print('Total train audio files:',len(meta_audio_files),'\n')
# print('Total val files:',len(val_audio_files),'\n')
# print('Total pitch files:',len(pitch_files),'\n')

# train_audio_files = glob('/hdd_storage/data/kavyars/Datasets/spec_data/Mir1K/audio/'+'*.npy')[:1800]
# val_audio_files = glob('/hdd_storage/data/kavyars/Datasets/spec_data/Mir1K/audio/'+'*.npy')[1800:]

# pitch_files = sorted(glob('/hdd_storage/data/kavyars/Datasets/spec_data/Mir1K/pitch/'+'*.csv'))

# print('Total train audio files:',len(train_audio_files),'\n')
# print('Total val files:',len(val_audio_files),'\n')
# print('Total pitch files:',len(pitch_files),'\n')

##############################################################################
def get_cenfreq(StartFreq,StopFreq,NumPerOct):
    Nest = int(np.ceil(np.log2(StopFreq/StartFreq))*NumPerOct)
    central_freq = []
    for i in range(0, Nest):
        CenFreq = StartFreq*pow(2, float(i)/NumPerOct)
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

# pitch_range = get_cenfreq(61.74,830.61,bpo)
pitch_range = get_cenfreq(51.91,1975.53,bpo)    #in frequencies
pitch_range = np.concatenate([np.zeros(1),pitch_range])
onehot_pitch_range = onehotlabel(pitch_range)  #one hot vector 

class melody_extraction(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(filters=64,name='conv1',kernel_size=(5,5),input_shape=(win_size,513),padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-5),bias_regularizer=l2(1e-5))
        self.bn1 = BatchNormalization(name='bn1')
        self.conv2 = Conv2D(filters=128,name='conv2',kernel_size=(5,5),padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-5),bias_regularizer=l2(1e-5))
        self.bn2 = BatchNormalization(name='bn2')
        self.conv3 = Conv2D(filters=192,name='conv3',kernel_size=(5,5),padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-5),bias_regularizer=l2(1e-5))
        self.bn3 = BatchNormalization(name='bn3')
        self.conv4 = Conv2D(filters=256,name='conv4',kernel_size=(5,5),padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-5),bias_regularizer=l2(1e-5))
        self.bn4 = BatchNormalization(name='bn4')
        self.linear1 = Dense(512,name='dense1',activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(1e-5),bias_regularizer=l2(1e-5))
        self.final = TimeDistributed(Dense(len(pitch_range),name='dense2',activation='softmax',kernel_initializer='he_normal',kernel_regularizer=l2(1e-5),bias_regularizer=l2(1e-5)))
        

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.keras.activations.relu(x)
        x = MaxPooling2D((1,4))(x)
        # x = Dropout(0.2)(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.keras.activations.relu(x)
        x = MaxPooling2D((1,4))(x)
        # x = Dropout(0.2)(x) --
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.keras.activations.relu(x)
        x = MaxPooling2D((1,4))(x)
        # x = Dropout(0.2)(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = tf.keras.activations.relu(x)
        x = MaxPooling2D((1,4))(x)
        # x = Dropout(0.2)(x) --

        x = Reshape((-1,x.shape[2]*x.shape[3]))(x)     

        # x = Reshape((win_size,x.shape[2]*x.shape[3]))(x)     
        int_output = self.linear1(x)
        x = self.final(int_output)
        return x,int_output

    def build_graph(self,raw_shape):
        x = Input(shape=raw_shape)
        return Model(inputs=[x], outputs = self.call(x))    

model = melody_extraction()
model.build_graph([win_size,513,1])
#Load pre-train weights
model.load_weights('./models/weights-190')
#################################################

class ConfidenceModel(Model):
    def __init__(self):
        super().__init__()
        self.pretrain = model     
        self.dense2 = Dense(256,activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(1e-5),bias_regularizer=l2(1e-5))
        self.final = Dense(1,activation='sigmoid',kernel_initializer='he_normal',kernel_regularizer=l2(1e-5),bias_regularizer=l2(1e-5))
        
        
    def call(self,x):
        _,x = self.pretrain(x)     
        
        x = self.dense2(x)
        x = Dropout(0.2)(x)
        x = self.final(x)
        return x
    
    def build_graph(self,raw_shape):
        x = Input(shape=raw_shape)
        return Model(inputs=[x],outputs = self.call(x))

model_conf = ConfidenceModel()
model_conf.load_weights('./models/conf/conf-weights-6000')


##Meta-training

alpha = 1.e-5
beta = 1.e-5
base_optimizer = tf.keras.optimizers.Adam(learning_rate = alpha)
meta_optimizer = tf.keras.optimizers.Adam(learning_rate = beta)

base_acc_metric = tf.keras.metrics.CategoricalAccuracy()
meta_acc_metric = tf.keras.metrics.CategoricalAccuracy()

N = 5

batch_size = 2
loss_fn = tf.keras.losses.CategoricalCrossentropy()


def copy_model(model, x):
    copied_model = melody_extraction()
    copied_model.call(tf.convert_to_tensor(x))
    
    copied_model.set_weights(model.get_weights())
    return copied_model

def get_indx_conf(y_pred,cf):
           
    train_indx = np.zeros((y_pred.shape[0],y_pred.shape[1]))
    conf_indx = np.zeros((y_pred.shape[0],y_pred.shape[1]))
    cf_sc = np.zeros((y_pred.shape[0],y_pred.shape[1]))
    
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            # actual = np.argmax(y_true[i,j,:])
            # pred = np.argmax(y_pred[i,j,:])
            # conf_score = cf[i][j]
            
            if 0.45<cf[i][j]<0.55:
                train_indx[i,j] = 1
                cf_sc[i,j]=cf[i][j]
    
    for k in range(train_indx.shape[0]):
        row = train_indx[k,:]
        indx = [i for i,val in enumerate(row) if val==1]
        if len(indx)>=N:
            random_indx = random.sample(indx,N)        
            for l in random_indx:
                conf_indx[k,l]=1                  
                # print(cf_sc[k,l])     
    
    return conf_indx
    # return train_indx  

def support_custom_loss(yt,yp,tr_indx):   
    w1 = np.zeros((yt.shape[0],yt.shape[1]))    
    for i in range(yt.shape[0]):
        row = list(tr_indx[i,:])
        indx = [x for x in range(len(row)) if row[x]==float(1)]        
        if len(indx)>=1:
            w1[i][indx] = 1
    
    # classes = np.zeros(len(pitch_range))    
    # for i in range(yt.shape[0]):
    #     for j in range(yt.shape[1]):
    #         indx = np.argmax(yt[i][j])
    #         classes[indx]+=1    
    # cratio = [i/max(classes) for i in classes]
    # weights = [1/i if i!=0 else float(0) for i in cratio]
    
    # w2 = tf.cast(tf.gather(weights,tf.argmax(yt,axis=-1)),tf.float32)
    
    # w = w1*w2
    w = w1
    loss = loss_fn(yt,yp,sample_weight=tf.constant(w)) 
    return loss

def custom_loss(yt,yp):
    classes = np.zeros(len(pitch_range))    
    for i in range(yt.shape[0]):
        for j in range(yt.shape[1]):
            indx = np.argmax(yt[i][j])
            classes[indx]+=1
    
    cratio = [i/max(classes) for i in classes]
    weights = [1/i if i!=0 else float(0) for i in cratio]
    
    w = tf.cast(tf.gather(weights,tf.argmax(yt,axis=-1)),tf.float32)
    loss = loss_fn(yt,yp,sample_weight=tf.constant(w)) 
    return loss 


def get_rpa_rca(X,Y,Y_pred):
    tot_rpa = np.array([])
    tot_rca = np.array([])
    tot_oa = np.array([])
    
    efv = np.array([])
    gfv = np.array([])
    
    for j in range(X.shape[0]):  
        true_indx = np.argmax(Y[j],axis=1)
        gfv = pitch_range[true_indx]  
        gfv = np.array([librosa.midi_to_hz(i) if i!=0 else float(0) for i in gfv])
        
        pred_indx = np.argmax(Y_pred[j],axis=1)
        efv = pitch_range[pred_indx]  
        efv = np.array([librosa.midi_to_hz(i) if i!=0 else float(0) for i in efv])
        
        gtv = [(i+1)*0.01 for i in range(len(efv))]
        gtv = np.array(gtv)
          
        (ref_v, ref_c,est_v, est_c) = mir_eval.melody.to_cent_voicing(gtv,gfv,gtv,efv)

        RPA = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c,est_v, est_c)
        RCA = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c,est_v, est_c)
        OA = mir_eval.melody.overall_accuracy(ref_v, ref_c,est_v, est_c)
        tot_rpa = np.append(tot_rpa,RPA)
        tot_rca = np.append(tot_rca,RCA)
        tot_oa = np.append(tot_oa,OA)
    return np.mean(tot_rpa),np.mean(tot_rca),np.mean(tot_oa)
    # return RPA, RCA, OA
     
def compute_loss(model, x, y, train_indx):
    logits = model.call(x)
    loss = support_custom_loss(y, logits, train_indx)
    return loss, logits

def compute_gradients(model, x, y,train_indx):
    with tf.GradientTape() as tape:
        loss, logits = compute_loss(model, x, y,train_indx)
    return tape.gradient(loss, model.trainable_variables)


def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))
    
    
# def get_updated_model(model,cmodel):
#     model_fe = feature_extractor() 
#     model_fe.build_graph([win_size,513,1])
        
#     for j in range(0,len(model_fe.layers)):
#         model_fe.layers[j].kernel = model.layers[j].kernel
#         model_fe.layers[j].bias = model.layers[j].bias
            
#     model_c = ConfidenceModel(model_fe)
#     model_c.build_graph([win_size,513,1])
    
#     for i in range(1,len(model_c.layers)):
#         model_c.layers[i].kernel = model_conf.layers[i].kernel
#         model_c.layers[i].bias = model_conf.layers[i].bias
    
#     # print(model.layers[0].kernel[0][0],model_fe.layers[0].kernel[0][0])
#     # print(model_c.layers[1].kernel,model_conf.layers[1].kernel)
#     return model_c

def calc_spec(x,fs):
    X = np.abs(librosa.stft(x, n_fft=Nfft, hop_length=hop_len, win_length=win_len, window='hann'))
    X = librosa.power_to_db(X,ref=np.max)
    X = X[:,:win_size]
    return np.array(X)

mean = -18.800999505542258
std = 12.473455860620371

def plot_melody(model,S):
    """
    write the code for predicting the pitch contour for the given input
    
    Input: model, wavfile
    
    Output: Spectrogram(X), pitch contour (efv)
    
    """
    efv = []
    t = []
    # x, sr = librosa.load(input, sr=8000, mono=True)
    # X = calc_spec(x, sr)
    S = S.T
    S = tf.convert_to_tensor(S)
    S = (S-mean)/std
    S = S[tf.newaxis, :, :, tf.newaxis]
    print('shape:',S.shape)

    y_p,_ = model.call(S)
    # print('shape', y_p)
    for j in range(y_p.shape[0]):
        for i in range(y_p.shape[1]):
            indx = np.argmax(y_p[j, i, :])
            efv.append(pitch_range[indx])
            t.append(i * 0.01)
    return y_p, efv, t


def conf_values(model_conf,S):
    S = S.T
    S = tf.convert_to_tensor(S)
    S = (S-mean)/std
    S = S[tf.newaxis, :, :, tf.newaxis]
    
    conf_values = model_conf.call(S)
    return conf_values

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
    

# def closest(arr,K):
#     idx = (np.abs(arr - K)).argmin()
#     return idx

def getonehot(y):
    yhot = np.zeros((len(y),len(pitch_range)))
    for i in range(len(y)):
        idx = closest(pitch_range,y[i])
        yhot[i,:] = onehot_pitch_range[idx,:]
    return yhot

def conv_pitch_to_ypred(gtf):
    Y = []
    if(len(gtf)>500):
        gtf = gtf[:500]
    else:
        gtf = np.append(gtf,np.zeros(500-len(gtf)))
    gtf_midi = [librosa.hz_to_midi(x) if x!=float(0) else float(0) for x in gtf]

    for j in range(0, 500, win_size):
        y_train_tmp = getonehot(gtf_midi[j:j + win_size])
        Y.append(y_train_tmp)
    return np.array(Y)


def get_groundtruth(f):
    yhot = np.zeros((len(f),len(pitch_range)))
    # f_midi = [librosa.hz_to_midi(x) if x!=float(0) else float(0) for x in f]
    for i in range(len(f)):
        # print(f[i],f_midi[i])
        idx = closest(pitch_range,f[i])
        yhot[i,:] = onehot_pitch_range[idx,:]
    return yhot

def get_annotated_pitch_data():
    filename = os.listdir('UploadedWavFile')[0]
    melody_data = json.load(open('static/melody_data/melody_data_'+str(filename)+'.json'))
    return melody_data['f']

#this function will be called when each time the annotator will update the frequency value
# def train_maml(model,X,Y_pred,lr_inner=0.00001):
#     #adaptation by meta-learning
#     model.layers[0].trainable = False
#     model.layers[1].trainable = False
#     model.layers[2].trainable = False
#     model.layers[3].trainable = False
#     model.layers[4].trainable = False

#     with tf.GradientTape() as test_tape:
#         with tf.GradientTape() as train_tape:
#             # print(X.shape)
#             Y_pred = model.call(X)
#             Y_conf = model_conf.call(X)
#             # print(Y_conf.shape)
#             conf_frames = get_indx_conf(Y_pred,Y_conf)
#             # print('Conf frames:',conf_frames)
#             up_fv = get_annotated_pitch_data()
#             Y_gt = get_groundtruth(up_fv)
#             Y_gt = Y_gt[np.newaxis,:,:]
#             # print(Y_gt.shape)

#             loss = support_custom_loss(Y_gt,Y_pred,conf_frames)
#             # print(loss)
#             gradients = train_tape.gradient(loss, model.trainable_variables)      
#             k=0
#             model_copy = copy_model(model,X)                    
            
#             model_copy.layers[0].trainable=False
#             model_copy.layers[1].trainable=False
#             model_copy.layers[2].trainable=False
#             model_copy.layers[3].trainable=False
#             model_copy.layers[4].trainable=False
            
#             for j in range(0,len(model_copy.layers)):
#                 if model_copy.layers[j].trainable:
#                     model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel,
#                             tf.multiply(lr_inner, gradients[k]))
#                     model_copy.layers[j].bias = tf.subtract(model.layers[j].bias,
#                             tf.multiply(lr_inner, gradients[k+1]))
#                     k += 2
                    
#             for w in range(10):
#                 grads = compute_gradients(model_copy,X,Y_gt,conf_frames)
#                 apply_gradients(base_optimizer,grads,model_copy.trainable_variables)
                                           
#         logits = model_copy.call(X)
#         loss = custom_loss(Y_gt,logits)
#         gradients = test_tape.gradient(loss,model.trainable_variables)
#     meta_optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    
#     Y_pred = model.call(x)

def train_maml(model,X,support_indx,gfv,lr_inner=0.00001):
    X = X.T
    X = tf.convert_to_tensor(X)
    X = (X-mean)/std
    X = X[tf.newaxis, :, :, tf.newaxis]
    
    #adaptation by meta-learning
    model.layers[0].trainable = False
    model.layers[1].trainable = False
    model.layers[2].trainable = False
    model.layers[3].trainable = False
    model.layers[4].trainable = False

    with tf.GradientTape() as test_tape:
        with tf.GradientTape() as train_tape:
            # print(X.shape)
            Y_pred,_ = model.call(X)
            Y_conf = model_conf.call(X)
            # print(Y_conf.shape)
            conf_frames = np.array(support_indx)
            
            Y_gt = get_groundtruth(gfv)
            print(Y_gt[conf_frames[0]])
            Y_gt = Y_gt[np.newaxis,:,:]
            print(Y_gt.shape,Y_pred.shape)            
            
            conf_frames = conf_frames[np.newaxis,:]
            print('Conf frames:',conf_frames)

            # for i in range(Y_gt.shape[0]):
            #     for j in range(Y_gt.shape[1]):
            #         print(j, np.argmax(Y_gt[i,j],axis=-1),np.argmax(Y_pred[i,j],axis=-1))            

            loss = support_custom_loss(Y_gt,Y_pred,conf_frames)
            print('support loss:',loss)
            gradients = train_tape.gradient(loss, model.trainable_variables)      
            k=0
            model_copy = copy_model(model,X)                    
            
            model_copy.layers[0].trainable=False
            model_copy.layers[1].trainable=False
            model_copy.layers[2].trainable=False
            model_copy.layers[3].trainable=False
            model_copy.layers[4].trainable=False
            
    #         for j in range(0,len(model_copy.layers)):
    #             if model_copy.layers[j].trainable:
    #                 model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel,
    #                         tf.multiply(lr_inner, gradients[k]))
    #                 model_copy.layers[j].bias = tf.subtract(model.layers[j].bias,
    #                         tf.multiply(lr_inner, gradients[k+1]))
    #                 k += 2
                    
    #         for w in range(10):
    #             grads = compute_gradients(model_copy,X,Y_gt,conf_frames)
    #             apply_gradients(base_optimizer,grads,model_copy.trainable_variables)
                                           
    #     logits = model_copy.call(X)
    #     loss = custom_loss(Y_gt,logits)
    #     gradients = test_tape.gradient(loss,model.trainable_variables)
    # meta_optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    
    Y_pred,_ = model.call(X)
    ypred_gfv = np.argmax(Y_pred[0],axis=-1)
    Y_conf = model_conf.call(X)
    yconf_values = tf.reshape(Y_conf,[-1])
    
    return ypred_gfv,yconf_values
    

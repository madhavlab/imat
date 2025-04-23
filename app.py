import json
import re
import shutil
from scipy import signal
import numpy as np
import math
from flask import Flask, render_template, request, make_response, send_from_directory, jsonify, send_file
import os
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import mir_eval
import csv
from tcp_interface import *
from aml_test import *
import warnings
import socket

## weights to be used finally!!
# model.load_weights('./models/adaptive_weights/weights/meta_model/pre/meta_weights-500')
# model_conf.load_weights('./models/conf/conf-weights-6000')

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"

wavfileDir = 'UploadedWavFile'
chunkwavfileDir = './static/ChunkedWavFile'
resynthwavfileDir = './static/ResynthWavFile'
groundtruthDir = 'PredictedFile'
supportindxDir = 'supportindxfile'
filename = None

fileIndx = 0

@app.route('/')   #@app.route is the python decorator
def index():
  global fileIndx
  fileIndx = 0
  for filename in os.listdir(wavfileDir):
            file_path = os.path.join(wavfileDir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
  for filename in os.listdir(chunkwavfileDir):
          file_path = os.path.join(chunkwavfileDir, filename)
          if os.path.isfile(file_path):
              os.remove(file_path)
  for filename in os.listdir(resynthwavfileDir):
          file_path = os.path.join(resynthwavfileDir, filename)
          if os.path.isfile(file_path):
              os.remove(file_path)
  return render_template('index.html')
  
 
@app.route('/#browse')
def browse():
  return render_template('index.html')

def get_spectrogram_json(y,sr):
    N = 1024
    win_len = 512  # 256
    hop_len = 80
    Sxx = np.abs(librosa.stft(y, n_fft=N, hop_length=hop_len, win_length=win_len, window='hann'))
    Sxx = librosa.power_to_db(Sxx,ref=np.max)
    f = np.arange(0, 1 + N / 2) * sr / N
    t = [i * hop_len / sr for i in range(Sxx.shape[1])]
    data = {
        'x': t,
        'y': f.tolist(),
        'z': Sxx[:, :].tolist()
    }
    return Sxx,data
  
def calc_rpa(efv,filename,indexes):
  filename = filename.split('.')[0]
  gfv = []
  
  with open('./static/groundtruth/'+filename+'.csv','r') as fin:
    reader = csv.reader(fin)
    for row in reader:
        row = [float(i) for i in row]
        gfv.append(row[1])
  
  t = [0.01*i for i in range(len(efv))]
  t = np.array(t)
  gfv = np.array(gfv)
  efv = np.array(efv)
  
  if len(gfv)<len(efv):
    gfv = np.append(gfv,np.zeros(len(efv)-len(gfv)))
  else:
    efv = np.append(efv,np.zeros(len(gfv)-len(efv)))
  
  efv_trunc = [efv[i] for i in indexes]
  efv_trunc = np.array(efv_trunc)
  
  gfv_trunc = [gfv[i] for i in indexes]
  gfv_trunc = np.array(gfv_trunc)
  
  t_trunc = [0.01*i for i in range(len(efv_trunc))]
  t_trunc = np.array(t_trunc) 
  
  (ref_v, ref_c,est_v, est_c) = mir_eval.melody.to_cent_voicing(t_trunc,gfv_trunc,t_trunc,efv_trunc)
  RPA = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c,est_v, est_c)
  RCA = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c,est_v, est_c)
  OA = mir_eval.melody.overall_accuracy(ref_v, ref_c,est_v, est_c)
  print(RPA,OA)
  
  (ref_v, ref_c,est_v, est_c) = mir_eval.melody.to_cent_voicing(t,gfv,t,efv)
  RPA = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c,est_v, est_c)
  RCA = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c,est_v, est_c)
  OA = mir_eval.melody.overall_accuracy(ref_v, ref_c,est_v, est_c)
  print(RPA,OA)
  return gfv
  
def get_melody_json(Sxx,filename):
 
  _ , efv, t = plot_melody(model,Sxx)
  # print('efv len',len(efv))
  
  t = [round(x, 2) for x in t]
  data = {
        't': t,
        'f': efv
        }
  return data 
  

@app.route('/calculate_spectrogram', methods=['POST'])
def calculate_spectrogram():
  global fileIndx
  # print(f'File Index...{fileIndx}')
  file = request.files['file']
  fileName = file.filename
  
  file.save(os.path.join(wavfileDir, fileName))
  
  data, sr = librosa.load(os.path.join(wavfileDir, fileName), sr=8000)
  
  Sxx,spec_data = get_spectrogram_json(data,sr)
  # print('Melody spec',Sxx.shape)
  # print('Orignal RPA,RCA')
  melody_data = get_melody_json(Sxx,fileName) 
  return make_response([spec_data,melody_data])


@app.route('/get_conf_values',methods=['POST'])
def get_conf_values():
  file = request.files['file']
  fileName = file.filename
  
  data,sr = librosa.load(os.path.join(wavfileDir,fileName),sr=8000)
  
  Sxx,_ = get_spectrogram_json(data,sr)
  confValues = conf_values(model_conf,Sxx)
  confValues = confValues.numpy()
  confValues = confValues.reshape(-1)
  return confValues.tolist()

# @app.route('/get_sliced_audio_original',methods=['POST'])
# def get_sliced_audio_original():
#   file = request.files['file']
#   filename = file.filename
  
#   y,sr = librosa.load(os.path.join(wavfileDir,filename),sr=8000)
#   write(os.path.join(chunkwavfileDir,filename),sr,y)
#   # Set the correct Content-Length header
#   content_length = os.path.getsize(os.path.join(chunkwavfileDir, filename))
  
#   response = send_from_directory(chunkwavfileDir, filename)
#   response.headers["Content-Length"] = content_length  
#   return response


# @app.route('/get_sliced_audio_resynth',methods=['POST'])
# def get_sliced_audio_resynth():
#   file = request.files['file']
#   filename = file.filename
  
#   efv = request.form.get('array')
#   efv = json.loads(efv)
  
#   for key in efv:
#     efv = efv[key]
  
#   y,sr = librosa.load(os.path.join(wavfileDir,filename),sr=8000)
#   t = np.array([0.01*i for i in range(len(efv))])
#   t_new = np.array([i/sr for i in range(len(y))])  
  
#   v = np.array([1 if i!=0 else 0 for i in efv])
#   f_new,vc = mir_eval.melody.resample_melody_series(t,efv,v,t_new,kind='nearest')
#   y_resynth = pitch2wav(f_new,t_new)
 
#   write(os.path.join(resynthwavfileDir,filename),sr,y_resynth)
  
#   content_length = os.path.getsize(os.path.join(resynthwavfileDir, filename))
    
#   response = send_from_directory(resynthwavfileDir, filename)
#   response.headers["Content-Length"] = content_length
#   return response
   
  
  
  
@app.route('/get_sliced_audio_original',methods=['POST'])
def get_sliced_audio_original():
  file = request.files['file']
  filename = file.filename
  
  start_time = float(request.form.get('start_time'))
  end_time = float(request.form.get('end_time'))
  
  y,sr = librosa.load(os.path.join(wavfileDir,filename),sr=8000, offset= start_time, duration=end_time-start_time)
  write(os.path.join(chunkwavfileDir,filename),sr,y)
  # Set the correct Content-Length header
  content_length = os.path.getsize(os.path.join(chunkwavfileDir, filename))
  
  response = send_from_directory(chunkwavfileDir, filename)
  response.headers["Content-Length"] = content_length
  
  return response

@app.route('/get_sliced_audio_resynth',methods=['POST'])
def get_sliced_audio_resynth():
  file = request.files['file']
  filename = file.filename
  
  efv = request.form.get('array')
  efv = json.loads(efv)
  
  for key in efv:
    efv = efv[key]
  
  start_time = float(request.form.get('start_time'))
  end_time = float(request.form.get('end_time'))
  
  y,sr = librosa.load(os.path.join(wavfileDir,filename),sr=8000)
  t = np.array([0.01*i for i in range(len(efv))])
  t_new = np.array([i/sr for i in range(len(y))])  
  
  v = np.array([1 if i!=0 else 0 for i in efv])
  f_new,vc = mir_eval.melody.resample_melody_series(t,efv,v,t_new,kind='nearest')
  y_resynth = pitch2wav(f_new,t_new)
 
  write(os.path.join(resynthwavfileDir,filename),sr,y_resynth)
  
  y_resynth_chunked,sr = librosa.load(os.path.join(resynthwavfileDir,filename),sr=None, mono=True, offset= start_time, duration=end_time-start_time)

  write(os.path.join(resynthwavfileDir,filename),sr,y_resynth_chunked)
  
  content_length = os.path.getsize(os.path.join(resynthwavfileDir, filename))
    
  response = send_from_directory(resynthwavfileDir, filename)
  response.headers["Content-Length"] = content_length
  return response
  

def pitch2wav(f,t, FLAG_extend=False):
    if FLAG_extend:
        for n in range(1,len(f)):
            if f[n] == 0:
                f[n] = f[n-1]

    theta = [2*np.pi*f[0]*t[0]]
    for i in range(1,len(f)):
        delta_theta = 0.5* (2*np.pi*f[i]+2*np.pi*f[i-1]) * (t[i]-t[i-1])
        theta.append(theta[i-1]+delta_theta)
    return 0.5*np.sin(theta)
  
  
# @app.route('/retrain_model',methods=['POST'])
# def retrain_model():
#   file = request.files['file']
#   filename = file.filename
  
#   efv = request.form.get('orig_array')
#   efv = json.loads(efv)  
#   for key in efv:
#     efv = efv[key]
    
#   # conf_values = request.form.get('conf_values')
#   # conf_values = json.loads(conf_values)
 
    
#   retrainValues = request.form.get('retrain_values')
#   retrainValues = json.loads(retrainValues)
  
#   unique_retrainValues = set(map(tuple, retrainValues))
#   unique_retrainValues = [list(item) for item in unique_retrainValues]
  
#   unique_retrainValues = sorted(unique_retrainValues, key=lambda x: x[0]) 
#   support_indx, gfv = zip(*unique_retrainValues)
#   support_indx = list(support_indx)
#   print('support index',support_indx)
#   gfv = list(gfv)
  
#   # init_index = int(request.form.get('initial_index'))  
#   # end_index = int(request.form.get('end_index'))
  
#   data,sr = librosa.load(os.path.join(wavfileDir,filename),sr=8000)
  
#   Sxx,_ = get_spectrogram_json(data,sr)
    
#   final_gfv = [0] * len(efv)
  
#   for index,value in zip(support_indx,gfv):
#     final_gfv[index] = value       
  
#   updated_gfv, updated_conf = aml_test(Sxx,support_indx,final_gfv,init_index,end_index) 
#   # updated_conf = updated_conf[init_index:end_index]
#   # updated_conf = updated_conf.numpy()
#   return make_response([updated_gfv, updated_conf.tolist(), support_indx])



@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    """
    Retrain the model using user annotations.
    
    Returns:
        Updated melody frequencies
    """
    global fileIndx, model, model_conf
    
    file = request.files['file']
    filename = file.filename
    
    # Get original melody prediction
    original_gfv = request.form.get('orig_array')
    original_gfv = json.loads(original_gfv)
    for key in original_gfv:
        original_gfv = original_gfv[key]
    
    # Get confidence values
    conf_val = request.form.get('conf_values')
    conf_val = json.loads(conf_val)
    
    # Get user annotations
    retrainValues = request.form.get('retrain_values')
    retrainValues = json.loads(retrainValues)
    
    # Process annotations to get unique points
    unique_retrainValues = set(map(tuple, retrainValues))
    unique_retrainValues = [list(item) for item in unique_retrainValues]
    unique_retrainValues = sorted(unique_retrainValues, key=lambda x: x[0])
    
    # Extract support indices and ground truth frequencies
    support_indx, gfv = zip(*unique_retrainValues)
    support_indx = list(support_indx)
    support_indx = [int(i) for i in support_indx]
    gfv = list(gfv)
    
    # Find indices not in support set
    original_indexes = set(support_indx)
    query_indx = [x for x in range(len(original_gfv)) if x not in original_indexes]
    
    # Load audio and calculate spectrogram
    data, sr = librosa.load(os.path.join(wavfileDir, filename), sr=8000)
    Sxx, _ = get_spectrogram_json(data, sr)
    
    # Prepare for retraining
    final_gfv = [0] * len(original_gfv)
    active_frames = [0] * len(original_gfv)
    
    # Mark active frames and set ground truth values
    for index, value in zip(support_indx, gfv):
        final_gfv[index] = value
        active_frames[index] = 1
    
    # Retrain model and get updated predictions
    updated_gfv = aml(Sxx, active_frames, final_gfv, fileIndx, model, model_conf, updatedWeightsDirPre+'/weights')

    # IMPORTANT: Load the updated weights into the global models
    model.load_weights(updatedWeightsDirPre+'/weights')
    model_conf.load_weights(updatedWeightsDirConf+'/weights')

    fileIndx += 1

    return make_response([updated_gfv])

@app.route('/download', methods=['POST'])
def download():
    """
    Download the extracted melody as a CSV file.
    
    Returns:
        Empty response, file is saved server-side
    """
    file = request.files['file']
    filename = file.filename
    filename = filename.split('.')[0]
    
    efv = request.form.get('freq')
    efv = json.loads(efv)
    for key in efv:
        efv = efv[key]
    
    rows = []
    
    for i in range(1, len(efv)):
        rows.append([i * 0.01, efv[i]])
    
    filename = filename + '.csv'
    
    with open(os.path.join(groundtruthDir, filename), 'w') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
    
    return '', 204

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to any available port
        return s.getsockname()[1]

if __name__ == "__main__":
    port = find_free_port()
    print(f"Running on port {port}")
    app.run(debug=True, port=port)


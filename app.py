import json
import os
import csv
import numpy as np
import librosa
from scipy.io.wavfile import write
import mir_eval
from flask import Flask, render_template, request, make_response, send_from_directory, jsonify

# Import from utility modules

import utils as ut
import melody_processing as mp

# Initialize Flask app
app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"

# Directory configuration
wavfileDir = './AnnotatedData/UploadedWavFile'
chunkwavfileDir = './dynamic/ChunkedWavFile'
resynthwavfileDir = './dynamic/ResynthWavFile'
groundtruthDir = './AnnotatedData/AnnotatedPitch'

updatedWeightsDirPre = './models/updated_weights/pre'
updatedWeightsDirConf = './models/updated_weights/conf'

# Create directories if they don't exist
def ensure_directories_exist():
    directories = [
        wavfileDir,
        chunkwavfileDir,
        resynthwavfileDir, 
        groundtruthDir,
        updatedWeightsDirPre,
        updatedWeightsDirConf
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

# Create directories when app starts
ensure_directories_exist()

# Initialize global variables
fileIndx = 0

# Load pre-trained models
model = ut.melody_extraction()
model.build_graph([500, 513, 1])

model_conf = ut.ConfidenceModel(model)
model_conf.build_graph([500, 513, 1])
model.load_weights('./models/pre/weights')
model_conf.load_weights('./models/conf/weights')

@app.route('/')
def index():
    """Initialize the application and clean up directories."""
    global fileIndx
    fileIndx = 0
    
    # Clean up directories
    for directory in [wavfileDir, chunkwavfileDir, resynthwavfileDir]:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
    return render_template('index.html')

@app.route('/#browse')
def browse():
  """Alternative route to index."""
  # Clean up directories
  for directory in [updatedWeightsDirPre,updatedWeightsDirConf]:
      for filename in os.listdir(directory):
          file_path = os.path.join(directory, filename)
          if os.path.isfile(file_path):
              os.remove(file_path)
  return render_template('index.html')


@app.route('/calculate_spectrogram', methods=['POST'])
def calculate_spectrogram():
    """
    Calculate spectrogram and melody from uploaded audio file.
    
    Returns:
        JSON response with spectrogram and melody data
    """
    global fileIndx
    
    file = request.files['file']
    fileName = file.filename
    
    # Save uploaded file
    file.save(os.path.join(wavfileDir, fileName))
    
    # Load audio and calculate spectrogram
    data, sr = librosa.load(os.path.join(wavfileDir, fileName), sr=8000)
    Sxx, spec_data = mp.get_spectrogram_json(data, sr)
    
    # Calculate melody
    melody_data = mp.get_melody_json(model, Sxx)    
    return make_response([spec_data, melody_data])


@app.route('/get_conf_values', methods=['POST'])
def get_conf_values():
    """
    Get confidence values for melody prediction.
    
    Returns:
        List of confidence values
    """
    file = request.files['file']
    fileName = file.filename
    
    data, sr = librosa.load(os.path.join(wavfileDir, fileName), sr=8000)
    
    Sxx, _ = mp.get_spectrogram_json(data, sr)
    confValues = mp.conf_values(model_conf, Sxx)
    confValues = confValues.numpy()
    confValues = confValues.reshape(-1)
    return confValues.tolist()

@app.route('/get_sliced_audio_original', methods=['POST'])
def get_sliced_audio_original():
    """
    Extract and serve a slice of the original audio.
    
    Returns:
        Audio file response
    """
    file = request.files['file']
    filename = file.filename
    
    start_time = float(request.form.get('start_time'))
    end_time = float(request.form.get('end_time'))
    
    # Load and slice audio
    y, sr = librosa.load(os.path.join(wavfileDir, filename), sr=8000, offset=start_time, duration=end_time-start_time)
    write(os.path.join(chunkwavfileDir, filename), sr, y)
    
    # Set the correct Content-Length header
    content_length = os.path.getsize(os.path.join(chunkwavfileDir, filename))
    
    response = send_from_directory(chunkwavfileDir, filename)
    response.headers["Content-Length"] = content_length
    
    return response

@app.route('/get_sliced_audio_resynth', methods=['POST'])
def get_sliced_audio_resynth():
    """
    Extract, resynthesize, and serve a slice of audio based on melody.
    
    Returns:
        Audio file response with resynthesized melody
    """
    file = request.files['file']
    filename = file.filename
    
    efv = request.form.get('array')
    efv = json.loads(efv)
    
    for key in efv:
        efv = efv[key]
    
    start_time = float(request.form.get('start_time'))
    end_time = float(request.form.get('end_time'))
    
    # Load original audio
    y, sr = librosa.load(os.path.join(wavfileDir, filename), sr=8000)
    
    # Create time series
    t = np.array([0.01 * i for i in range(len(efv))])
    t_new = np.array([i/sr for i in range(len(y))])
    
    # Prepare voicing
    v = np.array([1 if i != 0 else 0 for i in efv])
    
    # Resample melody to audio time points
    f_new, vc = mir_eval.melody.resample_melody_series(t, efv, v, t_new, kind='nearest')
    
    # Create synthesized audio
    y_resynth = mp.pitch2wav(f_new, t_new)
    write(os.path.join(resynthwavfileDir, filename), sr, y_resynth)
    
    # Extract slice
    y_resynth_chunked, sr = librosa.load(os.path.join(resynthwavfileDir, filename), 
                                         sr=None, mono=True, offset=start_time, 
                                         duration=end_time-start_time)
    
    write(os.path.join(resynthwavfileDir, filename), sr, y_resynth_chunked)
    
    content_length = os.path.getsize(os.path.join(resynthwavfileDir, filename))
    
    response = send_from_directory(resynthwavfileDir, filename)
    response.headers["Content-Length"] = content_length
    return response


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
    Sxx, _ = mp.get_spectrogram_json(data, sr)
    
    # Prepare for retraining
    final_gfv = [0] * len(original_gfv)
    active_frames = [0] * len(original_gfv)
    
    # Mark active frames and set ground truth values
    for index, value in zip(support_indx, gfv):
        final_gfv[index] = value
        active_frames[index] = 1
    
    # Retrain model and get updated predictions
    updated_gfv = mp.aml(Sxx, active_frames, final_gfv, fileIndx, model, model_conf, updatedWeightsDirPre+'/weights')

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


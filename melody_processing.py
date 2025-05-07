import numpy as np
import librosa
import mir_eval
import tensorflow as tf
import os
import utils as ut
import csv


def get_spectrogram_json(y, sr):
    """
    Generate spectrogram data in JSON format for visualization.
    
    Args:
        y: Audio signal
        sr: Sample rate
    
    Returns:
        Sxx: Spectrogram matrix
        data: JSON-compatible dictionary with x, y, z values
    """
    N = 1024
    win_len = 512
    hop_len = 80
    Sxx = np.abs(librosa.stft(y, n_fft=N, hop_length=hop_len, win_length=win_len, window='hann'))
    Sxx = librosa.power_to_db(Sxx, ref=np.max)
    f = np.arange(0, 1 + N / 2) * sr / N
    t = [i * hop_len / sr for i in range(Sxx.shape[1])]
    data = {
        'x': t,
        'y': f.tolist(),
        'z': Sxx[:, :].tolist()
    }
    return Sxx, data


def plot_melody(model, S):
    """
    Predict melody contour from spectrogram.
    
    Args:
        model: Melody extraction model
        S: Spectrogram
    
    Returns:
        y_p: Raw prediction output
        efv: Estimated frequency values
        t: Time points
    """
    efv = []
    t = []
    
    S = S.T
    S = tf.convert_to_tensor(S)
    S = (S - ut.mean) / ut.std
    S = S[tf.newaxis, :, :, tf.newaxis]

    y_p, _ = model.call(S)
    
    for j in range(y_p.shape[0]):
        for i in range(y_p.shape[1]):
            indx = np.argmax(y_p[j, i, :])
            efv.append(ut.pitch_range[indx])
            t.append(i * 0.01)
            
    return y_p, efv, t

def get_melody_json(model, Sxx):
    """
    Get melody prediction as JSON for visualization.
    
    Args:
        model: Melody extraction model
        Sxx: Spectrogram
    
    Returns:
        data: JSON-compatible dictionary with time and frequency values
    """
    _, efv, t = plot_melody(model, Sxx)
    
    t = [round(x, 2) for x in t]
    data = {
        't': t,
        'f': efv
    }
    return data

def conf_values(model_conf, S):
    """
    Calculate confidence values for melody prediction.
    
    Args:
        model_conf: Confidence model
        S: Spectrogram
    
    Returns:
        conf_values: Confidence values
    """
    S = S.T
    S = tf.convert_to_tensor(S)
    S = (S - ut.mean) / ut.std
    S = S[tf.newaxis, :, :, tf.newaxis]
    
    conf_values = model_conf.call(S)
    return conf_values


def pitch2wav(f, t, FLAG_extend=False):
    """
    Convert pitch contour to audio waveform.
    
    Args:
        f: Frequency values
        t: Time points
        FLAG_extend: Whether to extend the last non-zero frequency
    
    Returns:
        Audio waveform
    """
    if FLAG_extend:
        for n in range(1, len(f)):
            if f[n] == 0:
                f[n] = f[n-1]

    theta = [2 * np.pi * f[0] * t[0]]
    for i in range(1, len(f)):
        delta_theta = 0.5 * (2 * np.pi * f[i] + 2 * np.pi * f[i-1]) * (t[i] - t[i-1])
        theta.append(theta[i-1] + delta_theta)
    return 0.5 * np.sin(theta)

def calc_rpa(efv, filename, indexes, groundtruth_dir='./static/groundtruth/'):
    """
    Calculate evaluation metrics for melody extraction.
    
    Args:
        efv: Estimated frequency values
        filename: Filename (for ground truth lookup)
        indexes: Indices to consider
        groundtruth_dir: Directory containing ground truth files
    
    Returns:
        gfv: Ground truth frequency values
    """
    filename = filename.split('.')[0]
    gfv = []
    
    with open(os.path.join(groundtruth_dir, filename + '.csv'), 'r') as fin:
        reader = csv.reader(fin)
        for row in reader:
            row = [float(i) for i in row]
            gfv.append(row[1])
    
    t = [0.01 * i for i in range(len(efv))]
    t = np.array(t)
    gfv = np.array(gfv)
    efv = np.array(efv)
    
    # Adjust array lengths if necessary
    if len(gfv) < len(efv):
        gfv = np.append(gfv, np.zeros(len(efv) - len(gfv)))
    else:
        efv = np.append(efv, np.zeros(len(gfv) - len(efv)))
    
    # Calculate metrics for specified indexes
    efv_trunc = [efv[i] for i in indexes]
    efv_trunc = np.array(efv_trunc)
    
    gfv_trunc = [gfv[i] for i in indexes]
    gfv_trunc = np.array(gfv_trunc)
    
    t_trunc = [0.01 * i for i in range(len(efv_trunc))]
    t_trunc = np.array(t_trunc)
    
    (ref_v, ref_c, est_v, est_c) = mir_eval.melody.to_cent_voicing(t_trunc, gfv_trunc, t_trunc, efv_trunc)
    RPA = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
    RCA = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
    OA = mir_eval.melody.overall_accuracy(ref_v, ref_c, est_v, est_c)
    
    # Calculate metrics for full dataset
    (ref_v, ref_c, est_v, est_c) = mir_eval.melody.to_cent_voicing(t, gfv, t, efv)
    RPA_full = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
    RCA_full = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
    OA_full = mir_eval.melody.overall_accuracy(ref_v, ref_c, est_v, est_c)
    
    return RPA_full, RCA_full, OA_full



def support_pretrain_step(x, y, ts, meta_model_pre, inner_optimizer, inner_step):
    """Train melody extraction model on support indices."""
    print("Training melody extraction model...")
       
    with tf.device('/gpu:0'):
        for step in range(inner_step):
            with tf.GradientTape(persistent=True) as tape:
                ys_hat, _ = meta_model_pre.call(x)
                loss = ut.custom_loss(y, ys_hat, ts)
                print(f"  Step {step}: loss = {loss:.4f}")
                
            grads = tape.gradient(loss, meta_model_pre.trainable_variables)
            inner_optimizer.apply_gradients(zip(grads, meta_model_pre.trainable_variables))
    print("Melody extraction model training complete.")
    print('\n')
    return loss

def support_conftrain_step(x, y, ts, meta_model_pre, meta_model_conf, conf_inner_optimizer, inner_step):
    """Train confidence model on support indices."""
    print("Training confidence model...")
    
    # Convert ts to tensor format expected by conf_loss
    ts_tensor = tf.convert_to_tensor(ts, dtype=tf.int32)
    
    with tf.device('/gpu:0'):
        for step in range(inner_step):
            with tf.GradientTape(persistent=True) as tape:
                ys_hat, _ = meta_model_pre.call(x)
                yc_hat = meta_model_conf.call(x)
                loss = ut.conf_loss(y, ys_hat, yc_hat, ts_tensor)
                print(f"  Step {step}: loss = {loss:.4f}")

            grads = tape.gradient(loss, meta_model_conf.trainable_variables)
            conf_inner_optimizer.apply_gradients(zip(grads, meta_model_conf.trainable_variables))    
    print("Confidence model training complete.")
    print('\n')
    return loss



def aml(S, active_frames, gfv, fileid, model, model_conf, weights_path='models/updated_weights/pre/weights'):
    # Calculate learning rate with decay
    base_alpha = 5.e-5  # 5.e-5  --> only 3 times retraining
    base_beta = 5.e-5

    alpha = base_alpha * (0.9 ** fileid) if fileid > 0 else base_alpha
    beta = base_beta * (0.9 ** fileid) if fileid > 0 else base_beta
    
    inner_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
    conf_inner_optimizer = tf.keras.optimizers.Adam(learning_rate=beta)

    inner_step = 10
    
    # Prepare data
    S = S.T
    S = tf.convert_to_tensor(S)
    S = (S-ut.mean)/ut.std
    S = S[tf.newaxis, :, :, tf.newaxis]    

    # Create models
    meta_trained_classifier = ut.melody_extraction()
    dummy_out, _ = meta_trained_classifier(S)
    
    # Set trainable flags - keep ONLY the final dense layers trainable
    for layer in meta_trained_classifier.layers:
        layer.trainable = False  # Start by making all layers non-trainable
    
    # Make specific layers trainable
    # meta_trained_classifier.linear1.trainable = True
    meta_trained_classifier.final.trainable = True
    
    # Create and build confidence model
    meta_trained_conf = ut.ConfidenceModel(meta_trained_classifier)
    _ = meta_trained_conf(S)
    
    # Set confidence model trainability
    # meta_trained_conf.dense2.trainable = True
    meta_trained_conf.final.trainable = True
    
    # Print diagnostics
    print(f"File ID: {fileid}")
    
    # Load weights based on training iteration
    if fileid == 0:
        meta_trained_classifier.set_weights(model.get_weights())
        meta_trained_conf.set_weights(model_conf.get_weights())
    else:
        print('moved to else part')
        try:
            meta_trained_classifier.load_weights(weights_path)
            meta_trained_conf.load_weights(weights_path.replace('pre', 'conf'))
            print('models loaded')
        except Exception as e:
            print(f"Error loading weights: {e}. Using original weights.")
            meta_trained_classifier.set_weights(model.get_weights())
            meta_trained_conf.set_weights(model_conf.get_weights())    

    # Prepare ground truth melody and determine support indices
    Y_true = ut.get_onehot(gfv)
    Y_true = Y_true[tf.newaxis, :, :]
    
    # Convert active_frames to indices for confidence training
    support_indices = [i for i, val in enumerate(active_frames) if val == 1]

    # First update the melody extraction model using support indices
    support_pretrain_step(S, Y_true, active_frames, meta_trained_classifier, inner_optimizer, inner_step)
    
    # Then update the confidence model using the same support indices
    support_conftrain_step(S, Y_true, support_indices, meta_trained_classifier, meta_trained_conf, conf_inner_optimizer, inner_step)
    
    # Save updated weights
    meta_trained_classifier.save_weights(weights_path)
    meta_trained_conf.save_weights(weights_path.replace('pre', 'conf'))
    
    # Generate melody prediction with updated model
    yq_hat, _ = meta_trained_classifier.call(S)
    yq_conf = meta_trained_conf.call(S)
   
    # Convert to frequency values
    efv = []
    for j in range(yq_hat.shape[0]):
        for i in range(yq_hat.shape[1]):
            indx = np.argmax(yq_hat[j, i, :])
            efv.append(ut.pitch_range[indx])

    return efv

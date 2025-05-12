import unittest
import numpy as np
import tensorflow as tf
import os
import json
import io
import sys
import librosa
import csv
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the path so we can import the modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import modules (but not app)
import utils as ut
import melody_processing as mp

class ImatTestsWithRealAudio(unittest.TestCase):
    """Test IMAT functionality using real audio files."""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment once for all tests."""
        # Path to the project root
        cls.project_root = project_root
        
        # Define paths to test data
        cls.test_data_dir = os.path.join(cls.project_root, 'test', 'test_data')
        cls.test_audio_dir = os.path.join(cls.test_data_dir, 'audio')
        cls.test_pitch_dir = os.path.join(cls.test_data_dir, 'ground_truth')
        
        # Create directories if they don't exist
        os.makedirs(cls.test_audio_dir, exist_ok=True)
        os.makedirs(cls.test_pitch_dir, exist_ok=True)
        
        # Correct paths to model weights (use absolute paths)
        cls.weights_path = os.path.join(cls.project_root, 'models', 'pre', 'weights')
        cls.conf_weights_path = os.path.join(cls.project_root, 'models', 'conf', 'weights')
        
        # Initialize models
        cls.model = ut.melody_extraction()
        cls.model.build_graph([500, 513, 1])
        
        # Try to load weights if they exist
        try:
            cls.model.load_weights(cls.weights_path)
            print(f"Successfully loaded melody model weights from {cls.weights_path}")
            weights_loaded = True
        except Exception as e:
            print(f"Warning: Could not load melody model weights: {e}")
            print("Tests will run with randomly initialized weights")
            weights_loaded = False
        
        # Initialize confidence model
        cls.model_conf = ut.ConfidenceModel(cls.model)
        cls.model_conf.build_graph([500, 513, 1])
        
        # Try to load confidence model weights if they exist
        if weights_loaded:
            try:
                cls.model_conf.load_weights(cls.conf_weights_path)
                print(f"Successfully loaded confidence model weights from {cls.conf_weights_path}")
            except Exception as e:
                print(f"Warning: Could not load confidence model weights: {e}")
        
        # Find available test files
        cls.available_audio_files = []
        if os.path.exists(cls.test_audio_dir):
            for filename in os.listdir(cls.test_audio_dir):
                if filename.endswith('.wav'):
                    cls.available_audio_files.append(filename)
        
        print(f"Found {len(cls.available_audio_files)} test audio files: {cls.available_audio_files}")
       
        # Create temporary directories for test outputs
        cls.temp_dir = os.path.join(cls.test_data_dir, 'temp')
        cls.results_dir = os.path.join(cls.test_data_dir, 'results')
        os.makedirs(cls.temp_dir, exist_ok=True)
        os.makedirs(cls.results_dir, exist_ok=True)
        
        # Set timestamp for this test run
        cls.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def test_audio_loading(self):
        """Test loading audio files."""
        if not self.available_audio_files:
            self.skipTest("No test audio files available")
        
        print("\n=== Testing Audio Loading ===")
        for filename in self.available_audio_files:
            file_path = os.path.join(self.test_audio_dir, filename)
            
            # Load audio
            y, sr = librosa.load(file_path, sr=8000)
            
            # Check audio properties
            self.assertIsNotNone(y)
            self.assertTrue(len(y) > 0)
            self.assertEqual(sr, 8000)
            
            print(f"Successfully loaded audio: {filename}, duration: {len(y)/sr:.2f}s")
    
    def test_spectrogram_generation(self):
        """Test spectrogram generation from audio files."""
        if not self.available_audio_files:
            self.skipTest("No test audio files available")
        
        print("\n=== Testing Spectrogram Generation ===")
        for filename in self.available_audio_files:
            file_path = os.path.join(self.test_audio_dir, filename)
            
            # Load audio
            y, sr = librosa.load(file_path, sr=8000)
            
            # Generate spectrogram
            Sxx, spec_data = mp.get_spectrogram_json(y, sr)
            
            # Check spectrogram dimensions
            self.assertEqual(Sxx.shape[0], 513)
            self.assertTrue(Sxx.shape[1] > 0)
            
            # Check JSON data
            self.assertIn('x', spec_data)
            self.assertIn('y', spec_data)
            self.assertIn('z', spec_data)
            
            print(f"Generated spectrogram for {filename}, shape: {Sxx.shape}")
    
    def test_melody_extraction(self):
        """Test melody extraction from audio files."""
        if not self.available_audio_files:
            self.skipTest("No test audio files available")
        
        print("\n=== Testing Initial Melody Extraction ===")
        results = []
        
        for filename in self.available_audio_files:
            file_path = os.path.join(self.test_audio_dir, filename)
            
            # Load audio
            y, sr = librosa.load(file_path, sr=8000)
            
            # Generate spectrogram
            Sxx, _ = mp.get_spectrogram_json(y, sr)
            
            # Extract melody
            melody_data = mp.get_melody_json(self.model, Sxx)
            
            # Check melody data structure
            self.assertIn('t', melody_data)
            self.assertIn('f', melody_data)
            self.assertEqual(len(melody_data['t']), len(melody_data['f']))
            
            # Check frequency values
            frequencies = melody_data['f']
            self.assertTrue(all(0 <= f < 2000 for f in frequencies))
            
            # Calculate metrics
            try:
                _, _, oa = mp.calc_rpa(np.array(frequencies), 
                                          filename.split('.')[0], 
                                          np.arange(len(frequencies)), 
                                          groundtruth_dir=self.test_pitch_dir)
                
                print(f"Extracted melody from {filename} - Initial OA: {oa:.4f}")
                results.append({
                    'filename': filename,
                    'initial_oa': oa
                })
            except Exception as e:
                print(f"Error calculating metrics for {filename}: {e}")
            
            # Save the extracted melody frequencies
            np.savetxt(os.path.join(self.results_dir, f"{os.path.splitext(filename)[0]}_melody.csv"),
                      np.array(frequencies), delimiter=",", fmt="%.2f")
        
        # Save overall results
        if results:
            avg_oa = sum(r['initial_oa'] for r in results) / len(results)
            
            print("\nAveraged Initial Metrics:")
            print(f"Average OA: {avg_oa:.4f}")
            
            # Save results to CSV
            with open(os.path.join(self.results_dir, f"initial_metrics_{self.timestamp}.csv"), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Filename', 'OA'])
                for r in results:
                    writer.writerow([r['filename'], r['initial_oa']])
    
    def test_confidence_estimation(self):
        """Test confidence estimation from audio files."""
        if not self.available_audio_files:
            self.skipTest("No test audio files available")
        
        print("\n=== Testing Confidence Estimation ===")
        
        for filename in self.available_audio_files:
            file_path = os.path.join(self.test_audio_dir, filename)
            
            # Load audio
            y, sr = librosa.load(file_path, sr=8000)
            
            # Generate spectrogram
            Sxx, _ = mp.get_spectrogram_json(y, sr)
            
            # Get confidence values
            conf_values = mp.conf_values(self.model_conf, Sxx)
            conf_values = conf_values.numpy().flatten()
            
            # Check confidence values
            self.assertTrue(all(0 <= c <= 1 for c in conf_values))
            self.assertEqual(len(conf_values), Sxx.shape[1])
            
            # Calculate statistics
            low_conf = sum(1 for c in conf_values if c < 0.5)
            high_conf = sum(1 for c in conf_values if c >= 0.5)
            avg_conf = np.mean(conf_values)
                        
            # Save confidence values
            np.savetxt(os.path.join(self.results_dir, f"{os.path.splitext(filename)[0]}_confidence.csv"),
                      conf_values, delimiter=",", fmt="%.4f")
            print(f"Calculated Confidence for {filename}")

    
    def test_model_adaptation(self):
        """Test model adaptation with user corrections."""
        if not self.available_audio_files:
            self.skipTest("No test audio files available")
        
        print("\n=== Testing Model Adaptation ===")
        
        # Parameters for adaptation
        n_corrections = 200  # Max number of corrections per file
        
        # Create a shared weights directory for this adaptation run
        adaptation_weights_dir = os.path.join(self.temp_dir, f"adaptation_{self.timestamp}")
        os.makedirs(adaptation_weights_dir, exist_ok=True)
        weights_path = os.path.join(adaptation_weights_dir, "model_weights")
        
        # Track metrics for all files
        all_metrics = []
        
        for file_index, filename in enumerate(self.available_audio_files):
            print(f"\nProcessing file {file_index+1}/{len(self.available_audio_files)}: {filename}")
            file_path = os.path.join(self.test_audio_dir, filename)
            
            # Load audio
            y, sr = librosa.load(file_path, sr=8000)
            
            # Generate spectrogram
            Sxx, _ = mp.get_spectrogram_json(y, sr)
            
            # Get initial melody prediction
            melody_data = mp.get_melody_json(self.model, Sxx)
            initial_freqs = np.array(melody_data['f'])
            
            # Get initial confidence prediction
            ypred_conf = mp.conf_values(self.model_conf, Sxx)
            ypred_conf = ypred_conf.numpy().flatten()
            
            # Load ground truth
            gt_path = os.path.join(self.test_pitch_dir, filename.split('.')[0] + '.csv')
            if not os.path.exists(gt_path):
                print(f"Ground truth file not found for {filename}, skipping")
                continue
                
            gfv = []
            try:
                with open(gt_path, 'r') as fin:
                    reader = csv.reader(fin)
                    for row in reader:
                        row = [float(i) for i in row]
                        gfv.append(row[1])
                
                # Ensure ground truth and prediction have the same length
                if len(gfv) < Sxx.shape[1]:
                    gfv.extend(np.zeros(int(Sxx.shape[1]-len(gfv))))
                elif len(gfv) > Sxx.shape[1]:
                    gfv = gfv[:Sxx.shape[1]]
            except Exception as e:
                print(f"Error loading ground truth for {filename}: {e}")
                continue
            
            # Calculate initial metrics
            try:
                _, _, initial_oa = mp.calc_rpa(
                    initial_freqs, filename.split('.')[0], 
                    np.arange(len(initial_freqs)), 
                    groundtruth_dir=self.test_pitch_dir
                )
                print(f"Initial metrics - OA: {initial_oa:.4f}")
            except Exception as e:
                print(f"Error calculating initial metrics: {e}")
                continue
            
            # Create mock user corrections
            # Select frames with low confidence for correction
            low_conf_indices = np.where(ypred_conf < 0.5)[0]
            if len(low_conf_indices) > 0:                
                # Take the frames with lowest confidence first
                time_indices = low_conf_indices[np.argsort(ypred_conf[low_conf_indices])]
                time_indices = time_indices[:min(n_corrections, len(time_indices))]
                
                corrected_freqs = initial_freqs.copy()
                corrections_count = 0
                
                for idx in time_indices:
                    if gfv[idx] > 0 and idx < len(corrected_freqs):
                        # For voiced frames: simulate user correction with slight randomness
                        # within half a semitone of ground truth
                        lower_bound = gfv[idx] * (2**(-0.5/12))
                        upper_bound = gfv[idx] * (2**(0.5/12))
                        corrected_freqs[idx] = np.random.uniform(lower_bound, upper_bound)
                        corrections_count += 1
                    elif gfv[idx] == 0 and idx < len(corrected_freqs):
                        # For unvoiced frames: make it unvoiced
                        corrected_freqs[idx] = 0.0
                        corrections_count += 1
                                
                # Create active frames array for AML
                active_frames = np.zeros_like(initial_freqs)
                active_frames[time_indices] = 1
                
                try:
                    # Apply active-meta-learning
                    print("Applying active meta-learning...")
                    updated_freqs = mp.aml(
                        Sxx, active_frames, corrected_freqs, 0,
                        self.model, self.model_conf, weights_path
                    )
                    
                    # Calculate metrics after adaptation
                    _, _, adapted_oa = mp.calc_rpa(
                        updated_freqs, filename.split('.')[0],
                        np.arange(len(updated_freqs)),
                        groundtruth_dir=self.test_pitch_dir
                    )
                    
                    # Save updated predictions
                    np.savetxt(
                        os.path.join(self.results_dir, f"{os.path.splitext(filename)[0]}_adapted.csv"),
                        updated_freqs, delimiter=",", fmt="%.2f"
                    )
                    
                    # Calculate improvements
                    oa_improvement = adapted_oa - initial_oa
                    
                    print(f"Adapted metrics - OA: {adapted_oa:.4f}")
                    print(f"Improvements - OA: {oa_improvement:.4f}")
                    
                    # Store metrics
                    all_metrics.append({
                        'filename': filename,
                        'initial_oa': initial_oa,
                        'adapted_oa': adapted_oa,
                        'oa_improvement': oa_improvement,
                        'corrections': corrections_count
                    })
                    
                except Exception as e:
                    print(f"AML failed for {filename}: {str(e)}")
            else:
                print(f"No low confidence frames found for {filename}, skipping adaptation")
        
        # Print overall results
        if all_metrics:
            print("\n\n=== OVERALL ADAPTATION RESULTS ===")
            avg_initial_oa = sum(m['initial_oa'] for m in all_metrics) / len(all_metrics)
            
            avg_adapted_oa = sum(m['adapted_oa'] for m in all_metrics) / len(all_metrics)
            
            avg_oa_improvement = sum(m['oa_improvement'] for m in all_metrics) / len(all_metrics)
            
            print(f"Average Improvements - OA: {avg_oa_improvement:.4f}")
            
            # Save results to CSV
            results_file = os.path.join(self.results_dir, f"adaptation_results_{self.timestamp}.csv")
            with open(results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Filename', 
                    'Initial OA',
                    'Adapted OA',
                    'OA Improvement',
                    'Corrections'
                ])
                for m in all_metrics:
                    writer.writerow([
                        m['filename'],
                        m['initial_oa'],
                        m['adapted_oa'],
                        m['oa_improvement'],
                        m['corrections']
                    ])
            
            print(f"\nResults saved to {results_file}")
            
    
if __name__ == '__main__':
    unittest.main()
import unittest
import numpy as np
import tensorflow as tf
import os
import json
import io
import sys
import librosa

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
        cls.test_data_dir = os.path.join(cls.project_root, 'test','test_data')
        cls.test_audio_dir = os.path.join(cls.test_data_dir, 'audio')
        cls.test_pitch_dir = os.path.join(cls.test_data_dir, 'pitch')
        
        # # Create directories if they don't exist
        # os.makedirs(cls.test_audio_dir, exist_ok=True)
        # os.makedirs(cls.test_pitch_dir, exist_ok=True)
        
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
        os.makedirs(cls.temp_dir, exist_ok=True)
    
    def test_audio_loading(self):
        """Test loading audio files."""
        if not self.available_audio_files:
            self.skipTest("No test audio files available")
        
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
            
            # Check if we have any voiced frames
            voiced_frames = sum(1 for f in frequencies if f > 0)
            print(f"Extracted melody from {filename}, {voiced_frames}/{len(frequencies)} voiced frames")
            
            # Save the extracted melody frequencies for use in other tests
            np.save(os.path.join(self.temp_dir, f"{os.path.splitext(filename)[0]}_melody.csv"), 
                    np.array(frequencies))
    
    def test_confidence_estimation(self):
        """Test confidence estimation from audio files."""
        if not self.available_audio_files:
            self.skipTest("No test audio files available")
        
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
            
            print(f"Confidence for {filename}: {low_conf} low, {high_conf} high confidence frames")
            
            # Save confidence values for use in other tests
            np.save(os.path.join(self.temp_dir, f"{os.path.splitext(filename)[0]}_confidence.csv"),
                    conf_values)
    
    def test_model_adaptation(self):
        """Test model adaptation with user corrections."""
        if not self.available_audio_files:
            self.skipTest("No test audio files available")
        
        for filename in self.available_audio_files:
            file_path = os.path.join(self.test_audio_dir, filename)
            
            # Load audio
            y, sr = librosa.load(file_path, sr=8000)
            
            # Generate spectrogram
            Sxx, _ = mp.get_spectrogram_json(y, sr)
            
            # Get initial melody prediction
            melody_data = mp.get_melody_json(self.model, Sxx)
            initial_freqs = np.array(melody_data['f'])
            
            # Create mock user corrections
            # Modify a few frames to simulate user corrections
            n_corrections = 5
            time_indices = np.random.choice(len(initial_freqs), n_corrections, replace=False)
            
            corrected_freqs = initial_freqs.copy()
            for idx in time_indices:
                # Apply a correction
                if corrected_freqs[idx] > 0:
                    corrected_freqs[idx] *= 1.5  # 50% change
                else:
                    corrected_freqs[idx] = 440.0  # Add a note
            
            # Create active frames array for AML
            active_frames = np.zeros_like(initial_freqs)
            active_frames[time_indices] = 1
            
            # Create temporary weights path
            weights_path = os.path.join(self.temp_dir, f"{os.path.splitext(filename)[0]}_weights")
            
            try:
                # Apply active-meta-learning
                updated_freqs = mp.aml(Sxx, active_frames, corrected_freqs, 0,
                                    self.model, self.model_conf, weights_path)

                for idx in time_indices:
                    print(f"Time index {idx}: corrected = {corrected_freqs[idx]}, updated = {updated_freqs[idx]}")

                
                # # Check that corrected frames match the ground truth
                # for idx in time_indices:
                #     self.assertAlmostEqual(updated_freqs[idx], corrected_freqs[idx], delta=1)
                
                # # Check that at least some other frames also changed
                # n_changed = sum(1 for i in range(len(initial_freqs)) 
                #                if i not in time_indices and 
                #                abs(updated_freqs[i] - initial_freqs[i]) > 1)
                
                # self.assertTrue(n_changed > 0, 
                #                 f"No frames changed after adaptation (besides corrections)")
                
            except Exception as e:
                self.fail(f"AML failed for {filename}: {e}")

if __name__ == '__main__':
    unittest.main()
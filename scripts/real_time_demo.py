"""
Real-time Audio Processing Demo

This script demonstrates real-time audio capture, feature extraction,
and classification using trained models.
"""

import numpy as np
import pyaudio
import threading
import time
import queue
import pickle
from pathlib import Path
import sys
import os
import argparse
from typing import Optional, Dict, List, Callable
import warnings

warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from audio_processing import AudioProcessor
    from feature_extraction import FeatureExtractor
    from visualization import AudioVisualizer
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.patches as patches
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed")
    print("Note: Real-time demo requires pyaudio: pip install pyaudio")
    sys.exit(1)


class RealTimeAudioProcessor:
    """
    Real-time audio processing and classification system.
    
    Captures audio from microphone, extracts features, and performs
    live classification with visualization.
    """
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 chunk_size: int = 1024,
                 buffer_duration: float = 3.0):
        """
        Initialize real-time audio processor.
        
        Args:
            sample_rate: Audio sample rate
            chunk_size: Size of audio chunks to process
            buffer_duration: Duration of audio buffer in seconds
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_duration = buffer_duration
        self.buffer_size = int(sample_rate * buffer_duration)
        
        # Audio components
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate)
        
        # Audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Data buffers
        self.audio_buffer = np.zeros(self.buffer_size)
        self.feature_queue = queue.Queue(maxsize=100)
        self.prediction_queue = queue.Queue(maxsize=100)
        
        # Processing state
        self.is_recording = False
        self.processing_thread = None
        
        # Model and classes
        self.model = None
        self.class_names = []
        self.feature_types = []
        
    def load_model(self, model_path: str, pipeline_state_path: str) -> None:
        """
        Load trained model and pipeline state.
        
        Args:
            model_path: Path to saved model
            pipeline_state_path: Path to pipeline state pickle file
        """
        
        try:
            # Load TensorFlow model
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
            
            # Load pipeline state
            with open(pipeline_state_path, 'rb') as f:
                pipeline_state = pickle.load(f)
            
            self.class_names = pipeline_state['class_names']
            self.feature_types = pipeline_state['feature_types']
            
            print(f"✅ Model loaded successfully!")
            print(f"   Classes: {self.class_names}")
            print(f"   Features: {self.feature_types}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def start_recording(self) -> None:
        """Start real-time audio recording."""
        
        if self.is_recording:
            print("⚠️ Already recording!")
            return
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            self.stream.start_stream()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            print("🎤 Started real-time audio recording")
            
        except Exception as e:
            print(f"❌ Error starting recording: {e}")
            raise
    
    def stop_recording(self) -> None:
        """Stop real-time audio recording."""
        
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        print("🛑 Stopped real-time audio recording")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream."""
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Update circular buffer
        self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
        self.audio_buffer[-len(audio_data):] = audio_data
        
        return (None, pyaudio.paContinue)
    
    def _processing_loop(self) -> None:
        """Main processing loop for feature extraction and classification."""
        
        while self.is_recording:
            try:
                # Get current audio buffer
                current_audio = self.audio_buffer.copy()
                
                # Extract features
                features = self._extract_features(current_audio)
                
                # Make prediction if model is loaded
                if self.model is not None:
                    prediction = self._predict(features)
                    
                    # Add to queues
                    if not self.feature_queue.full():
                        self.feature_queue.put(features)
                    
                    if not self.prediction_queue.full():
                        self.prediction_queue.put(prediction)
                
                # Sleep briefly to avoid overloading
                time.sleep(0.1)
                
            except Exception as e:
                print(f"⚠️ Processing error: {e}")
                time.sleep(0.5)
    
    def _extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract features from audio data."""
        
        # Normalize audio
        normalized = self.audio_processor.normalize_audio(audio_data, method='peak')
        
        # Extract specified features
        features = {}
        
        if 'mfcc' in self.feature_types:
            mfcc_features = self.feature_extractor.extract_mfcc(normalized)
            features.update(mfcc_features)
        
        if 'spectral' in self.feature_types:
            spectral_features = self.feature_extractor.extract_spectral_features(normalized)
            features.update(spectral_features)
        
        if 'chroma' in self.feature_types:
            chroma_features = self.feature_extractor.extract_chroma_features(normalized)
            features.update(chroma_features)
        
        if 'time_domain' in self.feature_types:
            time_features = self.feature_extractor.extract_time_domain_features(normalized)
            features.update(time_features)
        
        # Create feature vector
        feature_vector = self.feature_extractor.create_feature_vector(features, aggregate=True)
        
        return feature_vector
    
    def _predict(self, features: np.ndarray) -> Dict:
        """Make prediction using loaded model."""
        
        try:
            # Reshape for model input
            features_reshaped = features.reshape(1, -1)
            
            # Get prediction
            predictions = self.model.predict(features_reshaped, verbose=0)
            
            if predictions.shape[1] > 1:
                # Multi-class classification
                predicted_class = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class])
                class_name = self.class_names[predicted_class]
            else:
                # Binary classification
                confidence = float(predictions[0][0])
                predicted_class = int(confidence > 0.5)
                class_name = self.class_names[predicted_class] if predicted_class < len(self.class_names) else "Unknown"
            
            return {
                'class_name': class_name,
                'class_index': predicted_class,
                'confidence': confidence,
                'all_probabilities': predictions[0].tolist()
            }
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return {
                'class_name': 'Error',
                'class_index': -1,
                'confidence': 0.0,
                'all_probabilities': []
            }
    
    def get_latest_audio(self) -> np.ndarray:
        """Get the latest audio buffer."""
        return self.audio_buffer.copy()
    
    def get_latest_prediction(self) -> Optional[Dict]:
        """Get the latest prediction."""
        try:
            return self.prediction_queue.get_nowait()
        except queue.Empty:
            return None
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_recording()
        self.audio.terminate()


class RealTimeVisualizer:
    """
    Real-time visualization for audio processing.
    
    Shows waveform, spectrogram, and classification results in real-time.
    """
    
    def __init__(self, processor: RealTimeAudioProcessor):
        """
        Initialize real-time visualizer.
        
        Args:
            processor: RealTimeAudioProcessor instance
        """
        self.processor = processor
        
        # Setup matplotlib for real-time plotting
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Real-Time Audio Analysis', fontsize=16, fontweight='bold')
        
        # Initialize plots
        self.waveform_line, = self.axes[0, 0].plot([], [])
        self.axes[0, 0].set_title('Audio Waveform')
        self.axes[0, 0].set_xlabel('Time (s)')
        self.axes[0, 0].set_ylabel('Amplitude')
        
        self.spectrogram_im = None
        self.axes[0, 1].set_title('Spectrogram')
        self.axes[0, 1].set_xlabel('Time (s)')
        self.axes[0, 1].set_ylabel('Frequency (Hz)')
        
        # Classification results
        self.axes[1, 0].set_title('Classification Confidence')
        self.axes[1, 0].set_xlabel('Class')
        self.axes[1, 0].set_ylabel('Confidence')
        
        # Real-time metrics
        self.axes[1, 1].set_title('Audio Features')
        self.axes[1, 1].set_xlabel('Feature')
        self.axes[1, 1].set_ylabel('Value')
        
        plt.tight_layout()
        
        # Animation
        self.animation = FuncAnimation(
            self.fig, self._update_plots, interval=100, blit=False
        )
    
    def _update_plots(self, frame) -> None:
        """Update all plots with latest data."""
        
        # Get latest audio
        audio_data = self.processor.get_latest_audio()
        
        # Update waveform
        time_axis = np.linspace(0, self.processor.buffer_duration, len(audio_data))
        self.waveform_line.set_data(time_axis, audio_data)
        self.axes[0, 0].set_xlim(0, self.processor.buffer_duration)
        self.axes[0, 0].set_ylim(-1, 1)
        
        # Update spectrogram
        self._update_spectrogram(audio_data)
        
        # Update classification results
        self._update_classification()
        
        # Update feature visualization
        self._update_features()
    
    def _update_spectrogram(self, audio_data: np.ndarray) -> None:
        """Update spectrogram plot."""
        
        try:
            import librosa
            
            # Compute spectrogram
            S = librosa.stft(audio_data, n_fft=1024, hop_length=256)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            
            # Update plot
            if self.spectrogram_im is None:
                self.spectrogram_im = self.axes[0, 1].imshow(
                    S_db, aspect='auto', origin='lower', cmap='viridis'
                )
            else:
                self.spectrogram_im.set_array(S_db)
                self.spectrogram_im.set_clim(vmin=np.min(S_db), vmax=np.max(S_db))
                
        except Exception as e:
            pass  # Skip spectrogram update if error
    
    def _update_classification(self) -> None:
        """Update classification results plot."""
        
        prediction = self.processor.get_latest_prediction()
        
        if prediction and 'all_probabilities' in prediction:
            self.axes[1, 0].clear()
            self.axes[1, 0].set_title('Classification Confidence')
            
            probabilities = prediction['all_probabilities']
            class_names = self.processor.class_names
            
            if len(probabilities) == len(class_names):
                bars = self.axes[1, 0].bar(class_names, probabilities)
                
                # Highlight predicted class
                predicted_idx = prediction['class_index']
                if 0 <= predicted_idx < len(bars):
                    bars[predicted_idx].set_color('red')
                
                self.axes[1, 0].set_ylim(0, 1)
                self.axes[1, 0].tick_params(axis='x', rotation=45)
    
    def _update_features(self) -> None:
        """Update feature visualization."""
        
        # This is a simplified version - in practice you'd show
        # important features like spectral centroid, MFCC values, etc.
        
        try:
            audio_data = self.processor.get_latest_audio()
            
            # Calculate some basic features for visualization
            rms = np.sqrt(np.mean(audio_data**2))
            zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0) / len(audio_data)
            spectral_centroid = np.mean(np.abs(np.fft.fft(audio_data)[:len(audio_data)//2]))
            
            features = ['RMS', 'Zero Crossings', 'Spectral Centroid']
            values = [rms, zero_crossings, spectral_centroid / 1000]  # Normalize spectral centroid
            
            self.axes[1, 1].clear()
            self.axes[1, 1].set_title('Audio Features')
            self.axes[1, 1].bar(features, values)
            self.axes[1, 1].tick_params(axis='x', rotation=45)
            
        except Exception as e:
            pass  # Skip feature update if error
    
    def show(self) -> None:
        """Show the visualization."""
        plt.show()
    
    def close(self) -> None:
        """Close the visualization."""
        plt.close(self.fig)


def main():
    """Main function for real-time demo."""
    
    parser = argparse.ArgumentParser(description='Real-Time Audio Processing Demo')
    parser.add_argument('--model_path', type=str, default='models/best_cnn_1d_model.h5',
                       help='Path to trained model')
    parser.add_argument('--pipeline_state', type=str, default='models/cnn_1d_pipeline_state.pkl',
                       help='Path to pipeline state file')
    parser.add_argument('--duration', type=int, default=30,
                       help='Demo duration in seconds')
    parser.add_argument('--no_visualization', action='store_true',
                       help='Disable real-time visualization')
    
    args = parser.parse_args()
    
    print("🎤 Real-Time Audio Processing Demo")
    print("=" * 40)
    
    # Initialize processor
    processor = RealTimeAudioProcessor(
        sample_rate=22050,
        chunk_size=1024,
        buffer_duration=3.0
    )
    
    try:
        # Load model if available
        if Path(args.model_path).exists() and Path(args.pipeline_state).exists():
            processor.load_model(args.model_path, args.pipeline_state)
        else:
            print("⚠️ Model files not found. Running in feature extraction mode only.")
            print("   Train a model first using: python src/training.py")
        
        # Start recording
        processor.start_recording()
        
        if not args.no_visualization:
            # Start visualization
            print("🎨 Starting real-time visualization...")
            visualizer = RealTimeVisualizer(processor)
            
            # Show for specified duration
            start_time = time.time()
            while time.time() - start_time < args.duration:
                plt.pause(0.1)
            
            visualizer.close()
        else:
            # Run without visualization
            print(f"🎧 Recording for {args.duration} seconds...")
            
            start_time = time.time()
            while time.time() - start_time < args.duration:
                prediction = processor.get_latest_prediction()
                if prediction:
                    print(f"   Prediction: {prediction['class_name']} "
                          f"(confidence: {prediction['confidence']:.3f})")
                time.sleep(1.0)
    
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    
    except Exception as e:
        print(f"❌ Demo error: {e}")
    
    finally:
        # Cleanup
        processor.cleanup()
        print("✅ Demo completed!")


if __name__ == "__main__":
    main()

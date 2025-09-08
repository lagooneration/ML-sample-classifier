"""
Visualization Module for Audio Time Series ML Learning Project

This module provides comprehensive visualization capabilities for audio data,
features, and machine learning model performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from typing import Dict, List, Optional, Tuple, Union
import warnings

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AudioVisualizer:
    """
    Comprehensive visualization class for audio machine learning applications.
    
    Provides plotting functions for:
    - Audio waveforms and spectrograms
    - Feature visualizations
    - Model training progress
    - Classification results
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize the AudioVisualizer.
        
        Args:
            figsize: Default figure size for plots
            dpi: Resolution for plots
        """
        self.figsize = figsize
        self.dpi = dpi
        
    def plot_waveform(self, audio_data: np.ndarray, 
                     sample_rate: int = 22050,
                     title: str = "Audio Waveform",
                     save_path: Optional[str] = None) -> None:
        """
        Plot audio waveform in time domain.
        
        Args:
            audio_data: Input audio signal
            sample_rate: Sample rate of audio
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Create time axis
        time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        
        plt.plot(time, audio_data, linewidth=0.5, alpha=0.8)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = f'Duration: {len(audio_data)/sample_rate:.2f}s\n'
        stats_text += f'Sample Rate: {sample_rate} Hz\n'
        stats_text += f'Max Amplitude: {np.max(np.abs(audio_data)):.3f}\n'
        stats_text += f'RMS: {np.sqrt(np.mean(audio_data**2)):.3f}'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_spectrogram(self, audio_data: np.ndarray,
                        sample_rate: int = 22050,
                        n_fft: int = 2048,
                        hop_length: int = 512,
                        title: str = "Spectrogram",
                        save_path: Optional[str] = None) -> None:
        """
        Plot spectrogram of audio signal.
        
        Args:
            audio_data: Input audio signal
            sample_rate: Sample rate of audio
            n_fft: FFT window size
            hop_length: Number of samples between frames
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Compute spectrogram
        D = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Plot
        librosa.display.specshow(S_db, sr=sample_rate, hop_length=hop_length,
                                x_axis='time', y_axis='hz', cmap='viridis')
        
        plt.colorbar(format='%+2.0f dB', label='Magnitude (dB)')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Frequency (Hz)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_mel_spectrogram(self, audio_data: np.ndarray,
                            sample_rate: int = 22050,
                            n_mels: int = 128,
                            n_fft: int = 2048,
                            hop_length: int = 512,
                            title: str = "Mel Spectrogram",
                            save_path: Optional[str] = None) -> None:
        """
        Plot mel-scale spectrogram.
        
        Args:
            audio_data: Input audio signal
            sample_rate: Sample rate of audio
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Number of samples between frames
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(audio_data, sr=sample_rate, 
                                          n_mels=n_mels, n_fft=n_fft, 
                                          hop_length=hop_length)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Plot
        librosa.display.specshow(S_db, sr=sample_rate, hop_length=hop_length,
                                x_axis='time', y_axis='mel', cmap='viridis')
        
        plt.colorbar(format='%+2.0f dB', label='Power (dB)')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Mel Frequency', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_mfcc(self, mfcc_features: np.ndarray,
                  sample_rate: int = 22050,
                  hop_length: int = 512,
                  title: str = "MFCC Features",
                  save_path: Optional[str] = None) -> None:
        """
        Plot MFCC features as a heatmap.
        
        Args:
            mfcc_features: MFCC feature matrix
            sample_rate: Sample rate of audio
            hop_length: Number of samples between frames
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Plot MFCC
        librosa.display.specshow(mfcc_features, sr=sample_rate, 
                                hop_length=hop_length, x_axis='time', 
                                cmap='viridis')
        
        plt.colorbar(label='MFCC Coefficient Value')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('MFCC Coefficient', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_chroma(self, chroma_features: np.ndarray,
                   sample_rate: int = 22050,
                   hop_length: int = 512,
                   title: str = "Chroma Features",
                   save_path: Optional[str] = None) -> None:
        """
        Plot chroma features.
        
        Args:
            chroma_features: Chroma feature matrix
            sample_rate: Sample rate of audio
            hop_length: Number of samples between frames
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Plot chroma
        librosa.display.specshow(chroma_features, sr=sample_rate,
                                hop_length=hop_length, x_axis='time',
                                y_axis='chroma', cmap='viridis')
        
        plt.colorbar(label='Chroma Intensity')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Pitch Class', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_comparison(self, features_dict: Dict[str, np.ndarray],
                               title: str = "Feature Comparison",
                               save_path: Optional[str] = None) -> None:
        """
        Plot multiple features for comparison.
        
        Args:
            features_dict: Dictionary of features to plot
            title: Plot title
            save_path: Path to save the plot
        """
        n_features = len(features_dict)
        
        fig, axes = plt.subplots(n_features, 1, figsize=(self.figsize[0], 3 * n_features), 
                                dpi=self.dpi, sharex=True)
        
        if n_features == 1:
            axes = [axes]
        
        for idx, (feature_name, feature_data) in enumerate(features_dict.items()):
            if feature_data.ndim == 1:
                # 1D feature - plot as line
                axes[idx].plot(feature_data, linewidth=1)
                axes[idx].set_ylabel(feature_name)
            else:
                # 2D feature - plot as heatmap
                im = axes[idx].imshow(feature_data, aspect='auto', origin='lower', 
                                     cmap='viridis')
                plt.colorbar(im, ax=axes[idx])
                axes[idx].set_ylabel(feature_name)
            
            axes[idx].set_title(f"{feature_name} - Shape: {feature_data.shape}")
            axes[idx].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time Frames')
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_history(self, history: Dict[str, List[float]],
                             title: str = "Training History",
                             save_path: Optional[str] = None) -> None:
        """
        Plot training history (loss and accuracy).
        
        Args:
            history: Dictionary with 'loss', 'accuracy', 'val_loss', 'val_accuracy'
            title: Plot title
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        
        # Plot loss
        if 'loss' in history:
            ax1.plot(history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        if 'accuracy' in history:
            ax2.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history:
            ax2.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             title: str = "Confusion Matrix",
                             save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix for classification results.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            title: Plot title
            save_path: Path to save the plot
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6), dpi=self.dpi)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names or range(cm.shape[1]),
                   yticklabels=class_names or range(cm.shape[0]))
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importance_scores: np.ndarray,
                               title: str = "Feature Importance",
                               top_n: int = 20,
                               save_path: Optional[str] = None) -> None:
        """
        Plot feature importance scores.
        
        Args:
            feature_names: List of feature names
            importance_scores: Importance scores for each feature
            title: Plot title
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        # Sort features by importance
        indices = np.argsort(importance_scores)[::-1]
        top_indices = indices[:top_n]
        
        plt.figure(figsize=(10, max(6, top_n * 0.3)), dpi=self.dpi)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_indices))
        plt.barh(y_pos, importance_scores[top_indices])
        
        plt.yticks(y_pos, [feature_names[i] for i in top_indices])
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Invert y-axis to show most important at top
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()
    
    def plot_audio_comparison(self, audio_signals: Dict[str, Tuple[np.ndarray, int]],
                             title: str = "Audio Comparison",
                             save_path: Optional[str] = None) -> None:
        """
        Compare multiple audio signals visually.
        
        Args:
            audio_signals: Dict with signal names as keys and (audio_data, sample_rate) as values
            title: Plot title
            save_path: Path to save the plot
        """
        n_signals = len(audio_signals)
        
        fig, axes = plt.subplots(n_signals, 1, figsize=(self.figsize[0], 3 * n_signals),
                                dpi=self.dpi, sharex=True)
        
        if n_signals == 1:
            axes = [axes]
        
        for idx, (signal_name, (audio_data, sample_rate)) in enumerate(audio_signals.items()):
            time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
            
            axes[idx].plot(time, audio_data, linewidth=0.8, alpha=0.8)
            axes[idx].set_ylabel(f'{signal_name}\nAmplitude')
            axes[idx].grid(True, alpha=0.3)
            
            # Add signal statistics
            stats = f'RMS: {np.sqrt(np.mean(audio_data**2)):.3f}'
            axes[idx].text(0.02, 0.98, stats, transform=axes[idx].transAxes,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        axes[-1].set_xlabel('Time (seconds)')
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        plt.show()


def create_demo_plots():
    """Create demonstration plots to show visualization capabilities."""
    
    print("🎨 Creating Audio Visualization Demo")
    print("=" * 40)
    
    # Create sample audio data
    duration = 3.0
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create complex test signal
    audio_signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +     # A4 note
        0.3 * np.sin(2 * np.pi * 880 * t) +     # A5 note (octave)
        0.2 * np.sin(2 * np.pi * 1320 * t) +    # E6 note (harmonic)
        0.1 * np.random.normal(0, 1, len(t))     # Noise
    )
    
    # Add some amplitude modulation for visual interest
    audio_signal *= (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
    
    # Initialize visualizer
    visualizer = AudioVisualizer()
    
    print("📊 Creating waveform plot...")
    visualizer.plot_waveform(audio_signal, sample_rate, 
                            title="Demo Audio Waveform - Complex Musical Signal")
    
    print("📈 Creating spectrogram...")
    visualizer.plot_spectrogram(audio_signal, sample_rate,
                               title="Demo Spectrogram - Frequency Content Over Time")
    
    print("🎼 Creating mel spectrogram...")
    visualizer.plot_mel_spectrogram(audio_signal, sample_rate,
                                   title="Demo Mel Spectrogram - Perceptually Scaled")
    
    # Extract and visualize features
    print("🔍 Extracting features for visualization...")
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
    visualizer.plot_mfcc(mfcc, sample_rate, title="Demo MFCC Features")
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio_signal, sr=sample_rate)
    visualizer.plot_chroma(chroma, sample_rate, title="Demo Chroma Features - Pitch Classes")
    
    # Feature comparison
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_signal, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_signal, sr=sample_rate)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_signal)[0]
    
    features_dict = {
        'Spectral Centroid': spectral_centroid,
        'Spectral Rolloff': spectral_rolloff,
        'Zero Crossing Rate': zero_crossing_rate,
        'MFCC (first 5)': mfcc[:5]
    }
    
    visualizer.plot_feature_comparison(features_dict, 
                                      title="Demo Feature Comparison - Multiple Audio Features")
    
    # Create fake training history for demonstration
    print("📈 Creating training history demo...")
    epochs = 50
    train_loss = 2.0 * np.exp(-0.1 * np.arange(epochs)) + 0.1 * np.random.random(epochs)
    val_loss = 2.2 * np.exp(-0.08 * np.arange(epochs)) + 0.15 * np.random.random(epochs)
    train_acc = 1 - np.exp(-0.12 * np.arange(epochs)) + 0.05 * np.random.random(epochs)
    val_acc = 1 - np.exp(-0.10 * np.arange(epochs)) + 0.08 * np.random.random(epochs)
    
    history = {
        'loss': train_loss.tolist(),
        'val_loss': val_loss.tolist(),
        'accuracy': train_acc.tolist(),
        'val_accuracy': val_acc.tolist()
    }
    
    visualizer.plot_training_history(history, title="Demo Training History - Model Learning Progress")
    
    print("✅ Visualization demo completed!")


if __name__ == "__main__":
    create_demo_plots()

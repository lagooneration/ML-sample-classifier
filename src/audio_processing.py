"""
Audio Processing Module for Time Series ML Learning Project

This module provides comprehensive audio loading, preprocessing, and basic analysis
capabilities for machine learning applications.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import warnings
from typing import Tuple, Optional, Union

warnings.filterwarnings('ignore')


class AudioProcessor:
    """
    A comprehensive audio processing class for machine learning applications.
    
    This class handles audio loading, preprocessing, segmentation, and basic
    transformations commonly used in audio ML pipelines.
    """
    
    def __init__(self, sample_rate: int = 22050, duration: Optional[float] = None):
        """
        Initialize the AudioProcessor.
        
        Args:
            sample_rate (int): Target sample rate for audio processing
            duration (float, optional): Maximum duration to load (in seconds)
        """
        self.sample_rate = sample_rate
        self.duration = duration
        
    def load_audio(self, file_path: Union[str, Path], 
                   offset: float = 0.0) -> Tuple[np.ndarray, int]:
        """
        Load audio file with specified parameters.
        
        Args:
            file_path: Path to audio file
            offset: Start time offset in seconds
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        try:
            audio_data, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                duration=self.duration,
                offset=offset
            )
            
            print(f"✅ Loaded audio: {Path(file_path).name}")
            print(f"   Duration: {len(audio_data) / sr:.2f}s")
            print(f"   Sample rate: {sr} Hz")
            print(f"   Shape: {audio_data.shape}")
            
            return audio_data, sr
            
        except Exception as e:
            print(f"❌ Error loading audio file {file_path}: {str(e)}")
            raise
    
    def normalize_audio(self, audio_data: np.ndarray, 
                       method: str = 'peak') -> np.ndarray:
        """
        Normalize audio data using different methods.
        
        Args:
            audio_data: Input audio signal
            method: Normalization method ('peak', 'rms', 'lufs')
            
        Returns:
            Normalized audio data
        """
        if method == 'peak':
            # Peak normalization
            peak = np.max(np.abs(audio_data))
            if peak > 0:
                return audio_data / peak
            return audio_data
            
        elif method == 'rms':
            # RMS normalization
            rms = np.sqrt(np.mean(audio_data**2))
            if rms > 0:
                return audio_data / (rms * 4)  # Scale to reasonable level
            return audio_data
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def segment_audio(self, audio_data: np.ndarray, 
                     segment_length: float,
                     hop_length: Optional[float] = None,
                     pad_final: bool = True) -> list:
        """
        Segment audio into fixed-length chunks.
        
        Args:
            audio_data: Input audio signal
            segment_length: Length of each segment in seconds
            hop_length: Hop length between segments (default: segment_length)
            pad_final: Whether to pad the final segment if it's shorter
            
        Returns:
            List of audio segments
        """
        if hop_length is None:
            hop_length = segment_length
            
        samples_per_segment = int(segment_length * self.sample_rate)
        hop_samples = int(hop_length * self.sample_rate)
        
        segments = []
        
        for start in range(0, len(audio_data), hop_samples):
            end = start + samples_per_segment
            
            if end <= len(audio_data):
                segment = audio_data[start:end]
            elif pad_final and start < len(audio_data):
                # Pad the final segment
                segment = np.pad(
                    audio_data[start:],
                    (0, samples_per_segment - (len(audio_data) - start)),
                    mode='constant'
                )
            else:
                break
                
            segments.append(segment)
        
        print(f"📊 Created {len(segments)} segments of {segment_length}s each")
        return segments
    
    def apply_window(self, audio_data: np.ndarray, 
                    window_type: str = 'hann') -> np.ndarray:
        """
        Apply windowing function to audio data.
        
        Args:
            audio_data: Input audio signal
            window_type: Type of window ('hann', 'hamming', 'blackman')
            
        Returns:
            Windowed audio data
        """
        if window_type == 'hann':
            window = np.hanning(len(audio_data))
        elif window_type == 'hamming':
            window = np.hamming(len(audio_data))
        elif window_type == 'blackman':
            window = np.blackman(len(audio_data))
        else:
            raise ValueError(f"Unknown window type: {window_type}")
            
        return audio_data * window
    
    def add_noise(self, audio_data: np.ndarray, 
                  noise_factor: float = 0.01) -> np.ndarray:
        """
        Add white noise to audio for data augmentation.
        
        Args:
            audio_data: Input audio signal
            noise_factor: Noise amplitude factor
            
        Returns:
            Audio with added noise
        """
        noise = np.random.normal(0, noise_factor, audio_data.shape)
        return audio_data + noise
    
    def time_shift(self, audio_data: np.ndarray, 
                   shift_samples: int) -> np.ndarray:
        """
        Shift audio in time for data augmentation.
        
        Args:
            audio_data: Input audio signal
            shift_samples: Number of samples to shift (positive = right, negative = left)
            
        Returns:
            Time-shifted audio
        """
        if shift_samples > 0:
            # Shift right (delay)
            return np.pad(audio_data, (shift_samples, 0), mode='constant')[:-shift_samples]
        elif shift_samples < 0:
            # Shift left (advance)
            return np.pad(audio_data, (0, -shift_samples), mode='constant')[-shift_samples:]
        else:
            return audio_data
    
    def change_speed(self, audio_data: np.ndarray, 
                    speed_factor: float) -> np.ndarray:
        """
        Change playback speed without changing pitch.
        
        Args:
            audio_data: Input audio signal
            speed_factor: Speed multiplication factor (1.0 = normal, 1.5 = faster, 0.8 = slower)
            
        Returns:
            Speed-changed audio
        """
        return librosa.effects.time_stretch(audio_data, rate=speed_factor)
    
    def save_audio(self, audio_data: np.ndarray, 
                   output_path: Union[str, Path],
                   sample_rate: Optional[int] = None) -> None:
        """
        Save audio data to file.
        
        Args:
            audio_data: Audio signal to save
            output_path: Output file path
            sample_rate: Sample rate (uses class default if None)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        try:
            sf.write(output_path, audio_data, sample_rate)
            print(f"💾 Saved audio to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving audio: {str(e)}")
            raise
    
    def get_audio_info(self, audio_data: np.ndarray) -> dict:
        """
        Get comprehensive information about audio signal.
        
        Args:
            audio_data: Input audio signal
            
        Returns:
            Dictionary with audio information
        """
        return {
            'duration': len(audio_data) / self.sample_rate,
            'sample_rate': self.sample_rate,
            'samples': len(audio_data),
            'channels': 1 if audio_data.ndim == 1 else audio_data.shape[1],
            'max_amplitude': np.max(np.abs(audio_data)),
            'rms': np.sqrt(np.mean(audio_data**2)),
            'zero_crossings': np.sum(np.diff(np.sign(audio_data)) != 0) / 2
        }


def create_synthetic_audio(duration: float = 5.0, 
                          sample_rate: int = 22050,
                          frequency: float = 440.0,
                          wave_type: str = 'sine') -> np.ndarray:
    """
    Create synthetic audio for testing and demonstration.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate
        frequency: Frequency in Hz
        wave_type: Type of wave ('sine', 'square', 'sawtooth', 'noise')
        
    Returns:
        Synthetic audio signal
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    if wave_type == 'sine':
        audio = np.sin(2 * np.pi * frequency * t)
    elif wave_type == 'square':
        audio = np.sign(np.sin(2 * np.pi * frequency * t))
    elif wave_type == 'sawtooth':
        audio = 2 * (t * frequency - np.floor(t * frequency + 0.5))
    elif wave_type == 'noise':
        audio = np.random.normal(0, 0.3, len(t))
    else:
        raise ValueError(f"Unknown wave type: {wave_type}")
    
    return audio.astype(np.float32)


if __name__ == "__main__":
    # Demo usage
    print("🎵 Audio Processing Module Demo")
    print("=" * 40)
    
    # Create processor
    processor = AudioProcessor(sample_rate=22050)
    
    # Create synthetic audio for demonstration
    print("\n📊 Creating synthetic audio...")
    demo_audio = create_synthetic_audio(duration=3.0, frequency=440.0, wave_type='sine')
    
    # Display audio info
    info = processor.get_audio_info(demo_audio)
    print(f"\n📋 Audio Information:")
    for key, value in info.items():
        print(f"   {key}: {value:.4f}")
    
    # Demonstrate normalization
    print(f"\n🔧 Normalizing audio...")
    normalized = processor.normalize_audio(demo_audio, method='peak')
    print(f"   Original max: {np.max(np.abs(demo_audio)):.4f}")
    print(f"   Normalized max: {np.max(np.abs(normalized)):.4f}")
    
    # Demonstrate segmentation
    print(f"\n✂️ Segmenting audio...")
    segments = processor.segment_audio(demo_audio, segment_length=1.0, hop_length=0.5)
    
    print(f"\n✅ Demo completed successfully!")

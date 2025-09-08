"""
Feature Extraction Module for Audio Time Series ML

This module provides comprehensive feature extraction capabilities for audio signals,
including spectral features, MFCCs, chroma features, and time-domain characteristics.
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """
    Comprehensive feature extraction class for audio machine learning.
    
    Extracts various types of features commonly used in audio ML applications:
    - Time domain features
    - Spectral features  
    - MFCCs (Mel-Frequency Cepstral Coefficients)
    - Chroma features
    - Tonnetz features
    - Rhythm and tempo features
    """
    
    def __init__(self, sample_rate: int = 22050, n_fft: int = 2048, 
                 hop_length: int = 512, n_mfcc: int = 13):
        """
        Initialize the FeatureExtractor.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mfcc: Number of MFCC coefficients to extract
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        
    def extract_time_domain_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Extract time domain features from audio signal.
        
        Args:
            audio_data: Input audio signal
            
        Returns:
            Dictionary of time domain features
        """
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(audio_data)
        features['std'] = np.std(audio_data)
        features['var'] = np.var(audio_data)
        features['min'] = np.min(audio_data)
        features['max'] = np.max(audio_data)
        features['range'] = features['max'] - features['min']
        
        # Energy and power
        features['energy'] = np.sum(audio_data ** 2)
        features['power'] = features['energy'] / len(audio_data)
        features['rms'] = np.sqrt(features['power'])
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.sign(audio_data)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(audio_data)
        
        # Peak-related features
        features['peak_amplitude'] = np.max(np.abs(audio_data))
        features['crest_factor'] = features['peak_amplitude'] / features['rms'] if features['rms'] > 0 else 0
        
        return features
    
    def extract_mfcc(self, audio_data: np.ndarray, 
                     include_delta: bool = True) -> Dict[str, np.ndarray]:
        """
        Extract MFCC features and their derivatives.
        
        Args:
            audio_data: Input audio signal
            include_delta: Whether to include delta and delta-delta features
            
        Returns:
            Dictionary containing MFCC features
        """
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio_data,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        features = {'mfcc': mfccs}
        
        if include_delta:
            # First and second derivatives (delta and delta-delta)
            # Adjust width for short audio files - must be odd and >= 3
            max_width = mfccs.shape[1] // 2
            if max_width >= 3:
                width = min(9, max_width)
                # Ensure width is odd
                if width % 2 == 0:
                    width -= 1
                # Ensure width is at least 3
                width = max(3, width)
                
                delta_mfccs = librosa.feature.delta(mfccs, width=width)
                delta2_mfccs = librosa.feature.delta(mfccs, order=2, width=width)
                
                features['mfcc_delta'] = delta_mfccs
                features['mfcc_delta2'] = delta2_mfccs
            
        return features
    
    def extract_spectral_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract spectral features from audio signal.
        
        Args:
            audio_data: Input audio signal
            
        Returns:
            Dictionary of spectral features
        """
        # Compute magnitude spectrogram
        stft = librosa.stft(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        features = {}
        
        # Spectral centroid (center of mass of spectrum)
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            S=magnitude, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            S=magnitude, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral bandwidth (width of the spectrum)
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            S=magnitude, sr=self.sample_rate, hop_length=self.hop_length
        )[0]
        
        # Spectral contrast (difference in amplitude between peaks and valleys)
        features['spectral_contrast'] = librosa.feature.spectral_contrast(
            S=magnitude, sr=self.sample_rate, hop_length=self.hop_length
        )
        
        # Spectral flatness (measure of how noise-like vs tone-like)
        features['spectral_flatness'] = librosa.feature.spectral_flatness(
            S=magnitude, hop_length=self.hop_length
        )[0]
        
        return features
    
    def extract_chroma_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract chroma features (pitch class profiles).
        
        Args:
            audio_data: Input audio signal
            
        Returns:
            Dictionary of chroma features
        """
        features = {}
        
        # Standard chroma features
        features['chroma'] = librosa.feature.chroma_stft(
            y=audio_data,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # CQT-based chroma features (more accurate for harmonic content)
        features['chroma_cqt'] = librosa.feature.chroma_cqt(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Chromagram from onset envelope
        features['chroma_cens'] = librosa.feature.chroma_cens(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        return features
    
    def extract_rhythm_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Extract rhythm and tempo features.
        
        Args:
            audio_data: Input audio signal
            
        Returns:
            Dictionary of rhythm features
        """
        features = {}
        
        # Tempo estimation
        tempo, beats = librosa.beat.beat_track(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        features['tempo'] = float(tempo)
        features['beat_count'] = len(beats)
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(
            y=audio_data,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        features['onset_count'] = len(onset_frames)
        features['onset_rate'] = len(onset_frames) / (len(audio_data) / self.sample_rate)
        
        return features
    
    def extract_tonnetz_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extract Tonnetz features (tonal centroid features).
        
        Args:
            audio_data: Input audio signal
            
        Returns:
            Tonnetz feature matrix
        """
        return librosa.feature.tonnetz(
            y=audio_data,
            sr=self.sample_rate
        )
    
    def extract_mel_spectrogram(self, audio_data: np.ndarray, 
                               n_mels: int = 128) -> np.ndarray:
        """
        Extract mel-scale spectrogram.
        
        Args:
            audio_data: Input audio signal
            n_mels: Number of mel bands
            
        Returns:
            Mel spectrogram
        """
        return librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=n_mels
        )
    
    def extract_all_features(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all available features from audio signal.
        
        Args:
            audio_data: Input audio signal
            
        Returns:
            Dictionary containing all extracted features
        """
        print("🔍 Extracting comprehensive audio features...")
        
        all_features = {}
        
        # Time domain features
        print("   ⏱️ Time domain features...")
        time_features = self.extract_time_domain_features(audio_data)
        all_features.update(time_features)
        
        # MFCC features
        print("   🎼 MFCC features...")
        mfcc_features = self.extract_mfcc(audio_data)
        all_features.update(mfcc_features)
        
        # Spectral features
        print("   📊 Spectral features...")
        spectral_features = self.extract_spectral_features(audio_data)
        all_features.update(spectral_features)
        
        # Chroma features
        print("   🎵 Chroma features...")
        chroma_features = self.extract_chroma_features(audio_data)
        all_features.update(chroma_features)
        
        # Rhythm features
        print("   🥁 Rhythm features...")
        rhythm_features = self.extract_rhythm_features(audio_data)
        all_features.update(rhythm_features)
        
        # Tonnetz features
        print("   🎹 Tonnetz features...")
        all_features['tonnetz'] = self.extract_tonnetz_features(audio_data)
        
        # Mel spectrogram
        print("   🌡️ Mel spectrogram...")
        all_features['mel_spectrogram'] = self.extract_mel_spectrogram(audio_data)
        
        print("✅ Feature extraction completed!")
        return all_features
    
    def create_feature_vector(self, features: Dict[str, np.ndarray], 
                             aggregate: bool = True) -> np.ndarray:
        """
        Create a single feature vector from extracted features.
        
        Args:
            features: Dictionary of extracted features
            aggregate: Whether to aggregate time-series features to single values
            
        Returns:
            Flattened feature vector
        """
        feature_vector = []
        
        for key, value in features.items():
            if isinstance(value, (int, float)):
                # Scalar features
                feature_vector.append(value)
            elif isinstance(value, np.ndarray):
                if aggregate and value.ndim > 1:
                    # Aggregate 2D features (mean, std, min, max)
                    feature_vector.extend([
                        np.mean(value),
                        np.std(value),
                        np.min(value),
                        np.max(value)
                    ])
                elif aggregate and value.ndim == 1:
                    # Aggregate 1D features (mean, std)
                    feature_vector.extend([
                        np.mean(value),
                        np.std(value)
                    ])
                else:
                    # Flatten without aggregation
                    feature_vector.extend(value.flatten())
        
        return np.array(feature_vector)
    
    def get_feature_names(self, features: Dict[str, np.ndarray], 
                         aggregate: bool = True) -> List[str]:
        """
        Get names for features in the feature vector.
        
        Args:
            features: Dictionary of extracted features
            aggregate: Whether aggregation was used
            
        Returns:
            List of feature names
        """
        feature_names = []
        
        for key, value in features.items():
            if isinstance(value, (int, float)):
                feature_names.append(key)
            elif isinstance(value, np.ndarray):
                if aggregate and value.ndim > 1:
                    feature_names.extend([
                        f"{key}_mean",
                        f"{key}_std", 
                        f"{key}_min",
                        f"{key}_max"
                    ])
                elif aggregate and value.ndim == 1:
                    feature_names.extend([
                        f"{key}_mean",
                        f"{key}_std"
                    ])
                else:
                    if value.ndim == 1:
                        feature_names.extend([f"{key}_{i}" for i in range(len(value))])
                    else:
                        for i in range(value.shape[0]):
                            for j in range(value.shape[1]):
                                feature_names.append(f"{key}_{i}_{j}")
        
        return feature_names


class AudioFeatureExtractor:
    """
    High-level wrapper for audio feature extraction pipeline.
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.extractor = FeatureExtractor(sample_rate=sample_rate)
        
    def process_audio_file(self, audio_data: np.ndarray, 
                          feature_types: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Process audio file and extract specified features.
        
        Args:
            audio_data: Input audio signal
            feature_types: List of feature types to extract (None = all)
            
        Returns:
            Dictionary of extracted features
        """
        if feature_types is None:
            return self.extractor.extract_all_features(audio_data)
        
        features = {}
        
        if 'time_domain' in feature_types:
            features.update(self.extractor.extract_time_domain_features(audio_data))
        if 'mfcc' in feature_types:
            features.update(self.extractor.extract_mfcc(audio_data))
        if 'spectral' in feature_types:
            features.update(self.extractor.extract_spectral_features(audio_data))
        if 'chroma' in feature_types:
            features.update(self.extractor.extract_chroma_features(audio_data))
        if 'rhythm' in feature_types:
            features.update(self.extractor.extract_rhythm_features(audio_data))
        if 'tonnetz' in feature_types:
            features['tonnetz'] = self.extractor.extract_tonnetz_features(audio_data)
        if 'mel_spectrogram' in feature_types:
            features['mel_spectrogram'] = self.extractor.extract_mel_spectrogram(audio_data)
            
        return features


if __name__ == "__main__":
    # Demo usage
    print("🎵 Feature Extraction Module Demo")
    print("=" * 40)
    
    # Create sample audio data
    duration = 3.0
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a complex test signal with multiple components
    audio_signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # A4 note
        0.3 * np.sin(2 * np.pi * 880 * t) +  # A5 note (octave)
        0.1 * np.random.normal(0, 1, len(t))  # Noise
    )
    
    # Initialize feature extractor
    extractor = FeatureExtractor(sample_rate=sample_rate)
    
    # Extract all features
    print(f"\n📊 Processing {duration}s audio signal...")
    features = extractor.extract_all_features(audio_signal)
    
    # Display feature summary
    print(f"\n📋 Extracted Features Summary:")
    for feature_name, feature_data in features.items():
        if isinstance(feature_data, (int, float)):
            print(f"   {feature_name}: {feature_data:.4f}")
        elif isinstance(feature_data, np.ndarray):
            shape_str = " x ".join(map(str, feature_data.shape))
            print(f"   {feature_name}: {shape_str} array")
    
    # Create feature vector
    feature_vector = extractor.create_feature_vector(features, aggregate=True)
    feature_names = extractor.get_feature_names(features, aggregate=True)
    
    print(f"\n🔢 Feature Vector:")
    print(f"   Length: {len(feature_vector)}")
    print(f"   Shape: {feature_vector.shape}")
    print(f"   Sample features: {feature_names[:5]}...")
    
    print(f"\n✅ Demo completed successfully!")

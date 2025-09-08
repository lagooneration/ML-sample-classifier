"""
Data Download Script for Audio ML Project

This script downloads sample audio datasets for machine learning experiments.
"""

import requests
import os
import zipfile
import tarfile
from pathlib import Path
import numpy as np
import soundfile as sf
from typing import List, Dict
import argparse


def create_synthetic_music_dataset():
    """Create a synthetic music dataset with different characteristics."""
    
    print("🎼 Creating synthetic music dataset...")
    
    output_dir = Path("data/raw/synthetic_music")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define music styles with different characteristics
    styles = {
        'classical': {
            'base_freq': [261.63, 293.66, 329.63, 349.23, 392.00],  # C major scale
            'harmonics': [1, 0.5, 0.25, 0.125],
            'rhythm': 'steady',
            'tempo': 120
        },
        'electronic': {
            'base_freq': [130.81, 146.83, 164.81, 174.61, 196.00],  # Lower frequencies
            'harmonics': [1, 0.3, 0.8, 0.2, 0.1],
            'rhythm': 'synthetic',
            'tempo': 128
        },
        'folk': {
            'base_freq': [196.00, 220.00, 246.94, 261.63, 293.66],  # G major
            'harmonics': [1, 0.7, 0.4, 0.2],
            'rhythm': 'natural',
            'tempo': 100
        },
        'ambient': {
            'base_freq': [65.41, 73.42, 82.41, 87.31, 98.00],  # Very low frequencies
            'harmonics': [1, 0.8, 0.6, 0.4, 0.2],
            'rhythm': 'flowing',
            'tempo': 60
        }
    }
    
    samples_per_style = 25
    duration = 5.0
    sample_rate = 22050
    
    for style_name, style_params in styles.items():
        style_dir = output_dir / style_name
        style_dir.mkdir(exist_ok=True)
        
        print(f"   Creating {style_name} samples...")
        
        for i in range(samples_per_style):
            # Generate audio for this style
            audio = generate_musical_audio(
                duration=duration,
                sample_rate=sample_rate,
                base_frequencies=style_params['base_freq'],
                harmonics=style_params['harmonics'],
                tempo=style_params['tempo'],
                style=style_params['rhythm']
            )
            
            # Save audio file
            output_path = style_dir / f"{style_name}_{i:03d}.wav"
            sf.write(output_path, audio, sample_rate)
    
    print(f"✅ Created synthetic music dataset")
    print(f"   Location: {output_dir}")
    print(f"   Styles: {list(styles.keys())}")
    print(f"   Total files: {len(styles) * samples_per_style}")
    
    return output_dir


def generate_musical_audio(duration: float, sample_rate: int, 
                          base_frequencies: List[float], harmonics: List[float],
                          tempo: int, style: str) -> np.ndarray:
    """Generate synthetic musical audio with specified characteristics."""
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.zeros_like(t)
    
    # Time signature and note timing
    beat_duration = 60.0 / tempo  # Duration of one beat in seconds
    note_duration = beat_duration / 2  # Eighth notes
    
    num_notes = int(duration / note_duration)
    
    for note_idx in range(num_notes):
        note_start = note_idx * note_duration
        note_end = min((note_idx + 1) * note_duration, duration)
        
        start_sample = int(note_start * sample_rate)
        end_sample = int(note_end * sample_rate)
        
        if start_sample >= len(t):
            break
        
        # Select frequency for this note
        freq_idx = note_idx % len(base_frequencies)
        base_freq = base_frequencies[freq_idx]
        
        # Add variation based on style
        if style == 'steady':
            freq = base_freq
        elif style == 'synthetic':
            freq = base_freq * (1 + 0.1 * np.sin(2 * np.pi * 0.5 * note_start))
        elif style == 'natural':
            freq = base_freq * np.random.uniform(0.98, 1.02)
        elif style == 'flowing':
            freq = base_freq * (1 + 0.05 * np.sin(2 * np.pi * 0.2 * note_start))
        
        # Generate note with harmonics
        note_time = t[start_sample:end_sample] - note_start
        note_audio = np.zeros_like(note_time)
        
        for h_idx, harmonic_amp in enumerate(harmonics):
            harmonic_freq = freq * (h_idx + 1)
            if harmonic_freq < sample_rate / 2:  # Avoid aliasing
                note_audio += harmonic_amp * np.sin(2 * np.pi * harmonic_freq * note_time)
        
        # Apply envelope (ADSR-like)
        envelope = create_envelope(len(note_time), style)
        note_audio *= envelope
        
        # Add to main audio
        audio[start_sample:end_sample] += note_audio
    
    # Add some background characteristics
    if style == 'electronic':
        # Add some digital artifacts
        audio += 0.02 * np.random.choice([-1, 0, 1], size=len(audio))
    elif style == 'ambient':
        # Add reverb-like effect
        audio = add_simple_reverb(audio, sample_rate)
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Add slight noise for realism
    audio += 0.01 * np.random.normal(0, 1, len(audio))
    
    return audio


def create_envelope(length: int, style: str) -> np.ndarray:
    """Create amplitude envelope for musical notes."""
    
    envelope = np.ones(length)
    
    if length < 10:
        return envelope
    
    attack_len = max(1, length // 20)
    decay_len = max(1, length // 10)
    release_len = max(1, length // 15)
    
    if style == 'steady':
        # Simple attack and release
        envelope[:attack_len] = np.linspace(0, 1, attack_len)
        envelope[-release_len:] = np.linspace(1, 0, release_len)
    
    elif style == 'synthetic':
        # Sharp attack, sustained, quick release
        envelope[:attack_len] = np.linspace(0, 1, attack_len)
        envelope[-release_len//2:] = np.linspace(1, 0, release_len//2)
    
    elif style == 'natural':
        # Gradual attack, slight decay, gradual release
        envelope[:attack_len] = np.linspace(0, 1, attack_len)
        if attack_len + decay_len < length:
            envelope[attack_len:attack_len + decay_len] = np.linspace(1, 0.8, decay_len)
        envelope[-release_len:] = np.linspace(envelope[-release_len-1], 0, release_len)
    
    elif style == 'flowing':
        # Very gradual changes
        envelope[:attack_len*2] = np.linspace(0, 1, attack_len*2)
        envelope[-release_len*2:] = np.linspace(1, 0, release_len*2)
    
    return envelope


def add_simple_reverb(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Add simple reverb effect to audio."""
    
    # Simple delay-based reverb
    delay_samples = int(0.1 * sample_rate)  # 100ms delay
    decay_factor = 0.3
    
    reverb_audio = np.copy(audio)
    
    # Add delayed versions
    for delay_mult in [1, 2, 3]:
        delay = delay_samples * delay_mult
        if delay < len(audio):
            delayed = np.zeros_like(audio)
            delayed[delay:] = audio[:-delay] * (decay_factor ** delay_mult)
            reverb_audio += delayed
    
    return reverb_audio


def create_emotion_dataset():
    """Create synthetic audio dataset representing different emotions."""
    
    print("😊 Creating emotion audio dataset...")
    
    output_dir = Path("data/raw/emotions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    emotions = {
        'happy': {
            'pitch_range': (300, 600),
            'tempo': 140,
            'rhythm': 'bouncy',
            'harmonics': [1, 0.6, 0.3, 0.15]
        },
        'sad': {
            'pitch_range': (150, 300),
            'tempo': 60,
            'rhythm': 'slow',
            'harmonics': [1, 0.8, 0.4, 0.2]
        },
        'angry': {
            'pitch_range': (200, 500),
            'tempo': 120,
            'rhythm': 'aggressive',
            'harmonics': [1, 0.4, 0.7, 0.3, 0.1]
        },
        'calm': {
            'pitch_range': (100, 250),
            'tempo': 70,
            'rhythm': 'steady',
            'harmonics': [1, 0.7, 0.5, 0.3, 0.2]
        }
    }
    
    samples_per_emotion = 20
    duration = 4.0
    sample_rate = 22050
    
    for emotion_name, emotion_params in emotions.items():
        emotion_dir = output_dir / emotion_name
        emotion_dir.mkdir(exist_ok=True)
        
        print(f"   Creating {emotion_name} samples...")
        
        for i in range(samples_per_emotion):
            audio = generate_emotional_audio(
                duration=duration,
                sample_rate=sample_rate,
                emotion_params=emotion_params
            )
            
            output_path = emotion_dir / f"{emotion_name}_{i:03d}.wav"
            sf.write(output_path, audio, sample_rate)
    
    print(f"✅ Created emotion dataset")
    print(f"   Location: {output_dir}")
    return output_dir


def generate_emotional_audio(duration: float, sample_rate: int, 
                           emotion_params: Dict) -> np.ndarray:
    """Generate audio that represents specific emotions."""
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.zeros_like(t)
    
    pitch_low, pitch_high = emotion_params['pitch_range']
    tempo = emotion_params['tempo']
    rhythm = emotion_params['rhythm']
    harmonics = emotion_params['harmonics']
    
    beat_duration = 60.0 / tempo
    
    if rhythm == 'bouncy':
        # Quick, energetic notes
        note_duration = beat_duration / 4
        pitch_variation = 0.2
    elif rhythm == 'slow':
        # Long, sustained notes
        note_duration = beat_duration * 2
        pitch_variation = 0.05
    elif rhythm == 'aggressive':
        # Sharp, irregular notes
        note_duration = beat_duration / 3
        pitch_variation = 0.3
    else:  # steady
        note_duration = beat_duration
        pitch_variation = 0.1
    
    num_notes = int(duration / note_duration)
    
    for note_idx in range(num_notes):
        note_start = note_idx * note_duration
        note_end = min((note_idx + 1) * note_duration, duration)
        
        start_sample = int(note_start * sample_rate)
        end_sample = int(note_end * sample_rate)
        
        if start_sample >= len(t):
            break
        
        # Generate frequency for this note
        base_freq = np.random.uniform(pitch_low, pitch_high)
        freq_variation = 1 + pitch_variation * (np.random.random() - 0.5)
        freq = base_freq * freq_variation
        
        # Generate note
        note_time = t[start_sample:end_sample] - note_start
        note_audio = np.zeros_like(note_time)
        
        for h_idx, harmonic_amp in enumerate(harmonics):
            harmonic_freq = freq * (h_idx + 1)
            if harmonic_freq < sample_rate / 2:
                note_audio += harmonic_amp * np.sin(2 * np.pi * harmonic_freq * note_time)
        
        # Apply emotional envelope
        envelope = create_emotional_envelope(len(note_time), rhythm)
        note_audio *= envelope
        
        audio[start_sample:end_sample] += note_audio
    
    # Normalize and add noise
    audio = audio / np.max(np.abs(audio)) * 0.7
    audio += 0.005 * np.random.normal(0, 1, len(audio))
    
    return audio


def create_emotional_envelope(length: int, rhythm: str) -> np.ndarray:
    """Create emotion-specific amplitude envelopes."""
    
    envelope = np.ones(length)
    
    if length < 10:
        return envelope
    
    if rhythm == 'bouncy':
        # Quick attack, quick decay
        attack_len = max(1, length // 50)
        decay_len = max(1, length // 20)
        envelope[:attack_len] = np.linspace(0, 1, attack_len)
        envelope[attack_len:attack_len + decay_len] = np.linspace(1, 0.5, decay_len)
        envelope[-attack_len:] = np.linspace(envelope[-attack_len-1], 0, attack_len)
    
    elif rhythm == 'slow':
        # Very gradual attack and release
        attack_len = max(1, length // 10)
        release_len = max(1, length // 8)
        envelope[:attack_len] = np.linspace(0, 1, attack_len)
        envelope[-release_len:] = np.linspace(1, 0, release_len)
    
    elif rhythm == 'aggressive':
        # Sharp attack, plateau, sharp release
        attack_len = max(1, length // 100)
        release_len = max(1, length // 80)
        envelope[:attack_len] = np.linspace(0, 1, attack_len)
        envelope[-release_len:] = np.linspace(1, 0, release_len)
    
    return envelope


def download_real_datasets():
    """Download real audio datasets (if available)."""
    
    print("🌐 Checking for real audio datasets...")
    
    # This would contain code to download real datasets
    # For now, we'll just create placeholder directories
    
    datasets = [
        "data/raw/urbansound8k",
        "data/raw/speech_commands",
        "data/raw/musicnet"
    ]
    
    for dataset_path in datasets:
        Path(dataset_path).mkdir(parents=True, exist_ok=True)
    
    print("📁 Created placeholder directories for real datasets")
    print("   Note: To use real datasets, please download them manually")
    print("   Suggested datasets:")
    print("   - UrbanSound8K: https://urbansounddataset.weebly.com/")
    print("   - Speech Commands: https://www.tensorflow.org/datasets/catalog/speech_commands")
    print("   - GTZAN Music Genre: http://marsyas.info/downloads/datasets.html")


def main():
    """Main function for data download script."""
    
    parser = argparse.ArgumentParser(description='Download/Create Audio Datasets')
    parser.add_argument('--synthetic_music', action='store_true',
                       help='Create synthetic music dataset')
    parser.add_argument('--emotions', action='store_true',
                       help='Create emotion dataset')
    parser.add_argument('--real_datasets', action='store_true',
                       help='Setup real dataset directories')
    parser.add_argument('--all', action='store_true',
                       help='Create all datasets')
    
    args = parser.parse_args()
    
    if args.all:
        args.synthetic_music = True
        args.emotions = True
        args.real_datasets = True
    
    if not any([args.synthetic_music, args.emotions, args.real_datasets]):
        # Default: create all synthetic datasets
        args.synthetic_music = True
        args.emotions = True
    
    print("📥 Audio Dataset Setup")
    print("=" * 30)
    
    if args.synthetic_music:
        create_synthetic_music_dataset()
        print()
    
    if args.emotions:
        create_emotion_dataset()
        print()
    
    if args.real_datasets:
        download_real_datasets()
        print()
    
    print("✅ Dataset setup completed!")
    print("\nTo use the datasets:")
    print("  python src/training.py --data_dir data/raw/synthetic_music")
    print("  python src/training.py --data_dir data/raw/emotions")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test the trained drum classifier on new audio files.
"""

import os
import sys
import pickle
import argparse
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from audio_processing import AudioProcessor
from feature_extraction import FeatureExtractor

def extract_single_audio_features(audio_file: str) -> np.ndarray:
    """
    Extract features from a single audio file using the same method as training.
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        Feature vector as numpy array
    """
    # Initialize processors
    audio_processor = AudioProcessor()
    feature_extractor = FeatureExtractor()
    
    # Load and process audio
    audio, sr = audio_processor.load_audio(audio_file)
    audio = audio_processor.normalize_audio(audio)
    
    # Extract features (same as training)
    features = {}
    
    # Time domain features (important for drums)
    time_features = feature_extractor.extract_time_domain_features(audio)
    features.update(time_features)
    
    # Spectral features (kick vs snare have different frequency content)
    spectral_features = feature_extractor.extract_spectral_features(audio)
    features.update(spectral_features)
    
    # MFCC features (timbral characteristics)
    mfcc_features = feature_extractor.extract_mfcc(audio, include_delta=True)
    features.update(mfcc_features)
    
    # Create feature vector using the same method as training
    feature_vector = feature_extractor.create_feature_vector(features, aggregate=True)
    
    return feature_vector.reshape(1, -1)

def predict_drum_type(model_path: str, audio_file: str) -> tuple:
    """
    Predict the drum type for an audio file.
    
    Args:
        model_path: Path to the trained model
        audio_file: Path to the audio file to classify
        
    Returns:
        Tuple of (predicted_class, confidence)
    """
    # Load the trained model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    classes = model_data['class_names']
    
    # Extract features
    features = extract_single_audio_features(audio_file)
    
    # Make prediction
    prediction_idx = model.predict(features)[0]
    prediction = classes[prediction_idx]
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities)
    
    return prediction, confidence

def main():
    parser = argparse.ArgumentParser(description='Test drum classifier')
    parser.add_argument('audio_file', help='Path to audio file to classify')
    parser.add_argument('--model', default='models/drum_classifier.pkl', 
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"❌ Audio file not found: {args.audio_file}")
        return
    
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("💡 Train a model first using: python scripts/train_drums.py")
        return
    
    try:
        print(f"🎵 Testing audio file: {args.audio_file}")
        prediction, confidence = predict_drum_type(args.model, args.audio_file)
        
        print(f"🥁 Prediction: {prediction}")
        print(f"📊 Confidence: {confidence:.3f}")
        
        # Visual confidence indicator
        confidence_bar = "█" * int(confidence * 20)
        print(f"📈 Confidence: [{confidence_bar:<20}] {confidence:.1%}")
        
        if confidence > 0.8:
            print("✅ High confidence prediction!")
        elif confidence > 0.6:
            print("⚠️ Medium confidence prediction")
        else:
            print("❓ Low confidence prediction")
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

if __name__ == "__main__":
    main()

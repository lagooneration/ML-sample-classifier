#!/usr/bin/env python3
"""
Drum Classification Training Script
Specialized for Kick vs Snare drum sound classification

This script is optimized for short drum samples and percussion classification.
"""

import numpy as np
import os
import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from audio_processing import AudioProcessor
    from feature_extraction import FeatureExtractor
    from visualization import AudioVisualizer
    print("✅ Successfully imported custom audio modules!")
except ImportError as e:
    print(f"⚠️ Could not import custom modules: {e}")
    print("Please ensure you're running this from the project root directory")
    sys.exit(1)

def load_drum_dataset(data_dir):
    """Load kick and snare drum samples."""
    print(f"🥁 Loading drum dataset from: {data_dir}")
    
    processor = AudioProcessor(sample_rate=22050)
    
    audio_files = []
    labels = []
    class_names = []
    
    data_path = Path(data_dir)
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            class_names.append(class_name)
            class_index = len(class_names) - 1
            
            print(f"   Loading {class_name} samples...")
            
            # Load all WAV files in this class
            wav_files = list(class_dir.glob('*.wav'))
            print(f"     Found {len(wav_files)} files")
            
            for audio_file in wav_files:
                try:
                    audio_data, sr = processor.load_audio(audio_file)
                    
                    # For drum samples, we might want to focus on the attack portion
                    # Limit to first 1 second (drums are usually short)
                    max_samples = int(1.0 * sr)  # 1 second max
                    if len(audio_data) > max_samples:
                        audio_data = audio_data[:max_samples]
                    
                    audio_files.append(audio_data)
                    labels.append(class_index)
                    
                except Exception as e:
                    print(f"     ⚠️ Error loading {audio_file.name}: {e}")
    
    print(f"✅ Loaded dataset:")
    print(f"   Classes: {class_names}")
    print(f"   Total samples: {len(audio_files)}")
    for i, class_name in enumerate(class_names):
        count = sum(1 for label in labels if label == i)
        print(f"   {class_name}: {count} samples")
    
    return audio_files, labels, class_names

def extract_drum_features(audio_files, class_names):
    """Extract features optimized for drum classification."""
    print("🔍 Extracting drum-specific features...")
    
    extractor = FeatureExtractor(sample_rate=22050)
    
    all_features = []
    
    for i, audio in enumerate(audio_files):
        if i % 5 == 0:
            print(f"   Processing sample {i+1}/{len(audio_files)}")
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
        
        features = {}
        
        # Time domain features (important for drums)
        time_features = extractor.extract_time_domain_features(audio)
        features.update(time_features)
        
        # Spectral features (kick vs snare have different frequency content)
        spectral_features = extractor.extract_spectral_features(audio)
        features.update(spectral_features)
        
        # MFCC features (timbral characteristics)
        mfcc_features = extractor.extract_mfcc(audio, include_delta=True)
        features.update(mfcc_features)
        
        # Create feature vector
        feature_vector = extractor.create_feature_vector(features, aggregate=True)
        all_features.append(feature_vector)
    
    feature_matrix = np.array(all_features)
    print(f"✅ Feature extraction completed!")
    print(f"   Feature matrix shape: {feature_matrix.shape}")
    
    return feature_matrix

def visualize_drum_data(audio_files, labels, class_names):
    """Create visualizations specific to drum data."""
    print("📊 Creating drum data visualizations...")
    
    visualizer = AudioVisualizer()
    
    # 1. Show sample waveforms for each class
    fig, axes = plt.subplots(len(class_names), 1, figsize=(14, 8))
    if len(class_names) == 1:
        axes = [axes]
    
    for class_idx, class_name in enumerate(class_names):
        # Find first sample of this class
        sample_idx = next(i for i, label in enumerate(labels) if label == class_idx)
        audio_sample = audio_files[sample_idx]
        
        # Show first 0.5 seconds (typical drum hit duration)
        duration = min(0.5, len(audio_sample) / 22050)
        samples_to_show = int(duration * 22050)
        time_axis = np.linspace(0, duration, samples_to_show)
        
        axes[class_idx].plot(time_axis, audio_sample[:samples_to_show], linewidth=1.5)
        axes[class_idx].set_title(f'{class_name.capitalize()} Sample', fontsize=14, fontweight='bold')
        axes[class_idx].set_ylabel('Amplitude')
        axes[class_idx].grid(True, alpha=0.3)
        
        # Add RMS and peak info
        rms = np.sqrt(np.mean(audio_sample**2))
        peak = np.max(np.abs(audio_sample))
        axes[class_idx].text(0.02, 0.98, f'RMS: {rms:.3f}\nPeak: {peak:.3f}', 
                            transform=axes[class_idx].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.suptitle('Drum Sample Waveforms', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 2. Create spectrograms for comparison
    fig, axes = plt.subplots(1, len(class_names), figsize=(14, 6))
    if len(class_names) == 1:
        axes = [axes]
    
    for class_idx, class_name in enumerate(class_names):
        sample_idx = next(i for i, label in enumerate(labels) if label == class_idx)
        audio_sample = audio_files[sample_idx]
        
        # Create spectrogram
        import librosa
        import librosa.display
        
        D = librosa.stft(audio_sample, n_fft=1024, hop_length=256)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        librosa.display.specshow(S_db, sr=22050, hop_length=256,
                                x_axis='time', y_axis='hz', 
                                ax=axes[class_idx], cmap='viridis')
        
        axes[class_idx].set_title(f'{class_name.capitalize()} Spectrogram')
        axes[class_idx].set_ylim(0, 4000)  # Focus on lower frequencies for drums
    
    plt.suptitle('Drum Spectrograms Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def train_drum_classifier(X, y, class_names):
    """Train a classifier optimized for drum sounds."""
    print("🚀 Training drum classifier...")
    
    try:
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.preprocessing import StandardScaler
        
        # Split data (with small dataset, use larger test size)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Try multiple classifiers
        classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True)
        }
        
        best_model = None
        best_accuracy = 0
        best_name = ""
        
        for name, clf in classifiers.items():
            print(f"\n   Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=3)
            print(f"     CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Train and test
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"     Test Accuracy: {accuracy:.3f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = clf
                best_name = name
        
        print(f"\n✅ Best model: {best_name} with accuracy: {best_accuracy:.3f}")
        
        # Detailed evaluation of best model
        y_pred_best = best_model.predict(X_test_scaled)
        
        print(f"\n📊 Detailed Results for {best_name}:")
        print("=" * 50)
        print(classification_report(y_test, y_pred_best, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_best)
        
        plt.figure(figsize=(8, 6))
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{best_name} - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
        # Feature importance (for Random Forest)
        if best_name == 'Random Forest':
            feature_importance = best_model.feature_importances_
            
            # Show top 10 most important features
            top_indices = np.argsort(feature_importance)[-10:]
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(top_indices)), feature_importance[top_indices])
            plt.yticks(range(len(top_indices)), [f'Feature {i}' for i in top_indices])
            plt.xlabel('Feature Importance')
            plt.title('Top 10 Most Important Features for Drum Classification')
            plt.tight_layout()
            plt.show()
        
        return best_model, scaler, best_accuracy
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Please install scikit-learn: pip install scikit-learn")
        return None, None, 0

def save_drum_model(model, scaler, class_names, accuracy, save_dir="models"):
    """Save the trained drum classifier."""
    try:
        import pickle
        import os
        
        os.makedirs(save_dir, exist_ok=True)
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'class_names': class_names,
            'accuracy': accuracy,
            'model_type': 'drum_classifier',
            'features': ['time_domain', 'spectral', 'mfcc']
        }
        
        model_path = os.path.join(save_dir, 'drum_classifier.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to: {model_path}")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Classes: {class_names}")
        
        return model_path
        
    except Exception as e:
        print(f"⚠️ Error saving model: {e}")
        return None

def predict_drum_sample(model_path, audio_file):
    """Test the model on a new drum sample."""
    try:
        import pickle
        
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        class_names = model_data['class_names']
        
        # Load and process audio
        processor = AudioProcessor(sample_rate=22050)
        extractor = FeatureExtractor(sample_rate=22050)
        
        audio_data, sr = processor.load_audio(audio_file)
        
        # Extract features (same as training)
        features = {}
        features.update(extractor.extract_time_domain_features(audio_data))
        features.update(extractor.extract_spectral_features(audio_data))
        features.update(extractor.extract_mfcc(audio_data, include_delta=True))
        
        feature_vector = extractor.create_feature_vector(features, aggregate=True)
        feature_vector = feature_vector.reshape(1, -1)
        
        # Scale and predict
        feature_vector_scaled = scaler.transform(feature_vector)
        prediction = model.predict(feature_vector_scaled)[0]
        probability = model.predict_proba(feature_vector_scaled)[0]
        
        predicted_class = class_names[prediction]
        confidence = probability[prediction]
        
        print(f"🥁 Prediction for {Path(audio_file).name}:")
        print(f"   Predicted: {predicted_class}")
        print(f"   Confidence: {confidence:.3f}")
        
        for i, class_name in enumerate(class_names):
            print(f"   {class_name}: {probability[i]:.3f}")
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, 0

def main():
    """Main function for drum classification training."""
    parser = argparse.ArgumentParser(description='Train Kick vs Snare Drum Classifier')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Directory containing kick and snare folders')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--save_model', action='store_true', default=True,
                       help='Save trained model')
    parser.add_argument('--test_file', type=str, default=None,
                       help='Test model on specific audio file')
    
    args = parser.parse_args()
    
    print("🥁 Kick vs Snare Drum Classifier Training")
    print("=" * 50)
    
    try:
        # Step 1: Load drum dataset
        audio_files, labels, class_names = load_drum_dataset(args.data_dir)
        
        if len(audio_files) == 0:
            print("❌ No audio files found! Please check your data directory.")
            return
        
        # Step 2: Visualize data (optional)
        if args.visualize:
            visualize_drum_data(audio_files, labels, class_names)
        
        # Step 3: Extract features
        features = extract_drum_features(audio_files, class_names)
        
        # Step 4: Train classifier
        model, scaler, accuracy = train_drum_classifier(features, labels, class_names)
        
        if model is None:
            print("❌ Training failed!")
            return
        
        # Step 5: Save model
        if args.save_model:
            model_path = save_drum_model(model, scaler, class_names, accuracy)
        
        # Step 6: Test on specific file (optional)
        if args.test_file and model_path:
            predict_drum_sample(model_path, args.test_file)
        
        print(f"\n🎉 Drum classifier training completed!")
        print(f"   Final accuracy: {accuracy:.3f}")
        print(f"   Classes: {', '.join(class_names)}")
        
        print(f"\n💡 Usage examples:")
        print(f"   # Test on a specific file:")
        print(f"   python scripts/train_drums.py --test_file data/raw/kick/audio1.wav")
        print(f"   ")
        print(f"   # Train with visualizations:")
        print(f"   python scripts/train_drums.py --visualize")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

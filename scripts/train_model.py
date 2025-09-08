#!/usr/bin/env python3
"""
Simple Training Script for Audio ML Project

A streamlined script to train audio classification models with minimal setup.
Perfect for getting started quickly with the learning project.
"""

import numpy as np
import os
import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

print("🎵 Simple Audio ML Training Script")
print("=" * 40)

def create_demo_data():
    """Create simple demo audio data for training."""
    print("🎼 Creating demo audio data...")
    
    sample_rate = 22050
    duration = 2.0
    n_samples_per_class = 30
    
    # Create time axis
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    classes = ['low_tone', 'mid_tone', 'high_tone', 'noise']
    X = []
    y = []
    
    for class_idx, class_name in enumerate(classes):
        print(f"   Creating {class_name} samples...")
        
        for i in range(n_samples_per_class):
            if class_name == 'low_tone':
                freq = np.random.uniform(100, 200)
                audio = np.sin(2 * np.pi * freq * t)
            elif class_name == 'mid_tone':
                freq = np.random.uniform(400, 600)
                audio = np.sin(2 * np.pi * freq * t)
            elif class_name == 'high_tone':
                freq = np.random.uniform(800, 1200)
                audio = np.sin(2 * np.pi * freq * t)
            else:  # noise
                audio = np.random.normal(0, 0.3, len(t))
            
            # Add some variation
            audio += 0.1 * np.random.normal(0, 1, len(audio))
            audio = audio / np.max(np.abs(audio))  # Normalize
            
            X.append(audio)
            y.append(class_idx)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Created demo dataset:")
    print(f"   Shape: {X.shape}")
    print(f"   Classes: {classes}")
    print(f"   Samples per class: {n_samples_per_class}")
    
    return X, y, classes

def extract_simple_features(X):
    """Extract simple features from audio data."""
    print("🔍 Extracting features...")
    
    features = []
    
    for audio in X:
        # Simple time-domain features
        feature_vector = [
            np.mean(audio),              # Mean
            np.std(audio),               # Standard deviation
            np.max(audio),               # Maximum
            np.min(audio),               # Minimum
            np.mean(np.abs(audio)),      # Mean absolute value
            np.sqrt(np.mean(audio**2)),  # RMS
        ]
        
        # Add frequency domain features (simplified)
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft[:len(fft)//2])
        
        feature_vector.extend([
            np.argmax(magnitude),        # Dominant frequency bin
            np.max(magnitude),           # Peak magnitude
            np.mean(magnitude),          # Mean magnitude
            np.std(magnitude),           # Magnitude std
        ])
        
        features.append(feature_vector)
    
    features = np.array(features)
    print(f"✅ Extracted features shape: {features.shape}")
    
    return features

def train_simple_model(X_features, y, class_names):
    """Train a simple machine learning model."""
    print("🚀 Training machine learning model...")
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.preprocessing import StandardScaler
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained successfully!")
        print(f"   Test Accuracy: {accuracy:.3f}")
        
        # Detailed report
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        return model, scaler, accuracy
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Please install scikit-learn: pip install scikit-learn")
        return None, None, 0

def visualize_results(X, y, class_names, features):
    """Create simple visualizations."""
    print("📊 Creating visualizations...")
    
    try:
        import matplotlib.pyplot as plt
        
        # 1. Show sample waveforms
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, class_name in enumerate(class_names):
            class_samples = X[y == i]
            sample_audio = class_samples[0]  # First sample
            
            time_axis = np.linspace(0, 2.0, len(sample_audio))
            axes[i].plot(time_axis, sample_audio, linewidth=1)
            axes[i].set_title(f'{class_name.replace("_", " ").title()}')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Sample Waveforms by Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # 2. Feature scatter plot
        if features.shape[1] >= 2:
            plt.figure(figsize=(10, 6))
            
            colors = ['blue', 'green', 'red', 'purple']
            
            for i, class_name in enumerate(class_names):
                class_features = features[y == i]
                plt.scatter(class_features[:, 0], class_features[:, 1], 
                           c=colors[i], label=class_name.replace("_", " ").title(),
                           alpha=0.6)
            
            plt.xlabel('Feature 1 (Mean)')
            plt.ylabel('Feature 2 (Std)')
            plt.title('Feature Space Visualization')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        
        print("✅ Visualizations created!")
        
    except ImportError:
        print("⚠️ Matplotlib not available for visualization")
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Simple Audio ML Training')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--save-model', action='store_true',
                       help='Save trained model')
    
    args = parser.parse_args()
    
    try:
        # Step 1: Create demo data
        X, y, class_names = create_demo_data()
        
        # Step 2: Extract features
        features = extract_simple_features(X)
        
        # Step 3: Train model
        model, scaler, accuracy = train_simple_model(features, y, class_names)
        
        if model is None:
            print("❌ Training failed!")
            return
        
        # Step 4: Visualize (optional)
        if args.visualize:
            visualize_results(X, y, class_names, features)
        
        # Step 5: Save model (optional)
        if args.save_model:
            try:
                import pickle
                os.makedirs('models', exist_ok=True)
                
                with open('models/simple_audio_model.pkl', 'wb') as f:
                    pickle.dump({
                        'model': model,
                        'scaler': scaler,
                        'class_names': class_names,
                        'accuracy': accuracy
                    }, f)
                
                print("💾 Model saved to models/simple_audio_model.pkl")
                
            except Exception as e:
                print(f"⚠️ Error saving model: {e}")
        
        print(f"\n🎉 Training completed successfully!")
        print(f"   Final accuracy: {accuracy:.3f}")
        print(f"   Classes: {', '.join(class_names)}")
        
        print(f"\n💡 Next steps:")
        print(f"   - Try the full training pipeline: python src/training.py")
        print(f"   - Explore Jupyter notebooks: jupyter lab notebooks/")
        print(f"   - Check out the real-time demo: python scripts/real_time_demo.py")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print(f"   Make sure you have the required dependencies installed")
        print(f"   Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main()

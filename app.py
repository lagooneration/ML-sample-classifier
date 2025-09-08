#!/usr/bin/env python3
"""
Flask Web Application for Drum Classification
Upload audio files and get real-time classification results.
"""

import os
import sys
import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import librosa
import soundfile as sf

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from audio_processing import AudioProcessor
from feature_extraction import FeatureExtractor

app = Flask(__name__)
app.secret_key = 'drum_classifier_secret_key_2025'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
MODEL_PATH = 'models/drum_classifier.pkl'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model and processors
model_data = None
audio_processor = None
feature_extractor = None

def load_model():
    """Load the trained drum classifier model."""
    global model_data, audio_processor, feature_extractor
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        
        audio_processor = AudioProcessor()
        feature_extractor = FeatureExtractor()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Classes: {model_data['class_names']}")
        print(f"   Accuracy: {model_data['accuracy']:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio_features(audio_file_path):
    """Extract features from uploaded audio file."""
    try:
        # Load and process audio
        audio, sr = audio_processor.load_audio(audio_file_path)
        audio = audio_processor.normalize_audio(audio)
        
        # Get audio info
        duration = len(audio) / sr
        
        # Extract features (same as training)
        features = {}
        
        # Time domain features
        time_features = feature_extractor.extract_time_domain_features(audio)
        features.update(time_features)
        
        # Spectral features
        spectral_features = feature_extractor.extract_spectral_features(audio)
        features.update(spectral_features)
        
        # MFCC features
        mfcc_features = feature_extractor.extract_mfcc(audio, include_delta=True)
        features.update(mfcc_features)
        
        # Create feature vector
        feature_vector = feature_extractor.create_feature_vector(features, aggregate=True)
        
        return feature_vector, duration, sr, len(audio), features
        
    except Exception as e:
        raise Exception(f"Feature extraction failed: {str(e)}")

def predict_drum_type(feature_vector):
    """Predict drum type from feature vector."""
    try:
        model = model_data['model']
        classes = model_data['class_names']
        
        # Make prediction
        prediction_idx = model.predict(feature_vector.reshape(1, -1))[0]
        prediction = classes[prediction_idx]
        probabilities = model.predict_proba(feature_vector.reshape(1, -1))[0]
        confidence = max(probabilities)
        
        # Get probabilities for all classes
        class_probabilities = {classes[i]: prob for i, prob in enumerate(probabilities)}
        
        return prediction, confidence, class_probabilities
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

@app.route('/')
def index():
    """Main page with upload form."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and classification."""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Secure filename and save
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Extract features and make prediction
            feature_vector, duration, sample_rate, samples, raw_features = extract_audio_features(filepath)
            prediction, confidence, class_probabilities = predict_drum_type(feature_vector)
            
            # Prepare detailed results
            audio_info = {
                'filename': filename,
                'duration': f"{duration:.2f}",
                'sample_rate': sample_rate,
                'samples': samples,
                'size_mb': f"{os.path.getsize(filepath) / (1024*1024):.2f}"
            }
            
            # Feature analysis
            feature_analysis = {
                'total_features': len(feature_vector),
                'rms_energy': f"{raw_features.get('rms_energy', [0])[0]:.4f}" if 'rms_energy' in raw_features else "N/A",
                'zero_crossing_rate': f"{np.mean(raw_features.get('zero_crossing_rate', [0])):.4f}" if 'zero_crossing_rate' in raw_features else "N/A",
                'spectral_centroid': f"{np.mean(raw_features.get('spectral_centroid', [0])):.1f}" if 'spectral_centroid' in raw_features else "N/A",
                'spectral_rolloff': f"{np.mean(raw_features.get('spectral_rolloff', [0])):.1f}" if 'spectral_rolloff' in raw_features else "N/A"
            }
            
            # Classification results
            results = {
                'prediction': prediction,
                'confidence': f"{confidence:.3f}",
                'confidence_percent': f"{confidence*100:.1f}",
                'class_probabilities': {k: f"{v:.3f}" for k, v in class_probabilities.items()},
                'audio_info': audio_info,
                'feature_analysis': feature_analysis,
                'model_info': {
                    'accuracy': f"{model_data['accuracy']:.3f}",
                    'classes': model_data['class_names'],
                    'model_type': model_data.get('model_type', 'drum_classifier')
                }
            }
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return render_template('results.html', results=results)
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload WAV, MP3, FLAC, OGG, or M4A files.')
        return redirect(url_for('index'))

@app.route('/api/classify', methods=['POST'])
def api_classify():
    """API endpoint for programmatic access."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save temporary file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process and predict
        feature_vector, duration, sample_rate, samples, raw_features = extract_audio_features(filepath)
        prediction, confidence, class_probabilities = predict_drum_type(feature_vector)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'audio_info': {
                'duration': duration,
                'sample_rate': sample_rate,
                'samples': samples
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_data is not None,
        'model_accuracy': model_data['accuracy'] if model_data else None
    })

if __name__ == '__main__':
    print("🥁 Starting Drum Classifier Web App")
    print("=" * 50)
    
    # Load the trained model
    if load_model():
        print("🚀 Starting Flask server...")
        print("📱 Open your browser to: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Cannot start app - model not found!")
        print("💡 Train a model first using: python scripts/train_drums.py")

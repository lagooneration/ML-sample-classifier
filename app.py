#!/usr/bin/env python3
"""
Flask Web Application for Audio Classification
Upload audio files for training and get real-time classification results.
"""

import os
import sys
import pickle
import numpy as np
import json
import shutil
from datetime import datetime
from pathlib import Path
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
import librosa
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from audio_processing import AudioProcessor
from feature_extraction import FeatureExtractor

app = Flask(__name__)
app.secret_key = 'drum_classifier_secret_key_2025'

# Configuration
UPLOAD_FOLDER = 'uploads'
TRAINING_DATA_FOLDER = 'data/training'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
MODEL_PATH = 'models/drum_classifier.pkl'
TRAINING_STATUS_FILE = 'models/training_status.json'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRAINING_DATA_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

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

def get_training_status():
    """Get current training status and statistics."""
    status = {
        'classes': [],
        'total_samples': 0,
        'model_trained': False,
        'last_training': None,
        'accuracy': None
    }
    
    try:
        # Check training data
        training_path = Path(TRAINING_DATA_FOLDER)
        if training_path.exists():
            for class_dir in training_path.iterdir():
                if class_dir.is_dir():
                    audio_files = list(class_dir.glob('*.wav'))
                    status['classes'].append({
                        'name': class_dir.name,
                        'samples': len(audio_files)
                    })
                    status['total_samples'] += len(audio_files)
        
        # Check if model exists
        if os.path.exists(MODEL_PATH):
            status['model_trained'] = True
            if model_data:
                status['accuracy'] = model_data.get('accuracy', 0)
        
        # Check training status file
        if os.path.exists(TRAINING_STATUS_FILE):
            with open(TRAINING_STATUS_FILE, 'r') as f:
                saved_status = json.load(f)
                status.update(saved_status)
    
    except Exception as e:
        print(f"Error getting training status: {e}")
    
    return status

def save_training_file(file, class_name):
    """Save uploaded training file to appropriate class directory."""
    class_dir = os.path.join(TRAINING_DATA_FOLDER, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = secure_filename(file.filename)
    name, ext = os.path.splitext(original_name)
    new_filename = f"{name}_{timestamp}{ext}"
    
    filepath = os.path.join(class_dir, new_filename)
    file.save(filepath)
    return filepath

def train_model_from_data():
    """Train a new model from the uploaded training data."""
    global model_data, audio_processor, feature_extractor
    
    try:
        # Initialize processors
        local_audio_processor = AudioProcessor()
        local_feature_extractor = FeatureExtractor()
        
        # Load training data
        training_path = Path(TRAINING_DATA_FOLDER)
        
        features = []
        labels = []
        class_names = []
        
        for class_dir in training_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name not in class_names:
                    class_names.append(class_name)
                class_idx = class_names.index(class_name)
                
                # Process all audio files in this class
                audio_files = list(class_dir.glob('*.wav'))
                for audio_file in audio_files:
                    try:
                        # Load and process audio
                        audio, sr = local_audio_processor.load_audio(str(audio_file))
                        audio = local_audio_processor.normalize_audio(audio)
                        
                        # Extract features
                        feature_dict = {}
                        
                        # Time domain features
                        time_features = local_feature_extractor.extract_time_domain_features(audio)
                        feature_dict.update(time_features)
                        
                        # Spectral features
                        spectral_features = local_feature_extractor.extract_spectral_features(audio)
                        feature_dict.update(spectral_features)
                        
                        # MFCC features
                        mfcc_features = local_feature_extractor.extract_mfcc(audio, include_delta=True)
                        feature_dict.update(mfcc_features)
                        
                        # Create feature vector
                        feature_vector = local_feature_extractor.create_feature_vector(feature_dict, aggregate=True)
                        
                        features.append(feature_vector)
                        labels.append(class_idx)
                        
                    except Exception as e:
                        print(f"Error processing {audio_file}: {e}")
                        continue
        
        if len(features) < 2:
            raise Exception("Need at least 2 samples to train a model")
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        model_data_new = {
            'model': model,
            'class_names': class_names,
            'accuracy': accuracy,
            'model_type': 'audio_classifier',
            'features_used': X.shape[1],
            'training_samples': len(X),
            'trained_on': datetime.now().isoformat()
        }
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model_data_new, f)
        
        # Update global model
        model_data = model_data_new
        audio_processor = local_audio_processor
        feature_extractor = local_feature_extractor
        
        # Save training status
        status = {
            'last_training': datetime.now().isoformat(),
            'accuracy': accuracy,
            'classes_trained': class_names,
            'total_samples': len(X)
        }
        
        with open(TRAINING_STATUS_FILE, 'w') as f:
            json.dump(status, f, indent=2)
        
        return {
            'success': True,
            'accuracy': accuracy,
            'classes': class_names,
            'samples': len(X),
            'features': X.shape[1]
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/')
def index():
    """Main page with navigation."""
    return render_template('index.html')

@app.route('/train')
def train_page():
    """Training page."""
    status = get_training_status()
    return render_template('train.html', status=status)

@app.route('/test')
def test_page():
    """Testing page."""
    if not model_data:
        flash('No trained model available. Please train a model first.')
        return redirect(url_for('train_page'))
    return render_template('test.html')

@app.route('/upload_training', methods=['POST'])
def upload_training():
    """Handle training file upload."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files selected'}), 400
    
    class_name = request.form.get('class_name', '').strip()
    if not class_name:
        return jsonify({'error': 'Class name is required'}), 400
    
    # Validate class name (alphanumeric and underscores only)
    if not class_name.replace('_', '').isalnum():
        return jsonify({'error': 'Class name must contain only letters, numbers, and underscores'}), 400
    
    files = request.files.getlist('files')
    uploaded_files = []
    errors = []
    
    for file in files:
        if file.filename == '':
            continue
            
        if file and allowed_file(file.filename):
            try:
                filepath = save_training_file(file, class_name)
                uploaded_files.append({
                    'filename': file.filename,
                    'class': class_name,
                    'path': filepath
                })
            except Exception as e:
                errors.append(f"Error uploading {file.filename}: {str(e)}")
        else:
            errors.append(f"Invalid file type: {file.filename}")
    
    return jsonify({
        'success': len(uploaded_files) > 0,
        'uploaded': uploaded_files,
        'errors': errors,
        'total_uploaded': len(uploaded_files)
    })

@app.route('/start_training', methods=['POST'])
def start_training():
    """Start model training."""
    try:
        result = train_model_from_data()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/training_status')
def training_status():
    """Get current training status."""
    status = get_training_status()
    return jsonify(status)

@app.route('/delete_class/<class_name>', methods=['DELETE'])
def delete_class(class_name):
    """Delete all samples from a class."""
    try:
        class_dir = os.path.join(TRAINING_DATA_FOLDER, class_name)
        if os.path.exists(class_dir):
            shutil.rmtree(class_dir)
            return jsonify({'success': True, 'message': f'Class "{class_name}" deleted successfully'})
        else:
            return jsonify({'success': False, 'message': 'Class not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and classification for testing."""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('test_page'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('test_page'))
    
    if not model_data:
        flash('No trained model available. Please train a model first.')
        return redirect(url_for('train_page'))
    
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
                    'model_type': model_data.get('model_type', 'audio_classifier')
                }
            }
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return render_template('results.html', results=results)
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('test_page'))
    else:
        flash('Invalid file type. Please upload WAV, MP3, FLAC, OGG, or M4A files.')
        return redirect(url_for('test_page'))

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
    print("🎵 Starting Audio Classification Web App")
    print("=" * 50)
    
    # Load the trained model
    if load_model():
        print("🚀 Starting Flask server...")
        
        # Get port from environment variable (for Heroku/Railway)
        import os
        port = int(os.environ.get('PORT', 5000))
        
        print(f"📱 Server will run on port: {port}")
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("❌ Cannot start app - model not found!")
        print("💡 Train a model first using: python scripts/train_drums.py")

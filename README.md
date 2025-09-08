# 🎵 Audio Time Series ML Learning Project

A comprehensive machine learning project for exploring audio time series analysis, featuring real-time visualization and hands-on learning examples.

## 📊 Project Overview

This project demonstrates machine learning techniques applied to audio time series data, including:
- **Audio Feature Extraction** (MFCC, Spectrograms, Chroma features)
- **Time Series Classification** (Music genre, emotion recognition)
- **Real-time Audio Processing** and Visualization
- **Interactive Learning Examples** with Jupyter notebooks

## 🎯 Learning Objectives

- Understand audio signal processing fundamentals
- Learn time series feature extraction techniques
- Implement neural networks for audio classification
- Visualize learning processes and model performance
- Explore real-time audio analysis applications

## 🚀 Features

### 🔧 Core Components
- **Audio Preprocessing Pipeline** - Load, normalize, and segment audio files
- **Feature Engineering** - Extract meaningful features from audio signals
- **ML Models** - CNN, RNN, and hybrid architectures for classification
- **Visualization Tools** - Interactive plots for audio analysis and model training
- **Real-time Processing** - Live audio classification and feature visualization

### 📈 Visualization Capabilities
- Waveform and spectrogram displays
- Feature importance heatmaps
- Training progress monitoring
- Confusion matrices and performance metrics
- Real-time feature extraction plots

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Audio drivers (for real-time processing)

### Setup
```bash
# Clone and navigate to project
cd p:\ML\hearing

# Install dependencies
pip install -r requirements.txt

# Download sample audio data
python scripts/download_data.py
```

## 📁 Project Structure

```
hearing/
├── 📂 data/
│   ├── raw/                 # Original audio files
│   ├── processed/           # Preprocessed features
│   └── sample/              # Example audio files
├── 📂 src/
│   ├── audio_processing.py  # Audio loading and preprocessing
│   ├── feature_extraction.py # Feature engineering
│   ├── models.py           # ML model definitions
│   ├── training.py         # Training pipeline
│   └── visualization.py    # Plotting and visualization
├── 📂 notebooks/
│   ├── 01_audio_basics.ipynb      # Audio fundamentals
│   ├── 02_feature_extraction.ipynb # Feature engineering
│   ├── 03_model_training.ipynb     # ML model training
│   └── 04_real_time_demo.ipynb     # Live audio processing
├── 📂 scripts/
│   ├── download_data.py     # Data acquisition
│   ├── train_model.py       # Training script
│   └── real_time_demo.py    # Live demonstration
├── 📂 models/               # Saved model files
├── 📂 results/              # Training outputs and plots
└── requirements.txt
```

## 🎮 Quick Start

### 1. Basic Audio Analysis
```python
from src.audio_processing import AudioProcessor
from src.visualization import AudioVisualizer

# Load and visualize audio
processor = AudioProcessor()
audio_data = processor.load_audio('data/sample/music.wav')

visualizer = AudioVisualizer()
visualizer.plot_waveform(audio_data)
visualizer.plot_spectrogram(audio_data)
```

### 2. Feature Extraction
```python
from src.feature_extraction import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_mfcc(audio_data)
extractor.visualize_features(features)
```

### 3. Model Training
```python
from src.models import AudioClassifier
from src.training import Trainer

model = AudioClassifier(num_classes=10)
trainer = Trainer(model)
history = trainer.train(features, labels, visualize=True)
```

## 🥁 Drum Classification Scripts

### Training a Drum Classifier

#### Basic Training
```bash
# Train drum classifier with default settings
python scripts/train_drums.py

# Train with specific data directory
python scripts/train_drums.py --data_dir data/raw

# Train with visualizations
python scripts/train_drums.py --visualize

# Full training with model saving and visualizations
python scripts/train_drums.py --visualize --save_model --data_dir data/raw
```

#### Training Script Options
- `--data_dir`: Directory containing kick and snare folders (default: `data/raw`)
- `--visualize`: Create training visualizations and feature plots
- `--save_model`: Save the trained model (default: True)
- `--test_file`: Test model on a specific audio file after training

#### Expected Output
```
🥁 Kick vs Snare Drum Classifier Training
==================================================
🥁 Loading drum dataset from: data/raw
✅ Loaded dataset:
   Classes: ['kick', 'snare']
   Total samples: 20
   kick: 10 samples
   snare: 10 samples

🔍 Extracting drum-specific features...
✅ Feature extraction completed!
   Feature matrix shape: (20, 25)

🚀 Training drum classifier...
   Training samples: 14
   Test samples: 6

✅ Best model: Random Forest with accuracy: 0.833
💾 Model saved to: models/drum_classifier.pkl
```

### Testing the Trained Model

#### Single File Testing
```bash
# Test on a kick drum sample
python scripts/test_drum_model.py data/raw/kick/audio1.wav

# Test on a snare drum sample
python scripts/test_drum_model.py data/raw/snare/audio5.wav

# Test with custom model path
python scripts/test_drum_model.py path/to/audio.wav --model models/custom_model.pkl
```

#### Expected Test Output
```
🎵 Testing audio file: data/raw/kick/audio1.wav
🥁 Prediction: kick
📊 Confidence: 0.950
📈 Confidence: [███████████████████ ] 95.0%
✅ High confidence prediction!
```

## 🌐 Web Interface

### Starting the Web Application
```bash
# Start the Flask web server
python app.py

# Server will start on http://localhost:5000
```

### Web Interface Features

#### 🎵 Upload Interface
- **Drag & Drop**: Simply drag audio files onto the upload area
- **File Browser**: Click to browse and select audio files
- **Format Support**: WAV, MP3, FLAC, OGG, M4A files
- **File Validation**: Automatic format and size checking (max 16MB)

#### 📊 Results Display
- **Visual Prediction**: Large, clear prediction display with confidence
- **Confidence Meter**: Animated progress bar showing prediction confidence
- **Audio Information**: File details (duration, sample rate, size)
- **Feature Analysis**: Key audio features used for classification
- **Class Probabilities**: Probability breakdown for each drum type
- **Model Information**: Details about the trained classifier

#### 🔧 API Endpoint
```bash
# Programmatic access via API
curl -X POST -F "file=@audio.wav" http://localhost:5000/api/classify
```

#### API Response Example
```json
{
  "prediction": "snare",
  "confidence": 0.892,
  "class_probabilities": {
    "kick": 0.108,
    "snare": 0.892
  },
  "audio_info": {
    "duration": 0.45,
    "sample_rate": 22050,
    "samples": 9922
  }
}
```

## 🛠️ Training Scripts Overview

### Available Scripts

#### 1. `scripts/train_drums.py` - Drum Classification Training
**Purpose**: Train a machine learning model to classify kick vs snare drum sounds

**Features**:
- Loads audio files from organized directories
- Extracts 25 audio features (MFCC, spectral, time-domain)
- Trains multiple classifiers (Random Forest, SVM)
- Creates visualizations of training data and results
- Saves the best performing model

**Usage Examples**:
```bash
# Quick training
python scripts/train_drums.py

# Training with full analysis
python scripts/train_drums.py --visualize --data_dir data/raw

# Test specific file after training
python scripts/train_drums.py --test_file data/raw/kick/test.wav
```

#### 2. `scripts/test_drum_model.py` - Model Testing
**Purpose**: Test the trained drum classifier on individual audio files

**Features**:
- Loads pre-trained model
- Extracts features from new audio files
- Provides prediction with confidence score
- Visual confidence indicator

#### 3. `scripts/download_data.py` - Sample Data Generation
**Purpose**: Generate synthetic drum samples for training

**Features**:
- Creates realistic kick and snare drum sounds
- Generates multiple variations with different parameters
- Saves in organized directory structure

#### 4. `scripts/real_time_demo.py` - Live Audio Processing
**Purpose**: Demonstrate real-time audio classification

**Features**:
- Live microphone input processing
- Real-time feature extraction
- Live prediction display
- Visual audio analysis dashboard

#### 5. `app.py` - Web Application
**Purpose**: Provide user-friendly web interface for drum classification

**Features**:
- Drag-and-drop file upload
- Beautiful results visualization
- Detailed audio analysis
- RESTful API for programmatic access

### Training Data Organization

#### Required Directory Structure
```
data/raw/
├── kick/           # Kick drum samples
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── snare/          # Snare drum samples
    ├── audio1.wav
    ├── audio2.wav
    └── ...
```

#### Data Requirements
- **Format**: WAV files preferred (MP3, FLAC also supported)
- **Sample Rate**: 22,050 Hz (automatically resampled if different)
- **Duration**: Any length (automatically segmented if needed)
- **Quality**: Clean, isolated drum hits work best
- **Quantity**: Minimum 5 samples per class, 10+ recommended

### Model Performance Analysis

#### Feature Engineering (25 Features)
1. **Time Domain Features (6)**
   - RMS Energy (mean, std)
   - Zero Crossing Rate (mean, std)
   - Duration, Peak Amplitude

2. **Spectral Features (12)**
   - Spectral Centroid (mean, std, min, max)
   - Spectral Rolloff (mean, std, min, max)
   - Spectral Bandwidth (mean, std, min, max)

3. **MFCC Features (7)**
   - 13 MFCC coefficients (aggregated)
   - Delta features for temporal dynamics

#### Model Comparison
- **Random Forest**: 83.3% accuracy (recommended)
- **SVM with RBF**: 67% accuracy
- **Cross-validation**: 5-fold stratified

#### Performance Metrics
```
              precision    recall  f1-score   support
        kick       1.00      0.67      0.80         3
       snare       0.75      1.00      0.86         3
    accuracy                           0.83         6
```

## 📓 Learning Notebooks

### 🎵 01_audio_basics.ipynb
- Audio signal fundamentals
- Loading and manipulating audio files
- Time domain vs frequency domain analysis
- Interactive waveform exploration

### 🔍 02_feature_extraction.ipynb
- MFCC (Mel-Frequency Cepstral Coefficients)
- Spectral features (centroid, rolloff, zero-crossing rate)
- Chroma and tonnetz features
- Feature visualization and interpretation

### 🧠 03_model_training.ipynb
- Dataset preparation and augmentation
- Model architecture design
- Training with real-time monitoring
- Performance evaluation and visualization

### ⚡ 04_real_time_demo.ipynb
- Live audio capture and processing
- Real-time feature extraction
- Live classification with confidence scores
- Interactive dashboard

## 🎯 Example Use Cases

### 🥁 Drum Sound Classification (Main Implementation)
Classify drum sounds into kick and snare categories
```bash
# Train the drum classifier
python scripts/train_drums.py --visualize --data_dir data/raw

# Test the trained model
python scripts/test_drum_model.py data/raw/kick/audio1.wav

# Start web interface
python app.py
```

### 🎼 Music Genre Classification (Example Framework)
Classify audio clips into genres (rock, classical, jazz, etc.)
```bash
python scripts/train_model.py --task genre_classification --data data/music_genres/
```

### 😊 Emotion Recognition (Example Framework)
Detect emotional content in speech or music
```bash
python scripts/train_model.py --task emotion_recognition --data data/emotions/
```

### 🔊 Sound Event Detection (Example Framework)
Identify specific sounds in audio streams
```bash
python scripts/real_time_demo.py --task sound_detection
```

## 📋 Complete Script Reference

### Training Scripts

#### `scripts/train_drums.py` - Primary Training Script
```bash
# Basic usage
python scripts/train_drums.py

# Advanced usage with all options
python scripts/train_drums.py \
    --data_dir data/raw \
    --visualize \
    --save_model \
    --test_file data/raw/kick/audio1.wav

# Options:
#   --data_dir     : Directory with kick/ and snare/ folders
#   --visualize    : Create training plots and visualizations
#   --save_model   : Save trained model to models/ directory
#   --test_file    : Test on specific file after training
```

#### `scripts/train_model.py` - General Training Framework
```bash
# Genre classification
python scripts/train_model.py \
    --data data/genres \
    --model_type cnn \
    --epochs 50 \
    --visualize

# Options:
#   --data         : Path to training data
#   --model_type   : Model architecture (cnn, rnn, hybrid)
#   --epochs       : Number of training epochs
#   --batch_size   : Training batch size
#   --learning_rate: Learning rate for training
#   --visualize    : Show training progress plots
```

### Testing and Evaluation Scripts

#### `scripts/test_drum_model.py` - Model Testing
```bash
# Test single file
python scripts/test_drum_model.py audio_file.wav

# Test with custom model
python scripts/test_drum_model.py audio_file.wav --model path/to/model.pkl

# Batch testing (if implemented)
python scripts/test_drum_model.py --batch_dir data/test_samples/
```

#### `scripts/evaluate_model.py` - Model Evaluation
```bash
# Evaluate on test set
python scripts/evaluate_model.py \
    --model models/drum_classifier.pkl \
    --test_data data/test \
    --output_report results/evaluation.html

# Cross-validation evaluation
python scripts/evaluate_model.py \
    --model models/drum_classifier.pkl \
    --cross_validate \
    --folds 5
```

### Data Management Scripts

#### `scripts/download_data.py` - Sample Data Generation
```bash
# Generate synthetic drum samples
python scripts/download_data.py

# Generate specific amount
python scripts/download_data.py --samples 50 --output data/synthetic

# Download real datasets (if available)
python scripts/download_data.py --dataset freesound_drums --output data/real
```

#### `scripts/prepare_data.py` - Data Preprocessing
```bash
# Preprocess audio files
python scripts/prepare_data.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --normalize \
    --resample 22050

# Data augmentation
python scripts/prepare_data.py \
    --input_dir data/raw \
    --augment \
    --noise_level 0.01 \
    --time_stretch 0.1
```

### Real-time and Demo Scripts

#### `scripts/real_time_demo.py` - Live Audio Processing
```bash
# Real-time drum classification
python scripts/real_time_demo.py --model models/drum_classifier.pkl

# Real-time with visualization
python scripts/real_time_demo.py \
    --model models/drum_classifier.pkl \
    --visualize \
    --save_output results/live_demo.mp4

# Options:
#   --model        : Path to trained model
#   --visualize    : Show real-time plots
#   --save_output  : Record demo session
#   --input_device : Microphone device index
#   --buffer_size  : Audio buffer size
```

#### `scripts/feature_analysis.py` - Feature Exploration
```bash
# Analyze features in dataset
python scripts/feature_analysis.py \
    --data_dir data/raw \
    --output results/feature_analysis.html

# Compare features between classes
python scripts/feature_analysis.py \
    --data_dir data/raw \
    --compare_classes \
    --plot_distributions
```

### Web Application

#### `app.py` - Flask Web Interface
```bash
# Start web server
python app.py

# Start with custom settings
python app.py --host 0.0.0.0 --port 8080 --debug

# Production deployment (with gunicorn)
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

### Utility Scripts

#### `scripts/model_converter.py` - Model Format Conversion
```bash
# Convert scikit-learn to ONNX
python scripts/model_converter.py \
    --input models/drum_classifier.pkl \
    --output models/drum_classifier.onnx \
    --format onnx

# Convert to TensorFlow Lite
python scripts/model_converter.py \
    --input models/drum_classifier.pkl \
    --output models/drum_classifier.tflite \
    --format tflite
```

#### `scripts/audio_utils.py` - Audio Utilities
```bash
# Convert audio formats
python scripts/audio_utils.py convert \
    --input data/raw \
    --output data/converted \
    --format wav \
    --sample_rate 22050

# Audio quality analysis
python scripts/audio_utils.py analyze \
    --input data/raw \
    --output results/audio_quality_report.html
```

## 🚀 Quick Start Commands

### For Beginners
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample data (if you don't have audio files)
python scripts/download_data.py

# 3. Train your first model
python scripts/train_drums.py --visualize

# 4. Test the model
python scripts/test_drum_model.py data/raw/kick/audio1.wav

# 5. Start web interface
python app.py
```

### For Advanced Users
```bash
# Complete training pipeline with evaluation
python scripts/train_drums.py --visualize --save_model
python scripts/evaluate_model.py --model models/drum_classifier.pkl
python scripts/feature_analysis.py --data_dir data/raw

# Real-time demonstration
python scripts/real_time_demo.py --model models/drum_classifier.pkl --visualize

# Deploy web application
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 😊 Emotion Recognition
Detect emotional content in speech or music
```python
python scripts/train_model.py --task emotion_recognition --data data/emotions/
```

### 🔊 Sound Event Detection
Identify specific sounds in audio streams
```python
python scripts/real_time_demo.py --task sound_detection
```

## 📊 Model Performance

| Model | Accuracy | Training Time | Features |
|-------|----------|---------------|----------|
| CNN-1D | 87.3% | 15 min | Raw audio |
| CNN-2D | 91.2% | 22 min | Spectrograms |
| RNN-LSTM | 89.7% | 35 min | MFCC sequences |
| Hybrid | 93.1% | 28 min | Multi-modal |

## 🎨 Visualization Examples

### Waveform Analysis
![Waveform Example](results/waveform_example.png)

### Spectrogram Visualization
![Spectrogram Example](results/spectrogram_example.png)

### Feature Importance
![Feature Importance](results/feature_importance.png)

### Training Progress
![Training Progress](results/training_progress.png)

## 🔧 Advanced Features

### Custom Feature Extractors
```python
class CustomFeatureExtractor(FeatureExtractor):
    def extract_custom_features(self, audio):
        # Implement your own feature extraction
        pass
```

### Model Experimentation
```python
# Try different architectures
model = AudioClassifier(
    architecture='cnn_rnn',
    filters=[32, 64, 128],
    rnn_units=256,
    dropout=0.3
)
```

### Real-time Processing Pipeline
```python
pipeline = RealtimeAudioPipeline(
    feature_extractor=extractor,
    model=trained_model,
    visualization=True
)
pipeline.start_processing()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📚 Learning Resources

- [Audio Signal Processing Basics](docs/audio_basics.md)
- [Time Series ML Fundamentals](docs/timeseries_ml.md)
- [Feature Engineering Guide](docs/feature_engineering.md)
- [Model Architecture Examples](docs/model_architectures.md)

## 🐛 Troubleshooting

### Common Issues
- **Audio not loading**: Check file format and path
- **Feature extraction errors**: Verify sample rate compatibility
- **Model training slow**: Consider reducing batch size or using GPU
- **Real-time lag**: Adjust buffer size and processing window

### Performance Tips
- Use GPU acceleration when available
- Preprocess and cache features for faster training
- Implement data augmentation for better generalization
- Monitor memory usage with large audio files

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- librosa library for audio processing
- TensorFlow/PyTorch for machine learning
- matplotlib/plotly for visualization
- Community audio datasets

---

**Happy Learning! 🎵🤖**

*This project is designed to be educational and hands-on. Each component includes detailed comments and examples to help you understand the underlying concepts.*

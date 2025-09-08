# 🚀 Quick Start Guide

Welcome to the Audio Time Series ML Learning Project! This guide will get you up and running in just a few minutes.

## ⚡ 5-Minute Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Setup (Optional)
```bash
python setup.py --full-setup
```

### 3. Try Your First Audio ML Model
```bash
python scripts/train_model.py --visualize
```

### 4. Explore Learning Notebooks
```bash
jupyter lab notebooks/01_audio_basics.ipynb
```

## 🎯 What You'll Learn

- **Audio Signal Processing**: Understanding sound waves and digital audio
- **Feature Extraction**: Converting audio to ML-ready features (MFCC, spectrograms)
- **Classification Models**: Building CNNs and RNNs for audio tasks
- **Real-time Processing**: Live audio analysis and prediction
- **Visualization**: Creating beautiful plots to understand your data

## 📚 Learning Path

| Notebook | Topic | Time | Difficulty |
|----------|-------|------|------------|
| `01_audio_basics.ipynb` | Audio fundamentals | 30 min | Beginner |
| `02_feature_extraction.ipynb` | Feature engineering | 45 min | Intermediate |
| `03_model_training.ipynb` | ML model training | 60 min | Intermediate |
| `04_real_time_demo.ipynb` | Live audio processing | 30 min | Advanced |

## 🛠️ Available Tools

### Simple Training Script
Perfect for beginners - trains a model in one command:
```bash
python scripts/train_model.py --visualize --save-model
```

### Full Training Pipeline
Advanced training with deep learning models:
```bash
python src/training.py --architecture cnn_1d --epochs 50
```

### Real-time Audio Demo
See your models work live:
```bash
python scripts/real_time_demo.py --duration 30
```

### Data Creation
Generate synthetic datasets for learning:
```bash
python scripts/download_data.py --all
```

## 🎵 Example Use Cases

### 1. Music Genre Classification
```python
from src.audio_processing import AudioProcessor
from src.feature_extraction import FeatureExtractor

processor = AudioProcessor()
extractor = FeatureExtractor()

# Load your music
audio, sr = processor.load_audio('your_song.wav')

# Extract features
features = extractor.extract_all_features(audio)
```

### 2. Emotion Recognition in Speech
```python
# Train on emotion dataset
python src/training.py --data_dir data/raw/emotions --architecture rnn
```

### 3. Real-time Sound Classification
```python
# Run live classification
python scripts/real_time_demo.py --model_path models/best_model.h5
```

## 🔧 Troubleshooting

### Common Issues

**ImportError: No module named 'librosa'**
```bash
pip install librosa soundfile
```

**TensorFlow not found**
```bash
pip install tensorflow
```

**Jupyter notebooks won't start**
```bash
pip install jupyter jupyterlab ipywidgets
```

**Audio files won't load**
- Make sure audio files are in WAV format
- Check file paths are correct
- Install additional audio codecs: `pip install audioread`

### Performance Tips

- **GPU Acceleration**: Install `tensorflow-gpu` for faster training
- **Memory Issues**: Reduce batch size or segment length
- **Slow Processing**: Use smaller audio files for learning

## 📖 Project Structure Quick Reference

```
hearing/
├── 📓 notebooks/          # Learning materials
├── 🧠 src/               # Core ML modules  
├── 📊 data/              # Audio datasets
├── 🤖 models/            # Trained models
├── 📈 results/           # Outputs & plots
└── 🛠️ scripts/           # Utility scripts
```

## 🎓 Learning Tips

1. **Start Simple**: Begin with `01_audio_basics.ipynb`
2. **Experiment**: Modify code and see what happens
3. **Visualize**: Always plot your data to understand it
4. **Practice**: Try different audio files and parameters
5. **Ask Questions**: Use the code comments to understand concepts

## 🌟 Next Steps

Once you complete the basics:

1. **Try Real Data**: Download actual audio datasets
2. **Build Custom Models**: Modify the neural network architectures
3. **Create Applications**: Build your own audio classification app
4. **Explore Advanced Topics**: Dive into audio synthesis, speech recognition
5. **Share Your Work**: Contribute improvements back to the project

## 📞 Need Help?

- 📖 Check the main `README.md` for detailed documentation
- 🐛 Issues with code? Review the error messages carefully
- 💡 Want to understand concepts? Start with the Jupyter notebooks
- 🚀 Ready for advanced topics? Explore the full training pipeline

Happy learning! 🎵🤖

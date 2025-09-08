"""
Audio ML Learning Project - Source Module

This package provides comprehensive audio processing and machine learning
capabilities for educational and research purposes.
"""

__version__ = "1.0.0"
__author__ = "Audio ML Learning Project"

# Import main classes for easy access
try:
    from .audio_processing import AudioProcessor
    from .feature_extraction import FeatureExtractor, AudioFeatureExtractor
    from .visualization import AudioVisualizer
    from .models import AudioClassifier, AudioTrainer
    
    __all__ = [
        'AudioProcessor',
        'FeatureExtractor', 
        'AudioFeatureExtractor',
        'AudioVisualizer',
        'AudioClassifier',
        'AudioTrainer'
    ]
    
except ImportError:
    # If imports fail, make a note but don't crash
    print("⚠️ Some dependencies may not be installed.")
    print("Run: pip install -r requirements.txt")
    __all__ = []

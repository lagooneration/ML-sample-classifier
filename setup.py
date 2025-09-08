#!/usr/bin/env python3
"""
Setup and Quick Start Script for Audio ML Learning Project

This script helps you get started with the Audio ML Learning Project by:
- Checking dependencies
- Creating sample data
- Running basic tests
- Providing usage examples
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible!")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn',
        'librosa', 'soundfile', 'scikit-learn',
        'tensorflow', 'jupyter', 'plotly', 'ipywidgets'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are installed!")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n📥 Installing dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def create_sample_data():
    """Create sample audio data for learning."""
    print("\n🎼 Creating sample audio data...")
    
    try:
        # Import and run the data creation script
        sys.path.append('scripts')
        from download_data import create_synthetic_music_dataset, create_emotion_dataset
        
        # Create datasets
        create_synthetic_music_dataset()
        create_emotion_dataset()
        
        print("✅ Sample data created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False


def test_installation():
    """Test the installation by running basic functionality."""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        sys.path.append('src')
        from audio_processing import AudioProcessor, create_synthetic_audio
        from feature_extraction import FeatureExtractor
        from visualization import AudioVisualizer
        
        print("✅ Module imports successful")
        
        # Test basic functionality
        processor = AudioProcessor(sample_rate=22050)
        extractor = FeatureExtractor(sample_rate=22050)
        visualizer = AudioVisualizer()
        
        # Create test audio
        test_audio = create_synthetic_audio(duration=1.0, sample_rate=22050, frequency=440)
        
        # Test processing
        info = processor.get_audio_info(test_audio)
        features = extractor.extract_time_domain_features(test_audio)
        
        print("✅ Basic audio processing works")
        print(f"   Audio duration: {info['duration']:.2f}s")
        print(f"   Extracted {len(features)} time domain features")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


def show_usage_examples():
    """Show usage examples and next steps."""
    print("\n📚 Usage Examples and Next Steps:")
    print("=" * 50)
    
    print("\n1. 🎓 Start Learning with Jupyter Notebooks:")
    print("   jupyter lab notebooks/01_audio_basics.ipynb")
    
    print("\n2. 🎼 Create Sample Datasets:")
    print("   python scripts/download_data.py --all")
    
    print("\n3. 🚀 Train Your First Model:")
    print("   python src/training.py --create_synthetic --epochs 20")
    
    print("\n4. 🎤 Try Real-time Demo:")
    print("   python scripts/real_time_demo.py --duration 10")
    
    print("\n5. 📊 Explore the Modules:")
    print("   # In Python:")
    print("   from src.audio_processing import AudioProcessor")
    print("   from src.feature_extraction import FeatureExtractor")
    print("   from src.visualization import AudioVisualizer")
    
    print("\n🎯 Recommended Learning Path:")
    print("   1. notebooks/01_audio_basics.ipynb        - Audio fundamentals")
    print("   2. notebooks/02_feature_extraction.ipynb  - Feature engineering")
    print("   3. notebooks/03_model_training.ipynb      - ML model training")
    print("   4. notebooks/04_real_time_demo.ipynb      - Real-time processing")
    
    print("\n🔗 Resources:")
    print("   - README.md - Complete project documentation")
    print("   - src/ - Source code modules")
    print("   - data/ - Audio datasets")
    print("   - models/ - Trained model storage")
    print("   - results/ - Outputs and visualizations")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Audio ML Learning Project Setup')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install required dependencies')
    parser.add_argument('--create-data', action='store_true',
                       help='Create sample datasets')
    parser.add_argument('--test', action='store_true',
                       help='Test installation')
    parser.add_argument('--full-setup', action='store_true',
                       help='Run complete setup (install, create data, test)')
    
    args = parser.parse_args()
    
    print("🎵 Audio ML Learning Project Setup")
    print("=" * 40)
    
    # Check Python version first
    if not check_python_version():
        return False
    
    success = True
    
    # Install dependencies if requested or in full setup
    if args.install_deps or args.full_setup:
        if not install_dependencies():
            success = False
    else:
        # Just check dependencies
        if not check_dependencies():
            print("\n💡 To install dependencies, run:")
            print("   python setup.py --install-deps")
            success = False
    
    # Create sample data if requested or in full setup
    if (args.create_data or args.full_setup) and success:
        if not create_sample_data():
            success = False
    
    # Test installation if requested or in full setup
    if (args.test or args.full_setup) and success:
        if not test_installation():
            success = False
    
    if success:
        print("\n🎉 Setup completed successfully!")
        show_usage_examples()
    else:
        print("\n⚠️ Setup completed with some issues.")
        print("Please check the error messages above and try again.")
    
    return success


if __name__ == "__main__":
    main()

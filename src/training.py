"""
Training Pipeline for Audio Classification Models

This script provides a complete training pipeline with data loading,
preprocessing, model training, and evaluation capabilities.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import argparse
from typing import Tuple, Dict, List, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from audio_processing import AudioProcessor
    from feature_extraction import FeatureExtractor
    from models import AudioClassifier, AudioTrainer
    from visualization import AudioVisualizer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


class AudioMLPipeline:
    """
    Complete machine learning pipeline for audio classification.
    
    Handles data loading, feature extraction, model training, and evaluation.
    """
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 segment_duration: float = 3.0,
                 feature_types: List[str] = None):
        """
        Initialize the audio ML pipeline.
        
        Args:
            sample_rate: Audio sample rate
            segment_duration: Length of audio segments in seconds
            feature_types: Types of features to extract
        """
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.feature_types = feature_types or ['mfcc', 'spectral', 'chroma']
        
        # Initialize components
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate)
        self.visualizer = AudioVisualizer()
        
        # Data storage
        self.features = []
        self.labels = []
        self.class_names = []
        
    def load_audio_dataset(self, data_dir: Path, 
                          class_subdirs: bool = True) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load audio dataset from directory structure.
        
        Args:
            data_dir: Path to data directory
            class_subdirs: Whether classes are in subdirectories
            
        Returns:
            Tuple of (audio_files, labels)
        """
        
        print(f"📁 Loading audio dataset from: {data_dir}")
        
        audio_files = []
        labels = []
        
        if class_subdirs:
            # Each subdirectory is a class
            for class_dir in data_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    if class_name not in self.class_names:
                        self.class_names.append(class_name)
                    
                    class_index = self.class_names.index(class_name)
                    
                    # Load all audio files in this class
                    for audio_file in class_dir.glob('*.wav'):
                        try:
                            audio_data, sr = self.audio_processor.load_audio(audio_file)
                            audio_files.append(audio_data)
                            labels.append(class_index)
                        except Exception as e:
                            print(f"⚠️ Error loading {audio_file}: {e}")
        else:
            # All files in single directory, labels from filename or metadata
            for audio_file in data_dir.glob('*.wav'):
                try:
                    audio_data, sr = self.audio_processor.load_audio(audio_file)
                    audio_files.append(audio_data)
                    # Extract label from filename (assuming format: label_filename.wav)
                    label = audio_file.stem.split('_')[0]
                    if label not in self.class_names:
                        self.class_names.append(label)
                    labels.append(self.class_names.index(label))
                except Exception as e:
                    print(f"⚠️ Error loading {audio_file}: {e}")
        
        print(f"✅ Loaded {len(audio_files)} audio files")
        print(f"   Classes: {self.class_names}")
        print(f"   Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        return audio_files, labels
    
    def preprocess_audio(self, audio_files: List[np.ndarray]) -> List[np.ndarray]:
        """
        Preprocess audio files (normalization, segmentation).
        
        Args:
            audio_files: List of audio arrays
            
        Returns:
            List of preprocessed audio segments
        """
        
        print(f"🔧 Preprocessing audio files...")
        
        processed_segments = []
        processed_labels = []
        
        for i, audio_data in enumerate(audio_files):
            # Normalize audio
            normalized = self.audio_processor.normalize_audio(audio_data, method='peak')
            
            # Segment audio
            segments = self.audio_processor.segment_audio(
                normalized, 
                segment_length=self.segment_duration,
                hop_length=self.segment_duration,  # No overlap for now
                pad_final=True
            )
            
            # Add segments and replicate labels
            processed_segments.extend(segments)
            processed_labels.extend([self.labels[i]] * len(segments))
        
        # Update labels
        self.labels = processed_labels
        
        print(f"✅ Created {len(processed_segments)} audio segments")
        
        return processed_segments
    
    def extract_features(self, audio_segments: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from audio segments.
        
        Args:
            audio_segments: List of audio segments
            
        Returns:
            Feature matrix
        """
        
        print(f"🔍 Extracting features from {len(audio_segments)} segments...")
        
        feature_vectors = []
        
        for i, segment in enumerate(audio_segments):
            if i % 100 == 0:
                print(f"   Processing segment {i+1}/{len(audio_segments)}")
            
            # Extract specified features
            features = {}
            
            if 'mfcc' in self.feature_types:
                mfcc_features = self.feature_extractor.extract_mfcc(segment)
                features.update(mfcc_features)
            
            if 'spectral' in self.feature_types:
                spectral_features = self.feature_extractor.extract_spectral_features(segment)
                features.update(spectral_features)
            
            if 'chroma' in self.feature_types:
                chroma_features = self.feature_extractor.extract_chroma_features(segment)
                features.update(chroma_features)
            
            if 'time_domain' in self.feature_types:
                time_features = self.feature_extractor.extract_time_domain_features(segment)
                features.update(time_features)
            
            # Create feature vector
            feature_vector = self.feature_extractor.create_feature_vector(features, aggregate=True)
            feature_vectors.append(feature_vector)
        
        feature_matrix = np.array(feature_vectors)
        
        print(f"✅ Feature extraction completed!")
        print(f"   Feature matrix shape: {feature_matrix.shape}")
        
        return feature_matrix
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15) -> Tuple[np.ndarray, ...]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature matrix
            y: Labels
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        
        from sklearn.model_selection import train_test_split
        
        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_ratio + test_ratio), 
            random_state=42, stratify=y
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_size),
            random_state=42, stratify=y_temp
        )
        
        print(f"📊 Data split:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Validation: {X_val.shape[0]} samples")
        print(f"   Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   architecture: str = 'cnn_1d',
                   epochs: int = 50,
                   batch_size: int = 32) -> AudioTrainer:
        """
        Train audio classification model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            architecture: Model architecture
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Trained AudioTrainer instance
        """
        
        print(f"🚀 Training {architecture} model...")
        
        # Prepare input shape based on architecture
        if architecture == 'cnn_2d':
            # Reshape for 2D CNN (assuming we can reshape 1D features to 2D)
            side_length = int(np.sqrt(X_train.shape[1]))
            if side_length * side_length != X_train.shape[1]:
                # Pad to make square
                pad_size = (side_length + 1) ** 2 - X_train.shape[1]
                X_train = np.pad(X_train, ((0, 0), (0, pad_size)), mode='constant')
                X_val = np.pad(X_val, ((0, 0), (0, pad_size)), mode='constant')
                side_length += 1
            
            X_train = X_train.reshape(-1, side_length, side_length, 1)
            X_val = X_val.reshape(-1, side_length, side_length, 1)
            input_shape = (side_length, side_length, 1)
        
        elif architecture == 'rnn':
            # Reshape for RNN (split features into time steps)
            n_timesteps = 10
            n_features = X_train.shape[1] // n_timesteps
            if n_features * n_timesteps != X_train.shape[1]:
                # Truncate to fit
                X_train = X_train[:, :n_features * n_timesteps]
                X_val = X_val[:, :n_features * n_timesteps]
            
            X_train = X_train.reshape(-1, n_timesteps, n_features)
            X_val = X_val.reshape(-1, n_timesteps, n_features)
            input_shape = (n_timesteps, n_features)
        
        else:
            # 1D architectures
            input_shape = (X_train.shape[1],)
        
        # Create and compile model
        model = AudioClassifier(
            input_shape=input_shape,
            num_classes=len(self.class_names),
            architecture=architecture,
            filters=[32, 64, 128],
            dense_units=[128, 64],
            dropout_rate=0.3
        )
        
        model.compile_model(learning_rate=0.001)
        
        # Create trainer and train
        trainer = AudioTrainer(model)
        
        callbacks_config = {
            'save_best': True,
            'early_stopping': True,
            'reduce_lr': True,
            'model_checkpoint_path': f'models/best_{architecture}_model.h5'
        }
        
        history = trainer.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            callbacks_config=callbacks_config
        )
        
        return trainer
    
    def run_complete_pipeline(self, data_dir: str,
                             architecture: str = 'cnn_1d',
                             epochs: int = 50,
                             visualize: bool = True) -> Dict:
        """
        Run the complete ML pipeline.
        
        Args:
            data_dir: Path to data directory
            architecture: Model architecture to use
            epochs: Number of training epochs
            visualize: Whether to create visualizations
            
        Returns:
            Dictionary with results
        """
        
        print("🎵 Starting Complete Audio ML Pipeline")
        print("=" * 50)
        
        results = {}
        
        # Step 1: Load data
        audio_files, self.labels = self.load_audio_dataset(Path(data_dir))
        
        if len(audio_files) == 0:
            print("❌ No audio files found! Please check the data directory.")
            return results
        
        # Step 2: Preprocess audio
        audio_segments = self.preprocess_audio(audio_files)
        
        # Step 3: Extract features
        X = self.extract_features(audio_segments)
        y = np.array(self.labels)
        
        # Step 4: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Step 5: Train model
        trainer = self.train_model(X_train, y_train, X_val, y_val, 
                                  architecture=architecture, epochs=epochs)
        
        # Step 6: Evaluate model
        test_results = trainer.evaluate(X_test, y_test)
        results.update(test_results)
        
        # Step 7: Visualizations
        if visualize:
            print("🎨 Creating visualizations...")
            
            # Training history
            if trainer.history:
                self.visualizer.plot_training_history(
                    trainer.history.history,
                    title=f"{architecture.upper()} Training History",
                    save_path=f"results/{architecture}_training_history.png"
                )
            
            # Confusion matrix
            y_pred, _ = trainer.predict(X_test)
            self.visualizer.plot_confusion_matrix(
                y_test, y_pred,
                class_names=self.class_names,
                title=f"{architecture.upper()} Confusion Matrix",
                save_path=f"results/{architecture}_confusion_matrix.png"
            )
        
        # Step 8: Save results
        results['class_names'] = self.class_names
        results['feature_types'] = self.feature_types
        results['architecture'] = architecture
        
        # Save pipeline state
        pipeline_state = {
            'class_names': self.class_names,
            'feature_types': self.feature_types,
            'sample_rate': self.sample_rate,
            'segment_duration': self.segment_duration
        }
        
        with open(f'models/{architecture}_pipeline_state.pkl', 'wb') as f:
            pickle.dump(pipeline_state, f)
        
        print("✅ Pipeline completed successfully!")
        print(f"📊 Final Results: {results}")
        
        return results


def create_synthetic_dataset():
    """Create a synthetic audio dataset for demonstration."""
    
    print("🎼 Creating synthetic audio dataset...")
    
    data_dir = Path("data/sample/synthetic")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create different "classes" of synthetic audio
    classes = ['sine_wave', 'square_wave', 'noise', 'chirp']
    samples_per_class = 20
    duration = 3.0
    sample_rate = 22050
    
    for class_name in classes:
        class_dir = data_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        for i in range(samples_per_class):
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            if class_name == 'sine_wave':
                # Random frequency sine waves
                freq = np.random.uniform(200, 800)
                audio = np.sin(2 * np.pi * freq * t)
                
            elif class_name == 'square_wave':
                # Random frequency square waves
                freq = np.random.uniform(200, 800)
                audio = np.sign(np.sin(2 * np.pi * freq * t))
                
            elif class_name == 'noise':
                # White noise with different amplitudes
                audio = np.random.normal(0, 0.3, len(t))
                
            elif class_name == 'chirp':
                # Frequency sweeps
                f0, f1 = np.random.uniform(200, 400), np.random.uniform(600, 1000)
                audio = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duration) * t)
            
            # Add some noise for realism
            audio += 0.05 * np.random.normal(0, 1, len(audio))
            
            # Normalize
            audio = audio / np.max(np.abs(audio))
            
            # Save
            import soundfile as sf
            output_path = class_dir / f"{class_name}_{i:03d}.wav"
            sf.write(output_path, audio, sample_rate)
    
    print(f"✅ Created synthetic dataset with {len(classes)} classes")
    print(f"   Location: {data_dir}")
    print(f"   Total files: {len(classes) * samples_per_class}")
    
    return str(data_dir)


def main():
    """Main function for running the training pipeline."""
    
    parser = argparse.ArgumentParser(description='Audio ML Training Pipeline')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to audio data directory')
    parser.add_argument('--architecture', type=str, default='cnn_1d',
                       choices=['cnn_1d', 'cnn_2d', 'rnn', 'hybrid'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--create_synthetic', action='store_true',
                       help='Create synthetic dataset for demo')
    
    args = parser.parse_args()
    
    # Create synthetic dataset if requested
    if args.create_synthetic or args.data_dir is None:
        args.data_dir = create_synthetic_dataset()
    
    # Initialize pipeline
    pipeline = AudioMLPipeline(
        sample_rate=22050,
        segment_duration=3.0,
        feature_types=['mfcc', 'spectral', 'chroma', 'time_domain']
    )
    
    # Run pipeline
    results = pipeline.run_complete_pipeline(
        data_dir=args.data_dir,
        architecture=args.architecture,
        epochs=args.epochs,
        visualize=True
    )
    
    print(f"\n🎉 Training pipeline completed!")
    print(f"Results saved in: models/ and results/")


if __name__ == "__main__":
    main()

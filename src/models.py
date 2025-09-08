"""
Machine Learning Models Module for Audio Time Series Classification

This module provides various neural network architectures optimized for
audio classification tasks, including CNN, RNN, and hybrid models.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from typing import Tuple, List, Optional, Dict, Any
import warnings

warnings.filterwarnings('ignore')


class AudioClassifier:
    """
    Flexible audio classification model with multiple architecture options.
    
    Supports:
    - 1D CNN for raw audio or time series features
    - 2D CNN for spectrogram-like inputs
    - RNN/LSTM for sequential data
    - Hybrid CNN-RNN architectures
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, ...],
                 num_classes: int,
                 architecture: str = 'cnn_1d',
                 **kwargs):
        """
        Initialize the AudioClassifier.
        
        Args:
            input_shape: Shape of input data (excluding batch dimension)
            num_classes: Number of output classes
            architecture: Model architecture ('cnn_1d', 'cnn_2d', 'rnn', 'hybrid')
            **kwargs: Additional architecture-specific parameters
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.architecture = architecture
        self.model = None
        
        # Architecture parameters
        self.filters = kwargs.get('filters', [32, 64, 128])
        self.kernel_sizes = kwargs.get('kernel_sizes', [3, 3, 3])
        self.pool_sizes = kwargs.get('pool_sizes', [2, 2, 2])
        self.rnn_units = kwargs.get('rnn_units', 128)
        self.dense_units = kwargs.get('dense_units', [128, 64])
        self.dropout_rate = kwargs.get('dropout_rate', 0.3)
        self.l2_reg = kwargs.get('l2_reg', 0.001)
        
        self._build_model()
    
    def _build_model(self) -> None:
        """Build the neural network model based on specified architecture."""
        
        print(f"🏗️ Building {self.architecture} model...")
        
        if self.architecture == 'cnn_1d':
            self.model = self._build_cnn_1d()
        elif self.architecture == 'cnn_2d':
            self.model = self._build_cnn_2d()
        elif self.architecture == 'rnn':
            self.model = self._build_rnn()
        elif self.architecture == 'hybrid':
            self.model = self._build_hybrid()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        print(f"✅ Model built successfully!")
        print(f"   Total parameters: {self.model.count_params():,}")
    
    def _build_cnn_1d(self) -> keras.Model:
        """Build 1D CNN model for time series data."""
        
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            # Convolutional blocks
            *self._create_conv_blocks_1d(),
            
            # Global pooling and dense layers
            layers.GlobalAveragePooling1D(),
            layers.Dropout(self.dropout_rate),
            
            *self._create_dense_layers(),
            
            # Output layer
            layers.Dense(self.num_classes, 
                        activation='softmax' if self.num_classes > 2 else 'sigmoid',
                        kernel_regularizer=keras.regularizers.l2(self.l2_reg))
        ])
        
        return model
    
    def _build_cnn_2d(self) -> keras.Model:
        """Build 2D CNN model for spectrogram data."""
        
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            # Convolutional blocks
            *self._create_conv_blocks_2d(),
            
            # Global pooling and dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dropout(self.dropout_rate),
            
            *self._create_dense_layers(),
            
            # Output layer
            layers.Dense(self.num_classes,
                        activation='softmax' if self.num_classes > 2 else 'sigmoid',
                        kernel_regularizer=keras.regularizers.l2(self.l2_reg))
        ])
        
        return model
    
    def _build_rnn(self) -> keras.Model:
        """Build RNN/LSTM model for sequential data."""
        
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            # RNN layers
            layers.LSTM(self.rnn_units, return_sequences=True, dropout=self.dropout_rate),
            layers.LSTM(self.rnn_units // 2, dropout=self.dropout_rate),
            
            # Dense layers
            *self._create_dense_layers(),
            
            # Output layer
            layers.Dense(self.num_classes,
                        activation='softmax' if self.num_classes > 2 else 'sigmoid',
                        kernel_regularizer=keras.regularizers.l2(self.l2_reg))
        ])
        
        return model
    
    def _build_hybrid(self) -> keras.Model:
        """Build hybrid CNN-RNN model."""
        
        inputs = layers.Input(shape=self.input_shape)
        
        # CNN feature extraction
        x = inputs
        for i, (filters, kernel_size, pool_size) in enumerate(
            zip(self.filters, self.kernel_sizes, self.pool_sizes)):
            
            x = layers.Conv1D(filters, kernel_size, padding='same',
                             kernel_regularizer=keras.regularizers.l2(self.l2_reg))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling1D(pool_size)(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # RNN processing
        x = layers.LSTM(self.rnn_units, return_sequences=True, dropout=self.dropout_rate)(x)
        x = layers.LSTM(self.rnn_units // 2, dropout=self.dropout_rate)(x)
        
        # Dense layers
        for units in self.dense_units:
            x = layers.Dense(units, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(self.l2_reg))(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes,
                              activation='softmax' if self.num_classes > 2 else 'sigmoid',
                              kernel_regularizer=keras.regularizers.l2(self.l2_reg))(x)
        
        return keras.Model(inputs, outputs)
    
    def _create_conv_blocks_1d(self) -> List[layers.Layer]:
        """Create 1D convolutional blocks."""
        conv_layers = []
        
        for i, (filters, kernel_size, pool_size) in enumerate(
            zip(self.filters, self.kernel_sizes, self.pool_sizes)):
            
            conv_layers.extend([
                layers.Conv1D(filters, kernel_size, padding='same',
                             kernel_regularizer=keras.regularizers.l2(self.l2_reg)),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.MaxPooling1D(pool_size),
                layers.Dropout(self.dropout_rate)
            ])
        
        return conv_layers
    
    def _create_conv_blocks_2d(self) -> List[layers.Layer]:
        """Create 2D convolutional blocks."""
        conv_layers = []
        
        for i, (filters, kernel_size, pool_size) in enumerate(
            zip(self.filters, self.kernel_sizes, self.pool_sizes)):
            
            conv_layers.extend([
                layers.Conv2D(filters, (kernel_size, kernel_size), padding='same',
                             kernel_regularizer=keras.regularizers.l2(self.l2_reg)),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.MaxPooling2D((pool_size, pool_size)),
                layers.Dropout(self.dropout_rate)
            ])
        
        return conv_layers
    
    def _create_dense_layers(self) -> List[layers.Layer]:
        """Create fully connected layers."""
        dense_layers = []
        
        for units in self.dense_units:
            dense_layers.extend([
                layers.Dense(units, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(self.l2_reg)),
                layers.Dropout(self.dropout_rate)
            ])
        
        return dense_layers
    
    def compile_model(self, 
                     learning_rate: float = 0.001,
                     optimizer: str = 'adam',
                     loss: Optional[str] = None,
                     metrics: List[str] = ['accuracy']) -> None:
        """
        Compile the model with specified parameters.
        
        Args:
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'rmsprop', 'sgd')
            loss: Loss function (auto-selected if None)
            metrics: List of metrics to track
        """
        
        # Auto-select loss function if not specified
        if loss is None:
            if self.num_classes == 2:
                loss = 'binary_crossentropy'
            else:
                loss = 'sparse_categorical_crossentropy'
        
        # Select optimizer
        if optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        print(f"✅ Model compiled with {optimizer} optimizer")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Loss function: {loss}")
        print(f"   Metrics: {metrics}")
    
    def get_model_summary(self) -> None:
        """Print model architecture summary."""
        return self.model.summary()
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        self.model.save(filepath)
        print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a pre-trained model."""
        self.model = keras.models.load_model(filepath)
        print(f"📁 Model loaded from: {filepath}")


class AudioTrainer:
    """
    Training pipeline for audio classification models.
    
    Handles training with callbacks, data augmentation, and monitoring.
    """
    
    def __init__(self, model: AudioClassifier):
        """
        Initialize the AudioTrainer.
        
        Args:
            model: AudioClassifier instance to train
        """
        self.model = model
        self.history = None
        
    def create_callbacks(self, 
                        save_best: bool = True,
                        early_stopping: bool = True,
                        reduce_lr: bool = True,
                        model_checkpoint_path: str = 'best_model.h5') -> List[callbacks.Callback]:
        """
        Create training callbacks.
        
        Args:
            save_best: Whether to save best model
            early_stopping: Whether to use early stopping
            reduce_lr: Whether to reduce learning rate on plateau
            model_checkpoint_path: Path to save best model
            
        Returns:
            List of callbacks
        """
        callback_list = []
        
        if save_best:
            checkpoint = callbacks.ModelCheckpoint(
                model_checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            callback_list.append(checkpoint)
        
        if early_stopping:
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
            callback_list.append(early_stop)
        
        if reduce_lr:
            lr_scheduler = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
            callback_list.append(lr_scheduler)
        
        return callback_list
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 32,
              validation_split: float = 0.2,
              callbacks_config: Optional[Dict[str, Any]] = None,
              verbose: int = 1) -> keras.callbacks.History:
        """
        Train the audio classification model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of training data to use for validation
            callbacks_config: Configuration for callbacks
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        
        print(f"🚀 Starting model training...")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Input shape: {X_train.shape[1:]}")
        print(f"   Number of classes: {len(np.unique(y_train))}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = None
            print(f"   Validation samples: {X_val.shape[0]}")
        elif validation_split > 0:
            print(f"   Validation split: {validation_split}")
        
        # Setup callbacks
        if callbacks_config is None:
            callbacks_config = {}
        
        callback_list = self.create_callbacks(**callbacks_config)
        
        # Train the model
        self.history = self.model.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )
        
        print(f"✅ Training completed!")
        
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        
        print(f"📊 Evaluating model on test set...")
        print(f"   Test samples: {X_test.shape[0]}")
        
        # Get predictions
        y_pred_proba = self.model.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1) if y_pred_proba.shape[1] > 1 else (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        print(f"📈 Evaluation Results:")
        for metric, value in results.items():
            print(f"   {metric.capitalize()}: {value:.4f}")
        
        return results
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predicted_classes, prediction_probabilities)
        """
        
        y_pred_proba = self.model.model.predict(X)
        
        if y_pred_proba.shape[1] > 1:
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
        return y_pred, y_pred_proba


def create_sample_model_demo():
    """Demonstrate model creation and compilation."""
    
    print("🤖 Audio Classification Model Demo")
    print("=" * 40)
    
    # Demo parameters
    input_shapes = {
        'cnn_1d': (1000,),          # 1D time series
        'cnn_2d': (128, 128, 1),    # Spectrogram
        'rnn': (100, 13),           # MFCC sequence
        'hybrid': (500,)            # 1D for hybrid
    }
    
    num_classes = 5
    
    for arch_name, input_shape in input_shapes.items():
        print(f"\n🏗️ Creating {arch_name} model...")
        
        try:
            model = AudioClassifier(
                input_shape=input_shape,
                num_classes=num_classes,
                architecture=arch_name,
                filters=[16, 32, 64],  # Smaller for demo
                dense_units=[64, 32],
                dropout_rate=0.3
            )
            
            model.compile_model(learning_rate=0.001)
            
            print(f"   ✅ {arch_name} model created successfully!")
            print(f"   📊 Parameters: {model.model.count_params():,}")
            
        except Exception as e:
            print(f"   ❌ Error creating {arch_name} model: {str(e)}")
    
    print(f"\n✅ Model demo completed!")


if __name__ == "__main__":
    create_sample_model_demo()

import numpy as np
import logging
import yaml
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SWaTAutoencoder:
    """
    Autoencoder model for Anomaly Detection.
    """
    def __init__(self, input_dim: int, config_path: str = "configs/config.yaml"):
        with open(config_path, "r", encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
            
        self.input_dim = input_dim
        self.model_path = "models/autoencoder.h5"
        self.model = self._build_model()
        self.threshold = None # هيتحدد بعد التدريب
        
        Path("models").mkdir(parents=True, exist_ok=True)

    def _build_model(self) -> Model:
        """Builds the Autoencoder architecture."""
        logger.info(f"Building Autoencoder with input dimension: {self.input_dim}")
        
        # Input Layer
        input_layer = Input(shape=(self.input_dim,))
        
        # Encoder Layer
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        
        # Bottleneck Layer (الضغط بيحصل هنا)
        bottleneck = Dense(16, activation='relu')(encoded)
        
        # Decoder Layer
        decoded = Dense(32, activation='relu')(bottleneck)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(64, activation='relu')(decoded)
        
        # Output Layer (بيحاول يطلع نفس الداتا اللي دخلت)
        output_layer = Dense(self.input_dim, activation='linear')(decoded)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse')
        
        return model

    def train(self, X_train: np.ndarray, epochs: int = 50, batch_size: int = 256, validation_split: float = 0.1):
        """Trains the Autoencoder on normal data."""
        logger.info("Starting model training...")
        
        # هنوقف التدريب لو الموديل بطل يتحسن عشان نوفر وقت
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        # هنسيف أحسن موديل بس
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True)
        
        history = self.model.fit(
            X_train, X_train, # في الـ Autoencoder الـ input هو هو الـ target
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        logger.info(f"Model saved to {self.model_path}")
        self._calculate_threshold(X_train)
        return history

    def _calculate_threshold(self, X_train: np.ndarray):
        """
        Calculates the anomaly threshold based on the reconstruction error 
        of the normal training data.
        """
        logger.info("Calculating anomaly threshold...")
        reconstructions = self.model.predict(X_train)
        # بنحسب الـ Mean Squared Error لكل صف
        mse = np.mean(np.power(X_train - reconstructions, 2), axis=1)
        
        # Threshold = المتوسط + (3 * الانحراف المعياري)
        self.threshold = np.mean(mse) + 3 * np.std(mse)
        
        # هنسيف الـ Threshold عشان نستخدمه في الـ Inference
        threshold_path = "models/threshold.yaml"
        with open(threshold_path, 'w', encoding='utf-8') as f:
            yaml.dump({'anomaly_threshold': float(self.threshold)}, f)
            
        logger.info(f"Calculated Threshold: {self.threshold}")
        logger.info(f"Threshold saved to {threshold_path}")

    def load(self):
        """Loads a pre-trained model and its threshold."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError("Model file not found!")
            
        self.model = load_model(self.model_path)
        
        threshold_path = "models/threshold.yaml"
        if Path(threshold_path).exists():
            with open(threshold_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                self.threshold = data.get('anomaly_threshold')
        logger.info("Model and threshold loaded successfully.")

    def predict_anomalies(self, X_test: np.ndarray) -> np.ndarray:
        """Predicts if data points are anomalies based on the threshold."""
        if self.threshold is None:
            raise ValueError("Threshold is not set. Train or load the model first.")
            
        reconstructions = self.model.predict(X_test)
        mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
        
        # لو الـ MSE أكبر من الـ Threshold تبقى Anomaly (1), لو أقل تبقى Normal (0)
        anomalies = (mse > self.threshold).astype(int)
        return anomalies
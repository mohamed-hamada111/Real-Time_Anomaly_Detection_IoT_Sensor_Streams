import numpy as np
import logging
import yaml
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, RepeatVector, TimeDistributed, 
    Dense, Dropout, MultiHeadAttention, LayerNormalization, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SWaTAutoencoder:
    """
    Hybrid LSTM-Transformer (Attention) Autoencoder for Anomaly Detection.
    """
    def __init__(self, input_dim: int, time_steps: int = 5, config_path: str = "configs/config.yaml"):
        with open(config_path, "r", encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
            
        self.input_dim = input_dim
        self.time_steps = time_steps
        self.model_path = "models/autoencoder.keras"
        self.model = self._build_model()
        self.threshold = None 
        
        Path("models").mkdir(parents=True, exist_ok=True)

    def _build_model(self) -> Model:
        """Builds the Hybrid LSTM-Attention Autoencoder architecture."""
        logger.info(f"Building LSTM-Attention Model | Input features: {self.input_dim} | Time Steps: {self.time_steps}")
        
        # Input Layer
        input_layer = Input(shape=(self.time_steps, self.input_dim))
        
        # --- Encoder ---
        x = LSTM(64, activation='relu', return_sequences=True)(input_layer)
        
        
        attention_out = MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
        # Residual Connection & Normalization
        x = LayerNormalization()(Add()([x, attention_out]))
        x = Dropout(0.2)(x)
        
        # Bottleneck (Compression)
        encoded = LSTM(16, activation='relu', return_sequences=False)(x)
        
        # --- Bridge ---
        repeated = RepeatVector(self.time_steps)(encoded)
        
        # --- Decoder ---
        x = LSTM(16, activation='relu', return_sequences=True)(repeated)
        
        
        attention_out_dec = MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
        x = LayerNormalization()(Add()([x, attention_out_dec]))
        
        decoded = LSTM(64, activation='relu', return_sequences=True)(x)
        decoded = Dropout(0.2)(decoded)
        
        # Output Layer
        output_layer = TimeDistributed(Dense(self.input_dim, activation='linear'))(decoded)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse')
        
        return model

    def train(self, X_train: np.ndarray, epochs: int = 50, batch_size: int = 256, validation_split: float = 0.1):
        logger.info("Starting LSTM-Attention model training...")
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True)
        
        history = self.model.fit(
            X_train, X_train, 
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        logger.info(f"Model saved to {self.model_path}")
        self._calculate_threshold(X_train)
        return history

    def _calculate_threshold(self, X_train):
        import numpy as np
        
        # التوقع باستخدام الباتشات عشان ميسحبش رامات
        reconstructions = self.model.predict(X_train, batch_size=256)
        
        # حساب نسبة الخطأ (MSE) على أجزاء (Chunks) بدل مصفوفة واحدة ضخمة
        chunk_size = 10000
        mse_list = []
        
        for i in range(0, len(X_train), chunk_size):
            # تقليل حجم الداتا لـ float32 بيوفر نص الرامات بالظبط
            chunk_X = X_train[i:i+chunk_size].astype(np.float32)
            chunk_recon = reconstructions[i:i+chunk_size].astype(np.float32)
            
            # حساب الخطأ للجزء ده بس
            chunk_mse = np.mean(np.square(chunk_X - chunk_recon), axis=(1, 2))
            mse_list.append(chunk_mse)
            
        # تجميع الأجزاء كلها
        mse = np.concatenate(mse_list)
        
        # تحديد الـ Threshold (أعلى 99% من الأخطاء مثلاً)
        self.threshold = np.percentile(mse, 99)
        print(f"Calculated Anomaly Threshold: {self.threshold}")

    def load(self, model_path="models/autoencoder.keras"):
        import os
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        
        
        self.threshold = 0.00509513433245199
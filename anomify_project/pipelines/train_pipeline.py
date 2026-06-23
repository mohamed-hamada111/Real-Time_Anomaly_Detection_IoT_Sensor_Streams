import sys
import os
import logging
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import SWaTDataLoader
from src.preprocess import SWaTPreProcessor
from src.features import SWaTFeatureEngineer
from src.model import SWaTAutoencoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sequences(X, time_steps=5):
    Xs = []
    
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
    return np.array(Xs)

def run_full_pipeline():
    try:
        # 1. Loading Data
        logger.info("--- STEP 1: Data Loading ---")
        loader = SWaTDataLoader()
        df_normal = loader.load_csv_robust(loader.config['data']['normal_data_path'])
        
        # 2. Preprocessing
        logger.info("--- STEP 2: Data Preprocessing ---")
        preprocessor = SWaTPreProcessor()
        df_clean = preprocessor.run_pipeline(df_normal)
        
        # 3. Feature Engineering
        logger.info("--- STEP 3: Feature Engineering ---")
        engineer = SWaTFeatureEngineer()
        df_final = engineer.run_pipeline(df_clean, is_train=True)
        
        # Remove label column if it exists since Autoencoder is unsupervised
        if 'label' in df_final.columns:
            df_final = df_final.drop(columns=['label'])
            
        X_train = df_final.values
        
       
        logger.info("--- STEP 3.5: Sequence Generation for LSTM ---")
        TIME_STEPS = 5  
        logger.info(f"Converting 2D data to 3D sequences with TIME_STEPS={TIME_STEPS}...")
        X_train_seq = create_sequences(X_train, TIME_STEPS)
        logger.info(f"New data shape for LSTM: {X_train_seq.shape}") 
        # 4. Model Training
        logger.info("--- STEP 4: Model Training ---")
        n_features = X_train_seq.shape[2]
        
        
        autoencoder = SWaTAutoencoder(input_dim=n_features, time_steps=TIME_STEPS)
        
        logger.info(f"Training on sequence data shape: {X_train_seq.shape}")
        autoencoder.train(X_train_seq, epochs=20, batch_size=256)
        
        logger.info("=========================================")
        logger.info("✅ Pipeline Completed Successfully! ✅")
        logger.info("=========================================")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    run_full_pipeline()
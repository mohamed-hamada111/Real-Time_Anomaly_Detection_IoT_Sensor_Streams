import sys
import os
import logging
from pathlib import Path

# تظبيط مسار المشروع
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import SWaTDataLoader
from src.preprocess import SWaTPreProcessor
from src.features import SWaTFeatureEngineer
from src.model import SWaTAutoencoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
        # نشيل عمود الـ label لو موجود عشان الـ Autoencoder بيتدرب Unsupervised
        if 'label' in df_final.columns:
            df_final = df_final.drop(columns=['label'])
            
        # نحول الداتا لـ Numpy Array عشان Keras
        X_train = df_final.values
        
        # 4. Model Training
        logger.info("--- STEP 4: Model Training ---")
        input_dim = X_train.shape[1]
        autoencoder = SWaTAutoencoder(input_dim=input_dim)
        
        # هندرب على عينة بس لو جهازك ضعيف، بس في الـ Production هندرب على كله
        logger.info(f"Training on data shape: {X_train.shape}")
        autoencoder.train(X_train, epochs=20, batch_size=256)
        
        logger.info("=========================================")
        logger.info("✅ Pipeline Completed Successfully! ✅")
        logger.info("=========================================")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    run_full_pipeline()
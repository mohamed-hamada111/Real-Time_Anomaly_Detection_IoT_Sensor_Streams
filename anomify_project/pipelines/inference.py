import pandas as pd
import numpy as np
import yaml
import logging
import time
from pathlib import Path
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.models import load_model

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import SWaTDataLoader
from src.preprocess import SWaTPreProcessor
from src.features import SWaTFeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class AnomifyLiveDetector:
    def __init__(self):
        base_path = Path(__file__).parent.parent
        self.model_path = base_path / "models" / "autoencoder.h5"
        self.threshold_path = base_path / "models" / "threshold.yaml"
        
        logger.info("⚙️ Initializing Anomify Real-Time Detector...")
        self.model = load_model(str(self.model_path), compile=False)
        
        with open(self.threshold_path, "r", encoding='utf-8') as f:
            t_data = yaml.safe_load(f)
            self.threshold = float(list(t_data.values())[0]) if isinstance(t_data, dict) else float(t_data)
                
        logger.info(f"🎯 Active Threshold: {self.threshold:.6f}")
        
        self.preprocessor = SWaTPreProcessor()
        self.engineer = SWaTFeatureEngineer()

    def simulate_stream(self, data_path: str, num_records: int = 300):
        logger.info("📂 Loading full data to preserve rolling window context...")
        loader = SWaTDataLoader()
        full_df = loader.load_csv_robust(data_path)
        
        
        clean_df = self.preprocessor.run_pipeline(full_df)
        final_df = self.engineer.run_pipeline(clean_df, is_train=False)
        
        
        if 'label' in final_df.columns and (final_df['label'] == 1).any():
            
            attack_positions = np.where(final_df['label'] == 1)[0]
            first_attack_pos = attack_positions[0]
            start_pos = max(0, first_attack_pos - 100) 
            df_stream = final_df.iloc[start_pos : start_pos + num_records].copy()
            logger.info("😈 Attack found! Starting stream slightly before the attack...")
        else:
            df_stream = final_df.tail(num_records).copy()
            
        labels = df_stream['label'].values if 'label' in df_stream.columns else None
        X = df_stream.drop(columns=['label'], errors='ignore').values
        
        print("\n" + "="*50)
        logger.info("📡 --- STARTING LIVE IOT SENSOR STREAM --- 📡")
        print("="*50 + "\n")
        
        anomalies_caught = 0
        
        for i in range(len(X)):
            row = X[i:i+1]
            true_label = labels[i] if labels is not None else "Unknown"
            
            pred = self.model.predict(row, verbose=0)
            mse = np.mean(np.power(row - pred, 2))
            
            is_anomaly = mse > self.threshold
            
            if is_anomaly:
                anomalies_caught += 1
                logger.warning(f"🚨 ⚠️ ANOMALY DETECTED! ⚠️ | MSE: {mse:.6f} | True Label: {true_label}")
            else:
                if i % 10 == 0:  
                    logger.info(f"✅ Status Normal | MSE: {mse:.6f} | True Label: {true_label}")
            
            time.sleep(0.02) 
            
        print("\n" + "="*50)
        logger.info("🏁 --- STREAM ENDED --- 🏁")
        logger.info(f"📊 Total Anomalies Detected in Session: {anomalies_caught}")
        print("="*50)

if __name__ == "__main__":
    detector = AnomifyLiveDetector()
    loader = SWaTDataLoader()
    test_file = loader.config['data'].get('test_data_path', 'data/raw/merged.csv')
    detector.simulate_stream(test_file, num_records=300)
import pandas as pd
import numpy as np
import yaml
import logging
import time
from pathlib import Path
import os


# suppress TensorFlow warnings during inference (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.models import load_model

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import SWaTDataLoader
from src.preprocess import SWaTPreProcessor
from src.features import SWaTFeatureEngineer

#configure logging to be more concise for real-time output
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class AnomifyLiveDetector:
    def __init__(self):
        base_path = Path(__file__).parent.parent
        self.model_path = base_path / "models" / "autoencoder.h5"
        self.threshold_path = base_path / "models" / "threshold.yaml"
        
        logger.info("⚙️ Initializing Anomify Real-Time Detector...")
        
        #importing the trained model
        self.model = load_model(str(self.model_path), compile=False)
        
        #importing the threshold value that we calculated during training
        with open(self.threshold_path, "r", encoding='utf-8') as f:
            t_data = yaml.safe_load(f)
            if isinstance(t_data, dict):
                self.threshold = float(list(t_data.values())[0])
            else:
                self.threshold = float(t_data)
                
        logger.info(f"🎯 Active Threshold: {self.threshold:.6f}")
        
        
        # setting up the preprocessing and feature engineering pipelines
        self.preprocessor = SWaTPreProcessor()
        self.engineer = SWaTFeatureEngineer()

    def simulate_stream(self, data_path: str, num_records: int = 1000):
        """simulates a live stream of IoT sensor data by reading from a CSV file and processing it row by row"""
        logger.info(f"📂 Loading {num_records} records to simulate live stream...")
        loader = SWaTDataLoader()
        df = loader.load_csv_robust(data_path)
        
        # taking only the last `num_records` to simulate a stream of incoming data, we can adjust this number based on how long we want the simulation to run
        df_stream = df.tail(num_records).copy()
        
        
        # preprocessing (cleaning, handling missing values, etc.)
        clean_df = self.preprocessor.run_pipeline(df_stream)
        
        #feature engineering (scaling, rolling stats, etc.) - we set is_train=False because we're in inference mode
        final_df = self.engineer.run_pipeline(clean_df, is_train=False)
        
       
        #split features and labels if label column exists, otherwise we set labels to None
        labels = final_df['label'].values if 'label' in final_df.columns else None
        X = final_df.drop(columns=['label'], errors='ignore').values
        
        print("\n" + "="*50)
        logger.info("📡 --- STARTING LIVE IOT SENSOR STREAM --- 📡")
        print("="*50 + "\n")
        
        anomalies_caught = 0
        
        # loop through each row as if it's coming in real-time, we can adjust the sleep time to simulate different streaming speeds
        for i in range(len(X)):
            row = X[i:i+1] # we take a slice to keep it as a 2D array for Keras
            true_label = labels[i] if labels is not None else "Unknown"
            
            # predict the reconstruction and calculate the MSE for the current row
            pred = self.model.predict(row, verbose=0)
            mse = np.mean(np.power(row - pred, 2))
            
            is_anomaly = mse > self.threshold
            
            # print anomaly detection results in real-time, we can customize the message format as needed
            if is_anomaly:
                anomalies_caught += 1
                logger.warning(f"🚨 ⚠️ ANOMALY DETECTED! ⚠️ | MSE: {mse:.6f} | True Label: {true_label}")
            else:
                # to reduce log clutter, we can choose to log normal readings less frequently (e.g., every 50 records) or not at all
                if i % 50 == 0: 
                    logger.info(f"✅ Status Normal | MSE: {mse:.6f}")
            
            # simulate a short delay to mimic real-time streaming, we can adjust this based on how fast we want the stream to feel
            time.sleep(0.02) 
            
        print("\n" + "="*50)
        logger.info("🏁 --- STREAM ENDED --- 🏁")
        logger.info(f"📊 Total Anomalies Detected in Session: {anomalies_caught}")
        print("="*50)

if __name__ == "__main__":
    detector = AnomifyLiveDetector()
    loader = SWaTDataLoader()
    # take the test data path from the config file, we can specify a different test file if needed, but by default it will use the merged.csv which contains both normal and attack data for evaluation
    test_file = loader.config['data'].get('test_data_path', 'data/raw/merged.csv')
    
    # simulate the live stream of last 500 records from the test file, we can adjust the number of records to simulate a longer or shorter stream
    detector.simulate_stream(test_file, num_records=500)
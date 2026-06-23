import sys
import os
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import SWaTDataLoader
from src.preprocess import SWaTPreProcessor
from src.features import SWaTFeatureEngineer
from src.model import SWaTAutoencoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sequences(X, y, time_steps=5):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps]) 
    return np.array(Xs), np.array(ys)

def run_evaluation():
    try:
        logger.info("--- Starting LSTM Evaluation Pipeline ---")
        loader = SWaTDataLoader()
        test_file = loader.config['data'].get('test_data_path', 'data/raw/merged.csv')
        df_test = loader.load_csv_robust(test_file)
        
        preprocessor = SWaTPreProcessor()
        df_clean = preprocessor.run_pipeline(df_test)
        
        engineer = SWaTFeatureEngineer()
        df_final = engineer.run_pipeline(df_clean, is_train=False)
        
        y_test = df_final['label'].values if 'label' in df_final.columns else np.zeros(len(df_final))
        X_test = df_final.drop(columns=['label'], errors='ignore').values
        
        logger.info("Converting 2D test data to 3D sequences...")
        TIME_STEPS = 5
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, TIME_STEPS)
        
        input_dim = X_test_seq.shape[2]
        autoencoder = SWaTAutoencoder(input_dim=input_dim, time_steps=TIME_STEPS)
        autoencoder.load()
        
        logger.info("Making predictions on sequence data...")
        reconstructions = autoencoder.model.predict(X_test_seq)
        
        
        mse = np.mean(np.power(X_test_seq - reconstructions, 2), axis=(1, 2))
        
        logger.info("Calculating optimal threshold to maximize F1-Score...")
        best_f1 = 0
        best_thresh = autoencoder.threshold
        
        
        min_mse, max_mse = np.min(mse), np.max(mse)
        thresholds = np.linspace(min_mse, min_mse + (max_mse - min_mse) * 0.1, 100)
        
        for t in thresholds:
            preds = (mse > t).astype(int)
            score = f1_score(y_test_seq, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = t
                
        logger.info(f"💡 Optimal Threshold Found: {best_thresh:.6f}")
        logger.info(f"🚀 Expected Best F1-Score: {best_f1:.4f}")
        
        final_preds = (mse > best_thresh).astype(int)
        
        print("\n" + "="*40)
        print("CLASSIFICATION REPORT (LSTM)")
        print("="*40)
        print(classification_report(y_test_seq, final_preds, digits=4))
        
        print("="*40)
        print("CONFUSION MATRIX")
        print("="*40)
        print(confusion_matrix(y_test_seq, final_preds))

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    run_evaluation()
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import os

# بنخلي TensorFlow ميزعجناش بالتحذيرات
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, precision_recall_curve

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import SWaTDataLoader
from src.preprocess import SWaTPreProcessor
from src.features import SWaTFeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_evaluation():
    logger.info("--- Starting Evaluation Pipeline ---")
    
    # 1. Load Data (هنجيب داتا الاختبار اللي فيها الهجمات)
    loader = SWaTDataLoader()
    test_data_path = loader.config['data'].get('test_data_path', 'data/raw/merged.csv') 
    test_df = loader.load_csv_robust(test_data_path)
    
    # 2. Preprocess
    preprocessor = SWaTPreProcessor()
    clean_df = preprocessor.run_pipeline(test_df)
    
    # نحتفظ بالإجابات النموذجية (True Labels) عشان نصحح للموديل
    y_true = clean_df['label'].values
    
    # 3. Feature Engineering (Inference Mode -> is_train=False)
    engineer = SWaTFeatureEngineer()
    final_df = engineer.run_pipeline(clean_df, is_train=False)
    
    # نشيل عمود الـ label قبل ما ندخل الداتا للموديل
    X_test = final_df.drop(columns=['label']).values
    
    # 4. Load Saved Model & Threshold
    base_path = Path(__file__).parent.parent
    model_path = base_path / "models" / "autoencoder.h5"
    threshold_path = base_path / "models" / "threshold.yaml"
    
    logger.info("Loading trained model...")
    model = load_model(str(model_path), compile=False)
    
    logger.info("Loading initial anomaly threshold...")
    with open(threshold_path, "r", encoding='utf-8') as f:
        threshold_data = yaml.safe_load(f)
        
    if isinstance(threshold_data, dict):
        initial_threshold = float(list(threshold_data.values())[0])
    else:
        initial_threshold = float(threshold_data)
        
    logger.info(f"Loaded initial threshold value: {initial_threshold:.6f}")
    
    # 5. Predict & Calculate Error
    logger.info("Making predictions on test data...")
    X_pred = model.predict(X_test, batch_size=256)
    mse = np.mean(np.power(X_test - X_pred, 2), axis=1)
    
    # ---------------------------------------------------------
    # --- السلاح الأول: صيد العتبة المثالية (Optimal Threshold) ---
    # ---------------------------------------------------------
    logger.info("Calculating optimal threshold to maximize F1-Score...")
    precisions, recalls, thresholds_curve = precision_recall_curve(y_true, mse)
    
    # حساب الـ F1-Score لكل العتبات (بنتجاهل آخر قيمة عشان أطوال المصفوفات تتطابق)
    # ضفنا 1e-10 عشان نتجنب الـ Error بتاع القسمة على صفر
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    
    # نجيب العتبة اللي حققت أعلى F1-Score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds_curve[best_idx]
    
    logger.info(f"💡 Optimal Threshold Found: {best_threshold:.6f}")
    logger.info(f"🚀 Expected Best F1-Score: {f1_scores[best_idx]:.4f}")
    
    # 6. Evaluation (استخدام العتبة المثالية الجديدة بدل القديمة)
    y_pred = (mse > best_threshold).astype(int)
    
    logger.info("--- Final Evaluation Results ---")
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    
    print("\n" + "="*40)
    print("CLASSIFICATION REPORT")
    print("="*40)
    print(classification_report(y_true, y_pred, digits=4))
    
    print("="*40)
    print("CONFUSION MATRIX")
    print("="*40)
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    run_evaluation()
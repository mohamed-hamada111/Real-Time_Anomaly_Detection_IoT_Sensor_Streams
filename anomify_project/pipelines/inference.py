import pandas as pd
import numpy as np
import yaml
import logging
import time
from pathlib import Path
import os

# قفل تحذيرات TensorFlow عشان منزحمش الشاشة
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.models import load_model

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import SWaTDataLoader
from src.preprocess import SWaTPreProcessor
from src.features import SWaTFeatureEngineer

# تظبيط شكل اللوجز عشان تبان كأنها شاشة مراقبة
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class AnomifyLiveDetector:
    def __init__(self):
        base_path = Path(__file__).parent.parent
        self.model_path = base_path / "models" / "autoencoder.h5"
        self.threshold_path = base_path / "models" / "threshold.yaml"
        
        logger.info("⚙️ Initializing Anomify Real-Time Detector...")
        
        # تحميل الموديل
        self.model = load_model(str(self.model_path), compile=False)
        
        # تحميل العتبة
        with open(self.threshold_path, "r", encoding='utf-8') as f:
            t_data = yaml.safe_load(f)
            if isinstance(t_data, dict):
                self.threshold = float(list(t_data.values())[0])
            else:
                self.threshold = float(t_data)
                
        logger.info(f"🎯 Active Threshold: {self.threshold:.6f}")
        
        # تجهيز بايبلاين المعالجة
        self.preprocessor = SWaTPreProcessor()
        self.engineer = SWaTFeatureEngineer()

    def simulate_stream(self, data_path: str, num_records: int = 1000):
        """بتحاكي استلام قراءات من السنسورز بشكل لحظي"""
        logger.info(f"📂 Loading {num_records} records to simulate live stream...")
        loader = SWaTDataLoader()
        df = loader.load_csv_robust(data_path)
        
        # هناخد آخر شوية قراءات (عشان غالباً بيبقى فيهم الهجمات في الداتا دي)
        df_stream = df.tail(num_records).copy()
        
        # معالجة مبدئية
        clean_df = self.preprocessor.run_pipeline(df_stream)
        
        # هندسة الميزات (بنديله is_train=False عشان ميعدلش الـ Scaler)
        final_df = self.engineer.run_pipeline(clean_df, is_train=False)
        
        # فصل الإجابات لو موجودة عشان نقارن بيها
        labels = final_df['label'].values if 'label' in final_df.columns else None
        X = final_df.drop(columns=['label'], errors='ignore').values
        
        print("\n" + "="*50)
        logger.info("📡 --- STARTING LIVE IOT SENSOR STREAM --- 📡")
        print("="*50 + "\n")
        
        anomalies_caught = 0
        
        # اللوب دي بتمشي على الداتا صف صف كأنها بتيجي في وقتها
        for i in range(len(X)):
            row = X[i:i+1] # بناخد قراءة واحدة ونحافظ عليها 2D
            true_label = labels[i] if labels is not None else "Unknown"
            
            # الموديل بيتوقع ويعرف الـ Error
            pred = self.model.predict(row, verbose=0)
            mse = np.mean(np.power(row - pred, 2))
            
            is_anomaly = mse > self.threshold
            
            # طباعة النتيجة
            if is_anomaly:
                anomalies_caught += 1
                logger.warning(f"🚨 ⚠️ ANOMALY DETECTED! ⚠️ | MSE: {mse:.6f} | True Label: {true_label}")
            else:
                # بنطبع بس كل 50 قراءة عشان منزحمش الشاشة لو الوضع طبيعي
                if i % 50 == 0: 
                    logger.info(f"✅ Status Normal | MSE: {mse:.6f}")
            
            # بنوقف الكود جزء من الثانية عشان نحس بالستريم
            time.sleep(0.02) 
            
        print("\n" + "="*50)
        logger.info("🏁 --- STREAM ENDED --- 🏁")
        logger.info(f"📊 Total Anomalies Detected in Session: {anomalies_caught}")
        print("="*50)

if __name__ == "__main__":
    detector = AnomifyLiveDetector()
    loader = SWaTDataLoader()
    # بنجيب مسار ملف الاختبار من الـ config
    test_file = loader.config['data'].get('test_data_path', 'data/raw/merged.csv')
    
    # هنشغل المحاكاة على آخر 500 قراءة
    detector.simulate_stream(test_file, num_records=500)
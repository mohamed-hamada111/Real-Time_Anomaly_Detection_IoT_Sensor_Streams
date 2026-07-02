import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # لمنع رسائل التحذير المزعجة

import sqlite3
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# ==========================================
# 1. إعدادات قاعدة البيانات (SQLite)
# ==========================================
DB_FILE = "anomify_alerts.db"

def init_db():
    """إنشاء الداتابيز وجدول الإنذارات لو مش موجودين"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            mse_score REAL,
            threshold REAL,
            is_anomaly BOOLEAN
        )
    ''')
    conn.commit()
    conn.close()

# تشغيل دالة بناء الداتابيز مع بداية تشغيل السيرفر
init_db()

# ==========================================
# 2. تحميل موديل الذكاء الاصطناعي
# ==========================================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "autoencoder.keras"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.yaml"

print("⚙️ Loading Anomify Model & Threshold into Server Memory...")
model = load_model(str(MODEL_PATH), compile=False)

with open(THRESHOLD_PATH, 'r', encoding='utf-8') as f:
    t_data = yaml.safe_load(f)
    THRESHOLD = float(t_data.get('anomaly_threshold', 0.010277))
print(f"✅ Server Ready! Active Threshold: {THRESHOLD:.6f}")

# ==========================================
# 3. إعدادات السيرفر (FastAPI)
# ==========================================
app = FastAPI(
    title="Anomify IoT Control API", 
    description="REST API for Real-Time Anomaly Detection using LSTM-Attention"
)

# هيكل البيانات المتوقع من الحساسات (باقة من 5 قراءات، كل قراءة 101 ميزة)
class SensorWindow(BaseModel):
    readings: list[list[float]]

def log_anomaly_to_db(mse: float, threshold: float):
    """دالة خفيفة بتسجل الإنذار في الداتابيز في الخلفية"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO alerts (timestamp, mse_score, threshold, is_anomaly)
        VALUES (?, ?, ?, ?)
    ''', (current_time, mse, threshold, True))
    
    conn.commit()
    conn.close()

# ==========================================
# 4. المسار الرئيسي (The Endpoint)
# ==========================================
@app.post("/predict")
def predict_anomaly(data: SensorWindow, background_tasks: BackgroundTasks):
    try:
        # 1. تحويل الداتا والتأكد من الأبعاد (5, 101)
        np_data = np.array(data.readings)
        if np_data.shape != (5, 101):
            raise HTTPException(status_code=400, detail=f"Expected shape (5, 101), got {np_data.shape}")
        
        # 2. تجهيز الباقة للموديل (1, 5, 101)
        sequence_3d = np.expand_dims(np_data, axis=0)
        
        # 3. التوقع الحسابي
        pred = model.predict(sequence_3d, verbose=0)
        mse = float(np.mean(np.power(sequence_3d - pred, 2)))
        
        # 4. اتخاذ القرار
        is_anomaly = bool(mse > THRESHOLD)
        
        # 5. لو في خطر، احفظ في الداتابيز فوراً (في الخلفية عشان منبطأش الرد)
        if is_anomaly:
            background_tasks.add_task(log_anomaly_to_db, mse, THRESHOLD)
            
        return {
            "status": "success",
            "mse_score": round(mse, 6),
            "is_anomaly": is_anomaly,
            "message": "🚨 ANOMALY DETECTED & LOGGED!" if is_anomaly else "✅ Status Normal"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
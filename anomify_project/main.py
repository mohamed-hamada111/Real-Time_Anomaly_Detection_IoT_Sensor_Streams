import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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

print("⚙️ Loading H.A.R.E.S Model & Threshold into Server Memory...")
model = load_model(str(MODEL_PATH), compile=False)

with open(THRESHOLD_PATH, 'r', encoding='utf-8') as f:
    t_data = yaml.safe_load(f)
    THRESHOLD = float(t_data.get('anomaly_threshold', 0.010277))
print(f"✅ Server Ready! Active Threshold: {THRESHOLD:.6f}")

# ==========================================
# 3. إعدادات السيرفر (FastAPI)
# ==========================================
app = FastAPI(
    title="H.A.R.E.S IoT Control API", 
    description="REST API for Real-Time Anomaly Detection using LSTM-Attention"
)

class SensorWindow(BaseModel):
    readings: list[list[float]]

# التعديل الجديد: Data Model لاستقبال بيانات الإنذار فقط من Streamlit
class AnomalyLog(BaseModel):
    mse_score: float
    threshold: float

def log_anomaly_to_db(mse: float, threshold: float):
    """function to log detected anomalies into the SQLite database"""
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
        np_data = np.array(data.readings)
        if np_data.shape != (5, 101):
            raise HTTPException(status_code=400, detail=f"Expected shape (5, 101), got {np_data.shape}")
        
        sequence_3d = np.expand_dims(np_data, axis=0)
        
        pred = model.predict(sequence_3d, verbose=0)
        mse = float(np.mean(np.power(sequence_3d - pred, 2)))
        
        is_anomaly = bool(mse > THRESHOLD)
        
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

# ==========================================
# 5. المسار الجديد لتسجيل الإنذارات المكتشفة محلياً (التعديل الجديد)
# ==========================================
@app.post("/log_anomaly")
def log_anomaly_endpoint(data: AnomalyLog, background_tasks: BackgroundTasks):
    """function to log detected anomalies into the SQLite database"""
    background_tasks.add_task(log_anomaly_to_db, data.mse_score, data.threshold)
    return {"status": "success", "message": "Anomaly logged to SQLite successfully"}
# ==========================================
# 6. مسار جديد لقراءة الإنذارات (عشان n8n)
# ==========================================
@app.get("/get_alerts")
def get_recent_alerts():
    """function to retrieve the most recent alerts from the SQLite database"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # عشان نرجع الداتا كـ Dictionary
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 50")
    rows = cursor.fetchall()
    conn.close()
    
    return {"alerts": [dict(row) for row in rows]}
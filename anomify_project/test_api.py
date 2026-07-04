import requests
import numpy as np
import time

# عنوان السيرفر بتاعك
API_URL = "http://127.0.0.1:8000/predict"

def send_reading(is_attack=False):
    # 1. توليد قراءات وهمية (5 خطوات، 101 حساس)
    if is_attack:
        # لو هجوم: هنعلي الأرقام جداً عشان الموديل يلقطها
        dummy_data = np.random.normal(loc=5.0, scale=2.0, size=(5, 101)).tolist()
        print("\n😈 Sending Malicious Attack Data...")
    else:
        # لو طبيعي: أرقام هادية جداً
        dummy_data = np.random.normal(loc=0.0, scale=0.1, size=(5, 101)).tolist()
        print("\n✅ Sending Normal Data...")

    # 2. تجهيز الداتا في شكل JSON زي ما السيرفر مستني
    payload = {
        "readings": dummy_data
    }

    # 3. إرسال الداتا للسيرفر
    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()
        print(f"Server Response: {result}")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Server is not running! Please start uvicorn first.")

# ==========================================
# تجربة السيستم
# ==========================================
if __name__ == "__main__":
    # 1. هنبعت قراءة طبيعية
    send_reading(is_attack=False)
    
    time.sleep(2) # نستنى ثانيتين
    
    # 2. هنبعت قراءة شاذة (هجوم)
    send_reading(is_attack=True)
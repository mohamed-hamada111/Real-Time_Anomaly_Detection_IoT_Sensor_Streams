import requests
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import SWaTDataLoader
from src.preprocess import SWaTPreProcessor
from src.features import SWaTFeatureEngineer

API_URL = "http://127.0.0.1:8000/predict"

print("⏳ Loading a small chunk of real normal data...")
# هنسحب أول 500 صف بس عشان السكريبت يخلص في ثانية
loader = SWaTDataLoader()
df = loader.load_csv_robust(loader.config['data']['normal_data_path']).head(500)

print("⚙️ Processing data (Cleaning & Feature Engineering)...")
df_clean = SWaTPreProcessor().run_pipeline(df)
df_final = SWaTFeatureEngineer().run_pipeline(df_clean, is_train=False)

# هناخد آخر 5 قراءات (اللي هما 100% طبيعيين)
X = df_final.drop(columns=['label'], errors='ignore').values
real_sequence = X[-5:].tolist()

payload = {
    "readings": real_sequence
}

print("🚀 Sending REAL Normal Data to API...")
try:
    response = requests.post(API_URL, json=payload)
    print("\n" + "="*50)
    print(f"🎯 Server Response: {response.json()}")
    print("="*50 + "\n")
except requests.exceptions.ConnectionError:
    print("❌ Error: Server is not running! Please start uvicorn first.")
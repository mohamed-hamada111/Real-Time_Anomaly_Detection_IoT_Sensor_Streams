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
        self.model_path = base_path / "models" / "autoencoder.keras"
        self.threshold_path = base_path / "models" / "threshold.yaml"
        print(f"the model path: {self.model_path}")

        logger.info("⚙️ Initializing H.A.R.E.S Real-Time Detector...")
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

        # The model is an LSTM autoencoder trained on (time_steps, n_features)
        # sequences, not single rows. Pull time_steps straight from the
        # loaded model so this always matches whatever it was trained with.
        time_steps = self.model.input_shape[1]

        if 'label' in final_df.columns and (final_df['label'] == 1).any():
            attack_positions = np.where(final_df['label'] == 1)[0]
            first_attack_pos = attack_positions[0]
            start_pos = max(0, first_attack_pos - 100)
            logger.info("😈 Attack found! Starting stream slightly before the attack...")
        else:
            start_pos = max(0, len(final_df) - num_records)

        # Grab extra rows *before* start_pos so the very first streamed
        # record still has a full window of real history behind it, instead
        # of being padded with copies of itself.
        lookback_start = max(0, start_pos - (time_steps - 1))
        df_window = final_df.iloc[lookback_start: start_pos + num_records].copy()

        labels = df_window['label'].values if 'label' in df_window.columns else None
        X = df_window.drop(columns=['label'], errors='ignore').values

        # Index within X where the actual reported stream begins.
        offset = start_pos - lookback_start
        stream_len = min(num_records, len(X) - offset)
        
        print("\n" + "="*50)
        logger.info("📡 --- STARTING LIVE IOT SENSOR STREAM --- 📡")
        print("="*50 + "\n")
        
        anomalies_caught = 0
        
        time_steps = 5
        
        for i in range(time_steps, len(X)):
            # السطرين دول هما اللي بيعرفوا الـ sequence_3d وبيظبطوا الأبعاد
            sequence = X[i - time_steps : i]
            sequence_3d = np.expand_dims(sequence, axis=0) 
            
            true_label = labels[i] if labels is not None else "Unknown"
            
            # دلوقتي الموديل هيلاقيه ويشتغل طبيعي
            pred = self.model.predict(sequence_3d, verbose=0)
            mse = np.mean(np.power(sequence_3d - pred, 2))
            
            is_anomaly = mse > self.threshold
            
            if is_anomaly:
                anomalies_caught += 1
                logger.warning(f"🚨 ⚠️ ANOMALY DETECTED! ⚠️ | MSE: {mse:.6f} | True Label: {true_label}")
            else:
                if i % 10 == 0:  
                    logger.info(f"✅ Status Normal | MSE: {mse:.6f} | True Label: {true_label}")
            
            time.sleep(0.02)

    def analyze_stream(self, data_path: str, num_records: int = 300, progress_callback=None) -> pd.DataFrame:
        """
        Same sliding-window logic as simulate_stream(), but returns a
        DataFrame of {step, mse, threshold, is_anomaly, true_label, feat_0..N}
        instead of printing to the console. Built for the Streamlit dashboard.
        feat_* columns hold per-feature mean squared error for the last timestep
        of each window — used by the dashboard to identify which sensors drove
        each anomaly.
        """
        loader = SWaTDataLoader()
        full_df = loader.load_csv_robust(data_path)

        clean_df = self.preprocessor.run_pipeline(full_df)
        final_df = self.engineer.run_pipeline(clean_df, is_train=False)

        time_steps = self.model.input_shape[1]

        if 'label' in final_df.columns and (final_df['label'] == 1).any():
            attack_positions = np.where(final_df['label'] == 1)[0]
            start_pos = max(0, attack_positions[0] - 100)
        else:
            start_pos = max(0, len(final_df) - num_records)

        lookback_start = max(0, start_pos - (time_steps - 1))
        df_window = final_df.iloc[lookback_start: start_pos + num_records].copy()

        labels = df_window['label'].values if 'label' in df_window.columns else None
        X = df_window.drop(columns=['label'], errors='ignore').values
        n_features = X.shape[1]

        offset = start_pos - lookback_start
        stream_len = min(num_records, len(X) - offset)

        rows = []
        for j in range(stream_len):
            end_idx = offset + j + 1
            seq = X[max(0, end_idx - time_steps): end_idx]
            if len(seq) < time_steps:
                pad = np.repeat(seq[:1], time_steps - len(seq), axis=0)
                seq = np.vstack([pad, seq])

            row = seq.reshape(1, time_steps, n_features)
            pred = self.model.predict(row, verbose=0)
            diff_sq = np.power(row - pred, 2)          # (1, time_steps, n_features)
            mse = float(np.mean(diff_sq))
            # Per-feature error: mean over the batch+time dims -> (n_features,)
            per_feat = diff_sq.mean(axis=(0, 1))

            entry = {
                "step": j,
                "mse": mse,
                "threshold": self.threshold,
                "is_anomaly": mse > self.threshold,
                "true_label": int(labels[offset + j]) if labels is not None else None,
            }
            for fi in range(n_features):
                entry[f"feat_{fi}"] = float(per_feat[fi])

            rows.append(entry)
            if progress_callback:
                progress_callback((j + 1) / stream_len)

        return pd.DataFrame(rows)

    def score_batch(self, raw_df) -> dict:
        """
        Stateless scoring entry point for a single window of RAW sensor rows.

        raw_df must contain >= rolling_window_size rows ending at the moment
        you want scored (Stream Analytics is what supplies that window in
        the Azure deployment - see windowing.asaql). This mirrors training:
        rolling mean/std need real history, not just the last 5 rows, or
        they won't match what the model was trained on.

        Returns a small JSON-serializable dict - this is exactly the
        payload the Azure Function forwards to the alerts/scores Event Hub.
        """
        clean_df = self.preprocessor.run_pipeline(raw_df)
        final_df = self.engineer.run_pipeline(clean_df, is_train=False)

        time_steps = self.model.input_shape[1]
        X = final_df.drop(columns=['label'], errors='ignore').values

        if len(X) < time_steps:
            raise ValueError(
                f"Need at least {time_steps} rows after feature engineering, got {len(X)}"
            )

        seq = X[-time_steps:].reshape(1, time_steps, X.shape[1])
        pred = self.model.predict(seq, verbose=0)
        mse = float(np.mean(np.power(seq - pred, 2)))
        is_anomaly = mse > self.threshold

        return {
            "mse": mse,
            "threshold": self.threshold,
            "is_anomaly": bool(is_anomaly),
            "rows_used": int(len(raw_df)),
        }

if __name__ == "__main__":
    detector = AnomifyLiveDetector()
    loader = SWaTDataLoader()
    test_file = loader.config['data'].get('test_data_path', 'data/raw/merged.csv')
    detector.simulate_stream(test_file, num_records=300)
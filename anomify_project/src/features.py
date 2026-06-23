import pandas as pd
import numpy as np
import yaml
import logging
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SWaTFeatureEngineer:
    def __init__(self, config_path: str = None):
        if config_path is None:
            base_path = Path(__file__).parent.parent
            self.config_path = base_path / "configs" / "config.yaml"
        else:
            self.config_path = Path(config_path)
            
        with open(self.config_path, "r", encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
            
        self.window_size = self.config.get('pipeline', {}).get('rolling_window_size', 300)
        self.scaler_path = "models/scaler.pkl" 
        self.scaler = MinMaxScaler()
        
        Path("models").mkdir(parents=True, exist_ok=True)

    def identify_feature_types(self, df: pd.DataFrame):
        feature_cols = [c for c in df.columns if c != 'label']
        binary_cols = [c for c in feature_cols if df[c].nunique() <= 3]
        continuous_cols = [c for c in feature_cols if c not in binary_cols]
        logger.info(f"Identified {len(continuous_cols)} continuous sensors and {len(binary_cols)} binary actuators.")
        return continuous_cols, binary_cols

    def scale_features(self, df: pd.DataFrame, continuous_cols: list, is_train: bool = True) -> pd.DataFrame:
        df_scaled = df.copy()
        
        if is_train:
            logger.info("Fitting and transforming scaler on training data...")
            df_scaled[continuous_cols] = self.scaler.fit_transform(df[continuous_cols])
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"Scaler saved to {self.scaler_path}")
        else:
            logger.info("Loading scaler for inference data...")
            if not Path(self.scaler_path).exists():
                raise FileNotFoundError("Scaler file not found! You must run training first.")
            
            loaded_scaler = joblib.load(self.scaler_path)
            df_scaled[continuous_cols] = loaded_scaler.transform(df[continuous_cols])
            
        return df_scaled

    def add_rolling_features(self, df: pd.DataFrame, continuous_cols: list) -> pd.DataFrame:
        logger.info(f"Adding rolling features (window={self.window_size})...")
        df_rolling = df.copy()
        
        for col in continuous_cols:
            df_rolling[f'{col}_roll_mean'] = df[col].rolling(window=self.window_size, min_periods=1).mean()
            df_rolling[f'{col}_roll_std']  = df[col].rolling(window=self.window_size, min_periods=1).std().fillna(0)
            
        return df_rolling

    def run_pipeline(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        logger.info("--- Starting Feature Engineering Pipeline ---")
        
        if is_train:
            #if we're training, we analyze the data to identify feature types and save the scaler based on the continuous features
            continuous_cols, binary_cols = self.identify_feature_types(df)
        else:
            #magic trick: during inference, we load the scaler and extract the feature names it was fitted on to ensure we use the same continuous features as during training
            loaded_scaler = joblib.load(self.scaler_path)
            continuous_cols = list(loaded_scaler.feature_names_in_)
            logger.info(f"Extracted {len(continuous_cols)} continuous features directly from saved scaler.")
        
        df_processed = self.scale_features(df, continuous_cols, is_train=is_train)
        df_processed = self.add_rolling_features(df_processed, continuous_cols)
        
        #fill any remaining NaN values with 0 (especially important for the rolling std which can have NaNs at the beginning)
        df_processed = df_processed.bfill().fillna(0)
        
        logger.info("--- Feature Engineering Completed ---")
        return df_processed
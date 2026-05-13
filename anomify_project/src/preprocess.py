import pandas as pd
import logging
import yaml
from pathlib import Path # ضفنا دي عشان مسار الملف الديناميكي

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SWaTPreProcessor:
    def __init__(self, config_path: str = None):
        # تظبيط المسار الديناميكي عشان يشتغل من أي مكان
        if config_path is None:
            base_path = Path(__file__).parent.parent
            self.config_path = base_path / "configs" / "config.yaml"
        else:
            self.config_path = Path(config_path)

        # ضفنا encoding='utf-8' لحل مشكلة الإيرور الأخير
        with open(self.config_path, "r", encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
            
        self.label_col = self.config['columns']['label_col']
        self.timestamp_col = self.config['columns']['timestamp_col']

    def parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Parsing timestamps from column: {self.timestamp_col}")
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], dayfirst=True, errors='coerce')
        df = df.sort_values(self.timestamp_col).reset_index(drop=True)
        df = df.set_index(self.timestamp_col)
        return df

    def handle_missing_and_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.isnull().sum().sum() > 0:
            df = df.ffill().bfill()
        if df.duplicated().sum() > 0:
            df = df.drop_duplicates()
        return df

    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.label_col in df.columns:
            df['label'] = df[self.label_col].apply(
                lambda x: 0 if str(x).strip().lower() == 'normal' else 1
            )
            df = df.drop(columns=[self.label_col])
        return df

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        df_clean = self.parse_timestamps(df_clean)
        df_clean = self.handle_missing_and_duplicates(df_clean)
        df_clean = self.encode_labels(df_clean)
        return df_clean
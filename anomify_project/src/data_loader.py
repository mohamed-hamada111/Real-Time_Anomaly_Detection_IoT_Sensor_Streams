import pandas as pd
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SWaTDataLoader:
    def __init__(self, config_path: str = None):
        if config_path is None:
            # take the config path dynamically based on where the script is run from
            base_path = Path(__file__).parent.parent
            self.config_path = base_path / "configs" / "config.yaml"
        else:
            self.config_path = Path(config_path)
            
        # this line is added to fix the last error related to encoding when loading the config file

        self.config = self._load_config()

    def _load_config(self) -> dict:
        try:
            
            with open(self.config_path, "r", encoding='utf-8') as file:
                config = yaml.safe_load(file)
            logger.info("Configuration loaded successfully.")
            return config
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise

    def load_csv_robust(self, file_path: str) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        for sep in [',', ';']:
            try:
               
                #added encoding='utf-8' here as well just in case the data has encoding issues
                df = pd.read_csv(file_path, sep=sep, low_memory=False, encoding='utf-8')
                if df.shape[1] > 2:
                    df.columns = df.columns.str.strip()
                    logger.info(f"Loaded {path.name} successfully with separator '{sep}'. Shape: {df.shape}")
                    return df
            except Exception:
                continue
        
        logger.error(f"Failed to parse {file_path} with known separators.")
        raise ValueError(f"Cannot parse {file_path}")
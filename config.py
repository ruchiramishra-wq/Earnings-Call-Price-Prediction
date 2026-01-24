# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# load .env if present
load_dotenv()

# project root
PROJECT_ROOT = Path(__file__).resolve().parent

# data paths (from env, fallback to relative paths)
MOTLEY_FOOL_DATA_PATH = os.getenv(
    "MOTLEY_FOOL_DATA_PATH",
    PROJECT_ROOT / "data" / "motley-fool-data.pkl"
)

STOCK_DATA_KAGGLE_PATH = os.getenv(
    "STOCK_DATA_KAGGLE_PATH",
    PROJECT_ROOT / "data" / "sp500_stock_data"
)

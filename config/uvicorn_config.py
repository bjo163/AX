import os
from dotenv import load_dotenv

# Load variabel dari .env jika ada
load_dotenv()

# Konfigurasi Supabase
APP_DB_URL = os.getenv("APP_DB_URL", "https://your-default.supabase.co")
APP_DB_KEY = os.getenv("APP_DB_KEY", "your-default-api-key")

# Konfigurasi Uvicorn
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))
APP_RELOAD = os.getenv("APP_RELOAD", "true").lower() == "true"
APP_WORKERS = int(os.getenv("APP_WORKERS", 4))
APP_LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "info")
APP_USE_COLORS = os.getenv("APP_USE_COLORS", "true").lower() == "true"


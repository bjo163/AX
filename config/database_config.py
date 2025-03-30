import os
from dotenv import load_dotenv

# Load variabel dari .env jika ada
load_dotenv()

# Konfigurasi Supabase
APP_DB_URL = os.getenv("APP_DB_URL", "https://your-default.supabase.co")
APP_DB_KEY = os.getenv("APP_DB_KEY", "your-default-api-key")
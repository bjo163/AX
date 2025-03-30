# config/supabase_handler.py
import logging
from datetime import datetime
from supabase import create_client
from  config.database_config import APP_DB_KEY, APP_DB_URL

# Initialize Supabase client
supabase = create_client(APP_DB_URL, APP_DB_KEY)

class SupabaseHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        supabase.table('logs').insert([
            {
                "services": record.name,
                "level": record.levelname,
                "message": record.getMessage()
            }
        ]).execute()
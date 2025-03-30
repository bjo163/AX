# config/logging_config.py
import os
import sys
import logging
import logging.config
import colorlog

# Ensure the logs directory exists
LOGS_DIR = "logs"
LOG_FILE = os.path.join(LOGS_DIR, "uvicorn.log")
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# Set stdout encoding to UTF-8 (important for Windows)
sys.stdout.reconfigure(encoding="utf-8")

# Custom log formatter
class CustomFormatter(colorlog.ColoredFormatter):
    def format(self, record):
        return super().format(record)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        },
        "detailed": {
            "format": "[%(asctime)s] %(levelname)s [%(name)s] %(filename)s:%(lineno)d - %(message)s",
        },
        "futuristic": {
            "()": CustomFormatter,
            "format": "%(log_color)s🌀 [%(asctime)s] %(levelname)-8s | 🚀 %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "log_colors": {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        },
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "futuristic",
            "stream": sys.stdout,
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "filename": LOG_FILE,
            "formatter": "detailed",
            "encoding": "utf-8",
        },
        "supabase": {
            "level": "INFO",
            "class": "config.supabase_handler.SupabaseHandler",
            "formatter": "default",
        },
    },
    "loggers": {
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console", "file", "supabase"],
            "propagate": False,
        },
    },
}

# Apply logging config
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("uvicorn")
logger.info("Futuristic logging is now active!")

# Test logging
logger.info("Setiap log akan ditulis ke Supabase dengan ID otomatis dan services dari nama logger!")
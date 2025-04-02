import uvicorn
import logging
from config.uvicorn_config import APP_HOST, APP_PORT, APP_RELOAD, APP_WORKERS, APP_LOG_LEVEL
from config.logging_config import LOGGING_CONFIG
from config.logging_config import logger
if __name__ == "__main__":
    # Apply logging configuration
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger("uvicorn")
    logger.info("Aplikasi dimulai!")
    logger.debug("Ini adalah pesan debug.")
    logger.warning("Ini adalah pesan warning.")
    logger.error("Ini adalah pesan error.")
    logger.critical("Ini adalah pesan critical.")

    uvicorn.run(
        "app.main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=APP_RELOAD,
        log_level=APP_LOG_LEVEL,
        workers=APP_WORKERS,
    )


# Test logging

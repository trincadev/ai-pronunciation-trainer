"""Get machine learning predictions from geodata raster images"""
import os
from pathlib import Path

import structlog
from dotenv import load_dotenv

from aip_trainer.utils import session_logger


load_dotenv()
PROJECT_ROOT_FOLDER = Path(globals().get("__file__", "./_")).absolute().parent.parent
LOG_JSON_FORMAT = bool(os.getenv("LOG_JSON_FORMAT", False))
log_level = os.getenv("LOG_LEVEL", "INFO")
session_logger.setup_logging(json_logs=LOG_JSON_FORMAT, log_level=log_level)
app_logger = structlog.stdlib.get_logger(__name__)

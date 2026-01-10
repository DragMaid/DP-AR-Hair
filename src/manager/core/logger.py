import logging
import sys
import json
from logging.handlers import RotatingFileHandler


class JsonFormatter(logging.Formatter):
    def format(self, record: dict):
        log_record = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


def setup_logging(
    level=logging.INFO,
    log_file: str = "dqs_app.log",
    max_bytes: int = 10*1024*1024,  # 10 MB
    backup_count: int = 5,
    json_format: bool = False,
) -> None:
    """
    Configure centralized logging
    - json_format: outputs logs in JSON
    - Rotating file logs for persistence
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove default handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
    root_logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count)
    if json_format:
        file_handler.setFormatter(JsonFormatter())
    else:
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
    root_logger.addHandler(file_handler)


def get_logger(name: str):
    return logging.getLogger(name)

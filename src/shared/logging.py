# -------------------------
# Logging configuration
# -------------------------
import logging
from logging.config import dictConfig

class ColorFormatter(logging.Formatter):
    """
    Custom formatter to add color to log messages based on the log level.
    """
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.FORMATS = {
            logging.DEBUG: self.blue + self._fmt + self.reset,
            logging.INFO: self.grey + self._fmt + self.reset,
            logging.WARNING: self.yellow + self._fmt + self.reset,
            logging.ERROR: self.red + self._fmt + self.reset,
            logging.CRITICAL: self.bold_red + self._fmt + self.reset
        }

    def format(self, record):
        return logging.Formatter(self.FORMATS.get(record.levelno)).format(record)

class ProbeFilter(logging.Filter):
    HIDDEN_PATHS = ("/healthz", "/ready", "/metrics")
    def filter(self, record: logging.LogRecord) -> bool:
        return not any(path in record.getMessage() for path in self.HIDDEN_PATHS)
    
def configure_logging():
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "root": {
            "handlers": ["console"], 
            "level": "DEBUG"
        },
        "filters": {
            "exclude_probes": {"()": ProbeFilter}
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "color",
                "level": "INFO",
            },
            "console_access": {
                "class": "logging.StreamHandler",
                "formatter": "color",
                "level": "INFO",
                "filters": ["exclude_probes"],
            },
        },
        "formatters": {
            "std_out": {
                "format": "%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "color": {
                "()": ColorFormatter,
                "format": "%(asctime)s : %(levelname)s : %(module)s : %(funcName)s : %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "loggers": {
            "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["console"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["console_access"], "level": "INFO", "propagate": False},
            "fastapi": {"handlers": ["console"], "level": "DEBUG", "propagate": False},
            "httpx":         {"handlers": ["console_access"], "level": "INFO", "propagate": False},
            "httpx._client": {"handlers": ["console_access"], "level": "INFO", "propagate": False},
            "httpcore":      {"handlers": ["console_access"], "level": "INFO", "propagate": False}
        }
    }
    dictConfig(logging_config)
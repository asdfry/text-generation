import os
import sys

from loguru import logger


def update_config(dirpath):
    filepath = f"{dirpath}/torch.log"
    if os.path.exists(filepath):
        os.remove(filepath)
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "<green>{time:YYYY-MM-DD at HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}<white>:</white>{line}</cyan> | {message}",
                "colorize": "true",
            },
            {
                "sink": filepath,
                "format": "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {name}:{line} | {message}",
            },
        ]
    }
    logger.configure(**config)

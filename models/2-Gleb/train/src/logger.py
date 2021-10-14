import sys

from loguru import logger
from config import cfg

def log_init(output_folder):
    handlers = []
    if cfg.PARALLEL.IS_MASTER:
        handlers = [
            {"sink": sys.stdout, "format": "{time:[MM-DD HH:mm:ss]} - {message}"},
            #{"sink": f"{output_folder}/logs.txt", "format": "{time:[MM-DD HH:mm:ss]} - {message}"},
        ]
    logger.configure(**{"handlers":handlers})

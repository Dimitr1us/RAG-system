import logging
import sys
from datetime import datetime
from pythonjsonlogger import jsonlogger
import json


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def jsonify_log_record(self, log_record):
        return json.dumps(log_record, ensure_ascii=False)


def setup_logger(name="rag_experiment", log_file="logs/experiments.log", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = CustomJsonFormatter(
        fmt='%(asctime)s %(levelname)s %(name)s %(message)s %(task)s %(function_name)s '
            '%(accuracy_without_rag)s %(accuracy_with_rag)s %(context_items)s %(elapsed_time)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
            "name": "logger"
        }
    )

    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()
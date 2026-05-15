# -*- coding: utf-8 -*-
import logging
import os
import sys
from datetime import datetime

_MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOGS_DIR = os.path.normpath(os.path.join(_MODULE_DIR, '..', 'logs'))


def setup_logging():
    # Create log directory if not exist
    logs_dir = _LOGS_DIR
    os.makedirs(logs_dir, exist_ok=True)

    script_name = os.path.splitext(os.path.basename(sys.modules['__main__'].__file__))[0]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(logs_dir, f"{script_name}_{timestamp}.log")

    # Configure log format
    log_format = "%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create file handler
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Create TTY handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Configure main logger
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])

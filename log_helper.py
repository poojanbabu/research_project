import logging.handlers
import sys


def get_logger(name):
    # Initialize logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter('%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s')
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(log_formatter)
    f_handler = logging.handlers.RotatingFileHandler(filename='../logs/out.log', maxBytes=(1048576 * 5), backupCount=100)
    f_handler.setFormatter(log_formatter)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

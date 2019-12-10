import logging
import os
import time


def createDirs(dir_path):
    """
    Create a new directory if it doesn't exist.
    Inputs:
        - dir_path: (str) path to the directory to create
    Returns:
        (None)
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file = open(dir_path / ".gitkeep", "w+")
    file.close()


def createLogger(log_dir, logger_name):
    createDirs(log_dir)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(log_dir / (time.strftime("%Y%m%d-%H%M%S")+'.log'))
    logger.addHandler(fh)

    return logger


"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

import os

from loguru import logger


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        mode(str): log file write mode, `append` or `override`. default is `a`.
    Return:
        logger instance.
    """
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    if distributed_rank > 0:
        logger.remove()
    logger.add(
        save_file, format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}", filter="", level="INFO", enqueue=True
    )

    return logger

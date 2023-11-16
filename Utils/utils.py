import argparse
import random
import numpy as np
import torch
import os
import logging
import datetime
import sys

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_local_time():
    """
    获取时间

    Return:
        datetime: 时间
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur

def get_logger(config, name=None):
    """
    获取Logger对象

    Args:
        config(ConfigParser): config
        name: specified name

    Returns:
        Logger: logger
    """
    log_dir = './Logger/log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = '{}-{}-{}-{}.log'.format(config.exp_id,
                                            config.model_name, config.data_name, get_local_time())
    logfilepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)

    log_level = 'INFO'

    if log_level.lower() == 'info':
        level = logging.INFO
    elif log_level.lower() == 'debug':
        level = logging.DEBUG
    elif log_level.lower() == 'error':
        level = logging.ERROR
    elif log_level.lower() == 'warning':
        level = logging.WARNING
    elif log_level.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logfilepath)
    file_handler.setFormatter(formatter)

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('Log directory: %s', log_dir)
    return logger

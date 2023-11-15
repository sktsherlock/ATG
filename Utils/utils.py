import argparse
import random
import numpy as np
import torch
import os
import logging
import datetime
import sys
import time

def parse_args():
    parser = argparse.ArgumentParser()
    # Deafault arguments
    parser.add_argument('--path', default='Data/', help='Dataset loading path')
    parser.add_argument('--data_name', default='cifar10', type=str, help='Dataset name')
    parser.add_argument('--model_name', default='resnet50d', type=str, help='Model name')
    parser.add_argument('--task', default='node', type=str, help='Task name')
    parser.add_argument('--attribute', default='image', type=str, help='The attribute user choose to use')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU index, -1 for CPU')
    parser.add_argument('--seed',
                        help='Seed used for splitting sets. None by default, set to an integer for reproducibility',
                        type=int, default=42)
    parser.add_argument('--runs', type=int, default=1, help='Number of runs to run')
    parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # Training process
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--val_interval', type=int, default=10,
                        help='Evaluate on validation set every this many epochs. Must be >= 1.')
    parser.add_argument('--lr', type=float, default=2e-05,
                        help='learning rate (default holds for batch size 64)')
    parser.add_argument('--lr_step', type=str, default='1000000',
                        help='Comma separated string of epochs when to reduce learning rate by a factor of 10.'
                             ' The default is a large value, meaning that the learning rate will not change.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    # Model
    parser.add_argument('--model', choices={"transformer", "LINEAR"}, default="transformer",
                        help="Model class")
    parser.add_argument('--d_model', type=int, default=256,
                        help='Dimension of the first linear layer')
    return parser.parse_args()

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

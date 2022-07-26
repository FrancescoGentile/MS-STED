##
##
##

from functools import reduce
from typing import Optional
import logging
import sys
import os
from munch import DefaultMunch
import yaml

class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

def load_config_file(path: str) -> dict:
    if not os.path.isfile(path):
        raise ValueError(f'Config file {path} does not exist.')

    try:
        with open(path, 'rb') as f: 
            options = yaml.safe_load(f)
    except:
        raise(f'An error occurred while trying to read config file.')
        
    options = DefaultMunch.fromDict(options, default=None)
        
    return options

def init_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

def init_logger(name: str, level: int = logging.INFO, file: Optional[str] = None) -> logging.Logger:
    """
    Initialize a logger.
    Args:
        name (str): name of the logger
        level (int): loggin level. Defaults to INFO.
        file (Optional[str], optional): file where to write log messages. If no file is specified, 
        log messages are only written to stderr.
        Defaults to None.

    Returns:
        logging.Logger: the initialized logger.
    """
    logger = logging.getLogger(name)
    for handler in logger.handlers:
        logger.removeHandler(handler)

    formatter = logging.Formatter(fmt='[ %(asctime)s ] %(levelname)s --> %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)

    if file is not None:
        file_handler = logging.FileHandler(
            filename=file, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger

def check_and_create_dir(name: str):
    if not os.path.exists(name):
        os.makedirs(name)
    elif not os.path.isdir(name):
        raise ValueError(f'The path {name} is not a directory.')
    
def check_class_exists(module: str, class_name: str) -> bool:
    try:
        cls = reduce(getattr, class_name.split("."), sys.modules[module])
    except AttributeError:
        cls = None

    return cls is not None

def get_class_by_name(module: str, class_name: str):
    cls = getattr(sys.modules[module], class_name)
    return cls
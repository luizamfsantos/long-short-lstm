import yaml
import logging
import os

def get_logger(logger_name='model_training'):
    # Ensure the logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Create a custom logger
    logger = logging.getLogger(logger_name)
    
    # Set the default logging level
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('logs/model_training.log')
    
    # Create formatters and add them to handlers
    c_format = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s [[%(filename)s:%(lineno)d]]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    f_format = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s [[%(filename)s:%(lineno)d]]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

def get_config():
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)
    return config['LSTM']
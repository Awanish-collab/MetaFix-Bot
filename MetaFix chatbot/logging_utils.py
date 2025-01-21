import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logger(name, log_folder='logs'):
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent / log_folder
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Create handlers
    # File handler with daily rotation
    current_date = datetime.now().strftime('%Y-%m-%d')
    file_handler = logging.FileHandler(
        log_dir / f'{current_date}.log',
        encoding='utf-8'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set formatter for handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
"""
Logging utilities for API Layer
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: str = "INFO",
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    logs_path: str = "logs",
    file_enabled: bool = True,
    console_enabled: bool = True,
    max_file_size: int = 10485760,
    backup_count: int = 5,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    logger.handlers.clear()
    
    formatter = logging.Formatter(format_str)
    
    if console_enabled:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if file_enabled:
        if log_file is None:
            log_file = Path(logs_path) / f"{name}.log"
        
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)

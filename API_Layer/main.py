"""
Main API Server - Entry point for Paarvai Vision AI API
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
sys.path.insert(0, str(parent_dir))

from api_endpoints import app
from utils.config_loader import APIConfigLoader
from utils.logger import setup_logger

# Initialize configuration and logger
config_loader = APIConfigLoader()
logger = setup_logger(__name__)

if __name__ == "__main__":
    import uvicorn
    
    api_config = config_loader.get_api_config()
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    debug = api_config.get('debug', False)
    
    logger.info(f"Starting Paarvai Vision AI API on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=debug)

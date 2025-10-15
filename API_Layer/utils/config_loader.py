"""
Configuration loader for API Layer
"""

import json
import os
from pathlib import Path
from typing import Dict, Any


class APIConfigLoader:
    """Load and manage configuration for API Layer"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.json"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {self.config_path}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self.get('api', {})
    
    def get_kafka_config(self) -> Dict[str, Any]:
        """Get Kafka configuration"""
        return self.get('kafka', {})
    
    def get_data_stream_config(self) -> Dict[str, Any]:
        """Get data stream configuration"""
        return self.get('data_stream', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.get('monitoring', {})
    
    def get_endpoints_config(self) -> Dict[str, Any]:
        """Get endpoints configuration"""
        return self.get('endpoints', {})
    
    def get_ai_layer_config(self) -> Dict[str, Any]:
        """Get AI layer configuration"""
        return self.get('ai_layer', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get('logging', {})

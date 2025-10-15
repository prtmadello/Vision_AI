"""
Configuration loader for AI Layer
"""

import json
import os
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Load and manage configuration for AI Layer"""
    
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
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection configuration"""
        return self.get('detection', {})
    
    def get_face_processing_config(self) -> Dict[str, Any]:
        """Get face processing configuration"""
        return self.get('face_processing', {})
    
    def get_vectorization_config(self) -> Dict[str, Any]:
        """Get vectorization configuration"""
        return self.get('vectorization', {})
    
    def get_tracking_config(self) -> Dict[str, Any]:
        """Get tracking configuration"""
        return self.get('tracking', {})
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration"""
        return self.get('storage', {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration"""
        return self.get('paths', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.get('logging', {})
    
    def get_video_config(self) -> Dict[str, Any]:
        """Get video configuration"""
        return self.get('video', {})
    
    def get_box_colors_config(self) -> Dict[str, Any]:
        """Get box colors configuration"""
        return self.get('box_colors', {})
    
    def get_data_output_config(self) -> Dict[str, Any]:
        """Get data output configuration"""
        return self.get('data_output', {})

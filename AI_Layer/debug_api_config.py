#!/usr/bin/env python3
"""
Debug script to check API configuration
"""

import sys
from pathlib import Path

# Add API Layer to path
api_layer_path = Path.cwd().parent / 'API_Layer'
sys.path.insert(0, str(api_layer_path.resolve()))

from utils.config_loader import APIConfigLoader

def debug_config():
    """Debug the API configuration"""
    
    print("🔍 Debugging API Configuration...")
    
    # Load config
    config_loader = APIConfigLoader()
    
    print(f"📁 Config path: {config_loader.config_path}")
    print(f"📄 Config exists: {config_loader.config_path.exists()}")
    
    # Check data stream source
    data_stream_source = config_loader.get('data_stream.source', 'csv')
    print(f"📡 Data stream source: '{data_stream_source}'")
    
    # Check full data stream config
    data_stream_config = config_loader.get_data_stream_config()
    print(f"📊 Data stream config: {data_stream_config}")
    
    # Check if it's reading CSV
    if data_stream_source == 'csv':
        csv_path = data_stream_config.get('csv_path', '')
        print(f"📄 CSV path: {csv_path}")
        
        # Check if CSV exists
        full_csv_path = Path(api_layer_path.parent / csv_path)
        print(f"📄 Full CSV path: {full_csv_path}")
        print(f"📄 CSV exists: {full_csv_path.exists()}")
        
        if full_csv_path.exists():
            print(f"📄 CSV size: {full_csv_path.stat().st_size} bytes")
    
    # Check Kafka config
    kafka_config = config_loader.get_kafka_config()
    print(f"📡 Kafka config: {kafka_config}")

if __name__ == '__main__':
    debug_config()

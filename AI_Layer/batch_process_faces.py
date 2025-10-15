#!/usr/bin/env python3
"""
Batch Face Processing CLI
Process a folder of face images for vectorization and database storage
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from core_ai_service import CoreAIService
from utils.config_loader import ConfigLoader
from utils.logger import get_logger


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='Batch process face images')
    parser.add_argument('--folder', '-f', type=str, help='Folder path containing face images')
    parser.add_argument('--config', '-c', type=str, default='config.json', help='Config file path')
    parser.add_argument('--stats', '-s', action='store_true', help='Show processing statistics')
    parser.add_argument('--cleanup', action='store_true', help='Clean up duplicate faces')
    parser.add_argument('--threshold', '-t', type=float, default=0.75, help='Similarity threshold for duplicates')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = get_logger(__name__)
    
    try:
        # Load configuration
        config_loader = ConfigLoader(args.config)
        
        # Initialize AI service
        ai_service = CoreAIService(config_loader)
        
        if args.stats:
            # Show processing statistics
            stats = ai_service.get_batch_processing_stats()
            print("\n=== Batch Processing Statistics ===")
            print(f"Total faces in database: {stats.get('total_faces', 0)}")
            print(f"Total persons: {stats.get('total_persons', 0)}")
            print(f"Average confidence: {stats.get('average_confidence', 0):.3f}")
            print(f"Database size: {stats.get('database_size_mb', 0):.2f} MB")
            return
        
        if args.cleanup:
            # Clean up duplicate faces
            print("Cleaning up duplicate faces...")
            result = ai_service.cleanup_duplicate_faces(args.threshold)
            if result['success']:
                print(f"Removed {result['removed']} duplicate faces")
                print(f"Found {result['duplicates_found']} duplicate pairs")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
            return
        
        if args.folder:
            # Process folder
            if not os.path.exists(args.folder):
                print(f"Error: Folder '{args.folder}' does not exist")
                return
            
            print(f"Processing face images from: {args.folder}")
            result = ai_service.process_face_folder(args.folder)
            
            if result['success']:
                print("\n=== Processing Results ===")
                print(f"Total images: {result['total_images']}")
                print(f"Processed: {result['processed']}")
                print(f"Skipped: {result['skipped']}")
                print(f"Errors: {result['errors']}")
                print(f"New faces: {result['new_faces']}")
                print(f"Duplicate faces: {result['duplicate_faces']}")
                
                # Show details for errors
                if result['errors'] > 0:
                    print("\n=== Error Details ===")
                    for detail in result['details']:
                        if detail['status'] == 'error':
                            print(f"  {detail['file']}: {detail.get('error', 'Unknown error')}")
                
                # Show details for skipped files
                if result['skipped'] > 0:
                    print("\n=== Skipped Files ===")
                    for detail in result['details']:
                        if detail['status'] == 'skipped':
                            print(f"  {detail['file']}: {detail.get('reason', 'Unknown reason')}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
        else:
            print("Please specify a folder to process with --folder or use --stats for statistics")
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
